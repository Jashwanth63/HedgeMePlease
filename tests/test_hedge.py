import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import alpaca.graph as graph_mod
from alpaca.config import SLEEVE_C
from alpaca.strategy.condor import parse_chain
from alpaca.strategy.hedge import build_hedge_puts

from conftest import FIXED_NOW, FakeBroker, synthetic_chain
from test_graph_cycle import make_services, run
from test_stress import make_condor

ET = ZoneInfo("America/New_York")


def hedge_contracts():
    chain = synthetic_chain("SPY", 650.0, "2026-09-09", iv=0.15,
                            t_years=9 / 365, strikes=range(570, 655, 5))
    return parse_chain("SPY", chain)


def test_builder_picks_otm_put_within_budget():
    proposal, diag = build_hedge_puts(hedge_contracts(), 650.0, FIXED_NOW)
    assert proposal is not None, diag
    leg = proposal.legs[0]
    assert leg.opt_type == "put" and leg.side == "buy"
    assert 650.0 * 0.95 <= leg.strike <= 650.0 * 0.97
    assert proposal.max_loss <= SLEEVE_C.budget + 25  # qty rounding tolerance
    assert proposal.position.sleeve == "C"
    assert proposal.position.structure == "hedge_puts"


def test_builder_rejects_without_band():
    proposal, diag = build_hedge_puts([], 650.0, FIXED_NOW)
    assert proposal is None


def test_hedge_view_parser():
    from alpaca.agents.desk import parse_hedge_view

    assert parse_hedge_view('{"buy_now": true, "note": "AVGO tonight"}')["buy_now"] is True
    assert parse_hedge_view('{"buy_now": "yes"}') is None
    assert parse_hedge_view("nope") is None


def _filling_broker():
    class FillingFakeBroker(FakeBroker):
        async def place_option_order(self, qty, legs, limit_price, client_order_id):
            order = {"id": f"ord-{len(self.placed_orders)}",
                     "client_order_id": client_order_id, "status": "filled",
                     "filled_qty": str(qty), "filled_avg_price": limit_price,
                     "limit_price": limit_price, "legs": legs}
            self.placed_orders.append(order)
            return order
    return FillingFakeBroker()


def test_agent_buy_now_opens_hedge_once(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    monkeypatch.setattr(graph_mod, "now_et", lambda: FIXED_NOW)
    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    async def eager_view(context, memo):
        return {"buy_now": True, "note": "book deployed, event nights ahead, vol cheap"}

    monkeypatch.setattr(graph_mod.desk, "hedge_view", eager_view)

    services = make_services(tmp_path, broker=_filling_broker(), dry_run=False)
    run(services)
    hedges = [p for p in services.ledger.open_positions() if p.sleeve == "C"]
    assert len(hedges) == 1, "hedge must open when the analyst says buy"
    assert services.db.get_state("sleeveC:bought") == "opened"

    run(services)
    assert len([p for p in services.ledger.open_positions() if p.sleeve == "C"]) == 1


def test_backstop_buys_with_loaded_book(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    wednesday = datetime(2026, 9, 2, 10, 5, tzinfo=ET)
    monkeypatch.setattr(graph_mod, "now_et", lambda: wednesday)
    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    services = make_services(tmp_path, broker=_filling_broker(), dry_run=False)
    for i in range(2):
        pos = make_condor()
        pos.position_id = f"SLA-SEED-{i}"
        pos.client_order_id = pos.position_id
        pos.status = "open"
        pos.max_loss = 350.0
        services.ledger.add(pos)

    run(services)  # hedge analyst offline in tests: only the backstop can buy
    hedges = [p for p in services.ledger.open_positions() if p.sleeve == "C"]
    assert len(hedges) == 1, "backstop must force the hedge before the event night"
    events = [m["event"] for m in services.db.recent_memos(60)]
    assert "hedge_backstop_buy" in events
