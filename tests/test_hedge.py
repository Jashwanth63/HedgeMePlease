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


def _seed_hedge(services):
    from alpaca.risk.ledger import Leg, Position

    legs = [Leg("SPY260911P00738000", "buy", 1, 738.0, "put", "2026-09-11", 0.15, -0.10)]
    pos = Position(
        position_id="SLC-SEED", sleeve="C", underlying="SPY",
        structure="hedge_puts", legs=legs, qty=1, credit=1.22, width=0.0,
        max_loss=122.0, client_order_id="SLC-SEED",
        opened_at="2026-09-01T15:23:29-04:00",
    )
    pos.status = "open"
    services.ledger.add(pos)
    return pos


def test_hedge_retires_when_book_flat_and_events_done(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    thursday = datetime(2026, 9, 3, 10, 5, tzinfo=ET)  # after AVGO crush_exit_by
    monkeypatch.setattr(graph_mod, "now_et", lambda: thursday)
    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )
    services = make_services(tmp_path, broker=_filling_broker(), dry_run=False)
    _seed_hedge(services)

    run(services)
    assert not [p for p in services.ledger.open_positions() if p.sleeve == "C"]
    closed = [p for p in services.ledger.all_positions() if p.sleeve == "C"][0]
    assert closed.close_reason == "hedge_retired_book_flat"


def test_hedge_stays_while_book_has_positions_or_events_pending(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    # Wednesday morning: book flat but AVGO night still ahead — hold
    wednesday = datetime(2026, 9, 2, 10, 35, tzinfo=ET)
    monkeypatch.setattr(graph_mod, "now_et", lambda: wednesday)
    (tmp_path / "a").mkdir()
    services = make_services(tmp_path / "a", broker=_filling_broker(), dry_run=False)
    _seed_hedge(services)
    run(services)
    assert [p for p in services.ledger.open_positions() if p.sleeve == "C"]

    # Thursday after events, but another position still open — hold
    thursday = datetime(2026, 9, 3, 10, 5, tzinfo=ET)
    monkeypatch.setattr(graph_mod, "now_et", lambda: thursday)
    (tmp_path / "b").mkdir()
    services2 = make_services(tmp_path / "b", broker=_filling_broker(), dry_run=False)
    _seed_hedge(services2)
    seed = make_condor()
    seed.position_id = seed.client_order_id = "SLA-STILL-OPEN"
    seed.status = "open"
    services2.ledger.add(seed)
    run(services2)
    assert [p for p in services2.ledger.open_positions() if p.sleeve == "C"]
