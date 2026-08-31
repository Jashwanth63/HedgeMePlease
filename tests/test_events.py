import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import alpaca.graph as graph_mod
from alpaca.config import SLEEVE_B, SLEEVE_B_EVENTS
from alpaca.strategy.condor import parse_chain
from alpaca.strategy.events import build_crush_condor, implied_move

from conftest import FakeBroker, synthetic_chain
from test_graph_cycle import make_services, run

ET = ZoneInfo("America/New_York")
DELL_EVENT = next(e for e in SLEEVE_B_EVENTS if e.symbol == "DELL")
CRUSH_NOW = datetime(2026, 9, 1, 15, 35, tzinfo=ET)  # inside DELL entry window


def dell_contracts(iv: float = 0.65):
    chain = synthetic_chain("DELL", spot=140.0, expiry="2026-09-04", iv=iv,
                            t_years=4 / 365, strikes=range(105, 176, 1))
    return parse_chain("DELL", chain)


def test_implied_move_from_straddle():
    move = implied_move(dell_contracts(), "2026-09-04", 140.0)
    assert move is not None
    assert 4.0 < move < 15.0  # roughly 0.8 * S * sigma * sqrt(T)


def test_crush_condor_beyond_move_and_capped():
    proposal, diag = build_crush_condor(DELL_EVENT, dell_contracts(), 140.0, CRUSH_NOW)
    assert proposal is not None, diag
    move = diag["implied_move"]
    short_put = next(l for l in proposal.legs if l.side == "sell" and l.opt_type == "put")
    short_call = next(l for l in proposal.legs if l.side == "sell" and l.opt_type == "call")
    assert short_put.strike <= 140.0 - move
    assert short_call.strike >= 140.0 + move
    assert proposal.max_loss <= SLEEVE_B.crush_max_loss
    assert proposal.position.sleeve == "B"


def test_crush_rejects_without_straddle():
    proposal, diag = build_crush_condor(DELL_EVENT, [], 140.0, CRUSH_NOW)
    assert proposal is None
    assert "implied move" in diag["reject"]


def test_graph_opens_crush_in_window_once(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    monkeypatch.setattr(graph_mod, "now_et", lambda: CRUSH_NOW)
    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    class FillingFakeBroker(FakeBroker):
        async def place_option_order(self, qty, legs, limit_price, client_order_id):
            order = {
                "id": f"ord-{len(self.placed_orders)}",
                "client_order_id": client_order_id,
                "status": "filled",
                "filled_qty": str(qty),
                "filled_avg_price": limit_price,
                "limit_price": limit_price,
                "legs": legs,
            }
            self.placed_orders.append(order)
            return order

    broker = FillingFakeBroker()
    services = make_services(tmp_path, broker=broker, dry_run=False)
    run(services)

    open_pos = [p for p in services.ledger.open_positions() if p.sleeve == "B"]
    assert len(open_pos) == 1, "the DELL crush condor must open inside its window"
    assert open_pos[0].underlying == "DELL"
    assert services.db.get_state("sleeveB:DELL:crush") == "opened"

    orders_before = len(broker.placed_orders)
    run(services)  # second cycle in the same window must not duplicate
    b_positions = [p for p in services.ledger.open_positions() if p.sleeve == "B"]
    assert len(b_positions) == 1
    new_entry_orders = [
        o for o in broker.placed_orders[orders_before:] if "SLB-DELL" in o["client_order_id"]
    ]
    assert not new_entry_orders


def test_crush_time_exit_next_morning(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    class FillingFakeBroker(FakeBroker):
        async def place_option_order(self, qty, legs, limit_price, client_order_id):
            order = {
                "id": f"ord-{len(self.placed_orders)}",
                "client_order_id": client_order_id,
                "status": "filled",
                "filled_qty": str(qty),
                "filled_avg_price": limit_price,
                "limit_price": limit_price,
                "legs": legs,
            }
            self.placed_orders.append(order)
            return order

    broker = FillingFakeBroker()
    services = make_services(tmp_path, broker=broker, dry_run=False)

    monkeypatch.setattr(graph_mod, "now_et", lambda: CRUSH_NOW)
    run(services)
    assert [p for p in services.ledger.open_positions() if p.sleeve == "B"]

    wednesday_morning = datetime(2026, 9, 2, 10, 5, tzinfo=ET)
    monkeypatch.setattr(graph_mod, "now_et", lambda: wednesday_morning)
    run(services)
    b_open = [p for p in services.ledger.open_positions() if p.sleeve == "B"]
    assert not b_open, "crush condor must be covered by the morning time exit"
    closed = [p for p in services.ledger.all_positions() if p.sleeve == "B"][0]
    assert closed.close_reason in ("event_crush_exit", "profit_target_50pct")
