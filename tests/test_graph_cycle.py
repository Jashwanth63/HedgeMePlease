"""Full offline cycle: the entire LangGraph pipeline runs against the fake
broker with synthetic data, no network and no keys. The dry run must walk
risk check, manage, gather, regime, gates, build, propose, veto, risk gate,
and reach a dry-run execution, journaling everything to SQLite.
"""

import asyncio

import alpaca.graph as graph_mod
from alpaca.data.db import Db
from alpaca.graph import Services, run_cycle
from alpaca.risk.ledger import Ledger

from conftest import FIXED_NOW, FakeBroker


def make_services(tmp_path, broker=None, dry_run=True) -> Services:
    db = Db(tmp_path / "cycle.db")
    return Services(broker=broker or FakeBroker(), db=db, ledger=Ledger(db), dry_run=dry_run)


def run(services):
    return asyncio.run(run_cycle(services))


def test_full_dry_cycle_reaches_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "now_et", lambda: FIXED_NOW)
    services = make_services(tmp_path)
    result = run(services)

    assert result.get("action") == "ok"
    assert result.get("passing"), f"gates failed: {result.get('gates')}"
    assert result.get("executed", {}).get("dry_run") is True
    events = [m["event"] for m in services.db.recent_memos(100)]
    for expected in ("cycle_start", "gates", "candidates", "proposal_chosen",
                     "news_veto", "risk_verdict", "dry_run_would_open", "cycle_end"):
        assert expected in events, f"missing {expected} in {events}"
    # agent-unavailable defaults must never be cached as the day's regime view
    assert services.db.get_state("regime:2026-08-31") is None


def test_kill_switch_path_halts_and_flattens(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "now_et", lambda: FIXED_NOW)
    services = make_services(tmp_path, broker=FakeBroker(equity=96_000.0))
    services.ledger.update_equity(100_000.0)  # establish the peak first
    result = run(services)

    assert result.get("skip") == "kill switch"
    assert services.ledger.halted
    events = [m["event"] for m in services.db.recent_memos(50)]
    assert "KILL_SWITCH" in events


def test_market_closed_skips_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "now_et", lambda: FIXED_NOW)
    services = make_services(tmp_path, broker=FakeBroker(is_open=False))
    result = run(services)
    assert result.get("skip") == "market closed"
    assert "executed" not in result


def test_live_cycle_places_order_via_fake_broker(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    monkeypatch.setattr(graph_mod, "now_et", lambda: FIXED_NOW)
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
    result = run(services)

    assert result.get("executed", {}).get("opened") is True
    assert broker.placed_orders, "no order reached the broker"
    order = broker.placed_orders[0]
    assert len(order["legs"]) == 4
    assert float(order["limit_price"]) < 0, "opening condor must be a net credit"
    open_pos = services.ledger.open_positions()
    assert len(open_pos) == 1
    assert open_pos[0].status == "open"

    stored = services.db.conn.execute(
        "SELECT entry_context FROM trades WHERE trade_id=?", (open_pos[0].position_id,)
    ).fetchone()
    assert stored["entry_context"], "entry context must be stored with the fill"

    second = run(services)
    assert str(second.get("skip", "")).startswith("entry spacing"), second.get("skip")

    services.db.set_state("last_entry_at", "2026-08-31T09:00:00-04:00")  # spacing elapsed
    third = run(services)
    gates = third.get("gates") or {}
    spy_fails = (gates.get("SPY") or {}).get("failed", [])
    assert any("underlying_cooldown" in f for f in spy_fails), gates
