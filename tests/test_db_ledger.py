from alpacha.data.db import Db
from alpacha.risk.ledger import Ledger, Leg, Position

from conftest import FIXED_NOW


def make_position(pid: str = "SLA-TEST-1") -> Position:
    legs = [
        Leg("SPY260903P00640000", "sell", 1, 640.0, "put", "2026-09-03", 0.14, -0.20),
        Leg("SPY260903P00635000", "buy", 1, 635.0, "put", "2026-09-03", 0.15, -0.14),
    ]
    return Position(
        position_id=pid, sleeve="A", underlying="SPY", structure="iron_condor",
        legs=legs, qty=1, credit=1.10, width=5.0, max_loss=390.0,
        client_order_id=pid, opened_at=FIXED_NOW.isoformat(),
    )


def test_trade_roundtrip(tmp_path):
    db = Db(tmp_path / "t.db")
    ledger = Ledger(db)
    pos = make_position()
    ledger.add(pos)
    loaded = ledger.open_positions()
    assert len(loaded) == 1
    got = loaded[0]
    assert got.position_id == pos.position_id
    assert got.legs[0].symbol == "SPY260903P00640000"
    assert got.legs[0].entry_delta == -0.20

    got.status = "closed"
    got.realized_pnl = 55.0
    got.close_reason = "profit_target_50pct"
    ledger.update(got)
    assert ledger.open_positions() == []
    assert ledger.all_positions()[0].realized_pnl == 55.0


def test_equity_anchors_and_halt(tmp_path):
    db = Db(tmp_path / "t.db")
    ledger = Ledger(db)
    ledger.update_equity(100_000.0)
    assert ledger.hwm == 100_000.0
    ledger.update_equity(101_000.0)
    assert ledger.hwm == 101_000.0
    ledger.update_equity(100_500.0)
    assert ledger.hwm == 101_000.0
    assert not ledger.halted
    ledger.halt("test")
    assert ledger.halted
    ledger.unhalt()
    assert not ledger.halted


def test_memos_and_state(tmp_path):
    db = Db(tmp_path / "t.db")
    db.memo("gates", {"underlying": "SPY", "pass": True})
    db.memo("cycle_end", {"open_positions": 0})
    rows = db.recent_memos(5)
    assert rows[0]["event"] == "cycle_end"
    db.set_state("regime:2026-08-31", {"stance": "normal"})
    assert db.get_state("regime:2026-08-31")["stance"] == "normal"
    assert db.get_state("missing", "dflt") == "dflt"


def test_forecast_vs_realized_join(tmp_path):
    from alpacha.model.volutils import DayStats

    db = Db(tmp_path / "t.db")
    db.conn.execute(
        "INSERT INTO forecasts (ts, symbol, horizon, rv_forecast, method) VALUES "
        "('2026-08-27T10:00:00', 'SPY', 2, 0.10, 'har')"
    )
    db.upsert_rv_daily("SPY", [DayStats("2026-08-28", 0.16, 0.14, 0.01, -0.01)])
    pairs = db.forecast_vs_realized("SPY")
    assert pairs == [(0.10, 0.16)]
