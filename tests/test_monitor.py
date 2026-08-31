from alpaca.data.db import Db
from alpaca.monitor import render
from alpaca.risk.ledger import Ledger

from test_db_ledger import make_position


def test_render_shows_account_trades_and_memos(tmp_path):
    db = Db(tmp_path / "m.db")
    db.record_risk_snapshot(97_863.64, 100_000.0, 0.0214, "ok")
    db.record_forecast("SPY", 2, 0.093, "har")
    db.memo("gates", {"underlying": "SPY", "pass": True, "iv_rv_ratio": 1.364})
    ledger = Ledger(db)
    pos = make_position("SLA-MON-1")
    ledger.add(pos, entry_context={"gates": {"iv_rv_ratio": 1.36}, "proposer_why": "best ratio"})

    ledger.mark_position("SLA-MON-1", 0.55, 55.0)

    page = render(db)
    assert "97,863.64" in page
    assert "SLA-MON-1" in page
    assert "best ratio" in page
    assert "gates" in page
    assert "9.3%" in page
    assert "unrealized" in page and "+55" in page


def test_render_empty_db_does_not_crash(tmp_path):
    page = render(Db(tmp_path / "e.db"))
    assert "no snapshots yet" in page
