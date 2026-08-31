from alpacha.risk.ledger import Leg, Position
from alpacha.risk.stress import position_pnl_under_shock, worst_cell

from conftest import FIXED_NOW


def make_condor(qty: int = 1) -> Position:
    legs = [
        Leg("SPY260903P00640000", "sell", 1, 640.0, "put", "2026-09-03", 0.14, -0.20),
        Leg("SPY260903P00635000", "buy", 1, 635.0, "put", "2026-09-03", 0.15, -0.14),
        Leg("SPY260903C00662000", "sell", 1, 662.0, "call", "2026-09-03", 0.12, 0.20),
        Leg("SPY260903C00667000", "buy", 1, 667.0, "call", "2026-09-03", 0.11, 0.14),
    ]
    return Position(
        position_id="T1", sleeve="A", underlying="SPY", structure="iron_condor",
        legs=legs, qty=qty, credit=1.10, width=5.0, max_loss=(5.0 - 1.10) * 100 * qty,
        client_order_id="T1", opened_at=FIXED_NOW.isoformat(),
    )


def test_zero_shock_is_zero_pnl():
    assert abs(position_pnl_under_shock(make_condor(), 650.0, 0.0, FIXED_NOW)) < 1e-9


def test_both_tails_lose_but_bounded():
    pos = make_condor()
    down = position_pnl_under_shock(pos, 650.0, -0.05, FIXED_NOW)
    up = position_pnl_under_shock(pos, 650.0, 0.05, FIXED_NOW)
    assert down < 0 and up < 0
    assert down >= -5.0 * 100 and up >= -5.0 * 100


def test_worst_cell_scales_with_qty():
    w1, _ = worst_cell([make_condor(1)], {"SPY": 650.0}, FIXED_NOW)
    w2, _ = worst_cell([make_condor(2)], {"SPY": 650.0}, FIXED_NOW)
    assert w2 < w1 < 0
    assert abs(w2 - 2 * w1) < 1.0


def test_worst_cell_missing_spot_contributes_zero():
    worst, _ = worst_cell([make_condor()], {}, FIXED_NOW)
    assert worst == 0.0
