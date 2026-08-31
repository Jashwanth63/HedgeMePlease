from alpaca.config import RISK
from alpaca.risk.engine import AccountAction, check_pre_trade, evaluate_account
from alpaca.strategy.condor import build_candidates, parse_chain

from conftest import FIXED_NOW, synthetic_chain

SPOTS = {"SPY": 650.0, "QQQ": 560.0}


def make_proposal():
    cands, diag = build_candidates("SPY", parse_chain("SPY", synthetic_chain(iv=0.30)), 650.0, FIXED_NOW)
    assert cands, diag
    return cands[0]


def check(open_positions, proposal, equity=100_000.0, hwm=100_000.0, anchor=100_000.0, halted=False):
    return check_pre_trade(open_positions, proposal, equity, hwm, anchor, halted, SPOTS, FIXED_NOW)


def test_happy_path_approved():
    verdict = check([], make_proposal())
    assert verdict.approved, verdict.reasons
    assert verdict.size_factor == 1.0


def test_per_trade_cap_rejects():
    p = make_proposal()
    p.max_loss = RISK.per_trade_max_loss + 1
    verdict = check([], p)
    assert not verdict.approved
    assert any("per-trade" in r for r in verdict.reasons)


def test_sleeve_budget_rejects():
    fillers = []
    for _ in range(5):
        q = make_proposal()
        q.position.max_loss = 500.0
        q.position.status = "open"
        fillers.append(q.position)
    verdict = check(fillers, make_proposal())
    assert not verdict.approved


def test_underlying_concentration_rejects():
    a, b, c = make_proposal(), make_proposal(), make_proposal()
    a.position.status = b.position.status = c.position.status = "open"
    verdict = check([a.position, b.position, c.position], make_proposal())
    assert not verdict.approved
    assert any("already has" in r for r in verdict.reasons)


def test_dte_floor_rejects():
    p = make_proposal()
    p.dte = 0
    verdict = check([], p)
    assert not verdict.approved


def test_kill_and_daily_ladder():
    assert evaluate_account(96_500.0, 100_000.0, 100_000.0, False) == AccountAction.KILL
    assert evaluate_account(96_600.0, 100_000.0, 100_000.0, False) != AccountAction.KILL
    assert evaluate_account(98_950.0, 100_000.0, 100_000.0, False) == AccountAction.NO_NEW
    assert evaluate_account(98_400.0, 100_000.0, 100_000.0, False) == AccountAction.REDUCE_ONLY
    assert evaluate_account(99_600.0, 100_000.0, 100_000.0, False) == AccountAction.OK
    assert evaluate_account(100_000.0, 100_000.0, 100_000.0, True) == AccountAction.KILL


def test_derisk_ladder_halves_size():
    verdict = check([], make_proposal(), equity=98_100.0, anchor=98_100.0)
    if verdict.approved:
        assert verdict.size_factor == 0.5


def test_derisk_freeze_rejects():
    verdict = check([], make_proposal(), equity=97_200.0, anchor=97_200.0)
    assert not verdict.approved
    assert any("de-risk" in r for r in verdict.reasons)
