from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.config import RISK, STRAT
from alpaca.risk.engine import check_pre_trade, cluster_delta_dollars, cluster_of
from alpaca.strategy.condor import build_candidates, parse_chain

from conftest import FIXED_NOW, synthetic_chain

ET = ZoneInfo("America/New_York")
SPOTS = {"SPY": 650.0, "QQQ": 560.0, "GLD": 310.0, "TLT": 90.0}


def spy_proposal():
    cands, diag = build_candidates("SPY", parse_chain("SPY", synthetic_chain(iv=0.30)), 650.0, FIXED_NOW)
    assert cands, diag
    return cands[0]


def tlt_proposal():
    chain = synthetic_chain("TLT", spot=90.0, iv=0.30, strikes=range(80, 101, 1))
    cands, diag = build_candidates("TLT", parse_chain("TLT", chain), 90.0, FIXED_NOW)
    assert cands, diag
    return cands[0]


def check(open_positions, proposal):
    return check_pre_trade(
        open_positions, proposal, 100_000.0, 100_000.0, 100_000.0, False, SPOTS, FIXED_NOW
    )


def test_every_underlying_has_cluster_and_wing_floor():
    for und in STRAT.underlyings:
        assert und in STRAT.clusters
        assert STRAT.wing_width_floors.get(und, STRAT.min_wing_width) > 0
        assert STRAT.clusters[und] in STRAT.cluster_delta_caps


def test_tlt_builds_with_narrow_wings():
    p = tlt_proposal()
    assert p.width <= 2.0, f"TLT wings must be narrow, got {p.width}"
    assert p.max_loss <= RISK.per_trade_max_loss
    assert p.qty >= 1


def test_cluster_budget_blocks_third_equity_but_allows_rates():
    fillers = []
    for _ in range(2):
        q = spy_proposal()
        q.position.max_loss = 380.0
        q.position.status = "open"
        fillers.append(q.position)
    # equity cluster now holds 760 of its 750 cap with the next equity trade
    equity_verdict = check(fillers, spy_proposal())
    assert not equity_verdict.approved
    assert any("equity cluster budget" in r for r in equity_verdict.reasons)

    rates_verdict = check(fillers, tlt_proposal())
    assert rates_verdict.approved, rates_verdict.reasons


def test_cluster_delta_separates_families():
    p = tlt_proposal()
    deltas = cluster_delta_dollars([p.position], SPOTS)
    assert "rates" in deltas
    assert "equity" not in deltas
    assert cluster_of("GLD") == "metals"
