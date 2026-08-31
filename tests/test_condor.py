from alpacha.config import RISK
from alpacha.strategy.condor import build_candidates, parse_chain, pick_near_expiry

from conftest import FIXED_NOW, synthetic_chain


def contracts(iv: float = 0.30):
    return parse_chain("SPY", synthetic_chain(iv=iv))


def test_parse_chain_occ():
    cs = contracts()
    sample = next(c for c in cs if c.opt_type == "put" and c.strike == 640.0)
    assert sample.expiry == "2026-09-03"
    assert sample.bid > 0 and sample.ask > sample.bid


def test_pick_near_expiry_prefers_contest_thursday():
    assert pick_near_expiry(contracts(), FIXED_NOW) == "2026-09-03"


def test_build_candidates_viable_at_high_iv():
    cands, diag = build_candidates("SPY", contracts(iv=0.30), 650.0, FIXED_NOW)
    assert cands, diag
    best = cands[0]
    assert best.structure == "iron_condor"
    assert len(best.legs) == 4
    assert best.credit > 0
    assert best.max_loss <= RISK.per_trade_max_loss
    shorts = [l for l in best.legs if l.side == "sell"]
    wings = [l for l in best.legs if l.side == "buy"]
    assert len(shorts) == 2 and len(wings) == 2
    for s in shorts:
        assert 0.12 <= abs(s.entry_delta) <= 0.28


def test_menu_sorted_by_credit_per_width():
    cands, _ = build_candidates("SPY", contracts(iv=0.30), 650.0, FIXED_NOW)
    ratios = [c.credit / c.width for c in cands]
    assert ratios == sorted(ratios, reverse=True)


def test_thin_iv_rejected():
    cands, diag = build_candidates("SPY", contracts(iv=0.05), 650.0, FIXED_NOW)
    assert not cands
    assert diag["rejects"]


def test_qty_respects_pref_loss():
    cands, _ = build_candidates("SPY", contracts(iv=0.30), 650.0, FIXED_NOW)
    for c in cands:
        unit = (c.width - c.credit) * 100
        assert c.qty >= 1
        if c.qty > 1:
            assert unit * c.qty <= RISK.per_trade_pref_loss + unit
