from alpacha.config import CLAMPS, STRAT
from alpacha.agents.desk import (
    RegimeView,
    extract_json,
    parse_choice,
    parse_regime,
    parse_veto,
)


def test_extract_json_tolerant():
    assert extract_json('noise {"a": 1} trailing')["a"] == 1
    assert extract_json("no json here") is None
    assert extract_json("") is None


def test_parse_regime_clamps():
    raw = '{"stance": "cautious", "edge_ratio": 9.9, "delta_target": 0.01, "size_factor": 0.1, "note": "x"}'
    view = parse_regime(raw)
    assert view.stance == "cautious"
    assert view.edge_ratio == CLAMPS.edge_ratio[1]
    assert view.delta_target == CLAMPS.delta_target[0]
    assert view.size_factor == CLAMPS.size_factor[0]


def test_parse_regime_bad_stance_defaults():
    view = parse_regime('{"stance": "yolo", "edge_ratio": 1.2}')
    assert view.stance == "normal"


def test_parse_regime_garbage_none():
    assert parse_regime("panic!") is None


def test_default_regime_matches_config():
    view = RegimeView()
    assert view.edge_ratio == STRAT.iv_over_rv_min_ratio
    assert view.delta_target == STRAT.short_delta_target
    assert view.size_factor == 1.0


def test_parse_choice_bounds():
    assert parse_choice('{"choice": 1}', 3) == 1
    assert parse_choice('{"choice": 7}', 3) == 0
    assert parse_choice("garbage", 3) == 0


def test_parse_veto():
    assert parse_veto('{"veto": true, "reason": "AVGO"}') == {"veto": True, "reason": "AVGO"}
    assert parse_veto('{"veto": "yes"}') is None
    assert parse_veto("nah") is None
