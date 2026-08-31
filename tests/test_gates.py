from datetime import datetime, timezone
from alpacha.config import Settings
from alpacha.data.macro_calendar import MacroCalendar
from alpacha.strategy.gates import StrategyGateManager


def test_macro_calendar_gate():
    settings = Settings.load(config_path="config/settings.yaml")
    cal = MacroCalendar("config/macro_calendar.json")
    gate_mgr = StrategyGateManager(settings, cal)

    # FOMC event on 2025-01-29 14:00:00 EST (19:00:00 UTC)
    # Test time 30 mins prior -> Should FAIL macro gate
    event_time = datetime.fromisoformat("2025-01-29T18:30:00+00:00")
    passed, reason = gate_mgr.evaluate_macro_gate(target_time=event_time)
    assert passed is False
    assert reason is not None

    # Test time 10 days away -> Should PASS macro gate
    calm_time = datetime.fromisoformat("2025-01-20T15:00:00+00:00")
    passed, reason = gate_mgr.evaluate_macro_gate(target_time=calm_time)
    assert passed is True
    assert reason is None


def test_edge_gate():
    settings = Settings.load(config_path="config/settings.yaml")
    cal = MacroCalendar("config/macro_calendar.json")
    gate_mgr = StrategyGateManager(settings, cal)

    # IV = 24%, RV = 20% -> 1.20x -> PASS
    passed, _ = gate_mgr.evaluate_edge_gate(market_iv=0.24, forecasted_rv=0.20)
    assert passed is True

    # IV = 21%, RV = 20% -> 1.05x < 1.20x -> FAIL
    passed, reason = gate_mgr.evaluate_edge_gate(market_iv=0.21, forecasted_rv=0.20)
    assert passed is False
    assert "Insufficient IV/RV edge" in reason


def test_contango_gate():
    settings = Settings.load(config_path="config/settings.yaml")
    cal = MacroCalendar("config/macro_calendar.json")
    gate_mgr = StrategyGateManager(settings, cal)

    # Near IV = 18%, Next IV = 20% -> Contango -> PASS
    passed, _ = gate_mgr.evaluate_contango_gate(near_atm_iv=0.18, next_atm_iv=0.20)
    assert passed is True

    # Near IV = 35%, Next IV = 20% -> Severe backwardation spike -> FAIL
    passed, reason = gate_mgr.evaluate_contango_gate(near_atm_iv=0.35, next_atm_iv=0.20)
    assert passed is False
    assert "backwardation" in reason
