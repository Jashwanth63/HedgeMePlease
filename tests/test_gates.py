from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.strategy.gates import evaluate_gates

from conftest import FIXED_NOW

ET = ZoneInfo("America/New_York")


def test_all_pass_in_calm_contango():
    report = evaluate_gates(0.14, 0.16, 0.10, now=FIXED_NOW)
    assert report.all_pass, report.failed()


def test_backwardation_blocks():
    report = evaluate_gates(0.22, 0.16, 0.10, now=FIXED_NOW)
    assert not report.contango and not report.all_pass


def test_thin_edge_blocks():
    report = evaluate_gates(0.11, 0.13, 0.10, now=FIXED_NOW)
    assert not report.iv_over_rv


def test_edge_ratio_parameter_is_respected():
    loose = evaluate_gates(0.115, 0.13, 0.10, now=FIXED_NOW, edge_ratio=1.10)
    tight = evaluate_gates(0.115, 0.13, 0.10, now=FIXED_NOW, edge_ratio=1.20)
    assert loose.iv_over_rv and not tight.iv_over_rv


def test_macro_blackout_blocks():
    tuesday_morning = datetime(2026, 9, 1, 8, 30, tzinfo=ET)  # ISM+JOLTS at 10:00
    report = evaluate_gates(0.14, 0.16, 0.10, now=tuesday_morning)
    assert not report.macro_clear
    assert "ISM" in report.details["blocking_events"][0]


def test_avgo_blackout_wednesday_afternoon():
    wed_after_lunch = datetime(2026, 9, 2, 14, 30, tzinfo=ET)  # AVGO at 16:05
    report = evaluate_gates(0.14, 0.16, 0.10, now=wed_after_lunch)
    assert not report.macro_clear


def test_windows_block_weekend_and_thursday():
    saturday = datetime(2026, 8, 29, 11, 0, tzinfo=ET)
    thursday = datetime(2026, 9, 3, 11, 0, tzinfo=ET)
    assert not evaluate_gates(0.14, 0.16, 0.10, now=saturday).entry_window
    assert not evaluate_gates(0.14, 0.16, 0.10, now=thursday).entry_window


def test_missing_data_fails_closed():
    report = evaluate_gates(None, 0.16, 0.10, now=FIXED_NOW)
    assert not report.contango and not report.iv_over_rv
