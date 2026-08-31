from datetime import datetime
from zoneinfo import ZoneInfo

from alpacha.model.volutils import daily_stats, expected_move

from conftest import FIXED_NOW, synthetic_bars

ET = ZoneInfo("America/New_York")


def test_daily_stats_reasonable():
    stats = daily_stats(synthetic_bars(10), FIXED_NOW)
    assert len(stats) == 10
    for s in stats:
        assert 0.01 < s.rv < 0.60
        assert s.bv >= 0
        assert s.jump >= 0


def test_partial_and_current_days_excluded():
    bars = synthetic_bars(3)
    today = FIXED_NOW.date().isoformat()
    for i in range(20):
        hh = 9 + (30 + 5 * i) // 60
        mm = (30 + 5 * i) % 60
        bars.append({"t": f"{today}T{hh:02d}:{mm:02d}:00-04:00", "c": 650.0})
    stats = daily_stats(bars, FIXED_NOW)
    assert len(stats) == 3
    assert all(s.day != today for s in stats)


def test_short_days_excluded():
    bars = synthetic_bars(2)
    for i in range(30):  # 30 bars: below the 60 bar completeness floor
        hh = 9 + (30 + 5 * i) // 60
        mm = (30 + 5 * i) % 60
        bars.append({"t": f"2026-08-24T{hh:02d}:{mm:02d}:00-04:00", "c": 650.0})
    stats = daily_stats(bars, FIXED_NOW)
    assert all(s.day != "2026-08-24" for s in stats)


def test_expected_move():
    em = expected_move(769.0, 0.11, 3)
    assert 7.0 < em < 9.0
