import math
import random

from alpacha.model.har import (
    MIN_TRAIN,
    best_forecast,
    fallback_forecast,
    fit,
    should_demote,
    walk_forward,
)
from alpacha.model.volutils import DayStats


def synthetic_stats(n: int = 300, seed: int = 7) -> list[DayStats]:
    rng = random.Random(seed)
    level = math.log(0.12)
    out = []
    for i in range(n):
        level = 0.97 * level + 0.03 * math.log(0.12) + rng.gauss(0, 0.08)
        rv = math.exp(level)
        out.append(DayStats(day=f"d{i}", rv=rv, bv=rv * 0.9, jump=rv * 0.05, ret=rng.gauss(0, 0.008)))
    return out


def test_fit_and_forecast_reasonable():
    stats = synthetic_stats()
    model = fit(stats)
    assert 0.01 < model.forecast(stats) < 1.0


def test_fit_requires_history():
    try:
        fit(synthetic_stats(MIN_TRAIN - 1))
        assert False
    except ValueError:
        pass


def test_best_forecast_fallback_on_short_history():
    stats = synthetic_stats(30)
    value, method = best_forecast(stats)
    assert method == "fallback20"
    assert abs(value - fallback_forecast(stats)) < 1e-12


def test_best_forecast_demoted_uses_fallback():
    stats = synthetic_stats(200)
    value, method = best_forecast(stats, demoted=True)
    assert method == "fallback20"


def test_walk_forward_har_close_to_baselines():
    report = walk_forward(synthetic_stats(400), min_train=150)
    assert report.n_forecasts > 100
    assert report.rmse_har < report.rmse_lag1 * 1.2


def test_demotion_on_consecutive_misses():
    assert should_demote([(0.10, 0.16), (0.10, 0.17)])
    assert not should_demote([(0.10, 0.11), (0.10, 0.16)])
    assert not should_demote([(0.10, 0.16)])


def test_demotion_on_chronic_error():
    pairs = [(0.10, 0.25)] * 5
    assert should_demote(pairs)
    calm = [(0.10, 0.105)] * 5
    assert not should_demote(calm)
