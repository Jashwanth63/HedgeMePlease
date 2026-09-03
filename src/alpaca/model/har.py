"""Enhanced HAR-RV: Corsi's daily/weekly/monthly structure plus a leverage
term (negative returns raise future vol) and a jump component from bipower
variation. Fit in logs with numpy least squares. Includes the trailing-mean
fallback, a walk-forward validator against dumb baselines, and the demotion
rule from plan.md: the model must keep beating the fallback or it steps aside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .volutils import DayStats

MIN_TRAIN = 60


@dataclass
class HarModel:
    coef: np.ndarray  # intercept, logRV_1, mean5, mean22, leverage, jump
    horizon: int

    def forecast(self, stats: list[DayStats]) -> float:
        x = _features_from_tail(stats)
        log_pred = float(np.dot(np.concatenate(([1.0], x)), self.coef))
        return math.exp(log_pred)


def fallback_forecast(stats: list[DayStats], window: int = 20) -> float:
    tail = [s.rv for s in stats[-window:]]
    return sum(tail) / len(tail)


def _features_from_tail(stats: list[DayStats]) -> np.ndarray:
    logs = np.log(np.asarray([s.rv for s in stats[-22:]], dtype=float))
    last = stats[-1]
    return np.array(
        [logs[-1], logs[-5:].mean(), logs.mean(), min(0.0, last.ret), last.jump]
    )


def _design(stats: list[DayStats], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    logs = np.log(np.asarray([s.rv for s in stats], dtype=float))
    rows, targets = [], []
    for i in range(22, len(logs) - horizon + 1):
        prev = stats[i - 1]
        rows.append(
            [
                1.0,
                logs[i - 1],
                logs[i - 5 : i].mean(),
                logs[i - 22 : i].mean(),
                min(0.0, prev.ret),
                prev.jump,
            ]
        )
        targets.append(logs[i : i + horizon].mean())
    return np.asarray(rows), np.asarray(targets)


def fit(stats: list[DayStats], horizon: int = 2) -> HarModel:
    if len(stats) < MIN_TRAIN:
        raise ValueError(f"need at least {MIN_TRAIN} daily points, got {len(stats)}")
    x, y = _design(stats, horizon)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return HarModel(coef=coef, horizon=horizon)


def best_forecast(stats: list[DayStats], horizon: int = 2, demoted: bool = False) -> tuple[float, str]:
    """Returns (annualized rv forecast, method). Demoted or short history uses the fallback."""
    if len(stats) < 25:
        raise ValueError(f"need at least 25 daily points, got {len(stats)}")
    if demoted or len(stats) < MIN_TRAIN:
        return fallback_forecast(stats), "fallback20"
    try:
        model = fit(stats, horizon)
        value = model.forecast(stats)
        if not math.isfinite(value) or value <= 0:
            return fallback_forecast(stats), "fallback20"
        return value, "har"
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return fallback_forecast(stats), "fallback20"


@dataclass
class WalkForwardReport:
    n_forecasts: int
    rmse_har: float
    rmse_lag1: float
    rmse_mean20: float

    @property
    def har_beats_fallback(self) -> bool:
        return self.rmse_har <= self.rmse_mean20


def walk_forward(stats: list[DayStats], horizon: int = 2, min_train: int = 120) -> WalkForwardReport:
    errs_har, errs_lag1, errs_mean20 = [], [], []
    for split in range(min_train, len(stats) - horizon):
        train = stats[:split]
        realized = [s.rv for s in stats[split : split + horizon]]
        target = math.log(sum(realized) / len(realized))
        try:
            model = fit(train, horizon)
        except ValueError:
            continue
        errs_har.append(math.log(max(model.forecast(train), 1e-9)) - target)
        errs_lag1.append(math.log(train[-1].rv) - target)
        errs_mean20.append(math.log(fallback_forecast(train)) - target)

    def rmse(errs: list[float]) -> float:
        return math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else float("nan")

    return WalkForwardReport(
        n_forecasts=len(errs_har),
        rmse_har=rmse(errs_har),
        rmse_lag1=rmse(errs_lag1),
        rmse_mean20=rmse(errs_mean20),
    )


def should_demote(pairs: list[tuple[float, float]]) -> bool:
    """Demotion rule over recent (forecast, realized) pairs, newest last.

    Trip if realized exceeded forecast by more than 50 percent on the two most
    recent days, or if the trailing squared log error says the model has been
    worse than useless lately (realized wildly off forecast on average).
    """
    if len(pairs) >= 2:
        recent = pairs[-2:]
        if all(r > 1.5 * f > 0 for f, r in recent):
            return True
    if len(pairs) >= 5:
        errs = [math.log(r / f) for f, r in pairs[-5:] if f > 0 and r > 0]
        if errs and math.sqrt(sum(e * e for e in errs) / len(errs)) > 0.6:
            return True
    return False
