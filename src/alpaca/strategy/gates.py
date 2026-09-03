"""Entry gates. Pure functions; fail closed on missing data.

The edge ratio is a parameter because the regime agent may tune it, but only
inside config.CLAMPS — the caller clamps before passing it here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..config import EARNINGS_EVENTS, ENTRY_WINDOWS, MACRO_EVENTS, STRAT, now_et


@dataclass
class GateReport:
    entry_window: bool = False
    macro_clear: bool = False
    contango: bool = False
    iv_over_rv: bool = False
    details: dict = field(default_factory=dict)

    @property
    def all_pass(self) -> bool:
        return self.entry_window and self.macro_clear and self.contango and self.iv_over_rv

    def failed(self) -> list[str]:
        names = ("entry_window", "macro_clear", "contango", "iv_over_rv")
        return [n for n in names if not getattr(self, n)]


def evaluate_gates(
    near_atm_iv: Optional[float],
    far_atm_iv: Optional[float],
    rv_forecast: Optional[float],
    now: Optional[datetime] = None,
    edge_ratio: Optional[float] = None,
) -> GateReport:
    now = now or now_et()
    edge_ratio = edge_ratio or STRAT.iv_over_rv_min_ratio
    report = GateReport()
    report.details["edge_ratio_required"] = round(edge_ratio, 3)

    window = ENTRY_WINDOWS.get(now.weekday())
    report.entry_window = bool(window and window[0] <= now.time() <= window[1])

    horizon_s = STRAT.macro_blackout_min * 60
    blocking = [
        label
        for when, label in (*MACRO_EVENTS, *EARNINGS_EVENTS)
        if 0 <= (when - now).total_seconds() <= horizon_s
    ]
    report.macro_clear = not blocking
    if blocking:
        report.details["blocking_events"] = blocking

    if near_atm_iv is not None and far_atm_iv is not None:
        report.contango = near_atm_iv <= far_atm_iv + STRAT.contango_tolerance
        report.details["near_atm_iv"] = round(near_atm_iv, 4)
        report.details["far_atm_iv"] = round(far_atm_iv, 4)

    if near_atm_iv is not None and rv_forecast is not None and rv_forecast > 0:
        ratio = near_atm_iv / rv_forecast
        report.iv_over_rv = ratio >= edge_ratio
        report.details["iv_rv_ratio"] = round(ratio, 3)
        report.details["rv_forecast"] = round(rv_forecast, 4)

    return report
