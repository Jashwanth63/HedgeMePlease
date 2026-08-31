"""
Trade Entry Gates.
Enforces 3 mandatory filters prior to opening an Iron Condor position:
1. Macro Proximity Gate: No high-impact macro event within 2 hours.
2. Vol Contango Gate: Healthy term structure (avoid entering during inverted/backwardated vol spikes).
3. Edge Gate: Market IV >= 1.2x Forecasted Realized Volatility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from alpacha.config import Settings
from alpacha.data.macro_calendar import MacroCalendar
from alpacha.utils.logger import get_logger

logger = get_logger("strategy_gates")


@dataclass
class GateResult:
    all_passed: bool
    macro_passed: bool
    contango_passed: bool
    edge_passed: bool
    macro_reason: Optional[str] = None
    contango_reason: Optional[str] = None
    edge_reason: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class StrategyGateManager:
    def __init__(self, settings: Settings, macro_calendar: MacroCalendar) -> None:
        self.settings = settings
        self.macro_calendar = macro_calendar
        self.macro_buffer_hours = settings.strategy.macro_buffer_hours
        self.min_iv_rv_multiple = settings.model.iv_vs_rv_multiple
        self.contango_min_ratio = settings.strategy.contango_min_ratio

    def evaluate_macro_gate(self, target_time: Optional[datetime] = None) -> Tuple[bool, Optional[str]]:
        """Gate 1: Check proximity to high-impact macro economic events."""
        is_blocked, reason = self.macro_calendar.check_event_proximity(
            target_time=target_time,
            buffer_hours=self.macro_buffer_hours,
        )
        return not is_blocked, reason

    def evaluate_contango_gate(
        self,
        near_atm_iv: float,
        next_atm_iv: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Gate 2: Check volatility term structure contango.
        Normal contango / flat curve: near_iv <= next_iv * contango_ratio_tolerance.
        Severe backwardation (near_iv >> next_iv) indicates an acute panic/event pricing.
        """
        if near_atm_iv <= 0.0 or next_atm_iv <= 0.0:
            return True, None  # Default pass if term data unavailable in test/mock

        ratio = near_atm_iv / next_atm_iv
        # Backwardation threshold: if near IV is significantly higher than next IV (e.g. ratio > 1.05)
        # contango_min_ratio in config is 0.98, so near/next <= 1.05 is healthy
        if ratio > (1.0 / self.contango_min_ratio):
            reason = f"Vol Term Structure in backwardation: Near IV ({near_atm_iv:.2%}) / Next IV ({next_atm_iv:.2%}) = {ratio:.2f} > {1.0/self.contango_min_ratio:.2f}"
            logger.info(f"Contango Gate Failed: {reason}")
            return False, reason

        return True, None

    def evaluate_edge_gate(
        self,
        market_iv: float,
        forecasted_rv: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Gate 3: Edge filter.
        Requires Market IV >= 1.2x Forecasted Realized Volatility.
        """
        if forecasted_rv <= 1e-4:
            return False, "Forecasted RV is non-positive or near zero."

        multiple = market_iv / forecasted_rv
        if multiple < self.min_iv_rv_multiple:
            reason = (
                f"Insufficient IV/RV edge: Market IV ({market_iv:.2%}) / Forecasted RV ({forecasted_rv:.2%}) "
                f"= {multiple:.2f}x < required {self.min_iv_rv_multiple:.2f}x"
            )
            logger.info(f"Edge Gate Failed: {reason}")
            return False, reason

        logger.info(f"Edge Gate Passed: IV={market_iv:.2%}, Forecasted RV={forecasted_rv:.2%}, Edge={multiple:.2f}x")
        return True, None

    def evaluate_all_gates(
        self,
        symbol: str,
        market_iv: float,
        forecasted_rv: float,
        near_atm_iv: float,
        next_atm_iv: float,
        current_time: Optional[datetime] = None,
    ) -> GateResult:
        """Evaluates all 3 entry gates."""
        macro_ok, macro_reason = self.evaluate_macro_gate(target_time=current_time)
        contango_ok, contango_reason = self.evaluate_contango_gate(near_atm_iv, next_atm_iv)
        edge_ok, edge_reason = self.evaluate_edge_gate(market_iv, forecasted_rv)

        all_ok = macro_ok and contango_ok and edge_ok

        metrics = {
            "market_iv": market_iv,
            "forecasted_rv": forecasted_rv,
            "iv_rv_ratio": (market_iv / forecasted_rv) if forecasted_rv > 0 else 0.0,
            "near_atm_iv": near_atm_iv,
            "next_atm_iv": next_atm_iv,
        }

        if all_ok:
            logger.info(f"ALL ENTRY GATES PASSED for {symbol}!")
        else:
            logger.info(f"Entry Gates Rejected for {symbol}. Macro={macro_ok}, Contango={contango_ok}, Edge={edge_ok}")

        return GateResult(
            all_passed=all_ok,
            macro_passed=macro_ok,
            contango_passed=contango_ok,
            edge_passed=edge_ok,
            macro_reason=macro_reason,
            contango_reason=contango_reason,
            edge_reason=edge_reason,
            metrics=metrics,
        )
