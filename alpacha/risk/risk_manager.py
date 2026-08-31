"""
Risk Manager for AlpachaBot.
Tracks equity high-water mark, evaluates drawdown ladder (2% warning, 3.5% kill switch),
manages position sizing, and enforces portfolio guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Tuple

from alpacha.config import Settings
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.utils.logger import get_logger

logger = get_logger("risk_manager")


class RiskLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    KILL = "KILL"


@dataclass
class RiskStatus:
    current_equity: float
    peak_equity: float
    drawdown_pct: float
    risk_level: RiskLevel
    can_trade: bool
    should_liquidate: bool
    message: str


class RiskManager:
    META_PEAK_EQUITY_KEY = "peak_account_equity"

    def __init__(self, settings: Settings, db_manager: SQLiteManager) -> None:
        self.settings = settings
        self.db = db_manager
        self.warn_drawdown = settings.risk.warn_drawdown_pct
        self.kill_drawdown = settings.risk.kill_drawdown_pct
        self.max_portfolio_bp_pct = settings.risk.max_portfolio_bp_pct
        self.max_contracts = settings.risk.max_contracts_per_trade

        # Load or initialize peak equity
        self._peak_equity = self._load_peak_equity()

    def _load_peak_equity(self) -> float:
        val = self.db.get_meta(self.META_PEAK_EQUITY_KEY)
        if val:
            try:
                return float(val)
            except ValueError:
                pass
        return 0.0

    def update_peak_equity(self, current_equity: float) -> float:
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self.db.set_meta(self.META_PEAK_EQUITY_KEY, str(current_equity))
            logger.info(f"New Equity High-Water Mark: ${self._peak_equity:,.2f}")
        return self._peak_equity

    def evaluate_risk(self, current_equity: float) -> RiskStatus:
        """
        Evaluates portfolio drawdown against risk ladder:
        - Drawdown < 2.0%: NORMAL
        - Drawdown >= 2.0% and < 3.5%: WARNING (block new entries, warn)
        - Drawdown >= 3.5%: KILL (liquidate all positions, halt daemon)
        """
        if current_equity <= 0.0:
            logger.critical("Current equity is 0 or negative!")
            return RiskStatus(
                current_equity=current_equity,
                peak_equity=self._peak_equity,
                drawdown_pct=1.0,
                risk_level=RiskLevel.KILL,
                can_trade=False,
                should_liquidate=True,
                message="Account equity is 0 or negative.",
            )

        # Update peak if equity grows
        peak = self.update_peak_equity(current_equity)

        # Calculate drawdown from peak
        drawdown_pct = max(0.0, (peak - current_equity) / peak) if peak > 0 else 0.0

        if drawdown_pct >= self.kill_drawdown:
            level = RiskLevel.KILL
            can_trade = False
            should_liquidate = True
            msg = f"CRITICAL: Drawdown {drawdown_pct:.2%} exceeds kill threshold {self.kill_drawdown:.2%}. Triggering liquidation."
            logger.critical(msg)
        elif drawdown_pct >= self.warn_drawdown:
            level = RiskLevel.WARNING
            can_trade = False
            should_liquidate = False
            msg = f"WARNING: Drawdown {drawdown_pct:.2%} exceeds warning threshold {self.warn_drawdown:.2%}. New entries halted."
            logger.warning(msg)
        else:
            level = RiskLevel.NORMAL
            can_trade = True
            should_liquidate = False
            msg = f"Risk normal: Equity=${current_equity:,.2f}, Peak=${peak:,.2f}, Drawdown={drawdown_pct:.2%}"
            logger.debug(msg)

        # Record snapshot in SQLite
        self.db.save_risk_snapshot(
            equity=current_equity,
            peak_equity=peak,
            drawdown_pct=drawdown_pct,
            risk_level=level.value,
        )

        return RiskStatus(
            current_equity=current_equity,
            peak_equity=peak,
            drawdown_pct=drawdown_pct,
            risk_level=level,
            can_trade=can_trade,
            should_liquidate=should_liquidate,
            message=msg,
        )

    def calculate_position_size(
        self,
        account_equity: float,
        available_bp: float,
        wing_width: float,
        credit_per_share: float,
    ) -> Tuple[int, Optional[str]]:
        """
        Calculates safe number of Iron Condor contracts to trade.
        Max loss per contract = (Wing Width - Credit) * 100
        """
        if account_equity <= 0 or wing_width <= 0:
            return 0, "Invalid equity or wing width."

        max_loss_per_share = max(0.01, wing_width - credit_per_share)
        max_loss_per_contract = max_loss_per_share * 100.0

        # Max loss budget per trade (e.g. 1% of account equity)
        risk_budget = account_equity * self.settings.risk.single_trade_max_loss_pct
        size_by_risk = max(1, int(risk_budget / max_loss_per_contract))

        # Max buying power allocation (e.g. 30% of available buying power)
        bp_budget = available_bp * self.max_portfolio_bp_pct
        size_by_bp = max(1, int(bp_budget / max_loss_per_contract))

        # Cap by configured maximum contracts per trade
        contracts = min(size_by_risk, size_by_bp, self.max_contracts)
        contracts = max(1, contracts)

        total_risk = contracts * max_loss_per_contract
        if total_risk > available_bp:
            return 0, f"Insufficient buying power: requires ${total_risk:,.2f}, available ${available_bp:,.2f}"

        logger.info(
            f"Sized Iron Condor: {contracts} contracts "
            f"(Max loss/contract=${max_loss_per_contract:.2f}, Total risk=${total_risk:.2f})"
        )
        return contracts, None
