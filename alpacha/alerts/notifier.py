"""
Alerting & Notification Service.
Sends webhook alerts (Slack/Discord/Custom) and structured log messages for trades, warnings, and kill switch events.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import requests

from alpacha.config import Settings
from alpacha.utils.logger import get_logger

logger = get_logger("notifier")


class Notifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.alerts.enabled
        self.webhook_url = settings.alerts.webhook_url

    def send_alert(self, title: str, message: str, level: str = "INFO") -> None:
        """Sends an alert to log and webhook (if configured)."""
        log_msg = f"[{level}] {title} - {message}"
        if level in ["CRITICAL", "ERROR"]:
            logger.error(log_msg)
        elif level == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        if not self.enabled or not self.webhook_url:
            return

        payload = {
            "title": f"[{self.settings.app.name}] {title}",
            "text": message,
            "level": level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            resp = requests.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5.0,
            )
            if resp.status_code >= 400:
                logger.warning(f"Webhook alert returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"Failed to deliver webhook notification: {e}")

    def notify_trade_opened(self, trade_id: str, symbol: str, credit: float, contracts: int) -> None:
        self.send_alert(
            title="Iron Condor Opened",
            message=f"Trade {trade_id}: {symbol} x {contracts} contracts. Collected ${credit:.2f} total credit.",
            level="INFO",
        )

    def notify_drawdown_warning(self, drawdown_pct: float, current_equity: float) -> None:
        self.send_alert(
            title="Drawdown Warning (2.0% Threshold)",
            message=f"Portfolio Drawdown has reached {drawdown_pct:.2%}. Current Equity: ${current_equity:,.2f}. New entries halted.",
            level="WARNING",
        )

    def notify_kill_switch_triggered(self, drawdown_pct: float, current_equity: float) -> None:
        self.send_alert(
            title="CRITICAL KILL SWITCH TRIGGERED (3.5% Threshold)",
            message=f"Portfolio Drawdown has breached {drawdown_pct:.2%}. Liquidating all positions and stopping bot. Current Equity: ${current_equity:,.2f}.",
            level="CRITICAL",
        )
