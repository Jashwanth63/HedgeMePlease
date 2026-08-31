"""
Order Execution & Trade Lifecycle Manager.
Executes Iron Condor legs, monitors fills, manages profit target / stop loss exits,
and persists trade history into SQLite.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpacha.config import Settings
from alpacha.data.alpaca_data import AlpacaDataClient
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.strategy.ironcondor import IronCondor, OptionLeg
from alpacha.utils.logger import get_logger

logger = get_logger("executor")


class OrderExecutor:
    def __init__(
        self,
        settings: Settings,
        alpaca_client: AlpacaDataClient,
        db_manager: SQLiteManager,
    ) -> None:
        self.settings = settings
        self.client = alpaca_client
        self.db = db_manager
        self.dry_run = settings.app.dry_run

    def execute_iron_condor(
        self,
        condor: IronCondor,
        contracts: int,
        gates_passed: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Executes the 4 legs of an Iron Condor.
        In dry-run mode, logs and records fill immediately.
        In live/paper mode, submits limit orders and tracks status.
        """
        condor.contracts = contracts
        trade_id = condor.trade_id
        symbol = condor.underlying

        logger.info(
            f"Executing Iron Condor [{trade_id}] for {symbol}: {contracts} contracts. "
            f"Net Credit=${condor.net_credit_per_share:.2f}/share, Total=${condor.net_credit_total * contracts:.2f}"
        )

        if self.dry_run or not self.client.is_connected:
            logger.info(f"[DRY RUN] Simulating immediate fill for Iron Condor {trade_id}")
            self.db.save_trade(
                trade_id=trade_id,
                symbol=symbol,
                status="OPEN",
                entry_timestamp=datetime.now(timezone.utc),
                legs=condor.to_dict()["legs"],
                credit_received=condor.net_credit_per_share * contracts * 100.0,
                gates_passed=gates_passed,
            )
            return True, None

        # Live / Paper Execution on Alpaca
        submitted_orders = []
        try:
            for leg in [condor.long_put, condor.short_put, condor.short_call, condor.long_call]:
                side = OrderSide.BUY if leg.action == "BUY" else OrderSide.SELL
                limit_price = round(leg.mid, 2)
                order_req = LimitOrderRequest(
                    symbol=leg.symbol,
                    qty=contracts,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
                order = self.client.trading_client.submit_order(order_data=order_req)
                submitted_orders.append(order)
                logger.info(f"Submitted order for {leg.symbol} ({leg.action} {contracts} @ ${limit_price})")

            # Poll for fills
            all_filled = self._wait_for_fills(submitted_orders, timeout_sec=self.settings.execution.order_poll_timeout_sec)
            if not all_filled:
                logger.warning(f"Orders for {trade_id} did not fill completely within timeout. Cancelling open orders.")
                for ord in submitted_orders:
                    try:
                        self.client.trading_client.cancel_order_by_id(ord.id)
                    except Exception:
                        pass
                return False, "Order fill timeout reached. Partial fill protection invoked."

            # Save filled trade in database
            self.db.save_trade(
                trade_id=trade_id,
                symbol=symbol,
                status="OPEN",
                entry_timestamp=datetime.now(timezone.utc),
                legs=condor.to_dict()["legs"],
                credit_received=condor.net_credit_per_share * contracts * 100.0,
                gates_passed=gates_passed,
            )
            return True, None

        except Exception as e:
            logger.error(f"Error executing Iron Condor {trade_id}: {e}", exc_info=True)
            return False, str(e)

    def _wait_for_fills(self, orders: List[Any], timeout_sec: int = 60) -> bool:
        """Polls submitted orders until all are filled or timeout occurs."""
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            all_done = True
            for ord in orders:
                try:
                    current_ord = self.client.trading_client.get_order_by_id(ord.id)
                    if str(current_ord.status).lower() not in ["filled"]:
                        all_done = False
                        break
                except Exception as e:
                    logger.warning(f"Error polling order {ord.id}: {e}")
                    all_done = False
                    break
            if all_done:
                return True
            time.sleep(self.settings.execution.order_poll_interval_sec)
        return False
