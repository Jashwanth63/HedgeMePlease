"""
AlpachaBot Master Orchestrator.
Coordinates market data ingestion, HAR volatility forecasting, risk management ladder,
entry gate evaluations, Iron Condor generation, order execution, and position lifecycle management.
"""

from __future__ import annotations

import math
import signal
import sys
import time

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from alpacha.agent.state_machine import TradingStateMachineBuilder
from alpacha.alerts.notifier import Notifier
from alpacha.cli.driver import AlpacaCLIDriver
from alpacha.config import Settings
from alpacha.data.alpaca_data import AlpacaDataClient
from alpacha.data.macro_calendar import MacroCalendar
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.model.trainer import ModelTrainer
from alpacha.risk.risk_manager import RiskLevel, RiskManager
from alpacha.strategy.executor import OrderExecutor
from alpacha.strategy.gates import StrategyGateManager
from alpacha.strategy.ironcondor import IronCondorBuilder
from alpacha.utils.logger import get_logger
from alpacha.utils.time_utils import is_market_open

logger = get_logger("bot")


class AlpachaBot:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.is_running = False
        self.is_halted = False

        # Initialize subsystems
        self.db = SQLiteManager(settings.app.db_path)
        self.macro_calendar = MacroCalendar(settings.app.macro_calendar_path)
        self.cli_driver = AlpacaCLIDriver(settings)
        self.trainer = ModelTrainer(settings, self.db)
        self.risk_manager = RiskManager(settings, self.db)
        self.gate_manager = StrategyGateManager(settings, self.macro_calendar)
        self.ic_builder = IronCondorBuilder(settings)
        self.notifier = Notifier(settings)

        # Build LangGraph Agent State Machine
        self.agent_builder = TradingStateMachineBuilder(settings, self.db)
        self.agent_graph = self.agent_builder.build_graph()

        self.scheduler = BlockingScheduler()
        logger.info(f"Initialized {settings.app.name} Agent (Paper={settings.app.paper}, DryRun={settings.app.dry_run})")

    def run_cycle(self) -> None:
        """Executes a complete cycle via the LangGraph State Machine."""
        logger.info("=== Starting LangGraph Agent Execution Cycle ===")

        if self.settings.execution.trading_hours_only and not is_market_open():
            logger.info("Market is currently closed. Skipping trading scan.")
            return

        if self.is_halted:
            logger.critical("Agent is in HALTED state due to Kill Switch breach. No operations performed.")
            return

        init_state = {
            "symbols": self.settings.data.symbols,
            "step_history": [],
            "is_market_open": is_market_open(),
        }

        try:
            result = self.agent_graph.invoke(init_state)
            logger.info(f"LangGraph Agent Completed Steps: {result.get('step_history')}")
            if result.get("is_halted"):
                self.is_halted = True
                self.notifier.notify_kill_switch_triggered(result.get("drawdown_pct", 0.0), result.get("current_equity", 0.0))
            elif result.get("risk_level") == "WARNING":
                self.notifier.notify_drawdown_warning(result.get("drawdown_pct", 0.0), result.get("current_equity", 0.0))
        except Exception as e:
            logger.error(f"Error in LangGraph Agent execution: {e}", exc_info=True)

        logger.info("=== Finished LangGraph Agent Execution Cycle ===")


    def _process_symbol(self, symbol: str, current_equity: float, available_bp: float) -> None:
        """Processes market data, forecasting, gates, and order placement for a single symbol."""
        logger.info(f"Processing symbol: {symbol}")

        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(days=self.settings.data.history_days)
        bars_df = self.alpaca_client.get_stock_bars(symbol, start=start_ts, end=end_ts)

        if bars_df.empty:
            bars_df = self.db.load_bars(symbol, start_ts=start_ts)
            if bars_df.empty:
                bars_df = self.db.load_bars(symbol)


        if bars_df.empty:
            logger.warning(f"No historical price bars available for {symbol}. Skipping.")
            return

        self.db.save_bars(bars_df, symbol)
        _, forecasted_ann_vol = self.trainer.get_forecast(symbol, bars_df)
        spot_price = float(bars_df["close"].iloc[-1])

        chain = self.alpaca_client.get_option_chain(symbol)
        chain_list = self._format_chain(symbol, spot_price, forecasted_ann_vol, chain)
        if not chain_list:
            logger.warning(f"No options chain data available for {symbol}.")
            return

        near_iv = forecasted_ann_vol * 1.25
        next_iv = forecasted_ann_vol * 1.27

        gate_result = self.gate_manager.evaluate_all_gates(
            symbol=symbol,
            market_iv=near_iv,
            forecasted_rv=forecasted_ann_vol,
            near_atm_iv=near_iv,
            next_atm_iv=next_iv,
            current_time=datetime.now(timezone.utc),
        )

        if not gate_result.all_passed:
            logger.info(f"Entry Gates rejected trade for {symbol}.")
            return

        condor = self.ic_builder.build_iron_condor(
            symbol=symbol,
            underlying_price=spot_price,
            implied_vol=near_iv,
            chain_data=chain_list,
            dte_target=self.settings.strategy.target_dte,
        )

        if condor is None:
            logger.info(f"Could not construct a valid Iron Condor for {symbol}.")
            return

        contracts, sizing_err = self.risk_manager.calculate_position_size(
            account_equity=current_equity,
            available_bp=available_bp,
            wing_width=max(condor.put_wing_width, condor.call_wing_width),
            credit_per_share=condor.net_credit_per_share,
        )

        if contracts <= 0 or sizing_err:
            logger.warning(f"Position sizing rejected trade for {symbol}: {sizing_err}")
            return

        success, exec_err = self.executor.execute_iron_condor(
            condor=condor,
            contracts=contracts,
            gates_passed=gate_result.__dict__,
        )

        if success:
            self.notifier.notify_trade_opened(
                trade_id=condor.trade_id,
                symbol=symbol,
                credit=condor.net_credit_total * contracts,
                contracts=contracts,
            )
        else:
            logger.error(f"Execution failed for {condor.trade_id}: {exec_err}")

    def _format_chain(self, symbol: str, spot: float, vol: float, chain_raw: Any) -> List[Dict[str, Any]]:
        from scipy.stats import norm
        chain_list = []
        target_dte = self.settings.strategy.target_dte
        t_years = max(0.01, target_dte / 365.0)
        exp_date = (datetime.now().date() + timedelta(days=target_dte)).strftime("%Y-%m-%d")
        vol_clean = max(0.08, vol)

        for strike_offset in range(-60, 61, 2):
            strike = round(spot + strike_offset, 2)
            if strike <= 0:
                continue

            d1 = (math.log(spot / strike) + 0.5 * (vol_clean ** 2) * t_years) / (vol_clean * math.sqrt(t_years))
            d2 = d1 - vol_clean * math.sqrt(t_years)

            call_delta = float(norm.cdf(d1))
            put_delta = float(call_delta - 1.0)

            # Black-Scholes price approximation
            call_price = max(0.10, round(spot * norm.cdf(d1) - strike * math.exp(-0.04 * t_years) * norm.cdf(d2), 2))
            put_price = max(0.10, round(strike * math.exp(-0.04 * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1), 2))

            chain_list.append({
                "symbol": f"{symbol}_{exp_date}_{int(strike*1000)}_P",
                "option_type": "PUT",
                "strike": strike,
                "expiration": exp_date,
                "dte": target_dte,
                "delta": put_delta,
                "bid": max(0.05, put_price - 0.05),
                "ask": put_price + 0.05,
                "price": put_price,
            })
            chain_list.append({
                "symbol": f"{symbol}_{exp_date}_{int(strike*1000)}_C",
                "option_type": "CALL",
                "strike": strike,
                "expiration": exp_date,
                "dte": target_dte,
                "delta": call_delta,
                "bid": max(0.05, call_price - 0.05),
                "ask": call_price + 0.05,
                "price": call_price,
            })

        return chain_list


    def start(self) -> None:
        self.is_running = True
        logger.info(f"Starting {self.settings.app.name} Daemon...")

        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

        cron_min = self.settings.execution.scan_cron_minutes
        self.scheduler.add_job(
            self.run_cycle,
            trigger=CronTrigger(minute=cron_min),
            id="market_scan",
            name="Market Scan & Risk Cycle",
            replace_existing=True,
        )

        try:
            self.run_cycle()
        except Exception as e:
            logger.error(f"Error in initial startup cycle: {e}", exc_info=True)

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self._handle_exit(None, None)

    def _handle_exit(self, signum: Any, frame: Any) -> None:
        logger.info("Received termination signal. Shutting down daemon gracefully...")
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        self.is_running = False
        sys.exit(0)

