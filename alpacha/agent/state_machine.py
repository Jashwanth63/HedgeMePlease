"""
LangGraph State Machine for Alpacha Options Trading Agent.
Coordinates risk evaluation, volatility forecasting, entry gating,
Iron Condor construction, and order execution via an Agent Graph.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from langgraph.graph import StateGraph, START, END

from alpacha.agent.state import TradingAgentState
from alpacha.cli.driver import AlpacaCLIDriver
from alpacha.config import Settings
from alpacha.data.macro_calendar import MacroCalendar
from alpacha.data.news_client import AlpacaNewsClient
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.model.news_analyzer import NewsSentimentAnalyzer
from alpacha.model.trainer import ModelTrainer
from alpacha.risk.risk_manager import RiskLevel, RiskManager
from alpacha.strategy.gates import StrategyGateManager
from alpacha.strategy.ironcondor import IronCondorBuilder
from alpacha.utils.logger import get_logger

logger = get_logger("langgraph_agent")


class TradingStateMachineBuilder:
    def __init__(self, settings: Settings, db_manager: Optional[SQLiteManager] = None) -> None:
        self.settings = settings
        self.db = db_manager or SQLiteManager(settings.app.db_path)
        self.cli_driver = AlpacaCLIDriver(settings)
        self.news_client = AlpacaNewsClient(settings)
        self.news_analyzer = NewsSentimentAnalyzer()
        self.macro_calendar = MacroCalendar(settings.app.macro_calendar_path)
        self.trainer = ModelTrainer(settings, self.db)
        self.risk_manager = RiskManager(settings, self.db)
        self.gate_manager = StrategyGateManager(settings, self.macro_calendar)
        self.ic_builder = IronCondorBuilder(settings)


    def risk_evaluation_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 1: Evaluates account equity against the drawdown ladder."""
        logger.info("[State Machine: Node 1] Evaluating Portfolio Risk...")
        acc = self.cli_driver.get_account()
        current_equity = float(acc.get("equity", 100000.0))
        buying_power = float(acc.get("buying_power", 400000.0))

        risk_status = self.risk_manager.evaluate_risk(current_equity)
        history = list(state.get("step_history", [])) + ["risk_evaluation"]

        return {
            "current_equity": current_equity,
            "peak_equity": risk_status.peak_equity,
            "buying_power": buying_power,
            "drawdown_pct": risk_status.drawdown_pct,
            "risk_level": risk_status.risk_level.value,
            "should_liquidate": risk_status.should_liquidate,
            "can_trade": risk_status.can_trade,
            "status_message": risk_status.message,
            "step_history": history,
        }

    def kill_switch_liquidation_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node: Liquidates all positions upon 3.5% drawdown breach."""
        logger.critical("[State Machine: Emergency Node] Kill switch triggered! Liquidating all positions.")
        self.cli_driver.close_all_positions(cancel_orders=True)
        history = list(state.get("step_history", [])) + ["kill_switch_liquidation"]
        return {
            "is_halted": True,
            "status_message": "CRITICAL KILL SWITCH EXECUTED: All positions liquidated.",
            "step_history": history,
        }

    def market_data_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 2: Ingests 1-minute bars for target symbols."""
        logger.info("[State Machine: Node 2] Ingesting Market Data...")
        symbols = state.get("symbols", self.settings.data.symbols)
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(days=self.settings.data.history_days)

        bars_data = {}
        for sym in symbols:
            df = self.cli_driver.get_stock_bars(sym, start=start_ts, end=end_ts)
            if df.empty:
                df = self.db.load_bars(sym, start_ts=start_ts)
                if df.empty:
                    df = self.db.load_bars(sym)
            if not df.empty:
                self.db.save_bars(df, sym)
                bars_data[sym] = df

        history = list(state.get("step_history", [])) + ["market_data"]
        return {"bars_data": bars_data, "step_history": history}

    def news_analysis_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 3: Ingests breaking news & analyzes sentiment/risk."""
        logger.info("[State Machine: Node 3] Ingesting & Analyzing Live Market News...")
        symbols = state.get("symbols", self.settings.data.symbols)
        news_analysis = {}

        try:
            articles = self.news_client.get_latest_news(symbols=symbols[:10], limit=30)
            for sym in symbols:
                sym_articles = [a for a in articles if sym in a.get("symbols", [])]
                res = self.news_analyzer.analyze_news(sym, sym_articles)
                news_analysis[sym] = res.__dict__
        except Exception as e:
            logger.error(f"Error in news analysis: {e}")

        history = list(state.get("step_history", [])) + ["news_analysis"]
        return {"news_analysis": news_analysis, "step_history": history}

    def volatility_forecasting_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 4: Fits Enhanced HAR model and generates 1-day RV forecasts."""
        logger.info("[State Machine: Node 4] Generating HAR Volatility Forecasts...")
        bars_data = state.get("bars_data", {})
        forecasts = {}

        for sym, df in bars_data.items():
            try:
                _, ann_vol = self.trainer.get_forecast(sym, df)
                forecasts[sym] = ann_vol
            except Exception as e:
                logger.error(f"Error forecasting for {sym}: {e}")

        history = list(state.get("step_history", [])) + ["volatility_forecasting"]
        return {"forecasts": forecasts, "step_history": history}

    def gate_filtering_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 5: Evaluates Macro, News Sentiment, Contango, and IV/RV Edge gates."""
        logger.info("[State Machine: Node 5] Evaluating Entry & News Gates...")
        forecasts = state.get("forecasts", {})
        news_analysis = state.get("news_analysis", {})
        gate_results = {}
        eligible_symbols = []

        for sym, ann_vol in forecasts.items():
            # Check News Risk Gate
            sym_news = news_analysis.get(sym, {})
            if sym_news.get("is_event_risk_high", False):
                logger.warning(f"News Gate Blocked for {sym}: High event shock risk detected!")
                continue

            near_iv = ann_vol * 1.25
            next_iv = ann_vol * 1.27
            gate_res = self.gate_manager.evaluate_all_gates(
                symbol=sym,
                market_iv=near_iv,
                forecasted_rv=ann_vol,
                near_atm_iv=near_iv,
                next_atm_iv=next_iv,
                current_time=datetime.now(timezone.utc),
            )
            gate_results[sym] = gate_res.__dict__
            if gate_res.all_passed:
                eligible_symbols.append(sym)

        history = list(state.get("step_history", [])) + ["gate_filtering"]
        return {
            "gate_results": gate_results,
            "eligible_symbols": eligible_symbols,
            "step_history": history,
        }


    def iron_condor_builder_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 5: Builds 0.20 delta Iron Condor structures with Expected Move wings."""
        logger.info("[State Machine: Node 5] Constructing Iron Condors...")
        eligible_symbols = state.get("eligible_symbols", [])
        bars_data = state.get("bars_data", {})
        forecasts = state.get("forecasts", {})
        built_condors = {}
        sized_contracts = {}

        current_equity = state.get("current_equity", 100000.0)
        available_bp = state.get("buying_power", 400000.0)

        for sym in eligible_symbols:
            df = bars_data.get(sym)
            if df is None or df.empty:
                continue
            spot = float(df["close"].iloc[-1])
            vol = forecasts.get(sym, 0.20)
            near_iv = vol * 1.25

            chain_list = self._format_chain(sym, spot, near_iv)
            condor = self.ic_builder.build_iron_condor(
                symbol=sym,
                underlying_price=spot,
                implied_vol=near_iv,
                chain_data=chain_list,
                dte_target=self.settings.strategy.target_dte,
            )

            if condor is not None:
                contracts, err = self.risk_manager.calculate_position_size(
                    account_equity=current_equity,
                    available_bp=available_bp,
                    wing_width=max(condor.put_wing_width, condor.call_wing_width),
                    credit_per_share=condor.net_credit_per_share,
                    symbol=sym,
                    forecasted_vol=vol,
                )
                if contracts > 0 and not err:
                    built_condors[sym] = condor.to_dict()
                    sized_contracts[sym] = contracts

        history = list(state.get("step_history", [])) + ["iron_condor_builder"]
        return {
            "built_condors": built_condors,
            "sized_contracts": sized_contracts,
            "step_history": history,
        }

    def order_executor_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 6: Executes limit orders via CLI/MCP Driver and records to SQLite."""
        logger.info("[State Machine: Node 6] Executing Trade Orders...")
        built_condors = state.get("built_condors", {})
        sized_contracts = state.get("sized_contracts", {})
        gate_results = state.get("gate_results", {})

        executed_trades = []
        errors = []

        for sym, condor_dict in built_condors.items():
            contracts = max(1, sized_contracts.get(sym, 1))
            trade_id = condor_dict["trade_id"]

            try:
                if not self.settings.app.dry_run:
                    # Construct native multi-leg (mleg) Iron Condor order for Alpaca Level 3 Options
                    mleg_legs = [
                        {
                            "symbol": leg["symbol"],
                            "ratio_qty": "1",
                            "side": "buy" if leg["action"] == "BUY" else "sell",
                        }
                        for leg in condor_dict["legs"]
                    ]
                    payload = {
                        "order_class": "mleg",
                        "qty": str(contracts),
                        "type": "limit",
                        "time_in_force": "day",
                        "limit_price": str(max(0.05, round(condor_dict.get("net_credit_per_share", 0.50), 2))),
                        "legs": mleg_legs,
                    }
                    self.cli_driver._request("POST", "v2/orders", data=payload)
                    logger.info(f"Successfully submitted Multi-Leg Iron Condor {trade_id} to Alpaca!")
                else:
                    logger.info(f"[DRY RUN] Simulating fill for trade {trade_id} ({sym} x {contracts})")

                self.db.save_trade(
                    trade_id=trade_id,
                    symbol=sym,
                    status="OPEN",
                    entry_timestamp=datetime.now(timezone.utc),
                    legs=condor_dict["legs"],
                    credit_received=condor_dict["net_credit_total"] * contracts,
                    gates_passed=gate_results.get(sym),
                )
                executed_trades.append({"trade_id": trade_id, "symbol": sym, "contracts": contracts})
            except Exception as e:
                logger.error(f"Execution error for {trade_id}: {e}")
                errors.append(str(e))

        history = list(state.get("step_history", [])) + ["order_executor"]
        return {
            "executed_trades": executed_trades,
            "execution_errors": errors,
            "step_history": history,
        }

    def _format_chain(self, symbol: str, spot: float, vol: float) -> List[Dict[str, Any]]:
        from scipy.stats import norm
        import math

        chain_list = []
        vol_clean = max(0.08, vol)
        today = datetime.now(timezone.utc).date()

        # Attempt to query live option contracts from Alpaca
        live_contracts = []
        if self.cli_driver.is_connected:
            try:
                today_str = today.strftime("%Y-%m-%d")
                calls_res = self.cli_driver._request(
                    "GET",
                    "v2/options/contracts",
                    params={
                        "underlying_symbols": symbol,
                        "type": "call",
                        "expiration_date_gte": today_str,
                        "limit": 500,
                    },
                )
                puts_res = self.cli_driver._request(
                    "GET",
                    "v2/options/contracts",
                    params={
                        "underlying_symbols": symbol,
                        "type": "put",
                        "expiration_date_gte": today_str,
                        "limit": 500,
                    },
                )
                live_contracts = calls_res.get("option_contracts", []) + puts_res.get("option_contracts", [])
            except Exception as e:
                logger.warning(f"Could not query live option contracts for {symbol}: {e}")


        if live_contracts:
            # Find best expiration
            expirations = sorted(list(set(c["expiration_date"] for c in live_contracts if c.get("expiration_date"))))
            target_dte = self.settings.strategy.target_dte
            best_exp = min(
                expirations,
                key=lambda exp: abs((datetime.fromisoformat(exp).date() - today).days - target_dte)
            )
            actual_dte = max(1, (datetime.fromisoformat(best_exp).date() - today).days)
            t_years = max(0.003, actual_dte / 365.0)

            exp_contracts = [c for c in live_contracts if c.get("expiration_date") == best_exp]
            for c in exp_contracts:
                strike = float(c["strike_price"])
                opt_type = c["type"].upper()

                d1 = (math.log(spot / strike) + 0.5 * (vol_clean ** 2) * t_years) / (vol_clean * math.sqrt(t_years))
                d2 = d1 - vol_clean * math.sqrt(t_years)

                call_delta = float(norm.cdf(d1))
                put_delta = float(call_delta - 1.0)
                delta = call_delta if opt_type == "CALL" else put_delta

                if opt_type == "CALL":
                    price = max(0.10, round(spot * norm.cdf(d1) - strike * math.exp(-0.04 * t_years) * norm.cdf(d2), 2))
                else:
                    price = max(0.10, round(strike * math.exp(-0.04 * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1), 2))

            has_calls_above = any(c["option_type"] == "CALL" and c["strike"] > spot for c in chain_list)
            has_puts_below = any(c["option_type"] == "PUT" and c["strike"] < spot for c in chain_list)
            if has_calls_above and has_puts_below and len(chain_list) >= 8:
                return chain_list

        # Fallback for offline / synthetic testing or when live chain lacks near-spot strikes
        target_dte = max(1, self.settings.strategy.target_dte)
        t_years = max(0.01, target_dte / 365.0)
        exp_date = (today + timedelta(days=target_dte)).strftime("%Y-%m-%d")
        yy, mm, dd = exp_date[2:4], exp_date[5:7], exp_date[8:10]
        chain_list = []

        for strike_offset in range(-60, 61, 2):
            strike = round(spot + strike_offset, 2)
            if strike <= 0:
                continue

            d1 = (math.log(spot / strike) + 0.5 * (vol_clean ** 2) * t_years) / (vol_clean * math.sqrt(t_years))
            d2 = d1 - vol_clean * math.sqrt(t_years)

            call_delta = float(norm.cdf(d1))
            put_delta = float(call_delta - 1.0)
            call_price = max(0.10, round(spot * norm.cdf(d1) - strike * math.exp(-0.04 * t_years) * norm.cdf(d2), 2))
            put_price = max(0.10, round(strike * math.exp(-0.04 * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1), 2))

            strike_int = int(round(strike * 1000))
            chain_list.append({
                "symbol": f"{symbol.upper()}{yy}{mm}{dd}P{strike_int:08d}",
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
                "symbol": f"{symbol.upper()}{yy}{mm}{dd}C{strike_int:08d}",
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


    def build_graph(self):
        """Constructs and compiles the LangGraph State Machine with News & Sentiment Nodes."""
        graph = StateGraph(TradingAgentState)

        graph.add_node("risk_node", self.risk_evaluation_node)
        graph.add_node("kill_node", self.kill_switch_liquidation_node)
        graph.add_node("market_data_node", self.market_data_node)
        graph.add_node("news_node", self.news_analysis_node)
        graph.add_node("volatility_node", self.volatility_forecasting_node)
        graph.add_node("gates_node", self.gate_filtering_node)
        graph.add_node("ic_builder_node", self.iron_condor_builder_node)
        graph.add_node("executor_node", self.order_executor_node)

        def route_risk(state: TradingAgentState) -> str:
            level = state.get("risk_level", "NORMAL")
            if level == "KILL" or state.get("should_liquidate"):
                return "kill_node"
            if level == "WARNING" or not state.get("can_trade"):
                return END
            return "market_data_node"

        def route_gates(state: TradingAgentState) -> str:
            eligible = state.get("eligible_symbols", [])
            if eligible:
                return "ic_builder_node"
            return END

        graph.add_edge(START, "risk_node")
        graph.add_conditional_edges("risk_node", route_risk, {
            "kill_node": "kill_node",
            "market_data_node": "market_data_node",
            END: END,
        })
        graph.add_edge("kill_node", END)
        graph.add_edge("market_data_node", "news_node")
        graph.add_edge("news_node", "volatility_node")
        graph.add_edge("volatility_node", "gates_node")
        graph.add_conditional_edges("gates_node", route_gates, {
            "ic_builder_node": "ic_builder_node",
            END: END,
        })
        graph.add_edge("ic_builder_node", "executor_node")
        graph.add_edge("executor_node", END)

        return graph.compile()


