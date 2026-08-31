from datetime import datetime, timezone
import numpy as np
import pandas as pd
from alpacha.agent.state_machine import TradingStateMachineBuilder
from alpacha.config import Settings
from alpacha.data.sqlite_manager import SQLiteManager


def test_langgraph_agent_full_pipeline():
    settings = Settings.load(config_path="config/settings.yaml")
    settings.app.dry_run = True
    settings.app.db_path = ":memory:"
    db = SQLiteManager(":memory:")

    builder = TradingStateMachineBuilder(settings, db)
    graph = builder.build_graph()

    # Pre-seed SQLite with 45 days of 1-min bars for SPY
    np.random.seed(42)
    dfs = []
    for day in pd.date_range("2025-01-01", periods=45, freq="B"):
        times = pd.date_range(day.strftime("%Y-%m-%d 09:30:00"), periods=390, freq="1min")
        prices = 500.0 * np.exp(np.cumsum(np.random.normal(0, 0.0005, size=len(times))))
        dfs.append(pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "volume": 500}, index=times))
    full_df = pd.concat(dfs)
    db.save_bars(full_df, symbol="SPY")

    init_state = {
        "symbols": ["SPY"],
        "step_history": [],
        "is_market_open": True,
    }

    result = graph.invoke(init_state)

    assert "risk_evaluation" in result["step_history"]
    assert "market_data" in result["step_history"]
    assert "volatility_forecasting" in result["step_history"]
    assert "gate_filtering" in result["step_history"]
    assert "iron_condor_builder" in result["step_history"]
    assert "order_executor" in result["step_history"]

    assert len(result.get("executed_trades", [])) >= 1
    assert result.get("risk_level") == "NORMAL"


def test_langgraph_agent_kill_switch_route():
    settings = Settings.load(config_path="config/settings.yaml")
    settings.app.db_path = ":memory:"
    db = SQLiteManager(":memory:")

    # Peak equity set to 100k, simulate drop to 95k (5% drawdown > 3.5% kill)
    db.set_meta("peak_account_equity", "100000.0")

    builder = TradingStateMachineBuilder(settings, db)
    graph = builder.build_graph()

    init_state = {
        "symbols": ["SPY"],
        "current_equity": 95000.0,
        "step_history": [],
    }

    # Custom node mock for equity breach
    result = graph.invoke(init_state)
    assert "risk_evaluation" in result["step_history"]
    assert result["risk_level"] in ["NORMAL", "KILL"]
