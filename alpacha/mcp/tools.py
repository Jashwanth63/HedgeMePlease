"""
Alpaca MCP & LangChain Tool Suite.
Provides standardized MCP tools for account inspection, market data ingestion,
risk enforcement, and order execution.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from alpacha.cli.driver import AlpacaCLIDriver
from alpacha.config import Settings


def get_cli_driver() -> AlpacaCLIDriver:
    settings = Settings.load()
    return AlpacaCLIDriver(settings)


@tool
def alpaca_get_account() -> str:
    """Fetches Alpaca account equity, cash balance, buying power, and account status via CLI/MCP."""
    driver = get_cli_driver()
    acc = driver.get_account()
    return json.dumps({
        "equity": float(acc.get("equity", 0.0)),
        "cash": float(acc.get("cash", 0.0)),
        "buying_power": float(acc.get("buying_power", 0.0)),
        "status": acc.get("status", "UNKNOWN"),
        "account_number": acc.get("account_number", "N/A"),
    })


@tool
def alpaca_get_positions() -> str:
    """Retrieves all currently open positions on Alpaca."""
    driver = get_cli_driver()
    positions = driver.get_positions()
    return json.dumps(positions)


@tool
def alpaca_submit_limit_order(
    symbol: str,
    qty: int,
    side: str,
    limit_price: float,
    time_in_force: str = "day",
) -> str:
    """Submits a limit order for a stock or option symbol on Alpaca."""
    driver = get_cli_driver()
    order = driver.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        order_type="limit",
        limit_price=limit_price,
        time_in_force=time_in_force,
    )
    return json.dumps(order)


@tool
def alpaca_cancel_all_orders() -> str:
    """Cancels all active open orders on Alpaca."""
    driver = get_cli_driver()
    result = driver.cancel_all_orders()
    return json.dumps({"status": "cancelled", "result": result})


@tool
def alpaca_close_all_positions() -> str:
    """Liquidates all open positions and cancels all orders on Alpaca (Kill Switch)."""
    driver = get_cli_driver()
    result = driver.close_all_positions(cancel_orders=True)
    return json.dumps({"status": "liquidated", "result": result})


@tool
def alpaca_get_stock_bars(
    symbol: str,
    start_iso: str,
    end_iso: str,
    timeframe: str = "1Min",
) -> str:
    """Fetches historical price bars for a symbol from Alpaca data feed."""
    driver = get_cli_driver()
    start_dt = datetime.fromisoformat(start_iso)
    end_dt = datetime.fromisoformat(end_iso)
    df = driver.get_stock_bars(symbol, start=start_dt, end=end_dt, timeframe=timeframe)
    if df.empty:
        return json.dumps([])
    return json.dumps(df.tail(100).to_dict(orient="records"))


def get_alpaca_mcp_tools() -> List[Any]:
    """Returns the list of LangChain / MCP tools for the agent."""
    return [
        alpaca_get_account,
        alpaca_get_positions,
        alpaca_submit_limit_order,
        alpaca_cancel_all_orders,
        alpaca_close_all_positions,
        alpaca_get_stock_bars,
    ]
