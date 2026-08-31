"""
Alpaca FastMCP Server.
Exposes Alpaca tools through the standard Model Context Protocol (MCP).
Can be run as a standalone MCP server for Claude, Cursor, Cline, or LangChain agents.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP

from alpacha.cli.driver import AlpacaCLIDriver
from alpacha.config import Settings

mcp = FastMCP("Alpaca-Trading-MCP")


def _get_driver() -> AlpacaCLIDriver:
    return AlpacaCLIDriver(Settings.load())


@mcp.tool()
def get_account() -> str:
    """Returns Alpaca account balance, cash, buying power, and status."""
    driver = _get_driver()
    return json.dumps(driver.get_account())


@mcp.tool()
def get_open_positions() -> str:
    """Returns all currently active positions."""
    driver = _get_driver()
    return json.dumps(driver.get_positions())


@mcp.tool()
def place_limit_order(symbol: str, qty: int, side: str, limit_price: float) -> str:
    """Submits a limit order for stock or option symbol."""
    driver = _get_driver()
    return json.dumps(driver.submit_order(symbol=symbol, qty=qty, side=side, limit_price=limit_price))


@mcp.tool()
def cancel_all_orders() -> str:
    """Cancels all active open orders."""
    driver = _get_driver()
    return json.dumps(driver.cancel_all_orders())


@mcp.tool()
def kill_switch_liquidate_all() -> str:
    """Emergency kill switch: Liquidates all open positions and cancels all pending orders."""
    driver = _get_driver()
    return json.dumps(driver.close_all_positions(cancel_orders=True))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
