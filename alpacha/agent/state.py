"""
Typed State Definition for LangGraph Options Trading Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
from alpacha.strategy.ironcondor import IronCondor


class TradingAgentState(TypedDict, total=False):
    # Execution metadata
    symbols: List[str]
    current_time_iso: str
    is_market_open: bool
    status_message: str

    # Account & Risk State
    current_equity: float
    peak_equity: float
    buying_power: float
    drawdown_pct: float
    risk_level: str  # "NORMAL", "WARNING", "KILL"
    should_liquidate: bool
    can_trade: bool

    # Volatility & Market Data State
    bars_data: Dict[str, Any]
    forecasts: Dict[str, float]  # symbol -> annualized RV forecast

    # Strategy & Gate State
    gate_results: Dict[str, Any]  # symbol -> GateResult dict
    eligible_symbols: List[str]

    # Iron Condor & Orders State
    built_condors: Dict[str, Any]  # symbol -> IronCondor dict
    sized_contracts: Dict[str, int]  # symbol -> contracts
    executed_trades: List[Dict[str, Any]]
    execution_errors: List[str]

    # System State
    is_halted: bool
    step_history: List[str]
