"""Central configuration: every hard number the agent obeys, in one file.

Sourced from the team research (see README evidence section) and plan.md v2.
The risk engine reads these; strategy code and LLM agents may never override
them. Agent-tunable soft parameters live in Clamps and are bounded there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ET = ZoneInfo("America/New_York")

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / "state"
DB_PATH = STATE_DIR / "alpacha.db"
CHECKPOINT_DB = STATE_DIR / "graph_checkpoints.db"

load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class RiskLimits:
    starting_equity: float = 100_000.0
    external_max_dd: float = 0.05          # hackathon mandate, never approached
    kill_switch_dd: float = 0.035          # flatten all, halt, from high-water mark
    daily_no_new_dd: float = 0.010         # no new trades for the day
    daily_reduce_only_dd: float = 0.015    # reduce-only for the day
    per_trade_max_loss: float = 1_000.0    # hard cap on (width - credit) * 100 * qty
    per_trade_pref_loss: float = 500.0     # sizing target
    book_worst_case: float = 2_500.0       # stress-grid worst cell across whole book
    sleeve_budget: float = 1_500.0         # sum of open max losses
    max_positions: int = 6
    max_positions_per_underlying: int = 2
    max_net_delta_dollars: float = 25_000.0
    min_net_vega: float = -200.0
    min_entry_dte: int = 2
    derisk_half_at: float = 0.50           # fraction of kill budget consumed
    derisk_freeze_at: float = 0.75


@dataclass(frozen=True)
class StrategyConfig:
    underlyings: tuple[str, ...] = ("SPY", "QQQ")
    betas: dict[str, float] = field(default_factory=lambda: {"SPY": 1.0, "QQQ": 1.18})
    short_delta_target: float = 0.20
    short_delta_band: tuple[float, float] = (0.12, 0.28)
    min_wing_width: float = 5.0            # dollars; EM wings shrink toward this
    min_credit_frac_condor: float = 0.12   # credit / width floor
    profit_take_frac: float = 0.50
    loss_close_mult: float = 2.5
    iv_over_rv_min_ratio: float = 1.15     # regime agent may tune inside Clamps
    contango_tolerance: float = 0.005      # near ATM IV may exceed 30d by 0.5 vol pt
    macro_blackout_min: int = 120
    near_dte_min: int = 2
    near_dte_max: int = 5
    far_dte_min: int = 21
    far_dte_max: int = 45
    forecast_horizon_days: int = 2


@dataclass(frozen=True)
class ExecutorConfig:
    """Time-boxed limit ladder agreed with the team."""
    improve_step: float = 0.02             # concession per requote, dollars
    max_improvements: int = 2
    wait_seconds: int = 40                 # per price level
    poll_seconds: int = 5
    close_extra_steps: int = 2             # closes may pay up a little further


@dataclass(frozen=True)
class StressConfig:
    spot_shocks: tuple[float, ...] = (-0.05, -0.03, -0.02, 0.0, 0.02, 0.03, 0.05)
    vol_mult_down: float = 1.30
    vol_mult_up: float = 1.15
    vol_add_floor: float = 0.05
    rate: float = 0.04
    div_yield: float = 0.012


@dataclass(frozen=True)
class Clamps:
    """Hard bounds on everything the regime agent may tune. Enforced in code."""
    edge_ratio: tuple[float, float] = (1.10, 1.35)
    delta_target: tuple[float, float] = (0.15, 0.25)
    size_factor: tuple[float, float] = (0.5, 1.0)


# Contest week calendar, all times ET (plan.md section 3).
MACRO_EVENTS: tuple[tuple[datetime, str], ...] = (
    (datetime(2026, 9, 1, 10, 0, tzinfo=ET), "ISM Manufacturing + JOLTS"),
    (datetime(2026, 9, 2, 8, 15, tzinfo=ET), "ADP Employment"),
    (datetime(2026, 9, 3, 8, 30, tzinfo=ET), "Initial Claims"),
    (datetime(2026, 9, 3, 10, 0, tzinfo=ET), "ISM Services (tentative)"),
)

EARNINGS_EVENTS: tuple[tuple[datetime, str], ...] = (
    (datetime(2026, 9, 2, 16, 5, tzinfo=ET), "AVGO earnings (after close)"),
)

ENTRY_WINDOWS: dict[int, tuple[time, time]] = {
    0: (time(9, 45), time(15, 30)),   # Monday
    1: (time(9, 45), time(15, 30)),   # Tuesday
    2: (time(9, 45), time(12, 0)),    # Wednesday morning only
}

FLATTEN_AT = datetime(2026, 9, 3, 15, 30, tzinfo=ET)
CONTEST_END = datetime(2026, 9, 3, 16, 0, tzinfo=ET)
PREFERRED_EXPIRIES = ("2026-09-03", "2026-09-04")

RISK = RiskLimits()
STRAT = StrategyConfig()
EXEC = ExecutorConfig()
STRESS = StressConfig()
CLAMPS = Clamps()

CYCLE_MINUTES = 5


def now_et() -> datetime:
    return datetime.now(tz=ET)


def mcp_server_dir() -> Path:
    raw = os.environ.get("ALPACA_MCP_DIR", "../alpaca-mcp-server")
    p = Path(raw)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def alpaca_env() -> dict[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY missing. Copy .env.example to .env "
            "and fill in paper credentials."
        )
    return {
        "ALPACA_API_KEY": key,
        "ALPACA_SECRET_KEY": secret,
        "ALPACA_PAPER_TRADE": os.environ.get("ALPACA_PAPER_TRADE", "true"),
    }


def openrouter_config() -> tuple[str, str]:
    """Returns (api_key, model). Empty key means the agent desk is disabled."""
    return (
        os.environ.get("OPENROUTER_API_KEY", ""),
        os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"),
    )
