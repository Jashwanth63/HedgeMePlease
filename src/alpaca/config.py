"""Central configuration: every hard number the agent obeys, in one file.

Sourced from the team research (see README evidence section) and plan.md v2.
The risk engine reads these; strategy code and LLM agents may never override
them. Agent-tunable soft parameters live in Clamps and are bounded there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ET = ZoneInfo("America/New_York")

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / "state"
DB_PATH = STATE_DIR / "alpaca.db"
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
    book_worst_case: float = 3_000.0       # stress-grid worst cell across whole book
    sleeve_budget: float = 2_200.0         # sum of open max losses (sleeve A)
    max_positions: int = 15
    max_positions_per_underlying: int = 3
    max_net_delta_dollars: float = 25_000.0
    min_net_vega: float = -200.0
    min_entry_dte: int = 2
    derisk_half_at: float = 0.50           # fraction of kill budget consumed
    derisk_freeze_at: float = 0.75


@dataclass(frozen=True)
class StrategyConfig:
    underlyings: tuple[str, ...] = ("SPY", "QQQ", "GLD", "TLT")
    betas: dict[str, float] = field(default_factory=lambda: {"SPY": 1.0, "QQQ": 1.18})
    # correlation clusters: budgets and direction limits apply per cluster so
    # the book cannot quietly become one equity bet wearing four tickers
    clusters: dict[str, str] = field(default_factory=lambda: {
        "SPY": "equity", "QQQ": "equity", "GLD": "metals", "TLT": "rates",
        "DELL": "single_name", "AVGO": "single_name",
    })
    cluster_budget_frac: float = 0.55      # max share of the A sleeve budget per cluster
    cluster_delta_caps: dict[str, float] = field(default_factory=lambda: {
        "equity": 25_000.0, "metals": 10_000.0, "rates": 10_000.0,
        "single_name": 10_000.0,
    })
    short_delta_target: float = 0.20
    short_delta_band: tuple[float, float] = (0.12, 0.28)
    # wing width floors scale to each product's price and vol; a 5 dollar wing
    # on a 90 dollar TLT would fail every credit floor by construction
    wing_width_floors: dict[str, float] = field(default_factory=lambda: {
        "SPY": 5.0, "QQQ": 5.0, "GLD": 3.0, "TLT": 1.0,
    })
    min_wing_width: float = 5.0            # fallback for unlisted underlyings
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
    # Entry pacing. The quality filters are the gates and the risk engine; these
    # exist for temporal diversification and as software circuit breakers.
    same_underlying_cooldown_min: int = 45  # re-entry spacing per underlying
    global_entry_spacing_min: int = 10      # thin circuit breaker across all entries
    max_entries_per_day: int = 6            # hard cap regardless of budget recycling


@dataclass(frozen=True)
class ExecutorConfig:
    """Time-boxed limit ladder agreed with the team."""
    improve_step: float = 0.02             # concession per requote, dollars
    max_improvements: int = 4              # calibrated live Aug 31: mid plus four
                                           # cents rested unfilled on the paper engine
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
    (datetime(2026, 9, 1, 16, 5, tzinfo=ET), "DELL earnings (after close)"),
    (datetime(2026, 9, 2, 16, 5, tzinfo=ET), "AVGO earnings (after close)"),
)


@dataclass(frozen=True)
class EventTrade:
    """One scheduled earnings event Sleeve B trades around.

    Only the facts live here (symbol, verified event time, post-event expiry,
    crush window). Phase viability is computed at runtime from the clock, and
    the event-analyst agent may decline viable phases — never extend them.
    """
    symbol: str
    event_time: datetime            # release moment (after close)
    post_expiry: str                # first expiry after the event
    crush_entry_start: datetime     # sell the condor in this window, same day
    crush_entry_end: datetime
    crush_exit_by: datetime         # cover next morning, unconditionally

    @property
    def runup_entry_start(self) -> datetime:
        """Long premium may enter from the prior trading day's open."""
        prior = self.event_time - timedelta(days=1)
        while prior.weekday() >= 5:
            prior -= timedelta(days=1)
        return prior.replace(hour=9, minute=45, second=0, microsecond=0)

    @property
    def runup_entry_end(self) -> datetime:
        """No fresh run-up entries within the final hours before the print."""
        return self.event_time.replace(hour=13, minute=0, second=0, microsecond=0)

    @property
    def runup_exit_by(self) -> datetime:
        """The strangle is sold before the print, unconditionally."""
        return self.event_time.replace(hour=15, minute=0, second=0, microsecond=0)

    def runup_viable(self, now: datetime) -> bool:
        return self.runup_entry_start <= now <= self.runup_entry_end

    def crush_viable(self, now: datetime) -> bool:
        return self.crush_entry_start <= now <= self.crush_entry_end


@dataclass(frozen=True)
class SleeveBConfig:
    budget: float = 900.0                  # total max loss across event positions
    crush_max_loss: float = 350.0          # per crush condor
    runup_max_debit: float = 450.0         # per run-up strangle; earnings-week
                                           # strangles on 100-300 dollar stocks cost 3-4.5
    crush_move_mult: float = 1.0           # shorts at >= 1x the implied move
    crush_profit_take: float = 0.50
    crush_loss_mult: float = 2.5
    runup_profit_mult: float = 1.5         # sell the strangle early if it 1.5x's
    min_credit_frac: float = 0.10          # single-name event credit floor
    quote_spread_frac: float = 0.35        # single names quote wider than index ETFs


SLEEVE_B_EVENTS: tuple[EventTrade, ...] = (
    EventTrade(
        symbol="DELL",
        event_time=datetime(2026, 9, 1, 16, 5, tzinfo=ET),
        post_expiry="2026-09-04",
        crush_entry_start=datetime(2026, 9, 1, 15, 30, tzinfo=ET),
        crush_entry_end=datetime(2026, 9, 1, 15, 55, tzinfo=ET),
        crush_exit_by=datetime(2026, 9, 2, 10, 0, tzinfo=ET),
    ),
    EventTrade(
        symbol="AVGO",
        event_time=datetime(2026, 9, 2, 16, 5, tzinfo=ET),
        post_expiry="2026-09-04",
        crush_entry_start=datetime(2026, 9, 2, 15, 30, tzinfo=ET),
        crush_entry_end=datetime(2026, 9, 2, 15, 55, tzinfo=ET),
        crush_exit_by=datetime(2026, 9, 3, 10, 0, tzinfo=ET),
    ),
)

SLEEVE_B = SleeveBConfig()


@dataclass(frozen=True)
class SleeveCConfig:
    """Insurance: far OTM SPY puts. The hedge analyst decides when; code
    guarantees a backstop purchase before the biggest event night."""
    budget: float = 150.0                  # total debit, the sleeve's whole risk
    underlying: str = "SPY"
    otm_band: tuple[float, float] = (0.03, 0.05)   # strike 3-5 percent below spot
    dte_min: int = 4
    dte_max: int = 11                      # next-week expiry keeps resale value
    backstop_time: datetime = datetime(2026, 9, 1, 14, 45, tzinfo=ET)  # pulled
    # forward from Wed 10:00 on 2026-09-01: US-Iran escalation, term structure
    # inverted on all four underlyings, DELL and AVGO nights still ahead
    backstop_min_book: float = 600.0       # committed max loss that forces the buy


SLEEVE_C = SleeveCConfig()

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
        os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v3.2"),
    )
