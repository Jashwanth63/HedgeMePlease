"""The deterministic risk engine. No model, no LLM, no exceptions.

Strategy code and agents propose; this module disposes. Pure functions over
plain inputs so every rule is unit-testable without a broker or database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol

from ..config import RISK, SLEEVE_B, STRAT, STRESS, now_et
from .bs import bs
from .ledger import Position
from .stress import t_years, worst_cell


class AccountAction(str, Enum):
    OK = "ok"
    NO_NEW = "no_new_trades"
    REDUCE_ONLY = "reduce_only"
    KILL = "kill_switch"


class ProposalLike(Protocol):
    underlying: str
    max_loss: float
    net_delta_dollars: float
    net_vega_dollars: float
    dte: int
    position: Position


@dataclass
class RiskVerdict:
    approved: bool
    reasons: list[str]
    size_factor: float = 1.0


def evaluate_account(
    equity: float, hwm: float, day_anchor: float, halted: bool
) -> AccountAction:
    if halted:
        return AccountAction.KILL
    peak = max(hwm, RISK.starting_equity)
    if equity <= peak * (1 - RISK.kill_switch_dd):
        return AccountAction.KILL
    anchor = day_anchor or equity
    daily_dd = (anchor - equity) / anchor if anchor > 0 else 0.0
    if daily_dd >= RISK.daily_reduce_only_dd:
        return AccountAction.REDUCE_ONLY
    if daily_dd >= RISK.daily_no_new_dd:
        return AccountAction.NO_NEW
    return AccountAction.OK


def budget_consumed_frac(equity: float, hwm: float) -> float:
    peak = max(hwm, RISK.starting_equity)
    kill_budget = peak * RISK.kill_switch_dd
    return max(0.0, peak - equity) / kill_budget if kill_budget > 0 else 0.0


def book_net_delta_dollars(positions: list[Position], spots: dict[str, float]) -> float:
    total = 0.0
    for pos in positions:
        spot = spots.get(pos.underlying, 0.0)
        beta = STRAT.betas.get(pos.underlying, 1.0)
        for leg in pos.legs:
            sign = 1 if leg.side == "buy" else -1
            total += sign * leg.entry_delta * 100.0 * leg.ratio_qty * pos.qty * spot * beta
    return total


def cluster_of(underlying: str) -> str:
    return STRAT.clusters.get(underlying, "other")


def cluster_delta_dollars(
    positions: list[Position], spots: dict[str, float]
) -> dict[str, float]:
    """Directional exposure per correlation cluster. Equity uses SPY-beta
    weighting; other clusters use raw delta dollars, because beta to SPY is
    meaningless for gold or duration."""
    out: dict[str, float] = {}
    for pos in positions:
        spot = spots.get(pos.underlying, 0.0)
        cluster = cluster_of(pos.underlying)
        beta = STRAT.betas.get(pos.underlying, 1.0) if cluster == "equity" else 1.0
        for leg in pos.legs:
            sign = 1 if leg.side == "buy" else -1
            out[cluster] = out.get(cluster, 0.0) + (
                sign * leg.entry_delta * 100.0 * leg.ratio_qty * pos.qty * spot * beta
            )
    return out


def book_net_vega_dollars(
    positions: list[Position], spots: dict[str, float], asof: Optional[datetime] = None
) -> float:
    asof = asof or now_et()
    total = 0.0
    for pos in positions:
        spot = spots.get(pos.underlying)
        if not spot:
            continue
        for leg in pos.legs:
            t = t_years(leg.expiry, asof)
            res = bs(leg.opt_type == "call", spot, leg.strike, t, leg.entry_iv,
                     STRESS.rate, STRESS.div_yield)
            sign = 1 if leg.side == "buy" else -1
            total += sign * res.vega * 0.01 * 100.0 * leg.ratio_qty * pos.qty
    return total


def check_pre_trade(
    open_positions: list[Position],
    proposal: ProposalLike,
    equity: float,
    hwm: float,
    day_anchor: float,
    halted: bool,
    spots: dict[str, float],
    asof: Optional[datetime] = None,
) -> RiskVerdict:
    reasons: list[str] = []
    asof = asof or now_et()

    action = evaluate_account(equity, hwm, day_anchor, halted)
    if action != AccountAction.OK:
        return RiskVerdict(False, [f"account state: {action.value}"])

    size_factor = 1.0
    consumed = budget_consumed_frac(equity, hwm)
    if consumed >= RISK.derisk_freeze_at:
        return RiskVerdict(False, [f"de-risk ladder: {consumed:.0%} of kill budget consumed"])
    if consumed >= RISK.derisk_half_at:
        size_factor = 0.5

    if proposal.dte < RISK.min_entry_dte:
        reasons.append(f"dte {proposal.dte} below minimum {RISK.min_entry_dte}")

    sleeve = getattr(proposal.position, "sleeve", "A")
    per_trade_cap = SLEEVE_B.crush_max_loss if sleeve == "B" else RISK.per_trade_max_loss
    if proposal.max_loss > per_trade_cap:
        reasons.append(
            f"per-trade max loss {proposal.max_loss:.0f} exceeds sleeve {sleeve} "
            f"cap {per_trade_cap:.0f}"
        )

    if len(open_positions) >= RISK.max_positions:
        reasons.append(f"position count {len(open_positions)} at cap {RISK.max_positions}")
    same_und = [p for p in open_positions if p.underlying == proposal.underlying]
    if len(same_und) >= RISK.max_positions_per_underlying:
        reasons.append(f"{proposal.underlying} already has {len(same_und)} positions")

    sleeve_budget = SLEEVE_B.budget if sleeve == "B" else RISK.sleeve_budget
    committed = sum(p.max_loss for p in open_positions if p.sleeve == sleeve)
    if committed + proposal.max_loss > sleeve_budget:
        reasons.append(
            f"sleeve {sleeve} budget: {committed:.0f} committed + {proposal.max_loss:.0f} "
            f"proposed exceeds {sleeve_budget:.0f}"
        )

    if sleeve == "A":
        prop_cluster = cluster_of(proposal.underlying)
        cluster_cap = STRAT.cluster_budget_frac * RISK.sleeve_budget
        cluster_committed = sum(
            p.max_loss for p in open_positions
            if p.sleeve == "A" and cluster_of(p.underlying) == prop_cluster
        )
        if cluster_committed + proposal.max_loss > cluster_cap:
            reasons.append(
                f"{prop_cluster} cluster budget: {cluster_committed:.0f} committed + "
                f"{proposal.max_loss:.0f} proposed exceeds {cluster_cap:.0f}"
            )

    deltas = cluster_delta_dollars(open_positions + [proposal.position], spots)
    for cluster, exposure in deltas.items():
        cap = STRAT.cluster_delta_caps.get(cluster, RISK.max_net_delta_dollars)
        if abs(exposure) > cap:
            reasons.append(
                f"{cluster} cluster delta {exposure:,.0f} beyond cap {cap:,.0f}"
            )

    vega = book_net_vega_dollars(open_positions, spots, asof) + proposal.net_vega_dollars
    if vega < RISK.min_net_vega:
        reasons.append(f"net vega {vega:,.0f} below floor {RISK.min_net_vega:,.0f}")

    worst, shock = worst_cell(open_positions + [proposal.position], spots, asof)
    if worst < -RISK.book_worst_case:
        reasons.append(
            f"stress grid worst cell {worst:,.0f} at {shock:+.0%} breaches -{RISK.book_worst_case:,.0f}"
        )

    return RiskVerdict(approved=not reasons, reasons=reasons, size_factor=size_factor)
