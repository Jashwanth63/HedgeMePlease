"""Sleeve B: earnings-event structures. Phase one (implemented): the IV crush
condor, sold in the final minutes before an after-close print with short
strikes at least one implied move away, covered the next morning.

The implied move comes from the post-event expiry's ATM straddle price, the
market's own bet on the overnight jump. Evidence: the implied move overstates
the realized move roughly 70 to 75 percent of the time. Everything is defined
risk and deliberately tiny.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..config import SLEEVE_B, STRAT, EventTrade
from ..risk.bs import bs
from ..risk.ledger import Leg, Position
from ..risk.stress import t_years
from .condor import Contract, Proposal, dte_of


def _quote_ok(c: Contract) -> bool:
    """Single names quote wider than index ETFs; relax the spread tolerance."""
    if c.bid <= 0 or c.ask <= 0 or c.ask < c.bid:
        return False
    return (c.ask - c.bid) <= max(0.15, SLEEVE_B.quote_spread_frac * c.mid)


def implied_move(contracts: list[Contract], expiry: str, spot: float) -> Optional[float]:
    """ATM straddle mid of the post-event expiry, in dollars of underlying."""
    pool = [c for c in contracts if c.expiry == expiry and _quote_ok(c)]
    if not pool:
        return None
    strikes = sorted({c.strike for c in pool}, key=lambda k: abs(k - spot))
    for strike in strikes[:3]:
        call = next((c for c in pool if c.strike == strike and c.opt_type == "call"), None)
        put = next((c for c in pool if c.strike == strike and c.opt_type == "put"), None)
        if call and put:
            return call.mid + put.mid
    return None


def _shorts_at_or_beyond(
    contracts: list[Contract], expiry: str, opt_type: str, boundary: float, below: bool, n: int = 3
) -> list[Contract]:
    """The n qualifying short strikes at or beyond the boundary, nearest first.

    Pre-print quote churn can kill the nearest strike on one side only; picking
    each side's single nearest then pairs a close short with a far one — a
    directional bet wearing a condor. The builder needs alternatives per side
    so it can choose the delta-balanced pairing.
    """
    pool = [
        c for c in contracts
        if c.expiry == expiry and c.opt_type == opt_type and _quote_ok(c)
        and (c.strike <= boundary if below else c.strike >= boundary)
    ]
    pool.sort(key=lambda c: -c.strike if below else c.strike)
    return pool[:n]


def _wing(contracts: list[Contract], short: Contract, width: float) -> Optional[Contract]:
    direction = -1 if short.opt_type == "put" else 1
    target = short.strike + direction * width
    pool = [
        c for c in contracts
        if c.expiry == short.expiry and c.opt_type == short.opt_type and _quote_ok(c)
        and (c.strike < short.strike if direction < 0 else c.strike > short.strike)
        and 0.5 * width <= abs(c.strike - short.strike) <= 1.5 * width
    ]
    if not pool:
        return None
    return min(pool, key=lambda c: abs(c.strike - target))


def _leg(contract: Contract, side: str, spot: float, now: datetime) -> Leg:
    delta, iv = contract.delta, contract.iv
    if not delta or not iv:
        t = t_years(contract.expiry, now)
        est = bs(contract.opt_type == "call", spot, contract.strike, t, max(iv, 0.4))
        delta = delta or est.delta
        iv = iv or 0.4
    return Leg(
        symbol=contract.symbol, side=side, ratio_qty=1, strike=contract.strike,
        opt_type=contract.opt_type, expiry=contract.expiry, entry_iv=iv, entry_delta=delta,
    )


def build_runup_strangle(
    event: EventTrade,
    contracts: list[Contract],
    spot: float,
    now: datetime,
) -> tuple[Optional[Proposal], dict]:
    """Long OTM strangle on the post-event expiry, bought ahead of the print
    to harvest the documented pre-earnings IV run-up. Max loss is the debit,
    capped by config; exits before the print, always."""
    diag: dict = {"symbol": event.symbol, "phase": "runup"}
    if spot <= 0:
        diag["reject"] = "no spot"
        return None, diag

    calls = sorted(
        (c for c in contracts if c.expiry == event.post_expiry and c.opt_type == "call"
         and _quote_ok(c) and c.strike > spot),
        key=lambda c: c.strike,
    )
    puts = sorted(
        (c for c in contracts if c.expiry == event.post_expiry and c.opt_type == "put"
         and _quote_ok(c) and c.strike < spot),
        key=lambda c: -c.strike,
    )
    if not calls or not puts:
        diag["reject"] = "no OTM strikes with usable quotes"
        return None, diag

    max_debit_unit = SLEEVE_B.runup_max_debit / 100.0
    for step in range(min(4, len(calls), len(puts))):
        call, put = calls[step], puts[step]
        debit = call.mid + put.mid
        if debit <= 0:
            continue
        if debit > max_debit_unit:
            diag[f"step_{step}"] = f"debit {debit:.2f} over cap {max_debit_unit:.2f}"
            continue
        legs = [_leg(put, "buy", spot, now), _leg(call, "buy", spot, now)]
        net_delta = sum(l.entry_delta * 100.0 * spot for l in legs)
        net_vega = 0.0
        for l in legs:
            t = t_years(l.expiry, now)
            est = bs(l.opt_type == "call", spot, l.strike, t, l.entry_iv)
            net_vega += est.vega * 0.01 * 100.0

        position_id = f"SLB-{event.symbol}-RU-{now.strftime('%m%d%H%M%S')}"
        position = Position(
            position_id=position_id, sleeve="B", underlying=event.symbol,
            structure="event_runup_strangle", legs=legs, qty=1,
            credit=round(debit, 2),            # for long structures: the debit paid
            width=abs(call.strike - put.strike),
            max_loss=round(debit * 100, 2),
            client_order_id=position_id, opened_at=now.isoformat(),
        )
        proposal = Proposal(
            underlying=event.symbol, structure="event_runup_strangle", legs=legs, qty=1,
            credit=round(debit, 2), width=position.width, max_loss=position.max_loss,
            net_delta_dollars=net_delta, net_vega_dollars=net_vega,
            dte=dte_of(event.post_expiry, now), position=position,
        )
        diag["proposal"] = proposal.summary()
        diag["debit"] = round(debit, 2)
        return proposal, diag

    diag.setdefault("reject", "no strangle inside the debit cap")
    return None, diag


def build_crush_condor(
    event: EventTrade,
    contracts: list[Contract],
    spot: float,
    now: datetime,
    max_abs_delta: Optional[float] = None,
) -> tuple[Optional[Proposal], dict]:
    """Among every valid strike pairing, take the most delta-balanced condor.

    A crush condor's edge is the vol collapse, not direction, so of the
    candidates that clear the credit floor and loss cap we sell the one whose
    net dollar delta is smallest — and refuse outright if none fits inside
    max_abs_delta (the cluster's raw cap; hit live on DELL Sep 1 at -16,397).
    """
    diag: dict = {"symbol": event.symbol, "phase": "crush"}
    if spot <= 0:
        diag["reject"] = "no spot"
        return None, diag

    move = implied_move(contracts, event.post_expiry, spot)
    if move is None or move <= 0:
        diag["reject"] = "no usable straddle for implied move"
        return None, diag
    diag["implied_move"] = round(move, 2)
    diag["implied_move_pct"] = round(move / spot, 4)

    offset = move * SLEEVE_B.crush_move_mult
    short_puts = _shorts_at_or_beyond(contracts, event.post_expiry, "put", spot - offset, below=True)
    short_calls = _shorts_at_or_beyond(contracts, event.post_expiry, "call", spot + offset, below=False)
    if not (short_puts and short_calls):
        diag["reject"] = "no short strikes beyond the implied move"
        return None, diag

    best = None  # (abs_delta, -credit, parts)
    rejects: dict[str, int] = {}
    for short_put in short_puts:
        for short_call in short_calls:
            for width in (5.0, 4.0, 3.0):
                put_wing = _wing(contracts, short_put, width)
                call_wing = _wing(contracts, short_call, width)
                if not (put_wing and call_wing):
                    rejects["no_wing"] = rejects.get("no_wing", 0) + 1
                    continue
                eff_width = max(short_put.strike - put_wing.strike, call_wing.strike - short_call.strike)
                credit = (short_put.mid - put_wing.mid) + (short_call.mid - call_wing.mid)
                if credit < SLEEVE_B.min_credit_frac * eff_width:
                    rejects["credit_floor"] = rejects.get("credit_floor", 0) + 1
                    continue
                unit_loss = (eff_width - credit) * 100.0
                if unit_loss <= 0 or unit_loss > SLEEVE_B.crush_max_loss:
                    rejects["loss_cap"] = rejects.get("loss_cap", 0) + 1
                    continue

                legs = [
                    _leg(short_put, "sell", spot, now),
                    _leg(put_wing, "buy", spot, now),
                    _leg(short_call, "sell", spot, now),
                    _leg(call_wing, "buy", spot, now),
                ]
                net_delta = sum(
                    (1 if l.side == "buy" else -1) * l.entry_delta * 100.0 * spot for l in legs
                )
                if max_abs_delta is not None and abs(net_delta) > max_abs_delta:
                    rejects["delta_cap"] = rejects.get("delta_cap", 0) + 1
                    continue
                key = (abs(net_delta), -credit)
                if best is None or key < best[0]:
                    best = (key, legs, credit, eff_width, unit_loss, net_delta)

    diag["rejected_pairings"] = rejects
    if best is None:
        diag["reject"] = (
            "no delta-balanced condor cleared credit floor and loss cap"
            if rejects.get("delta_cap") else "no width satisfied credit floor and loss cap"
        )
        return None, diag

    _, legs, credit, eff_width, unit_loss, net_delta = best
    net_vega = 0.0
    for l in legs:
        t = t_years(l.expiry, now)
        est = bs(l.opt_type == "call", spot, l.strike, t, l.entry_iv)
        net_vega += (1 if l.side == "buy" else -1) * est.vega * 0.01 * 100.0

    position_id = f"SLB-{event.symbol}-{event.post_expiry.replace('-', '')}-{now.strftime('%m%d%H%M%S')}"
    position = Position(
        position_id=position_id, sleeve="B", underlying=event.symbol,
        structure="event_crush_condor", legs=legs, qty=1,
        credit=round(credit, 2), width=eff_width,
        max_loss=round(unit_loss, 2), client_order_id=position_id,
        opened_at=now.isoformat(),
    )
    proposal = Proposal(
        underlying=event.symbol, structure="event_crush_condor", legs=legs, qty=1,
        credit=round(credit, 2), width=eff_width, max_loss=round(unit_loss, 2),
        net_delta_dollars=net_delta, net_vega_dollars=net_vega,
        dte=dte_of(event.post_expiry, now), position=position,
    )
    diag["proposal"] = proposal.summary()
    diag["net_delta"] = round(net_delta)
    return proposal, diag
