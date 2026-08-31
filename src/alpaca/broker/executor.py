"""Order lifecycle with the team's time-boxed ladder.

Multi-leg limit convention: positive limit price = net debit, negative = net
credit. Opening a condor posts a negative limit. The ladder: post at mid,
concede improve_step at wait_seconds intervals up to max_improvements, never
past the credit floor, then confirmed-cancel and walk away. No re-chasing
within a cycle. Partial quantity fills are kept: every filled unit is a
complete defined-risk condor. Leg risk cannot exist: mleg fills atomically.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..config import EXEC, STRAT, now_et
from ..risk.ledger import Position

TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day"}


def ladder_prices(mid_credit: float, width: float) -> list[float]:
    """Limit prices for an opening credit order, most aggressive first.

    mid_credit is negative (credit). Each concession moves toward zero by
    improve_step but never above the credit floor for the structure.
    """
    floor_credit = STRAT.min_credit_frac_condor * width
    prices: list[float] = []
    for i in range(EXEC.max_improvements + 1):
        price = round(mid_credit + EXEC.improve_step * i, 2)
        if -price < floor_credit - 1e-9:
            break
        prices.append(price)
    return prices


def open_legs(position: Position) -> list[dict]:
    return [
        {
            "symbol": leg.symbol,
            "ratio_qty": str(leg.ratio_qty),
            "side": leg.side,
            "position_intent": "sell_to_open" if leg.side == "sell" else "buy_to_open",
        }
        for leg in position.legs
    ]


def close_legs(position: Position) -> list[dict]:
    return [
        {
            "symbol": leg.symbol,
            "ratio_qty": str(leg.ratio_qty),
            "side": "buy" if leg.side == "sell" else "sell",
            "position_intent": "buy_to_close" if leg.side == "sell" else "sell_to_close",
        }
        for leg in position.legs
    ]


async def net_mid(mcp, position: Position, closing: bool) -> Optional[float]:
    """Net structure price at current mids. Positive = debit to us."""
    quotes = await mcp.option_quotes([leg.symbol for leg in position.legs])
    total = 0.0
    for leg in position.legs:
        q = quotes.get(leg.symbol) or {}
        bp, ap = float(q.get("bp") or 0.0), float(q.get("ap") or 0.0)
        if ap <= 0 or bp < 0:
            return None
        if bp == 0 and ap > 0.10:
            return None  # zero bid under a fat ask is a stale or garbage quote
        # zero bid with a tiny ask is the normal state of a worthless far OTM
        # wing near expiry; refusing it would block profit-taking and the
        # contest-end flatten exactly when the condor is winning
        mid = (bp + ap) / 2.0
        opening_sign = 1 if leg.side == "buy" else -1
        sign = -opening_sign if closing else opening_sign
        total += sign * mid * leg.ratio_qty
    return total


async def _await_terminal(mcp, client_order_id: str, timeout_s: int) -> tuple[str, Optional[dict]]:
    waited = 0
    step = max(EXEC.poll_seconds, 1)
    order: Optional[dict] = None
    while waited < timeout_s:
        await asyncio.sleep(EXEC.poll_seconds)
        waited += step
        try:
            order = await mcp.order_by_client_id(client_order_id)
        except Exception:
            continue
        status = str(order.get("status", "")).lower()
        if status in TERMINAL:
            return status, order
    return "working", order


async def _confirmed_cancel(mcp, memo, order: Optional[dict], client_order_id: str) -> Optional[dict]:
    """Cancel and verify. Returns the final order dict (may show a partial fill)."""
    if not order:
        try:
            order = await mcp.order_by_client_id(client_order_id)
        except Exception:
            memo("cancel_lookup_failed", {"client_order_id": client_order_id})
            return None
    order_id = order.get("id")
    if not order_id:
        return order
    try:
        await mcp.cancel_order(order_id)
    except Exception:
        pass
    for _ in range(6):
        await asyncio.sleep(EXEC.poll_seconds)
        try:
            order = await mcp.order_by_client_id(client_order_id)
        except Exception:
            continue
        if str(order.get("status", "")).lower() in TERMINAL:
            return order
    memo("cancel_unconfirmed", {"client_order_id": client_order_id})
    return order


def _filled_qty(order: Optional[dict]) -> int:
    try:
        return int(float(order.get("filled_qty") or 0)) if order else 0
    except (TypeError, ValueError):
        return 0


def is_long_structure(position: Position) -> bool:
    """All legs bought: a debit position (e.g. the run-up strangle)."""
    return all(leg.side == "buy" for leg in position.legs)


def _max_loss_for(position: Position) -> float:
    """Long structures risk exactly the debit paid (stored in credit);
    credit structures risk width minus credit."""
    if is_long_structure(position):
        return round(position.credit * 100 * position.qty, 2)
    return round((position.width - position.credit) * 100 * position.qty, 2)


def debit_ladder_prices(mid_debit: float, max_debit_price: float) -> list[float]:
    """Limit prices for an opening DEBIT order (long structures): start at the
    mid and pay up by improve_step, never beyond the per-unit debit cap."""
    prices: list[float] = []
    for i in range(EXEC.max_improvements + 1):
        price = round(mid_debit + EXEC.improve_step * i, 2)
        if price > max_debit_price + 1e-9:
            break
        prices.append(price)
    return prices


async def submit_open(
    mcp, memo, position: Position, max_debit_price: Optional[float] = None
) -> bool:
    """Work an opening order. Credit structures (net mid negative) concede
    toward zero; debit structures (net mid positive, max_debit_price required)
    pay up toward their cap. Fill handling adapts max_loss to the structure."""
    mid = await net_mid(mcp, position, closing=False)
    is_debit = max_debit_price is not None
    if mid is None or (not is_debit and mid >= 0) or (is_debit and mid <= 0):
        memo("open_abandoned", {"position": position.position_id, "sleeve": position.sleeve, "reason": "no usable mid"})
        position.status = "abandoned"
        return False

    if is_debit:
        prices = debit_ladder_prices(mid, max_debit_price)
        abandon_reason = "mid already above debit cap"
    else:
        prices = ladder_prices(mid, position.width)
        abandon_reason = "mid already below credit floor"
    if not prices:
        memo("open_abandoned", {"position": position.position_id, "sleeve": position.sleeve, "reason": abandon_reason})
        position.status = "abandoned"
        return False

    for attempt, price in enumerate(prices):
        coid = position.client_order_id if attempt == 0 else f"{position.client_order_id}-r{attempt}"
        try:
            await mcp.place_option_order(
                qty=position.qty, legs=open_legs(position), limit_price=price, client_order_id=coid
            )
        except Exception as exc:
            memo("open_rejected", {"position": position.position_id, "sleeve": position.sleeve, "error": str(exc)[:400]})
            position.status = "abandoned"
            return False

        status, order = await _await_terminal(mcp, coid, EXEC.wait_seconds)
        if status == "filled":
            fill = order.get("filled_avg_price") if order else None
            if fill is not None:
                position.credit = abs(float(fill))
                position.max_loss = _max_loss_for(position)
            position.status = "open"
            position.client_order_id = coid
            memo("opened", {"position": position.position_id, "sleeve": position.sleeve, "price": price, "fill": fill})
            return True

        final = await _confirmed_cancel(mcp, memo, order, coid)
        partial = _filled_qty(final)
        if partial > 0:
            fill = final.get("filled_avg_price") if final else None
            position.qty = partial
            if fill is not None:
                position.credit = abs(float(fill))
            position.max_loss = _max_loss_for(position)
            position.status = "open"
            position.client_order_id = coid
            memo("opened_partial", {"position": position.position_id, "sleeve": position.sleeve, "filled_qty": partial, "fill": fill})
            return True
        if final is None or str(final.get("status", "")).lower() not in TERMINAL:
            # the old order may still be live at the exchange; requoting now
            # risks a double fill, so abandon the ladder entirely
            memo("open_abandoned", {"position": position.position_id, "sleeve": position.sleeve, "reason": "cancel unconfirmed"})
            position.status = "abandoned"
            return False
        memo("open_requote", {"position": position.position_id, "sleeve": position.sleeve, "attempt": attempt, "price": price})

    memo("open_abandoned", {"position": position.position_id, "sleeve": position.sleeve, "reason": "time box expired unfilled"})
    position.status = "abandoned"
    return False


async def submit_close(mcp, memo, position: Position, reason: str) -> bool:
    mid = await net_mid(mcp, position, closing=True)
    if mid is None:
        memo("close_no_quotes", {"position": position.position_id, "sleeve": position.sleeve})
        return False
    position.status = "closing"
    long_close = mid < 0  # closing an all-long structure nets a credit to us

    steps = EXEC.max_improvements + EXEC.close_extra_steps + 1
    stamp = now_et().strftime("%H%M%S")  # close ladders may rerun across cycles;
    for attempt in range(steps):        # client order ids must never repeat
        if long_close:
            price = round(min(mid + EXEC.improve_step * attempt, -0.01), 2)
        else:
            price = round(max(mid, 0.01) + EXEC.improve_step * attempt, 2)
        coid = f"{position.position_id}-x{attempt}-{stamp}"
        try:
            await mcp.place_option_order(
                qty=position.qty, legs=close_legs(position), limit_price=price, client_order_id=coid
            )
        except Exception as exc:
            memo("close_rejected", {"position": position.position_id, "sleeve": position.sleeve, "error": str(exc)[:400]})
            return False

        status, order = await _await_terminal(mcp, coid, EXEC.wait_seconds)
        if status == "filled":
            fill = order.get("filled_avg_price") if order else None
            exit_price = abs(float(fill)) if fill is not None else abs(price)
            position.status = "closed"
            position.closed_at = now_et().isoformat()
            position.close_order_id = coid
            position.close_reason = reason
            position.exit_debit = exit_price
            if long_close:  # proceeds received minus debit paid
                position.realized_pnl = round((exit_price - position.credit) * 100 * position.qty, 2)
            else:           # credit received minus buy-back debit
                position.realized_pnl = round((position.credit - exit_price) * 100 * position.qty, 2)
            memo("closed", {
                "position": position.position_id, "reason": reason,
                "exit_price": exit_price, "realized_pnl": position.realized_pnl,
            })
            return True
        final = await _confirmed_cancel(mcp, memo, order, coid)
        if final is None or str(final.get("status", "")).lower() not in TERMINAL:
            memo("close_cancel_unconfirmed", {"position": position.position_id, "sleeve": position.sleeve, "reason": reason})
            return False

    memo("close_unfilled", {"position": position.position_id, "sleeve": position.sleeve, "reason": reason})
    return False


async def cost_to_close(mcp, position: Position) -> Optional[float]:
    return await net_mid(mcp, position, closing=True)
