"""Instantaneous stress grid over the whole book.

Revalues every leg under crossed spot and vol shocks with Black-Scholes and
reports the worst cell. Both base and shocked values come from the same model
so model error largely cancels; only the difference is trusted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from ..config import ET, STRESS
from .bs import bs
from .ledger import Leg, Position


def t_years(expiry: str, asof: datetime) -> float:
    exp = datetime(*[int(x) for x in expiry.split("-")], 16, 0, tzinfo=ET)
    seconds = (exp - asof).total_seconds()
    return max(seconds, 4 * 3600) / (365.0 * 24 * 3600)


def _leg_sign(side: str) -> int:
    return 1 if side == "buy" else -1


def _leg_value(leg: Leg, spot: float, iv: float, t: float) -> float:
    return bs(
        is_call=(leg.opt_type == "call"),
        spot=spot,
        strike=leg.strike,
        t_years=t,
        iv=iv,
        rate=STRESS.rate,
        div_yield=STRESS.div_yield,
    ).price


def _shocked_iv(iv: float, spot_shock: float) -> float:
    if spot_shock == 0.0:
        return iv
    mult = STRESS.vol_mult_down if spot_shock < 0 else STRESS.vol_mult_up
    return max(iv * mult, iv + STRESS.vol_add_floor)


def position_pnl_under_shock(
    pos: Position, spot: float, spot_shock: float, asof: datetime
) -> float:
    pnl = 0.0
    for leg in pos.legs:
        t = t_years(leg.expiry, asof)
        base = _leg_value(leg, spot, leg.entry_iv, t)
        shocked = _leg_value(leg, spot * (1 + spot_shock), _shocked_iv(leg.entry_iv, spot_shock), t)
        pnl += _leg_sign(leg.side) * (shocked - base) * 100.0 * leg.ratio_qty * pos.qty
    return pnl


def worst_cell(
    positions: Iterable[Position], spots: dict[str, float], asof: datetime | None = None
) -> tuple[float, float]:
    asof = asof or datetime.now(tz=ET)
    worst = 0.0
    worst_shock = 0.0
    for shock in STRESS.spot_shocks:
        total = 0.0
        for pos in positions:
            spot = spots.get(pos.underlying)
            if spot is None:
                continue
            total += position_pnl_under_shock(pos, spot, shock, asof)
        if total < worst:
            worst = total
            worst_shock = shock
    return worst, worst_shock
