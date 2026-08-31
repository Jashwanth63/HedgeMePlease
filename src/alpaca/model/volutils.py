"""Volatility math from intraday bars.

daily_stats turns 5 minute bars into one row per COMPLETED trading day:
annualized realized vol (overnight gap included), bipower variation,
a jump component, and the day's log return for the HAR leverage term.

The current (in-progress) day and any day with fewer than MIN_BARS_PER_DAY
bars are excluded: annualizing a partial day understates its vol, and that
biased point would be the heaviest-weighted HAR feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
RTH_START = time(9, 30)
RTH_END = time(16, 0)
ANNUALIZATION = 252.0
MIN_BARS_PER_DAY = 60  # full session is ~78 five-minute bars
BV_SCALE = math.pi / 2.0


@dataclass(frozen=True)
class DayStats:
    day: str
    rv: float        # annualized realized vol, overnight included
    bv: float        # annualized bipower (jump-robust) vol, intraday only
    jump: float      # annualized jump vol component, >= 0
    ret: float       # close-to-close log return (leverage input)


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(ET)


def expected_move(price: float, iv: float, dte_days: float) -> float:
    return price * iv * math.sqrt(max(dte_days, 0.0) / 365.0)


def daily_stats(bars: list[dict], asof: datetime | None = None) -> list[DayStats]:
    asof = asof or datetime.now(tz=ET)
    today = asof.date().isoformat()

    by_day: dict[str, list[tuple[datetime, float]]] = {}
    for bar in bars:
        ts = _parse_ts(bar["t"])
        if not (RTH_START <= ts.time() < RTH_END):
            continue
        by_day.setdefault(ts.date().isoformat(), []).append((ts, float(bar["c"])))

    out: list[DayStats] = []
    prev_close: float | None = None
    for day in sorted(by_day):
        closes = [c for _, c in sorted(by_day[day])]
        if day >= today or len(closes) < MIN_BARS_PER_DAY:
            prev_close = closes[-1] if closes else prev_close
            continue

        rets = [
            math.log(b / a)
            for a, b in zip(closes, closes[1:])
            if a > 0 and b > 0
        ]
        intraday_var = sum(r * r for r in rets)
        bv_var = BV_SCALE * sum(abs(a) * abs(b) for a, b in zip(rets, rets[1:]))
        jump_var = max(0.0, intraday_var - bv_var)

        total_var = intraday_var
        day_ret = 0.0
        if prev_close and prev_close > 0 and closes[0] > 0:
            gap = math.log(closes[0] / prev_close)
            total_var += gap * gap
            day_ret = math.log(closes[-1] / prev_close)

        out.append(
            DayStats(
                day=day,
                rv=math.sqrt(total_var * ANNUALIZATION),
                bv=math.sqrt(bv_var * ANNUALIZATION),
                jump=math.sqrt(jump_var * ANNUALIZATION),
                ret=day_ret,
            )
        )
        prev_close = closes[-1]
    return out
