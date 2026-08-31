"""
Trading hours and calendar utilities using pandas_market_calendars.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
import pandas_market_calendars as mcal
import pytz

US_EASTERN = pytz.timezone("America/New_York")


def get_nyse_calendar():
    """Returns the NYSE trading calendar."""
    return mcal.get_calendar("NYSE")


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Checks if the US market is currently open for regular trading."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = US_EASTERN.localize(dt).astimezone(timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    nyse = get_nyse_calendar()
    # Schedule for the day
    start_date = dt.strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start_date, end_date=start_date)

    if schedule.empty:
        return False

    market_open = schedule.iloc[0]["market_open"].to_pydatetime()
    market_close = schedule.iloc[0]["market_close"].to_pydatetime()

    return market_open <= dt <= market_close


def get_market_hours_for_date(date_str: str) -> Optional[tuple[datetime, datetime]]:
    """Returns (market_open, market_close) in UTC for a given YYYY-MM-DD."""
    nyse = get_nyse_calendar()
    schedule = nyse.schedule(start_date=date_str, end_date=date_str)
    if schedule.empty:
        return None
    return (
        schedule.iloc[0]["market_open"].to_pydatetime(),
        schedule.iloc[0]["market_close"].to_pydatetime(),
    )


def now_eastern() -> datetime:
    """Returns the current datetime in US/Eastern timezone."""
    return datetime.now(US_EASTERN)
