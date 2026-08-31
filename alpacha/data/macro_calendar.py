"""
Macro economic calendar loader and event filter.
Checks proximity to high-impact economic events (FOMC, CPI, NFP, GDP, etc.).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pytz

from alpacha.utils.logger import get_logger

logger = get_logger("macro_calendar")


class MacroCalendar:
    def __init__(self, calendar_path: str | Path = "config/macro_calendar.json") -> None:
        self.calendar_path = Path(calendar_path)
        self.events: List[Dict[str, Any]] = []
        self.load_events()

    def load_events(self) -> None:
        """Loads events from the JSON file."""
        if not self.calendar_path.exists():
            logger.warning(f"Macro calendar file not found at {self.calendar_path}. Operating with empty events list.")
            self.events = []
            return

        try:
            with open(self.calendar_path, "r", encoding="utf-8") as f:
                raw_events = json.load(f)
            
            parsed_events = []
            for item in raw_events:
                # Parse ISO timestamp
                ts_str = item.get("timestamp")
                if ts_str:
                    dt = datetime.fromisoformat(ts_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    parsed_events.append({
                        "event": item.get("event", "Unknown Event"),
                        "timestamp": dt,
                        "importance": item.get("importance", "HIGH").upper(),
                    })
            self.events = sorted(parsed_events, key=lambda x: x["timestamp"])
            logger.info(f"Loaded {len(self.events)} macro events from {self.calendar_path}")
        except Exception as e:
            logger.error(f"Failed to load macro calendar: {e}", exc_info=True)
            self.events = []

    def check_event_proximity(
        self,
        target_time: Optional[datetime] = None,
        buffer_hours: float = 2.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Checks if any high-impact macro event falls within ±buffer_hours of target_time.
        Returns:
            (is_blocked, reason)
        """
        if not self.events:
            return False, None

        if target_time is None:
            target_time = datetime.now(timezone.utc)
        elif target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        else:
            target_time = target_time.astimezone(timezone.utc)

        buffer_delta = timedelta(hours=buffer_hours)
        window_start = target_time - buffer_delta
        window_end = target_time + buffer_delta

        for ev in self.events:
            if ev["importance"] == "HIGH":
                ev_time = ev["timestamp"]
                if window_start <= ev_time <= window_end:
                    time_diff_min = int((ev_time - target_time).total_seconds() / 60)
                    reason = (
                        f"Macro event '{ev['event']}' at {ev_time.isoformat()} "
                        f"is within {buffer_hours}h window ({time_diff_min} min away)"
                    )
                    logger.warning(f"Macro Gate Blocked: {reason}")
                    return True, reason

        return False, None
