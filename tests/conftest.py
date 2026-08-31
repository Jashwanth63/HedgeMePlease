"""Shared fixtures: synthetic chains, synthetic bars, and a fake broker that
lets the whole graph run offline with zero network and zero keys.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from alpaca.risk.bs import bs

ET = ZoneInfo("America/New_York")

# Monday Aug 31 2026, 10:30 ET: inside the entry window, no macro blackout.
FIXED_NOW = datetime(2026, 8, 31, 10, 30, tzinfo=ET)


def occ(underlying: str, expiry: str, opt_type: str, strike: float) -> str:
    y, m, d = expiry.split("-")
    cp = "C" if opt_type == "call" else "P"
    return f"{underlying}{y[2:]}{m}{d}{cp}{int(round(strike * 1000)):08d}"


def synthetic_chain(
    underlying: str = "SPY",
    spot: float = 650.0,
    expiry: str = "2026-09-03",
    iv: float = 0.30,
    t_years: float = 3 / 365,
    strikes: range | None = None,
    spread: float = 0.04,
) -> dict:
    strikes = strikes or range(600, 705, 5)
    snapshots = {}
    for strike in strikes:
        for opt_type in ("put", "call"):
            res = bs(opt_type == "call", spot, float(strike), t_years, iv)
            mid = max(res.price, 0.02)
            snapshots[occ(underlying, expiry, opt_type, strike)] = {
                "latestQuote": {"bp": round(mid - spread / 2, 2), "ap": round(mid + spread / 2, 2)},
                "impliedVolatility": iv,
                "greeks": {"delta": res.delta},
            }
    return {"snapshots": snapshots}


def synthetic_bars(
    n_days: int = 90, end_day: datetime | None = None, seed: int = 7, base: float = 650.0
) -> list[dict]:
    """78 five-minute RTH bars per weekday, ending before FIXED_NOW."""
    end_day = end_day or (FIXED_NOW - timedelta(days=1))
    rng = random.Random(seed)
    days = []
    cursor = end_day
    while len(days) < n_days:
        if cursor.weekday() < 5:
            days.append(cursor.date())
        cursor -= timedelta(days=1)
    days.reverse()

    bars = []
    price = base
    for day in days:
        price *= 1 + rng.gauss(0, 0.004)
        for i in range(78):
            price *= 1 + rng.gauss(0, 0.0006)
            hh = 9 + (30 + 5 * i) // 60
            mm = (30 + 5 * i) % 60
            bars.append({"t": f"{day.isoformat()}T{hh:02d}:{mm:02d}:00-04:00", "c": round(price, 4)})
    return bars


class FakeBroker:
    """Implements the AlpacaMCP surface with synthetic data, no network."""

    def __init__(self, equity: float = 100_000.0, is_open: bool = True):
        self._equity = equity
        self._is_open = is_open
        self.placed_orders: list[dict] = []
        self.canceled: list[str] = []
        self.spot_map = {"SPY": 650.0, "QQQ": 560.0, "GLD": 310.0, "TLT": 90.0,
                         "DELL": 140.0, "AVGO": 300.0}

    async def clock(self) -> dict:
        return {"is_open": self._is_open}

    async def account(self) -> dict:
        return {"equity": self._equity, "cash": self._equity}

    async def equity(self) -> float:
        return self._equity

    async def positions(self) -> list[dict]:
        return []

    async def open_orders(self) -> list[dict]:
        return []

    async def spots(self, symbols: list[str]) -> dict[str, float]:
        return {s: self.spot_map.get(s, 0.0) for s in symbols}

    async def stock_bars_5min(self, symbol: str, days: int) -> list[dict]:
        return synthetic_bars(base=self.spot_map.get(symbol, 650.0))

    async def option_chain(self, underlying: str, **kwargs) -> dict:
        spot = self.spot_map.get(underlying, 650.0)
        if underlying in ("DELL", "AVGO"):
            step = 1 if spot < 200 else 5
            lo, hi = int(spot * 0.75), int(spot * 1.25) + step
            return synthetic_chain(
                underlying, spot, "2026-09-04", iv=0.65, t_years=4 / 365,
                strikes=range(lo, hi, step),
            )
        if kwargs.get("expiration_date_gte"):
            lte = str(kwargs.get("expiration_date_lte") or "")
            if lte and lte <= "2026-09-15":  # hedge-band request: next-week expiry
                lo = int(spot * 0.88) // 5 * 5
                hi = int(spot * 1.00) // 5 * 5 + 5
                return synthetic_chain(
                    underlying, spot, "2026-09-09", iv=0.15, t_years=9 / 365,
                    strikes=range(lo, hi, 5),
                )
            lo = int(spot * 0.95) // 5 * 5
            hi = int(spot * 1.05) // 5 * 5 + 5
            return synthetic_chain(
                underlying, spot, "2026-09-30", iv=0.32, t_years=30 / 365,
                strikes=range(lo, hi, 5),
            )
        lo = int(spot * 0.86) // 5 * 5
        hi = int(spot * 1.14) // 5 * 5 + 5
        return synthetic_chain(underlying, spot, "2026-09-03", iv=0.30, strikes=range(lo, hi, 5))

    async def option_quotes(self, symbols: list[str]) -> dict[str, dict]:
        import re

        out = {}
        for s in symbols:
            m = re.match(r"^([A-Z]{1,6})(\d{6})([CP])\d{8}$", s)
            if not m:
                out[s] = {"bp": 0.48, "ap": 0.52}
                continue
            und, yymmdd, cp = m.groups()
            strike = int(s[-8:]) / 1000.0
            spot = self.spot_map.get(und, 650.0)
            from datetime import date
            expiry = date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:]))
            t = max((expiry - FIXED_NOW.date()).days, 1) / 365
            if und in ("DELL", "AVGO"):
                iv = 0.65
            elif yymmdd == "260909":
                iv = 0.15  # matches the hedge-band chain
            elif yymmdd == "260930":
                iv = 0.32  # matches the far chain
            else:
                iv = 0.30
            mid = max(bs(cp == "C", spot, strike, t, iv).price, 0.02)
            out[s] = {"bp": round(mid - 0.02, 2), "ap": round(mid + 0.02, 2)}
        return out

    async def news(self, symbol: str, limit: int = 12) -> list[dict]:
        return []

    async def place_option_order(self, qty, legs, limit_price, client_order_id) -> dict:
        order = {
            "id": f"ord-{len(self.placed_orders)}",
            "client_order_id": client_order_id,
            "status": "accepted",
            "qty": str(qty),
            "legs": legs,
            "limit_price": limit_price,
        }
        self.placed_orders.append(order)
        return order

    async def order_by_client_id(self, client_order_id: str) -> dict:
        for order in self.placed_orders:
            if order["client_order_id"] == client_order_id:
                return order
        raise RuntimeError("unknown order")

    async def cancel_order(self, order_id: str):
        self.canceled.append(order_id)
        for order in self.placed_orders:
            if order["id"] == order_id:
                order["status"] = "canceled"
        return {"ok": True}


@pytest.fixture(autouse=True)
def _agents_offline(monkeypatch):
    """Tests are offline and free: no OpenRouter key ever reaches the desk,
    even when the developer's .env has a real one loaded."""
    import alpaca.agents.desk as desk

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    desk._llm = None
    desk._llm_checked = False
    yield
    desk._llm = None
    desk._llm_checked = False


@pytest.fixture
def fixed_now() -> datetime:
    return FIXED_NOW
