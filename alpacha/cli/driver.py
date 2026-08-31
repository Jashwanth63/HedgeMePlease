"""
Alpaca CLI & Direct REST Driver.
Direct API interface communicating over HTTP REST with Alpaca paper/live endpoints,
satisfying the requirement to avoid high-level broker SDK dependencies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import requests
import pandas as pd

from alpacha.config import Settings
from alpacha.utils.logger import get_logger

logger = get_logger("alpaca_cli")


class AlpacaCLIDriver:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        creds = settings.credentials

        self.api_key = creds.api_key_id if creds else ""
        self.api_secret = creds.api_secret_key if creds else ""
        self.base_url = creds.base_url.rstrip("/") if creds else "https://paper-api.alpaca.markets"
        self.data_base_url = "https://data.alpaca.markets"

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.is_connected = bool(self.api_key and self.api_secret)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        is_data_api: bool = False,
    ) -> Any:
        if not self.is_connected:
            raise ConnectionError("Alpaca credentials not configured.")

        base = self.data_base_url if is_data_api else self.base_url
        url = f"{base}/{endpoint.lstrip('/')}"

        resp = requests.request(
            method=method.upper(),
            url=url,
            headers=self.headers,
            params=params,
            json=data if data else None,
            timeout=15.0,
        )

        if resp.status_code >= 400:
            logger.error(f"Alpaca CLI Request failed [{resp.status_code}]: {resp.text}")
            resp.raise_for_status()

        return resp.json()

    def get_account(self) -> Dict[str, Any]:
        """GET /v2/account - Retrieves account equity, cash, and buying power."""
        if not self.is_connected:
            return {"equity": 100000.0, "buying_power": 400000.0, "cash": 100000.0, "status": "MOCK"}
        return self._request("GET", "v2/account")

    def get_positions(self) -> List[Dict[str, Any]]:
        """GET /v2/positions - Retrieves open positions."""
        if not self.is_connected:
            return []
        return self._request("GET", "v2/positions")

    def close_all_positions(self, cancel_orders: bool = True) -> List[Dict[str, Any]]:
        """DELETE /v2/positions - Liquidates all positions (Kill Switch)."""
        if not self.is_connected:
            logger.info("[MOCK CLI] Close all positions requested.")
            return []
        params = {"cancel_orders": str(cancel_orders).lower()}
        return self._request("DELETE", "v2/positions", params=params)

    def submit_order(
        self,
        symbol: str,
        qty: int | float,
        side: str,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """POST /v2/orders - Submits an order via direct HTTP REST."""
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force.lower(),
        }
        if limit_price is not None:
            payload["limit_price"] = str(round(limit_price, 2))

        if not self.is_connected:
            logger.info(f"[MOCK CLI] Submitting order: {payload}")
            return {"id": f"mock_{symbol}_{side}", "status": "accepted", **payload}

        return self._request("POST", "v2/orders", data=payload)

    def cancel_all_orders(self) -> List[Dict[str, Any]]:
        """DELETE /v2/orders - Cancels all open orders."""

    def get_stock_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
        feed: Optional[str] = None,
    ) -> pd.DataFrame:
        """GET /v2/stocks/bars - Fetches historical stock bars via direct REST API with pagination."""
        if not self.is_connected:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        feed_val = feed or ("iex" if self.settings.app.paper else "sip")
        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": feed_val,
            "limit": 10000,
        }

        all_bars = []
        try:
            page_token = None
            for _ in range(5):  # Up to 5 pages (50,000 bars ~ 125 trading days)
                if page_token:
                    params["page_token"] = page_token

                data = self._request("GET", "v2/stocks/bars", params=params, is_data_api=True)
                bars = data.get("bars", {}).get(symbol, [])
                if bars:
                    all_bars.extend(bars)

                page_token = data.get("next_page_token")
                if not page_token:
                    break

            if not all_bars:
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

            rows = [
                {
                    "timestamp": b["t"],
                    "open": float(b["o"]),
                    "high": float(b["h"]),
                    "low": float(b["l"]),
                    "close": float(b["c"]),
                    "volume": int(b["v"]),
                }
                for b in all_bars
            ]
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.drop_duplicates(subset=["timestamp"], inplace=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Alpaca CLI Data error for {symbol}: {e}")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


        if not self.is_connected:
            return []
        return self._request("DELETE", "v2/orders")

    def get_orders(self, status: str = "open") -> List[Dict[str, Any]]:
        """GET /v2/orders - Fetches orders by status."""
        if not self.is_connected:
            return []
        return self._request("GET", "v2/orders", params={"status": status})
