"""Client for the official Alpaca MCP server — the only path to Alpaca in the
codebase, per hackathon rules. Spawns the server as a stdio subprocess and
calls its tools programmatically.

Integration facts baked in from live verification:
- every result arrives wrapped in a security envelope with the payload under "data"
- the mcp SDK uses snake_case attributes (structured_content, is_error)
- the bars tool has no pagination parameter and the server page cap is ~2000
  bars, so 5 minute history is fetched in 14 day start/end windows
- the option chain tool paginates via page_token
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config import alpaca_env, mcp_server_dir


class McpToolError(RuntimeError):
    def __init__(self, tool: str, message: str):
        super().__init__(f"{tool}: {message}")
        self.tool = tool


def option_order_payload(
    qty: int, legs: list[dict], limit_price: float, client_order_id: str
) -> dict:
    """Translate an order into the shape Alpaca accepts.

    Alpaca's mleg order class requires 2-4 legs (API code 42210000, hit live
    Sep 1); a lone option — the sleeve C hedge put — must go as a plain
    single-leg order: symbol and side on the parent, unsigned limit price,
    direction carried by the side. Signed prices are an mleg-only convention.
    """
    payload: dict[str, Any] = {
        "qty": str(qty),
        "type": "limit",
        "time_in_force": "day",
        "client_order_id": client_order_id,
    }
    if len(legs) == 1:
        leg = legs[0]
        if leg.get("position_intent") == "sell_to_open":
            raise ValueError("single-leg sell_to_open is a naked short; refused")
        payload.update(
            {
                "symbol": leg["symbol"],
                "side": leg["side"],
                "position_intent": leg.get("position_intent"),
                "qty": str(qty * int(leg.get("ratio_qty", 1))),
                "limit_price": f"{abs(limit_price):.2f}",
            }
        )
    elif 2 <= len(legs) <= 4:
        payload.update(
            {
                "limit_price": f"{limit_price:.2f}",
                "order_class": "mleg",
                "legs": legs,
            }
        )
    else:
        raise ValueError(f"option order needs 1-4 legs, got {len(legs)}")
    return payload


class AlpacaMCP:
    """Async context manager owning one MCP session to the Alpaca server."""

    def __init__(self) -> None:
        self._session: Optional[ClientSession] = None
        self._cm = None
        self._session_cm = None

    async def __aenter__(self) -> "AlpacaMCP":
        server_dir = mcp_server_dir()
        if not server_dir.exists():
            raise RuntimeError(
                f"Alpaca MCP server clone not found at {server_dir}. "
                "Clone https://github.com/alpacahq/alpaca-mcp-server there or set ALPACA_MCP_DIR."
            )
        env = {**os.environ, **alpaca_env()}
        # under systemd the service PATH is minimal; resolve uv absolutely
        uv_name = "uv.exe" if os.name == "nt" else "uv"
        uv_cmd = shutil.which("uv") or str(Path.home() / ".local" / "bin" / uv_name)
        params = StdioServerParameters(
            command=uv_cmd,
            args=["run", "--directory", str(server_dir), "alpaca-mcp-server", "--transport", "stdio"],
            env=env,
        )
        self._cm = stdio_client(params)
        read, write = await self._cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(*exc)
        if self._cm is not None:
            await self._cm.__aexit__(*exc)

    async def call(self, tool: str, args: Optional[dict[str, Any]] = None) -> Any:
        assert self._session is not None, "use 'async with AlpacaMCP()'"
        result = await self._session.call_tool(tool, args or {})
        payload: Any = getattr(result, "structured_content", None)
        if payload is None:
            payload = getattr(result, "structuredContent", None)
        if payload is None:
            texts = [c.text for c in result.content if getattr(c, "text", None)]
            raw = "\n".join(texts)
            try:
                payload = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                payload = raw
        is_error = getattr(result, "is_error", None)
        if is_error is None:
            is_error = getattr(result, "isError", False)
        if is_error:
            raise McpToolError(tool, str(payload)[:2000])
        if isinstance(payload, dict) and set(payload.keys()) == {"result"}:
            payload = payload["result"]
        if isinstance(payload, dict) and "_alpaca_mcp_security" in payload:
            payload = payload.get("data")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                pass
        if isinstance(payload, dict) and payload.get("error"):
            raise McpToolError(tool, str(payload)[:2000])
        return payload

    # ---- typed helpers -------------------------------------------------

    async def account(self) -> dict:
        return await self.call("get_account_info")

    async def equity(self) -> float:
        return float((await self.account())["equity"])

    async def clock(self) -> dict:
        return await self.call("get_clock")

    async def positions(self) -> list[dict]:
        # the server wraps list payloads under "result" (verified live Sep 2;
        # the old "positions" fallback silently returned [] forever)
        out = await self.call("get_all_positions")
        return out if isinstance(out, list) else out.get("result", out.get("positions", []))

    async def spots(self, symbols: list[str]) -> dict[str, float]:
        payload = await self.call(
            "get_stock_snapshot", {"symbols": ",".join(symbols), "feed": "iex"}
        )
        snaps = payload.get("snapshots", payload) if isinstance(payload, dict) else {}
        out: dict[str, float] = {}
        for sym in symbols:
            snap = snaps.get(sym) or {}
            trade = snap.get("latestTrade") or {}
            quote = snap.get("latestQuote") or {}
            price = trade.get("p") or 0.0
            if not price and quote:
                bp, ap = quote.get("bp") or 0.0, quote.get("ap") or 0.0
                price = (bp + ap) / 2 if bp and ap else 0.0
            minute = snap.get("minuteBar") or {}
            daily = snap.get("dailyBar") or {}
            out[sym] = float(price or minute.get("c") or daily.get("c") or 0.0)
        return out

    async def stock_bars_5min(self, symbol: str, days: int) -> list[dict]:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=days)
        merged: dict[str, dict] = {}
        window = timedelta(days=14)
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + window, end)
            payload = await self.call(
                "get_stock_bars",
                {
                    "symbols": symbol,
                    "timeframe": "5Min",
                    "start": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": 10000,
                    "feed": "iex",
                },
            )
            bars = payload.get("bars", {}) if isinstance(payload, dict) else {}
            if isinstance(bars, dict):
                bars = bars.get(symbol, [])
            for bar in bars or []:
                merged[bar["t"]] = bar
            cursor = chunk_end
        return [merged[t] for t in sorted(merged)]

    async def option_chain(
        self,
        underlying: str,
        expiration_date_gte: Optional[str] = None,
        expiration_date_lte: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
    ) -> dict:
        merged: dict[str, Any] = {}
        page_token: Optional[str] = None
        for _ in range(20):
            args: dict[str, Any] = {"underlying_symbol": underlying, "limit": 1000}
            if expiration_date_gte:
                args["expiration_date_gte"] = expiration_date_gte
            if expiration_date_lte:
                args["expiration_date_lte"] = expiration_date_lte
            if strike_price_gte is not None:
                args["strike_price_gte"] = strike_price_gte
            if strike_price_lte is not None:
                args["strike_price_lte"] = strike_price_lte
            if page_token:
                args["page_token"] = page_token
            payload = await self.call("get_option_chain", args)
            merged.update(payload.get("snapshots") or {})
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        return {"snapshots": merged}

    async def option_quotes(self, symbols: list[str]) -> dict[str, dict]:
        payload = await self.call("get_option_latest_quote", {"symbols": ",".join(symbols)})
        return payload.get("quotes", {}) if isinstance(payload, dict) else {}

    async def news(self, symbol: str, limit: int = 12) -> list[dict]:
        payload = await self.call(
            "get_news",
            {"symbols": symbol, "limit": limit, "sort": "desc", "exclude_contentless": True},
        )
        articles = payload.get("news", payload) if isinstance(payload, dict) else []
        return articles if isinstance(articles, list) else []

    async def place_option_order(
        self, qty: int, legs: list[dict], limit_price: float, client_order_id: str
    ) -> dict:
        return await self.call("place_option_order", option_order_payload(qty, legs, limit_price, client_order_id))

    async def order_by_client_id(self, client_order_id: str) -> dict:
        return await self.call("get_order_by_client_id", {"client_order_id": client_order_id})

    async def cancel_order(self, order_id: str) -> Any:
        return await self.call("cancel_order_by_id", {"order_id": order_id})

    async def open_orders(self) -> list[dict]:
        # same "result" envelope as positions(). The old "orders" fallback made
        # the daemon's startup stray sweep a no-op all week — discovered when a
        # restart-orphaned requote rested through the AVGO print and filled.
        out = await self.call("get_orders", {"status": "open", "limit": 100, "nested": "true"})
        return out if isinstance(out, list) else out.get("result", out.get("orders", []))
