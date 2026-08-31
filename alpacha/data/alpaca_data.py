"""
Alpaca Data & Broker Client wrapper.
Interfaces with Alpaca SDK for market data, options chains, account state, and order execution.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame

from alpacha.config import Settings
from alpacha.utils.logger import get_logger

logger = get_logger("alpaca_data")


class AlpacaDataClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        creds = settings.credentials

        if creds and creds.api_key_id and creds.api_secret_key:
            self.trading_client = TradingClient(
                api_key=creds.api_key_id,
                secret_key=creds.api_secret_key,
                paper=settings.app.paper,
            )
            self.stock_data_client = StockHistoricalDataClient(
                api_key=creds.api_key_id,
                secret_key=creds.api_secret_key,
            )
            self.option_data_client = OptionHistoricalDataClient(
                api_key=creds.api_key_id,
                secret_key=creds.api_secret_key,
            )
            self.is_connected = True
            logger.info(f"Initialized Alpaca Client (Paper={settings.app.paper})")
        else:
            self.trading_client = None
            self.stock_data_client = None
            self.option_data_client = None
            self.is_connected = False
            logger.warning("Alpaca credentials not provided. Operating in offline/mock mode.")

    def get_stock_bars(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetches 1-minute historical bars for a stock symbol."""
        if not self.is_connected or not self.stock_data_client:
            logger.warning("Alpaca client not connected. Returning empty DataFrame.")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        if end is None:
            end = datetime.now(timezone.utc)

        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            barset = self.stock_data_client.get_stock_bars(req)
            if symbol in barset.data:
                bars = barset.data[symbol]
                data = [
                    {
                        "timestamp": b.timestamp,
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": int(b.volume),
                    }
                    for b in bars
                ]
                df = pd.DataFrame(data)
                if not df.empty:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                return df
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        except Exception as e:
            logger.error(f"Failed to fetch stock bars for {symbol}: {e}", exc_info=True)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Fetches the latest price for a symbol."""
        if not self.is_connected:
            return None
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(minutes=15)
            df = self.get_stock_bars(symbol, start=start, end=end)
            if not df.empty:
                return float(df["close"].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def get_account_equity(self) -> float:
        """Returns current account equity."""
        if not self.is_connected or not self.trading_client:
            return 100000.0
        try:
            account = self.trading_client.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Failed to get account equity: {e}", exc_info=True)
            return 0.0

    def get_buying_power(self) -> float:
        """Returns available options buying power."""
        if not self.is_connected or not self.trading_client:
            return 100000.0
        try:
            account = self.trading_client.get_account()
            return float(account.options_buying_power or account.buying_power or account.cash)
        except Exception as e:
            logger.error(f"Failed to get buying power: {e}", exc_info=True)
            return 0.0

    def get_open_positions(self) -> List[Any]:
        """Returns all open positions."""
        if not self.is_connected or not self.trading_client:
            return []
        try:
            return self.trading_client.get_all_positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}", exc_info=True)
            return []

    def close_all_positions(self) -> None:
        """Liquidates all positions and cancels open orders (Kill Switch action)."""
        if not self.is_connected or not self.trading_client:
            logger.warning("[MOCK] Liquidating all positions and canceling orders.")
            return
        try:
            self.trading_client.cancel_orders()
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.info("Successfully liquidated all positions and canceled open orders.")
        except Exception as e:
            logger.critical(f"Failed to close all positions during kill switch: {e}", exc_info=True)
            raise

    def get_option_chain(self, symbol: str) -> Dict[str, Any]:
        """Fetches options chain snapshots for an underlying symbol."""
        if not self.is_connected or not self.option_data_client:
            return {}
        try:
            req = OptionChainRequest(underlying_symbol=symbol)
            chain = self.option_data_client.get_option_chain(req)
            return chain
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {symbol}: {e}", exc_info=True)
            return {}

