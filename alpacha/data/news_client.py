"""
Alpaca Live News API Client.
Fetches breaking financial news, earnings headlines, and market commentary.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import requests

from alpacha.config import Settings
from alpacha.utils.logger import get_logger

logger = get_logger("news_client")


class AlpacaNewsClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        creds = settings.credentials

        self.api_key = creds.api_key_id if creds else ""
        self.api_secret = creds.api_secret_key if creds else ""
        self.data_base_url = "https://data.alpaca.markets"
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.is_connected = bool(self.api_key and self.api_secret)

    def get_latest_news(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 20,
        include_content: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetches breaking news articles from Alpaca News API.
        """
        if not self.is_connected:
            logger.warning("News client not connected. Returning empty list.")
            return []

        params: Dict[str, Any] = {
            "limit": limit,
            "include_content": str(include_content).lower(),
        }
        if symbols:
            params["symbols"] = ",".join(symbols)

        try:
            url = f"{self.data_base_url}/v1beta1/news"
            resp = requests.get(url, headers=self.headers, params=params, timeout=10.0)
            if resp.status_code >= 400:
                logger.error(f"Alpaca News API error [{resp.status_code}]: {resp.text}")
                return []

            data = resp.json()
            articles = data.get("news", [])
            logger.info(f"Retrieved {len(articles)} live news headlines from Alpaca")
            return articles
        except Exception as e:
            logger.error(f"Failed to fetch news from Alpaca: {e}")
            return []
