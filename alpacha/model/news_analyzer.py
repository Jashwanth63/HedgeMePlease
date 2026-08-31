"""
Quantitative News Sentiment & Volatility Edge Analyzer.
Evaluates breaking news risk scores, directional sentiment, and post-event IV crush opportunities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from alpacha.utils.logger import get_logger

logger = get_logger("news_analyzer")

# Lexicons for financial sentiment & risk analysis
HIGH_RISK_KEYWORDS = {
    "investigation", "subpoena", "fraud", "sec", "lawsuit", "indictment",
    "halted", "default", "bankruptcy", "restructuring", "warning", "plunge",
    "collapse", "recession", "war", "tariff", "sanction", "resigns", "scandal"
}

POSITIVE_KEYWORDS = {
    "beat", "record", "growth", "upgrade", "buyback", "dividend", "surge",
    "outperform", "profit", "expansion", "approval", "rally", "strong", "bullish"
}

NEGATIVE_KEYWORDS = {
    "miss", "downgrade", "cut", "decline", "drop", "lower", "loss", "bearish",
    "weak", "slump", "deficit", "fall", "warning", "layoffs", "delay"
}


@dataclass
class NewsAnalysisResult:
    symbol: str
    article_count: int
    sentiment_score: float         # [-1.0 (bearish) to +1.0 (bullish)]
    event_risk_score: float        # [0.0 (safe) to 1.0 (extreme risk)]
    is_event_risk_high: bool       # True if event_risk_score >= 0.65
    is_iv_crush_opportunity: bool  # True if high IV + stable post-news sentiment
    sizing_multiplier: float       # Multiplier for contract sizing [0.0 to 1.5]
    top_headlines: List[str] = field(default_factory=list)


class NewsSentimentAnalyzer:
    def __init__(self, risk_threshold: float = 0.65) -> None:
        self.risk_threshold = risk_threshold

    def analyze_news(self, symbol: str, articles: List[Dict[str, Any]]) -> NewsAnalysisResult:
        """
        Analyzes news articles for a specific symbol to derive sentiment, event risk, and sizing factors.
        """
        if not articles:
            return NewsAnalysisResult(
                symbol=symbol,
                article_count=0,
                sentiment_score=0.0,
                event_risk_score=0.10,
                is_event_risk_high=False,
                is_iv_crush_opportunity=False,
                sizing_multiplier=1.0,
                top_headlines=[],
            )

        pos_count = 0
        neg_count = 0
        risk_count = 0
        total_words = 0
        headlines = []

        for art in articles:
            headline = art.get("headline", "")
            summary = art.get("summary", "")
            text = f"{headline} {summary}".lower()
            headlines.append(headline)

            words = re.findall(r"\b[a-z]+\b", text)
            total_words += len(words)

            for w in words:
                if w in POSITIVE_KEYWORDS:
                    pos_count += 1
                elif w in NEGATIVE_KEYWORDS:
                    neg_count += 1
                if w in HIGH_RISK_KEYWORDS:
                    risk_count += 1

        # Calculate directional sentiment score [-1.0, +1.0]
        total_sent_words = pos_count + neg_count
        if total_sent_words > 0:
            sentiment_score = round((pos_count - neg_count) / total_sent_words, 2)
        else:
            sentiment_score = 0.0

        # Calculate event risk score [0.0, 1.0]
        risk_score = min(1.0, round((risk_count * 0.25) + (neg_count * 0.05), 2))
        is_high_risk = risk_score >= self.risk_threshold

        # Post-event IV Crush Opportunity: High news activity without toxic existential risk
        is_iv_crush = (len(articles) >= 1) and not is_high_risk and (risk_score < 0.40)

        # Sizing multiplier: Boost safe high-edge setups, reduce risky ones
        if is_high_risk:
            sizing_multiplier = 0.0  # Block trading
        elif is_iv_crush:
            sizing_multiplier = 1.35  # Scale up to capture fast premium collapse
        else:
            sizing_multiplier = 1.0


        logger.info(
            f"News Analysis for {symbol}: Sentiment={sentiment_score:+.2f}, "
            f"Risk={risk_score:.2f}, HighRisk={is_high_risk}, IVCrush={is_iv_crush}, SizingMult={sizing_multiplier:.2f}x"
        )

        return NewsAnalysisResult(
            symbol=symbol,
            article_count=len(articles),
            sentiment_score=sentiment_score,
            event_risk_score=risk_score,
            is_event_risk_high=is_high_risk,
            is_iv_crush_opportunity=is_iv_crush,
            sizing_multiplier=sizing_multiplier,
            top_headlines=headlines[:3],
        )
