import json
from alpacha.model.news_analyzer import NewsSentimentAnalyzer
from alpacha.mcp.tools import alpaca_get_news


def test_news_analyzer_positive_earnings():
    analyzer = NewsSentimentAnalyzer()
    articles = [
        {"headline": "NVIDIA Beats Earnings Estimates with Record Revenue and Strong Guidance", "summary": "Record growth across data centers and strong bullish forecast."},
        {"headline": "Analysts Upgrade NVIDIA Following Surge in AI Demand", "summary": "Outperform rating reiterated with price target upgrade."},
    ]
    res = analyzer.analyze_news("NVDA", articles)
    assert res.sentiment_score > 0.0
    assert res.is_event_risk_high is False
    assert res.is_iv_crush_opportunity is True
    assert res.sizing_multiplier >= 1.0


def test_news_analyzer_high_risk_shock():
    analyzer = NewsSentimentAnalyzer()
    articles = [
        {"headline": "SEC Launches Formal Fraud Investigation into Company Following Whistleblower Report", "summary": "Subpoena issued amid scandal and default warning."},
    ]
    res = analyzer.analyze_news("XYZ", articles)
    assert res.event_risk_score >= 0.25
    assert res.is_event_risk_high is True
    assert res.sizing_multiplier == 0.0  # Blocks trading


def test_mcp_news_tool():
    res_str = alpaca_get_news.invoke({"symbols": "SPY,GLD", "limit": 3})
    data = json.loads(res_str)
    assert isinstance(data, list)
