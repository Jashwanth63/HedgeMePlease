import numpy as np
import pandas as pd
from alpacha.bot import AlpachaBot
from alpacha.config import Settings
from alpacha.data.sqlite_manager import SQLiteManager


def test_bot_end_to_end_cycle():
    settings = Settings.load(config_path="config/settings.yaml")
    settings.app.dry_run = True
    settings.execution.trading_hours_only = False  # Disable market clock restriction for test
    settings.app.db_path = ":memory:"

    bot = AlpachaBot(settings)

    # Seed SQLite with synthetic historical 1-min bars for SPY & QQQ (50 days each)
    np.random.seed(42)
    for sym, base_price in [("SPY", 500.0), ("QQQ", 440.0)]:
        dfs = []
        for day in pd.date_range("2025-01-01", periods=50, freq="B"):
            times = pd.date_range(day.strftime("%Y-%m-%d 09:30:00"), periods=390, freq="1min")
            prices = base_price * np.exp(np.cumsum(np.random.normal(0, 0.0005, size=len(times))))
            dfs.append(pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "volume": 500}, index=times))
        full_df = pd.concat(dfs)
        bot.db.save_bars(full_df, symbol=sym)

    # Run execution cycle
    bot.run_cycle()

    # Verify that trades were generated and saved
    open_trades = bot.db.get_open_trades()
    assert len(open_trades) >= 1
    trade = open_trades[0]
    assert trade["status"] == "OPEN"
    assert trade["credit_received"] > 0
    assert len(trade["legs"]) == 4

    # Verify that risk snapshot was recorded
    risk_snap = bot.db.get_latest_risk_snapshot()
    assert risk_snap is not None
    assert risk_snap["equity"] > 0
    assert risk_snap["risk_level"] == "NORMAL"
