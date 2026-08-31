from datetime import datetime, timezone
import pandas as pd
from alpacha.data.sqlite_manager import SQLiteManager


def test_sqlite_bars_and_meta():
    db = SQLiteManager(":memory:")

    # Test meta key-value
    db.set_meta("test_key", "test_value_123")
    assert db.get_meta("test_key") == "test_value_123"
    assert db.get_meta("non_existent", "default_val") == "default_val"

    # Test saving & loading bars
    dates = pd.date_range("2025-01-01 09:30:00", periods=10, freq="1min")
    bars_df = pd.DataFrame({
        "open": [500.0] * 10,
        "high": [501.0] * 10,
        "low": [499.0] * 10,
        "close": [500.5] * 10,
        "volume": [1000] * 10,
    }, index=dates)

    saved_count = db.save_bars(bars_df, symbol="SPY")
    assert saved_count == 10

    loaded_df = db.load_bars("SPY")
    assert len(loaded_df) == 10
    assert "close" in loaded_df.columns


def test_sqlite_forecasts_and_trades():
    db = SQLiteManager(":memory:")

    # Forecasts
    db.save_forecast("SPY", forecasted_rv=0.185, metrics={"r2": 0.35})
    latest = db.get_latest_forecast("SPY")
    assert latest is not None
    assert latest["forecasted_rv"] == 0.185
    assert latest["metrics"]["r2"] == 0.35

    # Trades
    db.save_trade(
        trade_id="IC_TEST_001",
        symbol="SPY",
        status="OPEN",
        entry_timestamp=datetime.now(timezone.utc),
        legs=[{"strike": 480, "type": "PUT"}],
        credit_received=150.0,
    )
    open_trades = db.get_open_trades()
    assert len(open_trades) == 1
    assert open_trades[0]["trade_id"] == "IC_TEST_001"

    # Update exit
    db.update_trade_exit("IC_TEST_001", exit_timestamp=datetime.now(timezone.utc), exit_pnl=75.0, status="CLOSED")
    open_trades_after = db.get_open_trades()
    assert len(open_trades_after) == 0
