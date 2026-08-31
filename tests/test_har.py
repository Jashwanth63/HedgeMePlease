import numpy as np
import pandas as pd
import pytest
from alpacha.model.har import EnhancedHARModel
from alpacha.model.volutils import compute_daily_rv_and_jumps


@pytest.fixture
def sample_intraday_data():
    np.random.seed(42)
    dfs = []
    for day in pd.date_range("2025-01-01", periods=45, freq="B"):
        times = pd.date_range(day.strftime("%Y-%m-%d 09:30:00"), periods=390, freq="1min")
        prices = 500.0 * np.exp(np.cumsum(np.random.normal(0, 0.0005, size=len(times))))
        dfs.append(pd.DataFrame({"close": prices, "open": prices, "high": prices, "low": prices, "volume": 100}, index=times))
    return pd.concat(dfs)


def test_har_fit_and_predict(sample_intraday_data):
    daily_df = compute_daily_rv_and_jumps(sample_intraday_data)
    assert len(daily_df) == 45

    model = EnhancedHARModel("SPY", daily_lags=1, weekly_lags=5, monthly_lags=22, use_leverage=True, use_jumps=True)
    metrics = model.fit(daily_df)

    assert "r_squared" in metrics
    assert "aic" in metrics
    assert metrics["r_squared"] >= 0.0
    assert model.is_fitted is True

    daily_rv, ann_vol = model.predict_next_rv(daily_df)
    assert daily_rv > 0
    assert 0.05 <= ann_vol <= 0.60  # Typical equity volatility range


def test_har_insufficient_data():
    # Only 10 days
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    df = pd.DataFrame({"rv": [0.0001]*10, "daily_return": [0.0]*10, "jump": [0.0]*10}, index=dates)
    model = EnhancedHARModel("SPY")
    with pytest.raises(ValueError, match="Insufficient observations"):
        model.fit(df)
