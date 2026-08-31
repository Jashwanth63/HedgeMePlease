import math
import numpy as np
import pandas as pd
from alpacha.model.volutils import (
    compute_daily_rv_and_jumps,
    compute_expected_move,
    calculate_iv_rv_ratio,
)


def test_compute_expected_move():
    price = 500.0
    iv = 0.20
    dte = 30
    exp_move = compute_expected_move(price, iv, dte)
    expected = 500.0 * 0.20 * math.sqrt(30.0 / 365.0)
    assert math.isclose(exp_move, expected, rel_tol=1e-5)


def test_calculate_iv_rv_ratio():
    iv = 0.24
    forecasted_rv = 0.20
    ratio = calculate_iv_rv_ratio(iv, forecasted_rv)
    assert math.isclose(ratio, 1.20, rel_tol=1e-5)


def test_compute_daily_rv_and_jumps():
    dates = pd.date_range("2025-01-01 09:30:00", periods=390, freq="1min")
    np.random.seed(42)
    # Generate prices with known variance
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.001, size=len(dates))))
    df = pd.DataFrame({"close": prices}, index=dates)

    daily_df = compute_daily_rv_and_jumps(df, annualize=False)
    assert len(daily_df) == 1
    assert "rv" in daily_df.columns
    assert "bv" in daily_df.columns
    assert "jump" in daily_df.columns
    assert daily_df["rv"].iloc[0] > 0
    assert daily_df["bv"].iloc[0] > 0
    assert daily_df["jump"].iloc[0] >= 0
