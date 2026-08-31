"""
Volatility calculation utilities for high-frequency return series.
Computes Realized Variance (RV), Bipower Variation (BV), Jump Variation, and Expected Move.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Constants for Bipower and Quarticity calculations
MU_1 = math.sqrt(2.0 / math.pi)  # ~0.79788
MU_1_INV_SQ = 1.0 / (MU_1 ** 2)  # pi / 2 ~ 1.5707963
MU_4_3 = (2.0 ** (2.0 / 3.0)) * (math.gamma(7.0 / 6.0) / math.gamma(1.0 / 2.0))  # ~0.83086


def compute_daily_rv_and_jumps(
    intraday_bars: pd.DataFrame,
    annualize: bool = False,
) -> pd.DataFrame:
    """
    Computes daily Realized Variance (RV), Bipower Variation (BV),
    Jump Variation (J), Relative Jump (RJ), and Daily Log Return from 1-minute intraday bars.
    
    Args:
        intraday_bars: DataFrame with DatetimeIndex and 'close' column.
        annualize: If True, annualizes RV and BV by multiplying variance by 252.
        
    Returns:
        DataFrame indexed by Date with columns ['rv', 'bv', 'jump', 'rj', 'daily_return', 'daily_close']
    """
    if intraday_bars.empty or "close" not in intraday_bars.columns:
        return pd.DataFrame(columns=["rv", "bv", "jump", "rj", "daily_return", "daily_close"])

    df = intraday_bars.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort index
    df.sort_index(inplace=True)

    # Group by calendar date
    grouped = df.groupby(df.index.date)
    results = []

    for date, group in grouped:
        prices = group["close"].values
        if len(prices) < 5:
            continue

        # 1-minute log returns
        log_rets = np.diff(np.log(prices))
        n_obs = len(log_rets)
        if n_obs < 4:
            continue

        # Realized Variance: sum of squared returns
        rv = float(np.sum(log_rets ** 2))

        # Bipower Variation (jump-robust continuous variance)
        abs_rets = np.abs(log_rets)
        bv = float(MU_1_INV_SQ * np.sum(abs_rets[1:] * abs_rets[:-1]))

        # Jump component: non-negative excess variance
        jump = max(0.0, rv - bv)
        rj = (jump / rv) if rv > 1e-12 else 0.0

        # Daily Return from first to last bar
        daily_return = float(np.log(prices[-1] / prices[0]))
        daily_close = float(prices[-1])

        scale = 252.0 if annualize else 1.0

        results.append({
            "date": pd.to_datetime(date),
            "rv": rv * scale,
            "bv": bv * scale,
            "jump": jump * scale,
            "rj": rj,
            "daily_return": daily_return,
            "daily_close": daily_close,
        })

    if not results:
        return pd.DataFrame(columns=["rv", "bv", "jump", "rj", "daily_return", "daily_close"])

    res_df = pd.DataFrame(results)
    res_df.set_index("date", inplace=True)
    return res_df


def compute_expected_move(
    price: float,
    implied_volatility: float,
    dte: float,
) -> float:
    """
    Computes expected move for an underlying given IV and Days To Expiration:
    Expected Move = Price * IV * sqrt(DTE / 365.0)
    """
    if price <= 0 or implied_volatility <= 0 or dte <= 0:
        return 0.0
    return float(price * implied_volatility * math.sqrt(dte / 365.0))


def calculate_iv_rv_ratio(
    implied_vol: float,
    forecasted_annualized_rv: float,
) -> float:
    """
    Calculates the ratio of Implied Volatility to Forecasted Realized Volatility:
    Edge Ratio = IV / RV_forecast
    """
    if forecasted_annualized_rv <= 1e-6:
        return 1.0
    return float(implied_vol / forecasted_annualized_rv)
