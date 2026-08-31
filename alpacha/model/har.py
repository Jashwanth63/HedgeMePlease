"""
Enhanced HAR (Heterogeneous Autoregressive) Volatility Model.
Forecasts 1-day ahead Realized Volatility using daily, weekly, and monthly components,
augmented with asymmetric leverage effects and jump variations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

from alpacha.utils.logger import get_logger

logger = get_logger("har_model")


class EnhancedHARModel:
    def __init__(
        self,
        symbol: str,
        daily_lags: int = 1,
        weekly_lags: int = 5,
        monthly_lags: int = 22,
        use_leverage: bool = True,
        use_jumps: bool = True,
    ) -> None:
        self.symbol = symbol
        self.daily_lags = daily_lags
        self.weekly_lags = weekly_lags
        self.monthly_lags = monthly_lags
        self.use_leverage = use_leverage
        self.use_jumps = use_jumps

        self.model_results: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def _prepare_features(self, daily_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Constructs lag features for HAR estimation from daily realized variance DataFrame.
        """
        df = daily_df.copy()
        if "rv" not in df.columns:
            raise ValueError("Input DataFrame must contain 'rv' column.")

        # Ensure small positive value to avoid log(0)
        df["rv_clean"] = df["rv"].clip(lower=1e-8)
        df["log_rv"] = np.log(df["rv_clean"])

        # Target: 1-step ahead log RV
        df["target"] = df["log_rv"].shift(-1)

        # 1. Daily component (lag 1)
        df["rv_daily"] = df["log_rv"].shift(0)
        features = ["rv_daily"]

        # 2. Weekly component (mean over past 5 days)
        df["rv_weekly"] = df["log_rv"].rolling(window=self.weekly_lags).mean()
        features.append("rv_weekly")

        # 3. Monthly component (mean over past 22 days)
        df["rv_monthly"] = df["log_rv"].rolling(window=self.monthly_lags).mean()
        features.append("rv_monthly")

        # 4. Leverage effect: negative daily return shock
        if self.use_leverage and "daily_return" in df.columns:
            df["ret_neg"] = np.minimum(0.0, df["daily_return"])
            features.append("ret_neg")

        # 5. Jump component: log jump variation
        if self.use_jumps and "jump" in df.columns:
            df["jump_log"] = np.log(df["jump"].clip(lower=1e-8) + 1.0)
            features.append("jump_log")

        # Drop NaN rows due to lagging & target shift
        clean_df = df.dropna(subset=features + ["target"]).copy()
        return clean_df[features], clean_df["target"]

    def fit(self, daily_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Fits the Enhanced HAR model via OLS regression.
        """
        min_required = max(self.monthly_lags + 2, 20)
        if len(daily_df) < min_required:
            raise ValueError(f"Insufficient observations for HAR model. Required >= {min_required}, got {len(daily_df)}")

        X, y = self._prepare_features(daily_df)
        self.feature_names = list(X.columns)

        X_with_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_with_const)
        self.model_results = ols_model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        self.is_fitted = True

        r2 = float(self.model_results.rsquared)
        aic = float(self.model_results.aic)
        logger.info(f"Fitted HAR model for {self.symbol} (R²={r2:.4f}, AIC={aic:.2f}, Obs={len(X)})")

        return {
            "symbol": self.symbol,
            "r_squared": r2,
            "adj_r_squared": float(self.model_results.rsquared_adj),
            "aic": aic,
            "params": self.model_results.params.to_dict(),
            "pvalues": self.model_results.pvalues.to_dict(),
            "n_observations": int(self.model_results.nobs),
        }

    def predict_next_rv(self, daily_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Predicts 1-day ahead Realized Variance and annualized Realized Volatility.
        Returns:
            (forecasted_daily_rv, forecasted_annualized_volatility)
        """
        if not self.is_fitted or self.model_results is None:
            raise RuntimeError(f"HAR model for {self.symbol} is not fitted yet.")

        df = daily_df.copy()
        df["rv_clean"] = df["rv"].clip(lower=1e-8)
        df["log_rv"] = np.log(df["rv_clean"])

        row_dict: Dict[str, float] = {"const": 1.0}
        row_dict["rv_daily"] = float(df["log_rv"].iloc[-1])
        row_dict["rv_weekly"] = float(df["log_rv"].iloc[-self.weekly_lags:].mean())
        row_dict["rv_monthly"] = float(df["log_rv"].iloc[-self.monthly_lags:].mean())

        if self.use_leverage and "ret_neg" in self.feature_names:
            last_ret = float(df["daily_return"].iloc[-1]) if "daily_return" in df.columns else 0.0
            row_dict["ret_neg"] = min(0.0, last_ret)

        if self.use_jumps and "jump_log" in self.feature_names:
            last_jump = float(df["jump"].iloc[-1]) if "jump" in df.columns else 0.0
            row_dict["jump_log"] = math.log(max(1e-8, last_jump) + 1.0)

        feature_vector = [row_dict[name] for name in ["const"] + self.feature_names]
        log_rv_pred = float(self.model_results.predict(feature_vector)[0])

        # Convert log(RV) back to daily RV with Jensen's inequality correction
        residual_var = float(self.model_results.scale)
        pred_daily_rv = float(np.exp(log_rv_pred + 0.5 * residual_var))

        # Annualized Realized Volatility: sqrt(252 * daily_rv)
        annualized_vol = math.sqrt(252.0 * pred_daily_rv)

        return pred_daily_rv, annualized_vol
