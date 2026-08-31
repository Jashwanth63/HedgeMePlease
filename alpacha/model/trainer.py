"""
Model Trainer and Lifecycle Manager for Enhanced HAR models.
Manages model fitting, serialization to disk, staleness checks, and forecast generation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd

from alpacha.config import Settings
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.model.har import EnhancedHARModel
from alpacha.model.volutils import compute_daily_rv_and_jumps
from alpacha.utils.logger import get_logger

logger = get_logger("trainer")


class ModelTrainer:
    def __init__(self, settings: Settings, db_manager: SQLiteManager) -> None:
        self.settings = settings
        self.db = db_manager
        self.models_dir = Path(settings.app.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, EnhancedHARModel] = {}

    def _get_model_path(self, symbol: str) -> Path:
        return self.models_dir / f"har_{symbol.upper()}.joblib"

    def _get_meta_key(self, symbol: str) -> str:
        return f"har_model_last_fit_{symbol.upper()}"

    def should_retrain(self, symbol: str) -> bool:
        """Determines if the HAR model for a symbol needs retraining."""
        model_path = self._get_model_path(symbol)
        if not model_path.exists():
            return True

        last_fit_str = self.db.get_meta(self._get_meta_key(symbol))
        if not last_fit_str:
            return True

        try:
            last_fit_time = datetime.fromisoformat(last_fit_str)
            if last_fit_time.tzinfo is None:
                last_fit_time = last_fit_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            max_age = timedelta(days=self.settings.model.retrain_if_older_days)
            return (now - last_fit_time) > max_age
        except Exception:
            return True

    def train_and_save(self, symbol: str, intraday_bars: pd.DataFrame) -> Tuple[EnhancedHARModel, Dict[str, Any]]:
        """Fits the HAR model on intraday bars and saves it to disk."""
        daily_df = compute_daily_rv_and_jumps(intraday_bars, annualize=False)
        if len(daily_df) < self.settings.model.min_train_days:
            raise ValueError(
                f"Cannot train HAR model for {symbol}: only {len(daily_df)} daily points found, "
                f"minimum required is {self.settings.model.min_train_days}"
            )

        model = EnhancedHARModel(
            symbol=symbol,
            daily_lags=self.settings.model.daily_lags,
            weekly_lags=self.settings.model.weekly_lags,
            monthly_lags=self.settings.model.monthly_lags,
            use_leverage=self.settings.model.use_leverage,
            use_jumps=self.settings.model.use_jumps,
        )

        metrics = model.fit(daily_df)
        model_path = self._get_model_path(symbol)
        joblib.dump(model, model_path)

        now_str = datetime.now(timezone.utc).isoformat()
        self.db.set_meta(self._get_meta_key(symbol), now_str)
        self.loaded_models[symbol] = model

        logger.info(f"Saved trained HAR model for {symbol} to {model_path}")
        return model, metrics

    def load_model(self, symbol: str) -> Optional[EnhancedHARModel]:
        """Loads model from memory or disk."""
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]

        model_path = self._get_model_path(symbol)
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                self.loaded_models[symbol] = model
                return model
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
        return None

    def get_forecast(self, symbol: str, intraday_bars: pd.DataFrame) -> Tuple[float, float]:
        """
        Retrieves or retrains model, computes next-day RV forecast, and saves forecast to SQLite.
        Returns:
            (daily_rv_forecast, annualized_volatility_forecast)
        """
        model = self.load_model(symbol)
        if model is None or self.should_retrain(symbol):
            logger.info(f"Retraining HAR model for {symbol}...")
            model, _ = self.train_and_save(symbol, intraday_bars)

        daily_df = compute_daily_rv_and_jumps(intraday_bars, annualize=False)
        daily_rv, annualized_vol = model.predict_next_rv(daily_df)

        # Save forecast to SQLite
        self.db.save_forecast(
            symbol=symbol,
            forecasted_rv=annualized_vol,
            metrics={"daily_rv": daily_rv, "annualized_vol": annualized_vol},
        )
        logger.info(f"Generated Forecast for {symbol}: Daily RV={daily_rv:.6f}, Annualized Vol={annualized_vol:.4%}")
        return daily_rv, annualized_vol
