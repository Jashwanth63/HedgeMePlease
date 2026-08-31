"""
Configuration management for AlpachaBot.
Loads configuration from YAML with environment variable overrides for secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv


@dataclass
class AppConfig:
    name: str = "AlpachaBot"
    paper: bool = True
    dry_run: bool = False
    db_path: str = "data/alpacha.db"
    models_dir: str = "data/models"
    macro_calendar_path: str = "config/macro_calendar.json"


@dataclass
class DataConfig:
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    timeframe: str = "1Min"
    history_days: int = 90
    refresh_interval_sec: int = 60


@dataclass
class ModelConfig:
    daily_lags: int = 1
    weekly_lags: int = 5
    monthly_lags: int = 22
    use_leverage: bool = True
    use_jumps: bool = True
    jump_alpha: float = 0.01
    iv_vs_rv_multiple: float = 1.2
    retrain_if_older_days: int = 5
    min_train_days: int = 20


@dataclass
class AssetClassCapsConfig:
    equity_max_pct: float = 0.50
    fixed_income_max_pct: float = 0.25
    commodities_max_pct: float = 0.25
    single_symbol_max_pct: float = 0.08


@dataclass
class RiskConfig:
    warn_drawdown_pct: float = 0.02
    kill_drawdown_pct: float = 0.035
    max_portfolio_bp_pct: float = 0.30
    max_contracts_per_trade: int = 10
    single_trade_max_loss_pct: float = 0.03
    use_risk_parity: bool = True
    target_annualized_vol: float = 0.15
    caps: AssetClassCapsConfig = field(default_factory=AssetClassCapsConfig)




@dataclass
class StrategyConfig:
    target_delta: float = 0.20
    delta_tolerance: float = 0.08
    min_dte: int = 0
    max_dte: int = 45
    target_dte: int = 2
    macro_buffer_hours: float = 2.0
    min_credit: float = 0.20
    profit_target_pct: float = 0.50
    stop_loss_multiplier: float = 2.0
    close_dte_threshold: int = 2
    contango_min_ratio: float = 0.98


@dataclass
class ExecutionConfig:
    scan_cron_minutes: str = "*/5"
    trading_hours_only: bool = True
    order_poll_timeout_sec: int = 60
    order_poll_interval_sec: int = 2
    limit_price_offset: float = 0.01


@dataclass
class AlertsConfig:
    enabled: bool = True
    webhook_url: Optional[str] = None


@dataclass
class AlpacaCredentials:
    api_key_id: str
    api_secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"


@dataclass
class Settings:
    app: AppConfig = field(default_factory=AppConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    credentials: Optional[AlpacaCredentials] = None

    @classmethod
    def load(
        cls,
        config_path: str | Path = "config/settings.yaml",
        env_path: Optional[str | Path] = ".env",
    ) -> Settings:
        """Loads configuration from YAML file and overrides secrets from environment."""
        if env_path and Path(env_path).exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()

        config_dict: Dict[str, Any] = {}
        path_obj = Path(config_path)
        if path_obj.exists():
            with open(path_obj, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}

        app_cfg = AppConfig(**config_dict.get("app", {}))
        data_cfg = DataConfig(**config_dict.get("data", {}))
        model_cfg = ModelConfig(**config_dict.get("model", {}))
        risk_dict = dict(config_dict.get("risk", {}))
        if "caps" in risk_dict and isinstance(risk_dict["caps"], dict):
            risk_dict["caps"] = AssetClassCapsConfig(**risk_dict["caps"])
        risk_cfg = RiskConfig(**risk_dict)
        strategy_cfg = StrategyConfig(**config_dict.get("strategy", {}))
        exec_cfg = ExecutionConfig(**config_dict.get("execution", {}))

        alerts_raw = config_dict.get("alerts", {})
        webhook_from_env = os.getenv("ALERTS_WEBHOOK_URL")
        if webhook_from_env:
            alerts_raw["webhook_url"] = webhook_from_env
        alerts_cfg = AlertsConfig(**alerts_raw)

        # Alpaca Credentials from Environment
        api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or ""
        secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or ""
        base_url = (
            os.getenv("APCA_API_BASE_URL")
            or ("https://paper-api.alpaca.markets" if app_cfg.paper else "https://api.alpaca.markets")
        )

        creds = None
        if api_key and secret_key:
            creds = AlpacaCredentials(
                api_key_id=api_key,
                api_secret_key=secret_key,
                base_url=base_url,
            )

        return cls(
            app=app_cfg,
            data=data_cfg,
            model=model_cfg,
            risk=risk_cfg,
            strategy=strategy_cfg,
            execution=exec_cfg,
            alerts=alerts_cfg,
            credentials=creds,
        )
