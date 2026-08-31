import os
from alpacha.config import Settings


def test_default_config():
    settings = Settings.load(config_path="config/settings.yaml")
    assert settings.app.name == "AlpachaBot"
    assert settings.risk.warn_drawdown_pct == 0.02
    assert settings.risk.kill_drawdown_pct == 0.035
    assert settings.strategy.target_delta == 0.20
    assert settings.model.iv_vs_rv_multiple == 1.2


def test_env_override(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test_key_123")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test_secret_456")
    monkeypatch.setenv("ALERTS_WEBHOOK_URL", "https://hooks.example.com/alerts")

    settings = Settings.load(config_path="config/settings.yaml")
    assert settings.credentials is not None
    assert settings.credentials.api_key_id == "test_key_123"
    assert settings.credentials.api_secret_key == "test_secret_456"
    assert settings.alerts.webhook_url == "https://hooks.example.com/alerts"
