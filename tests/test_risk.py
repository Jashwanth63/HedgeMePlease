from alpacha.config import Settings
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.risk.risk_manager import RiskLevel, RiskManager


def test_risk_manager_drawdown_ladder():
    settings = Settings.load(config_path="config/settings.yaml")
    db = SQLiteManager(":memory:")
    rm = RiskManager(settings, db)

    # 1. Normal State ($100,000)
    s1 = rm.evaluate_risk(100000.0)
    assert s1.risk_level == RiskLevel.NORMAL
    assert s1.can_trade is True
    assert s1.should_liquidate is False

    # 2. Equity High Water Mark increases to $110,000
    s2 = rm.evaluate_risk(110000.0)
    assert s2.peak_equity == 110000.0

    # 3. 2.2% Drawdown from peak ($110,000 -> $107,500) -> WARNING
    s3 = rm.evaluate_risk(107500.0)
    assert s3.risk_level == RiskLevel.WARNING
    assert s3.can_trade is False
    assert s3.should_liquidate is False

    # 4. 3.8% Drawdown from peak ($110,000 -> $105,800) -> KILL SWITCH
    s4 = rm.evaluate_risk(105800.0)
    assert s4.risk_level == RiskLevel.KILL
    assert s4.can_trade is False
    assert s4.should_liquidate is True


def test_position_sizing():
    settings = Settings.load(config_path="config/settings.yaml")
    db = SQLiteManager(":memory:")
    rm = RiskManager(settings, db)

    # Account with $100k equity, $50k BP, wing width 10, credit 1.50
    # Max loss per contract = (10 - 1.5) * 100 = $850
    contracts, err = rm.calculate_position_size(
        account_equity=100000.0,
        available_bp=50000.0,
        wing_width=10.0,
        credit_per_share=1.50,
    )
    assert err is None
    assert 1 <= contracts <= settings.risk.max_contracts_per_trade
