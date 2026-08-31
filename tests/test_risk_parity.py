from alpacha.config import Settings
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.risk.risk_manager import RiskManager


def test_asset_class_classification():
    assert RiskManager.get_asset_class("TLT") == "FIXED_INCOME"
    assert RiskManager.get_asset_class("GLD") == "COMMODITIES"
    assert RiskManager.get_asset_class("USO") == "COMMODITIES"
    assert RiskManager.get_asset_class("XLE") == "COMMODITIES"
    assert RiskManager.get_asset_class("SPY") == "EQUITIES"
    assert RiskManager.get_asset_class("NVDA") == "EQUITIES"
    assert RiskManager.get_asset_class("COIN") == "EQUITIES"


def test_risk_parity_inverse_vol_sizing():
    settings = Settings.load(config_path="config/settings.yaml")
    settings.risk.use_risk_parity = True
    settings.risk.target_annualized_vol = 0.15
    db = SQLiteManager(":memory:")
    rm = RiskManager(settings, db)

    # 1. Low-vol asset (e.g. TLT or GLD with 10% vol) -> larger sizing
    low_vol_contracts, err1 = rm.calculate_position_size(
        account_equity=100000.0,
        available_bp=400000.0,
        wing_width=5.0,
        credit_per_share=0.80,
        symbol="TLT",
        forecasted_vol=0.10,
    )
    assert err1 is None

    # 2. High-vol asset (e.g. NVDA or COIN with 45% vol) -> smaller sizing
    high_vol_contracts, err2 = rm.calculate_position_size(
        account_equity=100000.0,
        available_bp=400000.0,
        wing_width=5.0,
        credit_per_share=0.80,
        symbol="COIN",
        forecasted_vol=0.45,
    )
    assert err2 is None

    # Risk parity condition: Low-vol asset gets more contracts than High-vol asset
    assert low_vol_contracts > high_vol_contracts
    print(f"Risk Parity verified: Low Vol (10%) = {low_vol_contracts} contracts vs High Vol (45%) = {high_vol_contracts} contracts")
