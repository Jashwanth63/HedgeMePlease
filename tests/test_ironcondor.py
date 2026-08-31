from alpacha.config import Settings
from alpacha.strategy.ironcondor import IronCondorBuilder


def test_iron_condor_builder():
    settings = Settings.load(config_path="config/settings.yaml")
    builder = IronCondorBuilder(settings)

    spot = 500.0
    vol = 0.20
    target_dte = 30
    exp_date = "2025-04-18"

    # Mock option chain
    chain = [
        # Puts
        {"symbol": "SPY_P_470", "option_type": "PUT", "strike": 470.0, "expiration": exp_date, "dte": 30, "delta": -0.10, "bid": 0.40, "ask": 0.50},
        {"symbol": "SPY_P_480", "option_type": "PUT", "strike": 480.0, "expiration": exp_date, "dte": 30, "delta": -0.20, "bid": 1.20, "ask": 1.30},
        {"symbol": "SPY_P_490", "option_type": "PUT", "strike": 490.0, "expiration": exp_date, "dte": 30, "delta": -0.35, "bid": 2.50, "ask": 2.60},
        # Calls
        {"symbol": "SPY_C_510", "option_type": "CALL", "strike": 510.0, "expiration": exp_date, "dte": 30, "delta": 0.35, "bid": 2.50, "ask": 2.60},
        {"symbol": "SPY_C_520", "option_type": "CALL", "strike": 520.0, "expiration": exp_date, "dte": 30, "delta": 0.20, "bid": 1.20, "ask": 1.30},
        {"symbol": "SPY_C_530", "option_type": "CALL", "strike": 530.0, "expiration": exp_date, "dte": 30, "delta": 0.10, "bid": 0.40, "ask": 0.50},
    ]

    ic = builder.build_iron_condor(
        symbol="SPY",
        underlying_price=spot,
        implied_vol=vol,
        chain_data=chain,
        dte_target=target_dte,
    )

    assert ic is not None
    assert ic.underlying == "SPY"
    assert ic.short_put.strike == 480.0
    assert ic.short_call.strike == 520.0
    assert ic.long_put.strike < 480.0
    assert ic.long_call.strike > 520.0
    assert ic.net_credit_per_share > 0.20
    assert ic.max_loss_total > 0
    assert ic.put_breakeven < 480.0
    assert ic.call_breakeven > 520.0
