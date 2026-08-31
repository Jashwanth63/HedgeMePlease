import math

from alpaca.risk.bs import bs


def test_put_call_parity():
    spot, strike, t, iv, r = 650.0, 645.0, 10 / 365, 0.2, 0.04
    call = bs(True, spot, strike, t, iv, rate=r)
    put = bs(False, spot, strike, t, iv, rate=r)
    assert abs((call.price - put.price) - (spot - strike * math.exp(-r * t))) < 1e-6


def test_delta_signs_and_bounds():
    assert 0.5 < bs(True, 650, 640, 5 / 365, 0.2).delta <= 1.0
    assert -1.0 <= bs(False, 650, 660, 5 / 365, 0.2).delta < -0.5


def test_deep_otm_never_negative():
    assert bs(False, 650, 585, 3 / 365, 0.15).price >= 0.0


def test_expired_returns_intrinsic():
    res = bs(False, 600, 650, 0.0, 0.2)
    assert res.price == 50.0 and res.delta == -1.0


def test_vega_gamma_theta_signs():
    res = bs(True, 650, 650, 5 / 365, 0.2)
    assert res.vega > 0 and res.gamma > 0 and res.theta < 0
