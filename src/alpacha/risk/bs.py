"""Black-Scholes pricing and greeks with continuous dividend yield.

Used for stress revaluation and sanity checks; live quotes always win for
marking positions. No scipy: the normal CDF comes from math.erf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


@dataclass(frozen=True)
class BsResult:
    price: float
    delta: float
    gamma: float
    vega: float    # per 1.00 change in vol
    theta: float   # per year


def bs(
    is_call: bool,
    spot: float,
    strike: float,
    t_years: float,
    iv: float,
    rate: float = 0.04,
    div_yield: float = 0.0,
) -> BsResult:
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
        delta = 0.0
        if intrinsic > 0:
            delta = 1.0 if is_call else -1.0
        return BsResult(price=intrinsic, delta=delta, gamma=0.0, vega=0.0, theta=0.0)

    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    disc_r = math.exp(-rate * t_years)
    disc_q = math.exp(-div_yield * t_years)

    if is_call:
        price = spot * disc_q * norm_cdf(d1) - strike * disc_r * norm_cdf(d2)
        delta = disc_q * norm_cdf(d1)
        theta = (
            -(spot * disc_q * norm_pdf(d1) * iv) / (2 * sqrt_t)
            - rate * strike * disc_r * norm_cdf(d2)
            + div_yield * spot * disc_q * norm_cdf(d1)
        )
    else:
        price = strike * disc_r * norm_cdf(-d2) - spot * disc_q * norm_cdf(-d1)
        delta = -disc_q * norm_cdf(-d1)
        theta = (
            -(spot * disc_q * norm_pdf(d1) * iv) / (2 * sqrt_t)
            + rate * strike * disc_r * norm_cdf(-d2)
            - div_yield * spot * disc_q * norm_cdf(-d1)
        )

    gamma = disc_q * norm_pdf(d1) / (spot * iv * sqrt_t)
    vega = spot * disc_q * norm_pdf(d1) * sqrt_t
    return BsResult(price=max(price, 0.0), delta=delta, gamma=gamma, vega=vega, theta=theta)
