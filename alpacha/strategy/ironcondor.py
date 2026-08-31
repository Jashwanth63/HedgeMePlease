"""
Iron Condor Structure Builder and Strike Selector.
Constructs 4-leg defined-risk Iron Condors targeting 0.20 short delta and wings sized by Expected Move.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from alpacha.config import Settings
from alpacha.model.volutils import compute_expected_move
from alpacha.utils.logger import get_logger

logger = get_logger("iron_condor")


@dataclass
class OptionLeg:
    symbol: str
    underlying: str
    option_type: str          # "CALL" or "PUT"
    action: str               # "BUY" or "SELL"
    strike: float
    expiration: str           # "YYYY-MM-DD"
    dte: int
    delta: float
    bid: float
    ask: float
    mid: float


@dataclass
class IronCondor:
    trade_id: str
    underlying: str
    underlying_price: float
    expiration: str
    dte: int
    long_put: OptionLeg
    short_put: OptionLeg
    short_call: OptionLeg
    long_call: OptionLeg
    put_wing_width: float
    call_wing_width: float
    net_credit_per_share: float
    net_credit_total: float
    max_loss_per_share: float
    max_loss_total: float
    put_breakeven: float
    call_breakeven: float
    expected_move: float
    target_profit: float
    stop_loss_amount: float
    contracts: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "underlying": self.underlying,
            "underlying_price": self.underlying_price,
            "expiration": self.expiration,
            "dte": self.dte,
            "contracts": self.contracts,
            "net_credit_per_share": self.net_credit_per_share,
            "net_credit_total": self.net_credit_total,
            "max_loss_total": self.max_loss_total,
            "put_breakeven": self.put_breakeven,
            "call_breakeven": self.call_breakeven,
            "expected_move": self.expected_move,
            "legs": [
                asdict(self.long_put),
                asdict(self.short_put),
                asdict(self.short_call),
                asdict(self.long_call),
            ],
        }


class IronCondorBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.target_delta = settings.strategy.target_delta
        self.delta_tolerance = settings.strategy.delta_tolerance
        self.min_credit = settings.strategy.min_credit
        self.profit_target_pct = settings.strategy.profit_target_pct
        self.stop_loss_multiplier = settings.strategy.stop_loss_multiplier

    def build_iron_condor(
        self,
        symbol: str,
        underlying_price: float,
        implied_vol: float,
        chain_data: List[Dict[str, Any]],
        dte_target: int = 30,
    ) -> Optional[IronCondor]:
        """
        Builds a 4-leg Iron Condor from available chain options.
        """
        if underlying_price <= 0 or implied_vol <= 0 or not chain_data:
            return None

        # Filter options by DTE range
        valid_options = [
            opt for opt in chain_data
            if self.settings.strategy.min_dte <= opt.get("dte", 0) <= self.settings.strategy.max_dte
        ]
        if not valid_options:
            return None

        # Group by expiration date and find closest expiration to target_dte
        expirations = sorted(list(set(opt["expiration"] for opt in valid_options)))
        if not expirations:
            return None

        best_exp = min(
            expirations,
            key=lambda exp: abs(
                next(opt["dte"] for opt in valid_options if opt["expiration"] == exp) - dte_target
            )
        )
        exp_options = [opt for opt in valid_options if opt["expiration"] == best_exp]
        actual_dte = exp_options[0]["dte"]

        # Expected Move for wing sizing
        exp_move = compute_expected_move(underlying_price, implied_vol, actual_dte)

        # Separate puts and calls
        puts = [opt for opt in exp_options if opt["option_type"].upper() == "PUT"]
        calls = [opt for opt in exp_options if opt["option_type"].upper() == "CALL"]

        if not puts or not calls:
            return None

        # 1. Select Short Put (target delta ~ -0.20)
        short_put_target_delta = -abs(self.target_delta)
        below_puts = [p for p in puts if p["strike"] < underlying_price]
        if not below_puts:
            return None
        short_put_raw = min(below_puts, key=lambda p: abs(p.get("delta", 0) - short_put_target_delta))

        # 2. Select Short Call (target delta ~ +0.20)
        short_call_target_delta = abs(self.target_delta)
        above_calls = [c for c in calls if c["strike"] > underlying_price]
        if not above_calls:
            return None
        short_call_raw = min(above_calls, key=lambda c: abs(c.get("delta", 0) - short_call_target_delta))

        # 3. Select Long Put wing (strike <= Short Put - Expected Move)
        target_long_put_strike = short_put_raw["strike"] - exp_move
        long_put_candidates = [p for p in puts if p["strike"] < short_put_raw["strike"]]
        if not long_put_candidates:
            return None
        long_put_raw = min(long_put_candidates, key=lambda p: abs(p["strike"] - target_long_put_strike))

        # 4. Select Long Call wing (strike >= Short Call + Expected Move)
        target_long_call_strike = short_call_raw["strike"] + exp_move
        long_call_candidates = [c for c in calls if c["strike"] > short_call_raw["strike"]]
        if not long_call_candidates:
            return None
        long_call_raw = min(long_call_candidates, key=lambda c: abs(c["strike"] - target_long_call_strike))

        def make_leg(raw: Dict[str, Any], action: str) -> OptionLeg:
            bid = float(raw.get("bid", 0.0))
            ask = float(raw.get("ask", 0.0))
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else float(raw.get("price", 0.0))
            return OptionLeg(
                symbol=raw.get("symbol", f"{symbol}_{raw['expiration']}_{raw['strike']}_{raw['option_type']}"),
                underlying=symbol,
                option_type=raw["option_type"].upper(),
                action=action,
                strike=float(raw["strike"]),
                expiration=raw["expiration"],
                dte=int(raw["dte"]),
                delta=float(raw.get("delta", 0.0)),
                bid=bid,
                ask=ask,
                mid=mid,
            )

        short_put_leg = make_leg(short_put_raw, "SELL")
        long_put_leg = make_leg(long_put_raw, "BUY")
        short_call_leg = make_leg(short_call_raw, "SELL")
        long_call_leg = make_leg(long_call_raw, "BUY")

        # Credit calculation: (Short Put Mid + Short Call Mid) - (Long Put Mid + Long Call Mid)
        credit_collected = (short_put_leg.mid + short_call_leg.mid) - (long_put_leg.mid + long_call_leg.mid)
        if credit_collected < self.min_credit:
            return None

        put_wing_width = short_put_leg.strike - long_put_leg.strike
        call_wing_width = long_call_leg.strike - short_call_leg.strike
        max_wing_width = max(put_wing_width, call_wing_width)

        if max_wing_width <= 0:
            return None

        max_loss_per_share = max_wing_width - credit_collected
        if max_loss_per_share <= 0:
            return None

        trade_id = f"IC_{symbol}_{best_exp}_{uuid.uuid4().hex[:8]}"

        return IronCondor(
            trade_id=trade_id,
            underlying=symbol,
            underlying_price=underlying_price,
            expiration=best_exp,
            dte=actual_dte,
            long_put=long_put_leg,
            short_put=short_put_leg,
            short_call=short_call_leg,
            long_call=long_call_leg,
            put_wing_width=put_wing_width,
            call_wing_width=call_wing_width,
            net_credit_per_share=credit_collected,
            net_credit_total=credit_collected * 100.0,
            max_loss_per_share=max_loss_per_share,
            max_loss_total=max_loss_per_share * 100.0,
            put_breakeven=short_put_leg.strike - credit_collected,
            call_breakeven=short_call_leg.strike + credit_collected,
            expected_move=exp_move,
            target_profit=credit_collected * 100.0 * self.profit_target_pct,
            stop_loss_amount=credit_collected * 100.0 * self.stop_loss_multiplier,
            contracts=1,
        )

