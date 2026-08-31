"""Sleeve C: the insurance builder. Far OTM SPY puts, next-week expiry,
sized to the tiny hedge budget. All-long, so max loss is the debit and the
executor's debit path applies. No exit logic: it rides to the Thursday
flatten, worthless on a calm week, priceless on a violent one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..config import SLEEVE_C
from ..risk.ledger import Leg, Position
from .condor import Contract, Proposal, dte_of


def build_hedge_puts(
    contracts: list[Contract], spot: float, now: datetime
) -> tuple[Optional[Proposal], dict]:
    diag: dict = {"sleeve": "C"}
    if spot <= 0:
        diag["reject"] = "no spot"
        return None, diag

    lo_strike = spot * (1 - SLEEVE_C.otm_band[1])
    hi_strike = spot * (1 - SLEEVE_C.otm_band[0])
    pool = [
        c for c in contracts
        if c.opt_type == "put" and c.quote_ok
        and lo_strike <= c.strike <= hi_strike
        and SLEEVE_C.dte_min <= dte_of(c.expiry, now) <= SLEEVE_C.dte_max
    ]
    if not pool:
        diag["reject"] = "no puts in the OTM and DTE bands with usable quotes"
        return None, diag

    # prefer the latest expiry in band (keeps value at the flatten), then the
    # cheapest strike that still fits at least one contract in the budget
    best_expiry = max({c.expiry for c in pool}, key=lambda e: dte_of(e, now))
    candidates = sorted(
        (c for c in pool if c.expiry == best_expiry), key=lambda c: -c.strike
    )
    per_unit_budget = SLEEVE_C.budget / 100.0
    pick = next((c for c in candidates if c.mid <= per_unit_budget), None)
    if pick is None:
        diag["reject"] = f"cheapest in-band put costs more than {per_unit_budget:.2f}"
        return None, diag

    qty = max(1, int(per_unit_budget // pick.mid))
    debit = pick.mid
    position_id = f"SLC-SPY-{now.strftime('%m%d%H%M%S')}"
    leg = Leg(
        symbol=pick.symbol, side="buy", ratio_qty=1, strike=pick.strike,
        opt_type="put", expiry=pick.expiry, entry_iv=pick.iv or 0.15,
        entry_delta=pick.delta or -0.10,
    )
    position = Position(
        position_id=position_id, sleeve="C", underlying="SPY",
        structure="hedge_puts", legs=[leg], qty=qty,
        credit=round(debit, 2),               # long structure: the debit paid
        width=0.0, max_loss=round(debit * 100 * qty, 2),
        client_order_id=position_id, opened_at=now.isoformat(),
    )
    proposal = Proposal(
        underlying="SPY", structure="hedge_puts", legs=[leg], qty=qty,
        credit=round(debit, 2), width=0.0, max_loss=position.max_loss,
        net_delta_dollars=(pick.delta or -0.10) * 100 * qty * spot,
        net_vega_dollars=0.0,  # long vega, helpful; zero keeps the floor check honest
        dte=dte_of(pick.expiry, now), position=position,
    )
    diag["proposal"] = proposal.summary()
    return proposal, diag
