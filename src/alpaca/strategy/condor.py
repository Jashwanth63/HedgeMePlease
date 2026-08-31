"""Iron condor construction from chain snapshots. Pure logic, no network.

Chain payload format matches the Alpaca MCP get_option_chain tool:
{"snapshots": {"SPY260903P00761000": {"latestQuote": {"bp":..,"ap":..},
"impliedVolatility": 0.11, "greeks": {"delta": -0.20}}, ...}}

Wing width starts at the expected move and shrinks toward the 5 dollar floor
until the credit floor and the per-trade loss cap are both satisfied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..config import PREFERRED_EXPIRIES, RISK, STRAT, now_et
from ..model.volutils import expected_move
from ..risk.bs import bs
from ..risk.ledger import Leg, Position, new_position_id
from ..risk.stress import t_years

OCC_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True)
class Contract:
    symbol: str
    underlying: str
    expiry: str
    opt_type: str
    strike: float
    bid: float
    ask: float
    iv: float
    delta: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def quote_ok(self) -> bool:
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            return False
        return (self.ask - self.bid) <= max(0.10, 0.25 * self.mid)


def parse_chain(underlying: str, payload: dict) -> list[Contract]:
    out: list[Contract] = []
    for symbol, snap in (payload.get("snapshots") or {}).items():
        m = OCC_RE.match(symbol)
        if not m:
            continue
        _, yymmdd, cp, strike_raw = m.groups()
        quote = snap.get("latestQuote") or {}
        greeks = snap.get("greeks") or {}
        out.append(
            Contract(
                symbol=symbol,
                underlying=underlying,
                expiry=f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:]}",
                opt_type="call" if cp == "C" else "put",
                strike=int(strike_raw) / 1000.0,
                bid=float(quote.get("bp") or 0.0),
                ask=float(quote.get("ap") or 0.0),
                iv=float(snap.get("impliedVolatility") or 0.0),
                delta=float(greeks.get("delta") or 0.0),
            )
        )
    return out


def dte_of(expiry: str, now: datetime) -> int:
    exp = datetime.strptime(expiry, "%Y-%m-%d").date()
    return (exp - now.date()).days


def atm_iv(contracts: list[Contract], expiry: str, spot: float) -> Optional[float]:
    at_exp = [c for c in contracts if c.expiry == expiry and c.iv > 0]
    if not at_exp:
        return None
    nearest = sorted(at_exp, key=lambda c: (abs(c.strike - spot), c.opt_type))[:4]
    return sum(c.iv for c in nearest) / len(nearest)


def pick_near_expiry(contracts: list[Contract], now: datetime) -> Optional[str]:
    expiries = sorted({c.expiry for c in contracts})
    in_band = [e for e in expiries if STRAT.near_dte_min <= dte_of(e, now) <= STRAT.near_dte_max]
    if not in_band:
        return None
    for preferred in PREFERRED_EXPIRIES:
        if preferred in in_band:
            return preferred
    return in_band[0]


def pick_far_expiry(contracts: list[Contract], now: datetime) -> Optional[str]:
    expiries = sorted({c.expiry for c in contracts})
    in_band = [e for e in expiries if STRAT.far_dte_min <= dte_of(e, now) <= STRAT.far_dte_max]
    if not in_band:
        return None
    return min(in_band, key=lambda e: abs(dte_of(e, now) - 30))


@dataclass
class Proposal:
    underlying: str
    structure: str
    legs: list[Leg]
    qty: int
    credit: float
    width: float
    max_loss: float
    net_delta_dollars: float
    net_vega_dollars: float
    dte: int
    position: Position

    def summary(self) -> dict:
        return {
            "underlying": self.underlying,
            "structure": self.structure,
            "qty": self.qty,
            "credit": round(self.credit, 2),
            "width": self.width,
            "max_loss": round(self.max_loss, 0),
            "net_delta_dollars": round(self.net_delta_dollars, 0),
            "net_vega_dollars": round(self.net_vega_dollars, 0),
            "dte": self.dte,
            "legs": [leg.symbol for leg in self.legs],
        }


def _short_candidate(
    contracts: list[Contract], expiry: str, opt_type: str, delta_target: float
) -> Optional[Contract]:
    lo, hi = STRAT.short_delta_band
    pool = [
        c for c in contracts
        if c.expiry == expiry and c.opt_type == opt_type and c.quote_ok
        and lo <= abs(c.delta) <= hi
    ]
    if not pool:
        return None
    return min(pool, key=lambda c: abs(abs(c.delta) - delta_target))


def _wing_for(contracts: list[Contract], short: Contract, width: float) -> Optional[Contract]:
    direction = -1 if short.opt_type == "put" else 1
    target = short.strike + direction * width
    pool = [
        c for c in contracts
        if c.expiry == short.expiry and c.opt_type == short.opt_type and c.quote_ok
        and (c.strike < short.strike if direction < 0 else c.strike > short.strike)
        and 0.6 * width <= abs(c.strike - short.strike) <= 1.4 * width
    ]
    if not pool:
        return None
    return min(pool, key=lambda c: abs(c.strike - target))


def _make_leg(contract: Contract, side: str, spot: float, now: datetime) -> Leg:
    delta, iv = contract.delta, contract.iv
    if not delta or not iv:
        t = t_years(contract.expiry, now)
        est = bs(contract.opt_type == "call", spot, contract.strike, t, max(iv, 0.12))
        delta = delta or est.delta
        iv = iv or 0.12
    return Leg(
        symbol=contract.symbol, side=side, ratio_qty=1, strike=contract.strike,
        opt_type=contract.opt_type, expiry=contract.expiry, entry_iv=iv, entry_delta=delta,
    )


def _assemble(
    underlying: str,
    contracts: list[Contract],
    expiry: str,
    spot: float,
    width: float,
    delta_target: float,
    now: datetime,
) -> tuple[Optional[Proposal], str]:
    short_put = _short_candidate(contracts, expiry, "put", delta_target)
    short_call = _short_candidate(contracts, expiry, "call", delta_target)
    if not (short_put and short_call):
        return None, "no short strikes in delta band with usable quotes"
    put_wing = _wing_for(contracts, short_put, width)
    call_wing = _wing_for(contracts, short_call, width)
    if not (put_wing and call_wing):
        return None, f"no wings near width {width:.0f}"

    put_width = abs(short_put.strike - put_wing.strike)
    call_width = abs(call_wing.strike - short_call.strike)
    eff_width = max(put_width, call_width)
    credit = (short_put.mid - put_wing.mid) + (short_call.mid - call_wing.mid)

    if credit < STRAT.min_credit_frac_condor * eff_width:
        return None, f"credit {credit:.2f} under floor {STRAT.min_credit_frac_condor:.0%} of width {eff_width:.0f}"
    unit_loss = (eff_width - credit) * 100.0
    if unit_loss <= 0:
        return None, "degenerate economics"
    if unit_loss > RISK.per_trade_max_loss:
        return None, f"one unit risks {unit_loss:.0f}, over the {RISK.per_trade_max_loss:.0f} cap"

    qty = max(1, int(RISK.per_trade_pref_loss // unit_loss))
    legs = [
        _make_leg(short_put, "sell", spot, now),
        _make_leg(put_wing, "buy", spot, now),
        _make_leg(short_call, "sell", spot, now),
        _make_leg(call_wing, "buy", spot, now),
    ]
    beta = STRAT.betas.get(underlying, 1.0)
    net_delta = sum(
        (1 if leg.side == "buy" else -1) * leg.entry_delta * 100.0 * qty * spot * beta
        for leg in legs
    )
    net_vega = 0.0
    for leg in legs:
        t = t_years(leg.expiry, now)
        est = bs(leg.opt_type == "call", spot, leg.strike, t, leg.entry_iv)
        net_vega += (1 if leg.side == "buy" else -1) * est.vega * 0.01 * 100.0 * qty

    position_id = new_position_id(underlying, expiry)
    position = Position(
        position_id=position_id, sleeve="A", underlying=underlying, structure="iron_condor",
        legs=legs, qty=qty, credit=round(credit, 2), width=eff_width,
        max_loss=round(unit_loss * qty, 2), client_order_id=position_id,
        opened_at=now.isoformat(),
    )
    return (
        Proposal(
            underlying=underlying, structure="iron_condor", legs=legs, qty=qty,
            credit=round(credit, 2), width=eff_width, max_loss=round(unit_loss * qty, 2),
            net_delta_dollars=net_delta, net_vega_dollars=net_vega,
            dte=dte_of(expiry, now), position=position,
        ),
        "ok",
    )


def build_candidates(
    underlying: str,
    contracts: list[Contract],
    spot: float,
    now: Optional[datetime] = None,
    delta_target: Optional[float] = None,
) -> tuple[list[Proposal], dict]:
    """Menu of viable condors: expected-move wings first, 5 dollar wings second."""
    now = now or now_et()
    delta_target = delta_target or STRAT.short_delta_target
    diag: dict = {"underlying": underlying, "rejects": {}}

    expiry = pick_near_expiry(contracts, now)
    if expiry is None:
        diag["reject"] = "no expiry in dte band"
        return [], diag
    diag["expiry"] = expiry

    iv = atm_iv(contracts, expiry, spot) or 0.12
    em_width = max(
        STRAT.min_wing_width,
        round(expected_move(spot, iv, dte_of(expiry, now))),
    )
    widths = sorted({em_width, STRAT.min_wing_width}, reverse=True)

    candidates: list[Proposal] = []
    for width in widths:
        proposal, why = _assemble(underlying, contracts, expiry, spot, width, delta_target, now)
        if proposal is not None:
            candidates.append(proposal)
        else:
            diag["rejects"][f"width_{width:.0f}"] = why

    candidates.sort(key=lambda p: p.credit / p.width, reverse=True)
    diag["candidates"] = [p.summary() for p in candidates]
    return candidates, diag
