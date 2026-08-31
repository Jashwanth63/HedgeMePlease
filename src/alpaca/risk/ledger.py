"""The agent's book of record, on SQLite.

Alpaca sees one anonymous account; position attribution, equity anchors, the
halt flag, and the audit trail live here. The Ledger is the only writer of the
trades table; the graph passes it around as a service object.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..config import RISK, now_et
from ..data.db import Db


@dataclass
class Leg:
    symbol: str            # OCC symbol
    side: str              # "buy" or "sell" as opened
    ratio_qty: int
    strike: float
    opt_type: str          # "put" | "call"
    expiry: str            # YYYY-MM-DD
    entry_iv: float
    entry_delta: float


@dataclass
class Position:
    position_id: str
    sleeve: str
    underlying: str
    structure: str
    legs: list[Leg]
    qty: int
    credit: float
    width: float
    max_loss: float
    client_order_id: str
    opened_at: str
    status: str = "pending"   # pending | open | closing | closed | abandoned
    close_order_id: Optional[str] = None
    closed_at: Optional[str] = None
    exit_debit: Optional[float] = None
    realized_pnl: Optional[float] = None
    close_reason: Optional[str] = None


def new_position_id(underlying: str, expiry: str) -> str:
    stamp = now_et().strftime("%m%d%H%M%S")
    return f"SLA-{underlying}-{expiry.replace('-', '')}-{stamp}"


class Ledger:
    def __init__(self, db: Db) -> None:
        self.db = db

    # ---- positions -----------------------------------------------------

    def add(self, pos: Position, entry_context: Optional[dict] = None) -> None:
        self.db.conn.execute(
            "INSERT INTO trades (trade_id, sleeve, symbol, structure, status, qty, credit, "
            "width, max_loss, legs_json, client_order_id, opened_at, entry_context) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pos.position_id, pos.sleeve, pos.underlying, pos.structure, pos.status,
                pos.qty, pos.credit, pos.width, pos.max_loss,
                json.dumps([asdict(leg) for leg in pos.legs]),
                pos.client_order_id, pos.opened_at,
                json.dumps(entry_context, default=str) if entry_context else None,
            ),
        )
        self.db.conn.commit()

    def update(self, pos: Position) -> None:
        self.db.conn.execute(
            "UPDATE trades SET status=?, qty=?, credit=?, max_loss=?, client_order_id=?, "
            "closed_at=?, close_order_id=?, exit_debit=?, realized_pnl=?, close_reason=? "
            "WHERE trade_id=?",
            (
                pos.status, pos.qty, pos.credit, pos.max_loss, pos.client_order_id,
                pos.closed_at, pos.close_order_id, pos.exit_debit, pos.realized_pnl,
                pos.close_reason, pos.position_id,
            ),
        )
        self.db.conn.commit()

    def open_positions(self) -> list[Position]:
        rows = self.db.conn.execute(
            "SELECT * FROM trades WHERE status IN ('pending', 'open', 'closing')"
        ).fetchall()
        return [self._decode(r) for r in rows]

    def all_positions(self) -> list[Position]:
        rows = self.db.conn.execute("SELECT * FROM trades ORDER BY opened_at").fetchall()
        return [self._decode(r) for r in rows]

    @staticmethod
    def _decode(row) -> Position:
        legs = [Leg(**leg) for leg in json.loads(row["legs_json"])]
        return Position(
            position_id=row["trade_id"],
            sleeve=row["sleeve"],
            underlying=row["symbol"],
            structure=row["structure"],
            legs=legs,
            qty=row["qty"],
            credit=row["credit"],
            width=row["width"],
            max_loss=row["max_loss"],
            client_order_id=row["client_order_id"],
            opened_at=row["opened_at"],
            status=row["status"],
            close_order_id=row["close_order_id"],
            closed_at=row["closed_at"],
            exit_debit=row["exit_debit"],
            realized_pnl=row["realized_pnl"],
            close_reason=row["close_reason"],
        )

    # ---- equity anchors and halt --------------------------------------

    @property
    def hwm(self) -> float:
        return float(self.db.get_state("hwm", RISK.starting_equity))

    @property
    def day_anchor(self) -> float:
        return float(self.db.get_state("day_anchor_equity", 0.0))

    @property
    def halted(self) -> bool:
        return bool(self.db.get_state("halted", False))

    def halt(self, reason: str) -> None:
        self.db.set_state("halted", True)
        self.db.set_state("halt_reason", reason)

    def unhalt(self) -> None:
        self.db.set_state("halted", False)
        self.db.set_state("halt_reason", "")

    def update_equity(self, equity: float) -> None:
        today = now_et().date().isoformat()
        if self.db.get_state("day_anchor_date") != today:
            self.db.set_state("day_anchor_date", today)
            self.db.set_state("day_anchor_equity", equity)
        self.db.set_state("last_equity", equity)
        self.db.set_state("hwm", max(self.hwm, equity))

    def memo(self, event: str, detail: dict) -> None:
        self.db.memo(event, detail)
