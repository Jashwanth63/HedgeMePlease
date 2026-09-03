"""SQLite persistence: trades, forecasts, daily RV, risk snapshots, app state,
and the memos audit trail. One file, WAL mode, judge-readable with any client.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from ..config import DB_PATH, STATE_DIR, now_et

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    sleeve TEXT NOT NULL DEFAULT 'A',
    symbol TEXT NOT NULL,
    structure TEXT NOT NULL,
    status TEXT NOT NULL,
    qty INTEGER NOT NULL,
    credit REAL NOT NULL,
    width REAL NOT NULL,
    max_loss REAL NOT NULL,
    legs_json TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    close_order_id TEXT,
    exit_debit REAL,
    realized_pnl REAL,
    close_reason TEXT,
    entry_context TEXT
);
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    rv_forecast REAL NOT NULL,
    method TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rv_daily (
    symbol TEXT NOT NULL,
    day TEXT NOT NULL,
    rv REAL NOT NULL,
    bv REAL NOT NULL,
    jump REAL NOT NULL,
    ret REAL NOT NULL,
    PRIMARY KEY (symbol, day)
);
CREATE TABLE IF NOT EXISTS risk_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    peak REAL NOT NULL,
    drawdown REAL NOT NULL,
    action TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
"""


class Db:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(trades)")}
        for col, ddl in (
            ("entry_context", "TEXT"),
            ("last_cost", "REAL"),
            ("unrealized_pnl", "REAL"),
            ("marked_at", "TEXT"),
        ):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {ddl}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- audit trail ---------------------------------------------------

    _SLEEVE_A_EVENTS = {
        "gates", "candidates", "proposal_chosen", "news_veto", "risk_verdict",
        "dry_run_would_open", "entry_skip", "rv_forecast", "regime_view",
    }
    _ROLE_SLEEVES = {
        "regime_analyst": "A", "proposer": "A", "news_analyst": "A",
        "event_analyst": "B", "hedge_analyst": "C", "journalist": "core",
    }

    @classmethod
    def _infer_sleeve(cls, event: str, detail: dict[str, Any]) -> str:
        if event.startswith("event_"):
            return "B"
        if event.startswith("hedge_"):
            return "C"
        if event in cls._SLEEVE_A_EVENTS:
            return "A"
        if event == "llm_call":
            return cls._ROLE_SLEEVES.get(str(detail.get("role", "")), "core")
        return "core"

    def memo(self, event: str, detail: dict[str, Any]) -> None:
        if "sleeve" not in detail:
            detail = {"sleeve": self._infer_sleeve(event, detail), **detail}
        self.conn.execute(
            "INSERT INTO memos (ts, event, detail_json) VALUES (?, ?, ?)",
            (now_et().isoformat(), event, json.dumps(detail, default=str)),
        )
        self.conn.commit()

    def sleeve_pnl(self) -> dict[str, dict[str, float]]:
        """Realized, unrealized, and committed risk per sleeve."""
        out: dict[str, dict[str, float]] = {}
        rows = self.conn.execute(
            "SELECT sleeve, "
            "SUM(CASE WHEN status='closed' THEN realized_pnl ELSE 0 END) AS realized, "
            "SUM(CASE WHEN status IN ('open','closing') THEN COALESCE(unrealized_pnl, 0) "
            "ELSE 0 END) AS unrealized, "
            "SUM(CASE WHEN status IN ('open','closing','pending') THEN max_loss ELSE 0 END) "
            "AS committed, "
            "COUNT(CASE WHEN status IN ('open','closing') THEN 1 END) AS open_count "
            "FROM trades GROUP BY sleeve"
        ).fetchall()
        for r in rows:
            out[r["sleeve"]] = {
                "realized": round(r["realized"] or 0.0, 2),
                "unrealized": round(r["unrealized"] or 0.0, 2),
                "net": round((r["realized"] or 0.0) + (r["unrealized"] or 0.0), 2),
                "committed": round(r["committed"] or 0.0, 2),
                "open": r["open_count"] or 0,
            }
        return out

    def recent_memos(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, event, detail_json FROM memos ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"ts": r["ts"], "event": r["event"], **json.loads(r["detail_json"])}
            for r in rows
        ]

    # ---- app state -----------------------------------------------------

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    # ---- model tables --------------------------------------------------

    def upsert_rv_daily(self, symbol: str, rows: list) -> None:
        self.conn.executemany(
            "INSERT INTO rv_daily (symbol, day, rv, bv, jump, ret) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol, day) DO UPDATE SET rv=excluded.rv, bv=excluded.bv, "
            "jump=excluded.jump, ret=excluded.ret",
            [(symbol, s.day, s.rv, s.bv, s.jump, s.ret) for s in rows],
        )
        self.conn.commit()

    def record_forecast(self, symbol: str, horizon: int, value: float, method: str) -> None:
        self.conn.execute(
            "INSERT INTO forecasts (ts, symbol, horizon, rv_forecast, method) VALUES (?, ?, ?, ?, ?)",
            (now_et().isoformat(), symbol, horizon, value, method),
        )
        self.conn.commit()

    def forecast_vs_realized(self, symbol: str, limit: int = 6) -> list[tuple[float, float]]:
        """Recent (forecast, next-day realized) pairs for the demotion rule."""
        rows = self.conn.execute(
            "SELECT f.rv_forecast AS f, r.rv AS r FROM forecasts f "
            "JOIN rv_daily r ON r.symbol = f.symbol AND r.day > substr(f.ts, 1, 10) "
            "WHERE f.symbol = ? AND f.method = 'har' "
            "GROUP BY f.id HAVING r.day = MIN(r.day) ORDER BY f.id DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [(row["f"], row["r"]) for row in reversed(rows)]

    def record_risk_snapshot(self, equity: float, peak: float, drawdown: float, action: str) -> None:
        self.conn.execute(
            "INSERT INTO risk_snapshots (ts, equity, peak, drawdown, action) VALUES (?, ?, ?, ?, ?)",
            (now_et().isoformat(), equity, peak, drawdown, action),
        )
        self.conn.commit()
