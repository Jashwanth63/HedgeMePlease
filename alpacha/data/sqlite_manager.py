"""
SQLite Storage Manager for AlpachaBot.
Handles persistence of 1-minute bars, model forecasts, executed trades, risk snapshots, and state metadata.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from alpacha.utils.logger import get_logger

logger = get_logger("sqlite")


class SQLiteManager:
    def __init__(self, db_path: str | Path = "data/alpacha.db") -> None:
        self.is_memory = str(db_path) == ":memory:"
        self.db_path = Path(db_path) if not self.is_memory else ":memory:"
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.is_memory:
            self._mem_conn = sqlite3.connect(":memory:", timeout=30.0)
            self._mem_conn.row_factory = sqlite3.Row
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def get_connection(self):
        if self.is_memory and self._mem_conn is not None:
            try:
                yield self._mem_conn
                self._mem_conn.commit()
            except Exception as e:
                self._mem_conn.rollback()
                logger.error(f"Database error: {e}", exc_info=True)
                raise
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}", exc_info=True)
                raise
            finally:
                conn.close()


    def _init_db(self) -> None:
        """Initializes database schema and indexes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bars (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    forecasted_rv REAL NOT NULL,
                    metrics_json TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entry_timestamp TEXT NOT NULL,
                    exit_timestamp TEXT,
                    legs_json TEXT NOT NULL,
                    credit_received REAL NOT NULL,
                    exit_pnl REAL,
                    gates_passed_json TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    peak_equity REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    risk_level TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON bars(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_ts ON forecasts(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_ts ON risk_snapshots(timestamp)")

    def save_bars(self, df: pd.DataFrame, symbol: str) -> int:
        """Saves a DataFrame of 1-minute bars into the database."""
        if df.empty:
            return 0

        rows = []
        for index, row in df.iterrows():
            ts_str = index.isoformat() if hasattr(index, "isoformat") else str(index)
            rows.append((
                symbol,
                ts_str,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            ))

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO bars (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            return len(rows)

    def load_bars(
        self,
        symbol: str,
        start_ts: Optional[datetime | str] = None,
        end_ts: Optional[datetime | str] = None,
    ) -> pd.DataFrame:
        """Loads historical 1-minute bars for a symbol as a Pandas DataFrame."""
        query = "SELECT timestamp, open, high, low, close, volume FROM bars WHERE symbol = ?"
        params: List[Any] = [symbol]

        if start_ts:
            query += " AND timestamp >= ?"
            params.append(start_ts.isoformat() if isinstance(start_ts, datetime) else str(start_ts))
        if end_ts:
            query += " AND timestamp <= ?"
            params.append(end_ts.isoformat() if isinstance(end_ts, datetime) else str(end_ts))

        query += " ORDER BY timestamp ASC"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        return df

    # -------------------------------------------------------------
    # Forecasts operations
    # -------------------------------------------------------------
    def save_forecast(
        self,
        symbol: str,
        forecasted_rv: float,
        timestamp: Optional[datetime] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> int:
        ts = (timestamp or datetime.now()).isoformat()
        metrics_json = json.dumps(metrics or {})

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO forecasts (timestamp, symbol, forecasted_rv, metrics_json)
                VALUES (?, ?, ?, ?)
            """, (ts, symbol, forecasted_rv, metrics_json))
            return cursor.lastrowid or 0

    def get_latest_forecast(self, symbol: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, forecasted_rv, metrics_json
                FROM forecasts
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            if row:
                return {
                    "timestamp": row["timestamp"],
                    "forecasted_rv": row["forecasted_rv"],
                    "metrics": json.loads(row["metrics_json"] or "{}"),
                }
            return None

    # -------------------------------------------------------------
    # Trades operations
    # -------------------------------------------------------------
    def save_trade(
        self,
        trade_id: str,
        symbol: str,
        status: str,
        entry_timestamp: datetime | str,
        legs: List[Dict[str, Any]],
        credit_received: float,
        gates_passed: Optional[Dict[str, Any]] = None,
        exit_timestamp: Optional[datetime | str] = None,
        exit_pnl: Optional[float] = None,
    ) -> None:
        entry_ts = entry_timestamp.isoformat() if isinstance(entry_timestamp, datetime) else str(entry_timestamp)
        exit_ts = exit_timestamp.isoformat() if isinstance(exit_timestamp, datetime) else (str(exit_timestamp) if exit_timestamp else None)
        legs_json = json.dumps(legs)
        gates_json = json.dumps(gates_passed or {})

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trades (
                    trade_id, symbol, status, entry_timestamp, exit_timestamp,
                    legs_json, credit_received, exit_pnl, gates_passed_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (trade_id, symbol, status, entry_ts, exit_ts, legs_json, credit_received, exit_pnl, gates_json))

    def update_trade_exit(self, trade_id: str, exit_timestamp: datetime | str, exit_pnl: float, status: str = "CLOSED") -> None:
        exit_ts = exit_timestamp.isoformat() if isinstance(exit_timestamp, datetime) else str(exit_timestamp)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trades
                SET status = ?, exit_timestamp = ?, exit_pnl = ?
                WHERE trade_id = ?
            """, (status, exit_ts, exit_pnl, trade_id))

    def get_open_trades(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status = 'OPEN'")
            rows = cursor.fetchall()
            trades = []
            for r in rows:
                trades.append({
                    "trade_id": r["trade_id"],
                    "symbol": r["symbol"],
                    "status": r["status"],
                    "entry_timestamp": r["entry_timestamp"],
                    "legs": json.loads(r["legs_json"]),
                    "credit_received": r["credit_received"],
                    "exit_pnl": r["exit_pnl"],
                    "gates_passed": json.loads(r["gates_passed_json"] or "{}"),
                })
            return trades

    # -------------------------------------------------------------
    # Risk snapshots operations
    # -------------------------------------------------------------
    def save_risk_snapshot(
        self,
        equity: float,
        peak_equity: float,
        drawdown_pct: float,
        risk_level: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = (timestamp or datetime.now()).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO risk_snapshots (timestamp, equity, peak_equity, drawdown_pct, risk_level)
                VALUES (?, ?, ?, ?, ?)
            """, (ts, equity, peak_equity, drawdown_pct, risk_level))

    def get_latest_risk_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM risk_snapshots ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    # -------------------------------------------------------------
    # Meta key-value store
    # -------------------------------------------------------------
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM meta WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

