"""Read-only status page for the team.

Serves the audit trail, trades with reasoning, forecasts, and the equity path
as a single auto-refreshing HTML page. No shell, no credentials: viewers need
only the URL with its token (MONITOR_TOKEN, default 'hedgeme26'). Strictly
read-only: this process never writes to the database and exposes no actions.
"""

from __future__ import annotations

import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import STRAT, now_et
from .data.db import Db

TOKEN = os.environ.get("MONITOR_TOKEN", "hedgeme26")


def _e(value) -> str:
    return html.escape(str(value))


def render(db: Db) -> str:
    parts: list[str] = []
    parts.append(
        "<meta http-equiv='refresh' content='60'><meta charset='utf-8'>"
        "<title>HedgeMePlease status</title>"
        "<style>body{font-family:ui-monospace,Consolas,monospace;font-size:13px;"
        "background:#101413;color:#d7e0dc;margin:24px;line-height:1.5}"
        "h2{color:#7fd3b8;font-size:15px;margin:20px 0 6px}"
        ".dim{color:#8a968f}.bad{color:#e0908a}.good{color:#9cc468}"
        "table{border-collapse:collapse}td,th{padding:2px 12px 2px 0;text-align:left}"
        "pre{white-space:pre-wrap;word-break:break-word;margin:2px 0}</style>"
    )
    parts.append(f"<p class='dim'>generated {_e(now_et().isoformat(timespec='seconds'))} "
                 f"(refreshes every 60s)</p>")

    row = db.conn.execute(
        "SELECT ts, equity, peak, drawdown, action FROM risk_snapshots ORDER BY id DESC LIMIT 1"
    ).fetchone()
    parts.append("<h2>account</h2>")
    if row:
        cls = "good" if row["action"] == "ok" else "bad"
        parts.append(
            f"<pre>equity {row['equity']:,.2f}   peak {row['peak']:,.2f}   "
            f"drawdown {row['drawdown']:.2%}   action <span class='{cls}'>{_e(row['action'])}</span>"
            f"   as of {_e(row['ts'][:19])}</pre>"
        )
    else:
        parts.append("<pre class='dim'>no snapshots yet</pre>")

    parts.append("<h2>forecasts</h2>")
    for symbol in STRAT.underlyings:
        latest = db.conn.execute(
            "SELECT ts, rv_forecast, method FROM forecasts WHERE symbol=? ORDER BY id DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        if latest:
            parts.append(
                f"<pre>{_e(symbol)}  rv {latest['rv_forecast']:.1%} ({_e(latest['method'])}) "
                f"at {_e(latest['ts'][:16])}</pre>"
            )
        else:
            parts.append(f"<pre class='dim'>{_e(symbol)}  none yet</pre>")

    parts.append("<h2>trades</h2>")
    trades = db.conn.execute("SELECT * FROM trades ORDER BY opened_at DESC").fetchall()
    if not trades:
        parts.append("<pre class='dim'>none yet</pre>")
    for t in trades:
        pnl = f"{t['realized_pnl']:+,.0f}" if t["realized_pnl"] is not None else "open"
        pnl_cls = "dim" if pnl == "open" else ("good" if (t["realized_pnl"] or 0) >= 0 else "bad")
        parts.append(
            f"<pre>{_e(t['trade_id'])}  {_e(t['symbol'])} x{t['qty']}  credit {t['credit']:.2f}  "
            f"max_loss {t['max_loss']:.0f}  [{_e(t['status'])}]  "
            f"pnl <span class='{pnl_cls}'>{_e(pnl)}</span>"
            + (f"  exit: {_e(t['close_reason'])}" if t["close_reason"] else "")
            + "</pre>"
        )
        if t["entry_context"]:
            ctx = json.loads(t["entry_context"])
            gates = ctx.get("gates") or {}
            parts.append(
                f"<pre class='dim'>   edge {_e(gates.get('iv_rv_ratio'))}  "
                f"rv {_e(gates.get('rv_forecast'))}  why: {_e(ctx.get('proposer_why'))}</pre>"
            )

    parts.append("<h2>audit trail (latest 40)</h2>")
    for m in db.recent_memos(40):
        detail = {k: v for k, v in m.items() if k not in ("ts", "event", "input")}
        line = f"{m['ts'][11:19]}  {m['event']:<22} {json.dumps(detail, default=str)[:400]}"
        cls = "bad" if ("exception" in m["event"] or "error" in m["event"]) else ""
        parts.append(f"<pre class='{cls}'>{_e(line)}</pre>")

    return "".join(parts)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        token = (parse_qs(url.query).get("k") or [""])[0]
        if token != TOKEN:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"forbidden")
            return
        try:
            body = render(self.server.db).encode("utf-8")  # type: ignore[attr-defined]
        except Exception as exc:
            body = f"render error: {exc}".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep journald quiet
        pass


def serve(port: int = 8080) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.db = Db()  # type: ignore[attr-defined]
    print(f"monitor serving on :{port}")
    server.serve_forever()
