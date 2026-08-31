"""Command line entry points.

alpaca status    account, book, drawdown state
alpaca rv SPY    daily RV series, HAR forecast, walk-forward check
alpaca preview   chain to candidates to risk verdict, no orders
alpaca scan      one full dry-run graph cycle (no orders)
alpaca once      one live graph cycle (places orders)
alpaca loop      the daemon until contest end
alpaca flatten   close every open position now
alpaca panic     cancel all orders, flatten, halt
alpaca unhalt    clear the halt after human review
alpaca memos     tail the audit trail
"""

from __future__ import annotations

import argparse
import asyncio
import json

from .broker.mcp import AlpacaMCP
from .config import RISK, STRAT, now_et
from .data.db import Db
from .graph import Services, run_cycle
from .model.har import best_forecast, fit, walk_forward
from .model.volutils import daily_stats
from .risk.engine import evaluate_account
from .risk.ledger import Ledger


def _services(dry_run: bool = False) -> Services:
    db = Db()
    return Services(broker=None, db=db, ledger=Ledger(db), dry_run=dry_run)


async def _status() -> None:
    services = _services()
    ledger = services.ledger
    async with AlpacaMCP() as mcp:
        acct = await mcp.account()
        equity = float(acct["equity"])
        action = evaluate_account(equity, ledger.hwm, ledger.day_anchor, ledger.halted)
        print(f"equity            {equity:,.2f}")
        print(f"cash              {float(acct.get('cash', 0)):,.2f}")
        print(f"options level     {acct.get('options_trading_level')}")
        print(f"high-water mark   {max(ledger.hwm, RISK.starting_equity):,.2f}")
        print(f"account action    {action.value}")
        print(f"halted            {ledger.halted}")
        open_pos = ledger.open_positions()
        print(f"open positions    {len(open_pos)}")
        for pos in open_pos:
            print(
                f"  {pos.position_id}  {pos.structure} x{pos.qty} credit {pos.credit:.2f} "
                f"max_loss {pos.max_loss:.0f} [{pos.status}]"
            )


async def _rv(symbol: str) -> None:
    async with AlpacaMCP() as mcp:
        bars = await mcp.stock_bars_5min(symbol, days=130)
    stats = daily_stats(bars, now_et())[1:]
    print(f"{symbol}: {len(stats)} completed daily points")
    if not stats:
        return
    tail = ", ".join(f"{s.rv:.1%}" for s in stats[-5:])
    print(f"last 5 daily RV: {tail}")
    value, method = best_forecast(stats, STRAT.forecast_horizon_days)
    print(f"forecast ({method}, {STRAT.forecast_horizon_days}d): {value:.1%}")
    if len(stats) >= 130:
        report = walk_forward(stats)
        print(
            f"walk-forward n={report.n_forecasts}: rmse har {report.rmse_har:.4f} | "
            f"lag1 {report.rmse_lag1:.4f} | mean20 {report.rmse_mean20:.4f} | "
            f"har_beats_fallback={report.har_beats_fallback}"
        )


async def _preview(symbol: str) -> None:
    from datetime import timedelta

    from .risk.engine import check_pre_trade
    from .strategy.condor import atm_iv, build_candidates, parse_chain, pick_far_expiry, pick_near_expiry
    from .strategy.gates import evaluate_gates

    services = _services()
    now = now_et()
    async with AlpacaMCP() as mcp:
        equity = await mcp.equity()
        spots = await mcp.spots([symbol])
        spot = spots.get(symbol, 0.0)
        print(f"{symbol} spot {spot:.2f}  equity {equity:,.0f}")
        near = parse_chain(symbol, await mcp.option_chain(
            symbol,
            expiration_date_lte=(now + timedelta(days=STRAT.near_dte_max)).date().isoformat(),
            strike_price_gte=spot * 0.88, strike_price_lte=spot * 1.12,
        ))
        far = parse_chain(symbol, await mcp.option_chain(
            symbol,
            expiration_date_gte=(now + timedelta(days=STRAT.far_dte_min)).date().isoformat(),
            expiration_date_lte=(now + timedelta(days=STRAT.far_dte_max)).date().isoformat(),
            strike_price_gte=spot * 0.96, strike_price_lte=spot * 1.04,
        ))
        near_exp, far_exp = pick_near_expiry(near, now), pick_far_expiry(far, now)
        n_iv = atm_iv(near, near_exp, spot) if near_exp else None
        f_iv = atm_iv(far, far_exp, spot) if far_exp else None
        print(f"near {near_exp} atm_iv {n_iv and round(n_iv, 4)} | far {far_exp} atm_iv {f_iv and round(f_iv, 4)}")
        bars = await mcp.stock_bars_5min(symbol, days=130)
        stats = daily_stats(bars, now)[1:]
        rv, method = (None, "insufficient")
        if len(stats) >= 25:
            rv, method = best_forecast(stats, STRAT.forecast_horizon_days)
        print(f"rv forecast {rv and round(rv, 4)} ({method})")
        gates = evaluate_gates(n_iv, f_iv, rv, now)
        print(f"gates pass={gates.all_pass} failed={gates.failed()} {gates.details}")
        cands, diag = build_candidates(symbol, near, spot, now)
        if not cands:
            print(f"no candidates: {diag.get('rejects') or diag.get('reject')}")
            return
        for i, c in enumerate(cands):
            print(f"candidate {i}: {json.dumps(c.summary())}")
        verdict = check_pre_trade(
            services.ledger.open_positions(), cands[0], equity,
            services.ledger.hwm, services.ledger.day_anchor, services.ledger.halted, spots, now,
        )
        print(f"risk verdict approved={verdict.approved} reasons={verdict.reasons} size={verdict.size_factor}")


async def _cycle(dry_run: bool) -> None:
    services = _services(dry_run=dry_run)
    async with AlpacaMCP() as mcp:
        services.broker = mcp
        result = await run_cycle(services)
    print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, indent=2, default=str))


async def _flatten(halt: bool) -> None:
    from .broker.executor import submit_close

    services = _services()
    ledger = services.ledger
    if halt:
        ledger.halt(f"manual panic at {now_et().isoformat()}")
        services.db.memo("PANIC", {"by": "operator"})
    async with AlpacaMCP() as mcp:
        for order in await mcp.open_orders():
            oid = order.get("id")
            if oid:
                try:
                    await mcp.cancel_order(oid)
                except Exception:
                    pass
        for pos in ledger.open_positions():
            await submit_close(mcp, services.db.memo, pos, "manual_flatten")
            ledger.update(pos)
    print("flatten pass complete")


def _report() -> None:
    """Contest evidence in one place: equity path, forecasts vs realized, trades with reasoning."""
    db = Db()

    rows = db.conn.execute(
        "SELECT ts, equity, peak, drawdown, action FROM risk_snapshots ORDER BY id"
    ).fetchall()
    if rows:
        first, last = rows[0], rows[-1]
        print("EQUITY")
        print(f"  first snapshot  {first['ts'][:16]}  {first['equity']:,.2f}")
        print(f"  last snapshot   {last['ts'][:16]}  {last['equity']:,.2f}  "
              f"dd {last['drawdown']:.2%}  action {last['action']}")

    print("FORECAST vs REALIZED (annualized vol)")
    for symbol in STRAT.underlyings:
        pairs = db.forecast_vs_realized(symbol, limit=10)
        latest = db.conn.execute(
            "SELECT ts, rv_forecast, method FROM forecasts WHERE symbol=? ORDER BY id DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        if latest:
            print(f"  {symbol}: latest forecast {latest['rv_forecast']:.1%} ({latest['method']}, {latest['ts'][:16]})")
        for f, r in pairs:
            print(f"    forecast {f:.1%} -> realized next day {r:.1%}  (miss {abs(r / f - 1):.0%})" if f else "")

    print("TRADES")
    trades = db.conn.execute("SELECT * FROM trades ORDER BY opened_at").fetchall()
    if not trades:
        print("  none yet")
    for t in trades:
        pnl = f"{t['realized_pnl']:+,.0f}" if t["realized_pnl"] is not None else "open"
        print(f"  {t['trade_id']}  {t['symbol']} {t['structure']} x{t['qty']}  "
              f"credit {t['credit']:.2f}  max_loss {t['max_loss']:.0f}  [{t['status']}]  pnl {pnl}"
              + (f"  exit: {t['close_reason']}" if t["close_reason"] else ""))
        if t["entry_context"]:
            ctx = json.loads(t["entry_context"])
            gates = ctx.get("gates") or {}
            regime = ctx.get("regime") or {}
            print(f"      edge ratio {gates.get('iv_rv_ratio')}  rv {gates.get('rv_forecast')}  "
                  f"near iv {gates.get('near_atm_iv')}  regime {regime.get('stance')}")
            if ctx.get("proposer_why"):
                print(f"      proposer: {ctx['proposer_why']}")
            veto = ctx.get("veto") or {}
            if veto.get("reason"):
                print(f"      news analyst: {str(veto['reason'])[:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="alpaca")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    sub.add_parser("status")
    for name in ("rv", "preview"):
        p = sub.add_parser(name)
        p.add_argument("symbol", nargs="?", default="SPY")
    sub.add_parser("scan")
    sub.add_parser("once")
    loop_p = sub.add_parser("loop")
    loop_p.add_argument("--dry-run", action="store_true")
    sub.add_parser("flatten")
    sub.add_parser("panic")
    sub.add_parser("unhalt")
    memos_p = sub.add_parser("memos")
    memos_p.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    if args.cmd == "report":
        _report()
    elif args.cmd == "status":
        asyncio.run(_status())
    elif args.cmd == "rv":
        asyncio.run(_rv(args.symbol.upper()))
    elif args.cmd == "preview":
        asyncio.run(_preview(args.symbol.upper()))
    elif args.cmd == "scan":
        asyncio.run(_cycle(dry_run=True))
    elif args.cmd == "once":
        asyncio.run(_cycle(dry_run=False))
    elif args.cmd == "loop":
        from .daemon import main as daemon_main
        daemon_main(dry_run=args.dry_run)
    elif args.cmd == "flatten":
        asyncio.run(_flatten(halt=False))
    elif args.cmd == "panic":
        asyncio.run(_flatten(halt=True))
    elif args.cmd == "unhalt":
        services = _services()
        services.ledger.unhalt()
        services.db.memo("unhalt", {"by": "operator"})
        print("halt cleared")
    elif args.cmd == "memos":
        db = Db()
        for row in reversed(db.recent_memos(args.limit)):
            print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()
