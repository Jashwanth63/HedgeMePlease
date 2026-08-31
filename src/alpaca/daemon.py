"""The long-running daemon: APScheduler fires one graph cycle every five
minutes during market hours until the contest end, with graceful shutdown
and LangGraph checkpointing to SQLite.
"""

from __future__ import annotations

import asyncio
import signal

from .broker.mcp import AlpacaMCP
from .config import CHECKPOINT_DB, CONTEST_END, CYCLE_MINUTES, now_et
from .data.db import Db
from .graph import Services, build_graph, run_cycle
from .risk.ledger import Ledger


async def _cycle_once(services: Services, graph) -> None:
    try:
        async with AlpacaMCP() as broker:
            services.broker = broker
            await run_cycle(services, graph)
    except Exception as exc:  # keep the daemon alive through transient failures
        services.db.memo("cycle_exception", {"error": repr(exc)[:500]})


async def main_async(dry_run: bool = False) -> None:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    db = Db()
    services = Services(broker=None, db=db, ledger=Ledger(db), dry_run=dry_run)

    from .graph import acquire_checkpointer

    checkpointer_cm, checkpointer = await acquire_checkpointer(db.memo)
    graph = build_graph(services, checkpointer)

    stop = asyncio.Event()

    def _shutdown(*_):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _shutdown)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: stop.set())

    scheduler = AsyncIOScheduler(timezone="America/New_York")
    scheduler.add_job(
        _cycle_once,
        "interval",
        minutes=CYCLE_MINUTES,
        args=[services, graph],
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    db.memo("daemon_start", {"dry_run": dry_run, "cycle_minutes": CYCLE_MINUTES})

    # a restart may have interrupted an order ladder mid-flight, leaving a
    # resting order nobody tracks; sweep strays before the first cycle
    try:
        async with AlpacaMCP() as broker:
            strays = await broker.open_orders()
            for order in strays:
                oid = order.get("id")
                if oid:
                    try:
                        await broker.cancel_order(oid)
                    except Exception:
                        pass
            if strays:
                db.memo("startup_canceled_stray_orders", {"count": len(strays)})
    except Exception as exc:
        db.memo("startup_hygiene_error", {"error": repr(exc)[:300]})

    await _cycle_once(services, graph)  # immediate first cycle

    while not stop.is_set() and now_et() < CONTEST_END:
        try:
            await asyncio.wait_for(stop.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass

    scheduler.shutdown(wait=False)
    db.memo("daemon_stop", {"reason": "signal" if stop.is_set() else "contest end"})
    if checkpointer_cm is not None:
        await checkpointer_cm.__aexit__(None, None, None)
    db.close()


def main(dry_run: bool = False) -> None:
    asyncio.run(main_async(dry_run))
