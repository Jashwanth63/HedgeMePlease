"""
Main entry point for AlpachaBot.
CLI interface to start daemon, run single cycles, check system risk status, or test strategies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alpacha.bot import AlpachaBot
from alpacha.config import Settings
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.utils.logger import get_logger, setup_logging

logger = get_logger("main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpacha: Alpaca Iron Condor Options Trading Bot")
    parser.add_argument("--config", type=str, default="config/settings.yaml", help="Path to settings.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending live broker orders")
    parser.add_argument("--single-run", action="store_true", help="Execute single risk & trade scan cycle and exit")
    parser.add_argument("--status", action="store_true", help="Show current portfolio risk status and open trades")
    args = parser.parse_args()

    # Initialize logging
    setup_logging()

    # Load configuration
    settings = Settings.load(config_path=args.config)
    if args.dry_run:
        settings.app.dry_run = True

    if args.status:
        db = SQLiteManager(settings.app.db_path)
        risk_snap = db.get_latest_risk_snapshot()
        open_trades = db.get_open_trades()
        peak_equity = db.get_meta("peak_account_equity", "N/A")

        print("========================================")
        print("          ALPACHABOT SYSTEM STATUS      ")
        print("========================================")
        print(f"Peak Equity:       ${float(peak_equity):,.2f}" if peak_equity != "N/A" else "Peak Equity:       N/A")
        if risk_snap:
            print(f"Current Equity:    ${risk_snap['equity']:,.2f}")
            print(f"Drawdown:          {risk_snap['drawdown_pct']:.2%}")
            print(f"Risk Level:        {risk_snap['risk_level']}")
            print(f"Last Snapshot:     {risk_snap['timestamp']}")
        else:
            print("No risk snapshots recorded yet.")
        print("----------------------------------------")
        print(f"Open Trades:       {len(open_trades)}")
        for t in open_trades:
            print(f"  - [{t['trade_id']}] {t['symbol']} | Credit: ${t['credit_received']:.2f} | Entered: {t['entry_timestamp']}")
        print("========================================")
        sys.exit(0)

    bot = AlpachaBot(settings)

    if args.single_run:
        logger.info("Executing single scan cycle (--single-run)...")
        bot.run_cycle()
        logger.info("Single cycle completed.")
        sys.exit(0)

    # Start long-running daemon
    bot.start()


if __name__ == "__main__":
    main()
