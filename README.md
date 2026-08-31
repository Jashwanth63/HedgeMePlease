# Alpacha: Alpaca Iron Condor Options Trading Bot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)]()

A robust, quantitative options trading daemon running on the **Alpaca** platform. The bot trades defined-risk **Iron Condors** on **SPY** and **QQQ**, forecasting 1-day realized volatility via an **Enhanced HAR Model** (Corsi 2009 + leverage effects + jump components), gating entries across three safety filters, and protecting capital with a **3.5% drawdown hard kill switch**.

---

## Architecture Overview

```
[ Market Clock Trigger ]
          │
          ▼
┌─────────────────────────┐
│   Risk Manager Check    │ ───► Peak Drawdown ≥ 3.5%? ──► [ KILL SWITCH: Liquidate & Halt ]
└───────────┬─────────────┘
            │ (Drawdown < 2.0%)
            ▼
┌─────────────────────────┐
│    Market Data Feed     │ ───► Fetch SPY/QQQ 1-min bars, Account Equity & Options Chain
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Enhanced HAR Vol Model  │ ───► Calculate Daily/Weekly/Monthly RV + Leverage + Jumps
└───────────┬─────────────┘      Generate 1-day RV Forecast
            │
            ▼
┌─────────────────────────┐
│    Trade Entry Gates    │
│ 1. Macro (<2h)?         │ ───► (FAIL) ──► Log Gate Rejection & Skip Entry
│ 2. Vol Contango?        │
│ 3. IV ≥ 1.2x RV Edge?   │
└───────────┬─────────────┘
            │ (ALL GATES PASSED)
            ▼
┌─────────────────────────┐
│   Build Iron Condor     │ ───► Find 0.20 Delta Short Strikes
└───────────┬─────────────┘      Set Long Wings at Short Strike ± Expected Move
            │
            ▼
┌─────────────────────────┐
│     Order Executor      │ ───► Place Limit Orders, Poll Fills & Record to SQLite
└─────────────────────────┘
```

---

## Key Features

1. **Enhanced HAR Volatility Model**:
   - High-frequency 1-minute log return realized variance ($\text{RV}$).
   - Jump-robust continuous variation via Bipower Variation ($\text{BV}$) and jump detection.
   - Heterogeneous autoregression across daily ($RV_d$), weekly ($RV_w$), and monthly ($RV_m$) frequencies.
   - Asymmetric **leverage effect** ($r_{d,-} = \min(0, r_d)$) and **jump variations** ($J_d$).

2. **Three-Gate Entry Filtering**:
   - **Gate 1: Macro Proximity**: Blocks trades within 2 hours of high-impact macroeconomic events.
   - **Gate 2: Vol Term Structure / Contango**: Prevents entering during acute backwardation spikes.
   - **Gate 3: Edge Filter**: Demands $IV_{\text{market}} \ge 1.2 \times RV_{\text{forecast}}$.

3. **Multi-Leg Iron Condor Construction**:
   - Short strikes targeted at $\approx 0.20$ delta.
   - Wing width dynamically sized by model Expected Move: $\text{Price} \times \text{IV} \times \sqrt{\text{DTE} / 365}$.

4. **Multi-Tier Risk Ladder**:
   - Rolling high-water mark (Peak Account Equity) stored in SQLite.
   - **2.0% Drawdown (Warning)**: Halts new entries, limits sizing, and sends alert.
   - **3.5% Drawdown (Kill Switch)**: Liquidates all open positions, cancels orders, halts daemon operations.


---

## Project Structure

```
alpacha/
├── plan.md                    # Implementation roadmap & specification
├── requirements.txt           # Python dependencies
├── .env.example               # Template for API credentials
├── config/
│   ├── settings.yaml          # Tunable strategy & model parameters
│   ├── logging.yaml           # Logging handler configuration
│   └── macro_calendar.json    # Preloaded economic calendar
├── alpacha/
│   ├── __init__.py
│   ├── bot.py                 # AlpachaBot master orchestrator
│   ├── config.py              # Typed dataclass config loader
│   ├── data/
│   │   ├── sqlite_manager.py  # SQLite persistence engine
│   │   ├── alpaca_data.py     # Alpaca SDK wrapper for market & account data
│   │   └── macro_calendar.py  # Macro calendar proximity filter
│   ├── model/
│   │   ├── volutils.py        # RV, BV, Jump variation & Expected Move math
│   │   ├── har.py             # Enhanced HAR regression model
│   │   └── trainer.py         # HAR model training, caching & forecasting
│   ├── risk/
│   │   └── risk_manager.py    # Equity high-water mark & drawdown ladder
│   ├── strategy/
│   │   ├── gates.py           # 3-tier trade entry filters
│   │   ├── ironcondor.py      # 0.20 delta IC builder with Expected Move wings
│   │   └── executor.py        # Order placement, fill polling & rollback
│   ├── alerts/
│   │   └── notifier.py        # Webhook & log alert dispatcher
│   └── utils/
│       ├── logger.py          # Logger setup
│       └── time_utils.py      # NYSE trading hours calendar
├── tests/                     # 16 unit & integration tests
└── main.py                    # CLI entry point
```

---

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Secrets

Copy `.env.example` to `.env` and set your Alpaca API credentials:

```bash
cp .env.example .env
```

---

## Usage

### Run Bot in Dry-Run Mode (Simulation)
```bash
python main.py --dry-run
```

### Run Single Scan Cycle
```bash
python main.py --single-run --dry-run
```

### Check System Risk & Portfolio Status
```bash
python main.py --status
```

### Start Production / Paper Daemon
```bash
python main.py
```

---

## Running Tests

Run all unit & integration tests:

```bash
python -m pytest -v
```
