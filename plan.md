# Alpaca Iron Condor Options Trading Bot — Implementation Plan

## TL;DR
An autonomous **LangGraph Agent State Machine** on the Alpaca platform (paper first, live later) interacting via **Alpaca MCP Tools & Direct CLI/REST**. It monitors account risk with a **3.5% drawdown kill switch**, fetches **SPY/QQQ 1-minute bars**, forecasts **1-day realized volatility (RV)** with an **enhanced HAR model** (leverage + jumps), gates every entry on **macro events / vol contango / IV ≥ 1.2×RV edge**, and places **Iron Condor limit orders** (short 0.20 delta, wings sized by expected move). All state persists to **SQLite**.

---

## 1. Scope & Goals

### In Scope
- **Agent Architecture**: LangGraph State Machine (`TradingAgentState`) coordinating modular agent nodes.
- **MCP & CLI Interface**: Alpaca Model Context Protocol (MCP) tool suite (`FastMCP`) & direct HTTP CLI engine (zero dependency on high-level broker SDKs).
- **Instruments**: SPY and QQQ options.
- **Strategy**: Iron Condor (IC) — four-leg defined-risk credit strategy.
- **Model**: Enhanced HAR volatility model (Corsi 2009 + leverage effect + jump component).
- **Risk**: Dynamic account equity peak tracking, drawdown warning ladder, and 3.5% hard kill switch.
- **Persistence**: SQLite (single-file, transactional, zero-config).

---

## 2. Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Agent Engine** | LangGraph State Machine (`StateGraph`) | Stateful, auditable multi-node agent pipeline designed for hackathon Agent frameworks |
| **Broker Interface** | Alpaca MCP Server + Direct REST CLI Driver | Fully compliant with MCP standards; zero proprietary broker SDK reliance in the strategy loop |
| **Language** | Python 3.11+ | Rich ecosystem for financial modeling, stats, and agent graphs |
| **Volatility Model** | Enhanced HAR (Corsi 2009 + leverage + jumps) via `statsmodels` OLS | Robust daily RV forecast incorporating asymmetric return shocks & jumps |
| **Wing Width Calculation** | Derived from Expected Move: `IV × √(DTE/365) × price` | Dynamic, model-consistent wing distance scaled to current market implied risk |
| **Storage Engine** | SQLite | Lightweight, single-file, transactional persistence for state and audit logs |
| **Vol Contango Gate** | Near vs. next SPY expiration ATM IV comparison | Ensures favorable options term structure before entering credit trades |
| **Macro Event Filtering** | Static JSON calendar (initial) with optional API fallback | Low overhead, reliable offline check for major market-moving events |
| **Risk Kill Switch** | 3.5% drawdown from peak equity → liquidate + halt | Enforces strictly capped portfolio risk |


---

## 3. Project Structure

```
alpacha/
├── plan.md
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.yaml          # All tunable parameters (no secrets)
│   ├── logging.yaml           # Structured logging configuration
│   └── loaders.py             # Dataclass config loader + env var overrides
├── alpacha/
│   ├── __init__.py
│   ├── bot.py                 # AlpachaBot master orchestrator (Phase 6)
│   ├── config.py              # Typed dataclass structures for application settings
│   ├── data/
│   │   ├── sqlite_manager.py  # SQLite storage manager for bars, forecasts, trades, risk
│   │   ├── alpaca_data.py     # Alpaca SDK wrapper for market data, account, positions, options chains
│   │   └── macro_calendar.py  # Macro economic calendar loader & filter
│   ├── model/
│   │   ├── volutils.py        # Volatility math: RV, bipower variation, jump metrics, expected move
│   │   ├── har.py             # Enhanced HAR regression model (daily, weekly, monthly, leverage, jumps)
│   │   └── trainer.py         # Trainer & model lifecycle manager (fit, save, load, retrain)
│   ├── risk/
│   │   └── risk_manager.py    # Equity peak tracking, drawdown ladder, sizing controls, kill switch
│   ├── strategy/
│   │   ├── gates.py           # Trade gate logic: Macro proximity, vol contango, IV ≥ 1.2× RV edge
│   │   ├── ironcondor.py      # Iron Condor 4-leg definition, strike & wing selector (0.20 delta)
│   │   └── executor.py        # 4-leg order builder, execution monitor, limit price poller
│   ├── alerts/
│   │   └── notifier.py        # Webhook / Email alerting & logging notifier
│   └── utils/
│       ├── time_utils.py      # Trading hours & calendar helpers (using pandas_market_calendar)
│       └── logger.py          # Centralized logger setup
├── data/                      # Local SQLite database files & raw cache (gitignored)
├── tests/
│   ├── test_volutils.py
│   ├── test_har.py
│   ├── test_risk.py
│   └── test_ironcondor.py
└── main.py                    # Main CLI entry point to launch daemon
```

---

## 4. Implementation Phases

### Phase 1: Project Scaffolding & Configuration
- Setup standard Python project structure and package layout.
- Create `requirements.txt` with core dependencies:
  - `alpaca-py`, `statsmodels`, `pandas`, `numpy`, `scipy`, `pandas-market-calendar`, `apscheduler`, `PyYAML`, `requests`, `pytest`, `python-dotenv`, `joblib`.
- Implement `config/settings.yaml` to store all non-sensitive configuration parameters.
- Implement typed configuration loader in `alpacha/config.py` with environment variable overrides for API keys (`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `APCA_API_BASE_URL`).
- Setup logging system in `config/logging.yaml` with file and console handlers.

### Phase 2: Data Layer *(Parallel with Phase 3)*
- **SQLite Storage Manager** (`sqlite_manager.py`):
  - Database table creation & migration schema.
  - CRUD helper methods for historical bars, model forecasts, executed trades, risk snapshots, and equity curves.
- **Alpaca Data Client** (`alpaca-data.py`):
  - Historical and live 1-minute bar fetching for SPY/QQQ.
  - Options market data queries: chains, snapshots, delta values, implied volatility.
  - Account state retrieval: total equity, buying power, active positions, open orders.

### Phase 3: Enhanced HAR Volatility Model *(Parallel with Phase 2)*
- **Volatility Utilities** (`volutils.py`):
  - Realized Volatility (RV) calculation from 1-minute log returns.
  - Bipower Variation (BV) calculation for jump-robust continuous volatility estimation.
  - Jump Detection: `Jump = max(0, RV - BV)` with Z-score significance testing.
  - Expected Move formula: `Expected Move = Price × IV × √(DTE / 365)`.
- **Enhanced HAR Model** (`har.py`):
  - Predict 1-day ahead log(RV) using log daily ($RV_d$), weekly ($RV_w$), and monthly ($RV_m$) components.
  - Incorporate **leverage effect**: $r_{d,-} = \min(0, r_d)$ (negative return shocks boost future volatility).
  - Incorporate **jump variation**: lagged jump component $J_d$.
  - Estimate parameters using ordinary least squares (`statsmodels.api.OLS`).
- **Model Trainer & Lifecycle** (`trainer.py`):
  - Fitting routine, persistence (`joblib`/`pickle`), and evaluation metrics (RMSE, QLIKE).
  - Automatic re-training trigger when models become stale (> N days) or error drifts beyond threshold.

### Phase 4: Risk Management *(Depends on Phase 2)*
- **Drawdown Ladder Logic** (`risk_manager.py`):
  - Monitor rolling high-water mark (Peak Account Equity) stored in SQLite.
  - Calculate Current Drawdown = `(Peak Equity - Current Equity) / Peak Equity`.
  - **Threshold 1 (2.0% Drawdown - WARNING)**:
    - Halt all new trade entries.
    - Reduce sizing for existing orders or tighten management stops.
    - Emit warning alert via notification module.
  - **Threshold 2 (3.5% Drawdown - KILL SWITCH)**:
    - Immediately liquidate all open positions.
    - Cancel all open orders.
    - Halt daemon operations for the remainder of the session.
    - Send critical alert.

### Phase 5: Strategy — Gating & Execution *(Depends on Phases 2, 3, 4)*
- **Macro Economic Calendar Gate** (`macro_calendar.py`):
  - Parse economic calendar JSON (FOMC interest rate decisions, CPI releases, NFP reports, GDP updates).
  - Block entries if scheduled high-impact events occur within **2 hours** of entry time.
- **Vol Contango Gate** (`gates.py`):
  - Fetch ATM implied volatility for nearest expiration ($IV_{near}$) and next expiration ($IV_{next}$).
  - Verify options term structure is in healthy contango ($IV_{near} \le IV_{next}$ or within acceptable ratio). Block entries during severe backwardation/volatility spike regimes.
- **IV / RV Edge Gate** (`gates.py`):
  - Compare market implied volatility against forecasted realized volatility.
  - Require edge condition: $IV_{market} \ge 1.2 \times RV_{forecasted}$.
- **Iron Condor Builder** (`ironcondor.py`):
  - Select short call and short put strikes target **0.20 delta**.
  - Calculate wing width using Expected Move distance: $WingWidth = \text{Price} \times IV \times \sqrt{DTE / 365}$.
  - Place long wings at $ShortStrike \pm WingWidth$.
  - Enforce minimum credit requirements and buying power limits.
- **Order Execution Manager** (`executor.py`):
  - Construct 4-leg limit orders (multi-leg structure or 4 individual leg limit orders).
  - Price polling & order modification loop to achieve target fill price within limit threshold.
  - Partial-fill fallback handling to prevent naked short options exposure.

### Phase 6: Main Daemon Loop *(Depends on Phases 4 & 5)*
- **AlpachaBot Orchestrator** (`bot.py`):
  - Integration of all subsystems into a unified class interface.
- **Scheduler Engine** (`APScheduler`):
  - Scheduled market-hours job execution (e.g., scan every 5 minutes during open market hours).
  - Periodic risk evaluation snapshots.
  - End-of-day summary generation.
- **Trading Hours Guard**:
  - Restrict scan/trade execution strictly to US equity market open hours using `pandas_market_calendar`.
- **Graceful Shutdown**:
  - Handle OS signals (`SIGINT`, `SIGTERM`) to flush SQLite transactions, cancel pending non-filled orders, and close broker client connections safely.

### Phase 7: Testing & Hardening
- **Unit Testing**:
  - Test harness for math routines in `volutils.py` and HAR regression in `har.py`.
  - Mocked tests for risk manager drawdown breaches and gate filters.
- **Dry-Run Mode**:
  - Execution mode where real-time market data is ingested and signals are generated, but orders are logged rather than sent to broker.
- **Integration & Paper Trading**:
  - End-to-end testing on Alpaca Paper Trading account.
  - Verification of full trade lifecycle (signal → gate check → order submission → fill monitoring → position tracking → exit).


---

## 5. Detailed Strategy Pipeline

```
[ Market Clock Trigger ]
          │
          ▼
┌──────────────────┐
│ Risk Manager Check│ ───► Peak Drawdown ≥ 3.5%? ──► [ KILL SWITCH: Liquidate & Halt ]
└─────────┬────────┘
          │ (Drawdown < 2.0%)
          ▼
┌──────────────────┐
│ Market Data Feed │ ───► Fetch SPY/QQQ 1-min bars, Account Equity & Options Chain
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ HAR Vol Model    │ ───► Calculate Daily/Weekly/Monthly RV + Leverage + Jumps
└─────────┬────────┘      Generate 1-day RV Forecast
          │
          ▼
┌──────────────────┐
│ Trade Entry Gates│
│ 1. Macro (<2h)?  │ ───► (FAIL) ──► Log Gate Rejection & Skip Entry
│ 2. Vol Contango? │
│ 3. IV ≥ 1.2x RV? │
└─────────┬────────┘
          │ (ALL GATES PASSED)
          ▼
┌──────────────────┐
│ Build Iron Condor│ ───► Find 0.20 Delta Short Strikes
└─────────┬────────┘      Set Long Wings at Short Strike ± Expected Move
          │
          ▼
┌──────────────────┐
│ Order Executor   │ ───► Place Limit Orders, Poll Fills & Record to SQLite
└──────────────────┘
```

---

## 6. SQLite Schema Design (v1)

- **`bars`**: Stores historical minute price data.
  - `symbol` (TEXT), `timestamp` (DATETIME), `open` (REAL), `high` (REAL), `low` (REAL), `close` (REAL), `volume` (INTEGER)
- **`forecasts`**: Stores HAR volatility model outputs.
  - `id` (INTEGER PRIMARY KEY), `timestamp` (DATETIME), `symbol` (TEXT), `forecasted_rv` (REAL), `metrics_json` (TEXT)
- **`trades`**: Tracks executed Iron Condor positions and trade life cycle.
  - `trade_id` (TEXT PRIMARY KEY), `symbol` (TEXT), `status` (TEXT), `entry_timestamp` (DATETIME), `exit_timestamp` (DATETIME), `legs_json` (TEXT), `credit_received` (REAL), `exit_pnl` (REAL), `gates_passed_json` (TEXT)
- **`risk_snapshots`**: Stores account risk history and equity peak markers.
  - `timestamp` (DATETIME), `equity` (REAL), `peak_equity` (REAL), `drawdown_pct` (REAL), `risk_level` (TEXT)

---

## 7. Verification & Success Criteria

1. **Unit Tests**: All test suites in `tests/` pass with zero failures.
2. **Paper Trading Validation**: Run daemon in dry-run/paper mode continuously without unhandled exceptions or connection errors.
3. **Risk Management Test**: Verify that simulated drawdown breaches trigger the correct warning and kill actions.
4. **Order Execution Verification**: Verify correct 4-leg Iron Condor generation and fill tracking in Alpaca paper account.

