# Alpaca Iron Condor Options Trading Agent — Implementation Plan (v2)

## TL;DR
An autonomous LangGraph agent on a light Azure VM, trading defined-risk Iron Condors on SPY and QQQ through the **official Alpaca MCP server** during the hackathon scoring window (Mon Aug 31 through equity mark at EOD Thu Sep 3). A deterministic core owns all money authority: drawdown ladder with a **3.5% kill switch**, per-trade and book-level caps, and a Black-Scholes **stress grid** veto. An **enhanced HAR-RV model** (leverage + jumps) forecasts realized vol; entries pass four gates (entry window, macro/earnings blackout, vol contango, IV at least 1.15x forecast RV). **Four Claude agents via OpenRouter** (regime analyst, proposer, news analyst, journalist) inform, choose within menus, veto, and narrate — all fail-open, never authoritative. Exits are mechanical: 50% profit take, 2.5x credit loss cut, full flatten Thursday 15:30 ET. State persists to SQLite, including LangGraph checkpoints.

Changes from v1 are marked **[v2]** with rationale.

---

## 1. Scope & Goals

### In Scope
- **Instruments**: SPY and QQQ options only
- **Strategy**: Iron Condor (4-leg defined-risk credit), short strikes ~0.20 delta
- **Execution**: **[v2] Official Alpaca MCP server (stdio subprocess)** — the hackathon rules require MCP or CLI; every quote, bar, chain, and order is an MCP tool call. alpaca-py is not used.
- **Orchestration**: **[v2] LangGraph state machine** invoked by APScheduler every 5 minutes during market hours; SQLite-checkpointed state
- **Agents**: **[v2] Four LLM agents via OpenRouter** with hard config clamps and fail-open wrappers
- **Model**: Enhanced HAR (Corsi 2009 + leverage + jump components), numpy OLS
- **Risk**: Peak + daily drawdown ladder, de-risk sizing ladder, 3.5% hard kill switch, per-trade caps, stress-grid book veto
- **Exits**: **[v2]** profit take at 50% of credit, loss cut at 2.5x credit, contest-end flatten Thu 15:30 ET, minimum 2 DTE at entry, never hold through expiry
- **Persistence**: SQLite (trades, forecasts, daily RV, risk snapshots, app state, memos) + LangGraph checkpoint db
- **Deployment**: **[v2]** light Azure VM, daemon with restart-on-failure, one-shot deploy script kept out of git

### Out of Scope (v1)
- Tickers beyond SPY/QQQ; strategies beyond Iron Condors
- Intraday delta rehedging; rolling/adjusting open positions on model updates (exits are price-rule driven only)
- Live (non-paper) trading — the contest is paper only

---

## 2. Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Broker interface | **[v2] Alpaca MCP server, stdio** | Hackathon requires MCP or CLI. Tool surface verified: get_option_chain, get_stock_bars, get_stock_snapshot, place_option_order (mleg), get_order_by_client_id, cancel_order_by_id, get_news, get_clock, get_account_info |
| Orchestration | **[v2] LangGraph StateGraph + APScheduler trigger** | Judged criterion is the agent workflow; graph gives conditional edges, checkpointing, replayable cycles |
| LLM provider | **[v2] OpenRouter (OpenAI-compatible endpoint)** | Team decision; one key, model configurable, langchain-openai client |
| Agent authority | **[v2] Inform / choose-from-menu / veto / narrate only** | Deterministic risk engine holds the only unconditional veto; all agent failures degrade to the deterministic path |
| Volatility model | Enhanced HAR (d/w/m log RV + leverage + jump), numpy lstsq | Academic standard, fits in ms; **[v2]** horizon = 2 days to match holding period; completed days only (never the in-progress day); statsmodels dropped for numpy to keep deps light |
| Model humility | **[v2] Error monitor + demotion** | Nightly forecast-vs-realized scoring; two consecutive days of realized > 1.5x forecast, or trailing QLIKE worse than the 20-day mean, demotes HAR to the fallback and widens the edge ratio |
| Edge gate | **[v2] IV >= 1.15x forecast RV (clamped 1.10–1.35)** | Friday's live reading was 1.152; a 1.2 floor risks zero trades at VIX 14. Regime agent may tune within clamps only |
| Contango gate | **[v2] Near ATM IV vs ~30d ATM IV** | Daily expirations make near-vs-next noisy; near-vs-30d is the stable regime read |
| Wing width | Expected move, capped | EM = price × IV × sqrt(DTE/365); **[v2]** width shrinks toward 5 if credit floor (12% of width) or per-trade cap ($1,000; target $500) would fail |
| Execution | **[v2] Atomic mleg limit only, time-boxed ladder** | Post at net-credit mid; two $0.02 concessions at ~40s intervals; never below credit floor; confirmed cancel at ~2 min; keep whole filled condors on partial qty; no re-chase within a cycle. Single-leg fallback is banned (legging risk; L3 rejects uncovered legs) |
| Risk kill switch | 3.5% below peak equity → cancel all, flatten, halt (sticky, human unhalt) | 1.5% buffer under the 5% mandate absorbs gap and exit slippage |
| Storage | SQLite single file + LangGraph SqliteSaver | Transactional, judge-readable; **[v2]** bars are fetched on demand (not stored); derived daily RV is stored |
| Config | **[v2] Typed dataclasses in config.py + .env for secrets** | YAML dropped for speed and type safety during the contest week |

---

## 3. Contest Calendar (hard-coded, ET) **[v2]**

- Scoring: Mon Aug 31 09:30 → snapshot of equity as of **EOD Thu Sep 3** (Fri Sep 4 09:30 snapshot time)
- Entry windows: Mon/Tue 09:45–15:30, Wed 09:45–12:00, Thu none (unwind only)
- Macro blackout (no entries within 120 min before): ISM Mfg + JOLTS Tue 10:00, ADP Wed 08:15, Claims Thu 08:30, ISM Services Thu 10:00 (tentative)
- Earnings: AVGO Wed Sep 2 after close — Wednesday entry cutoff exists for this reason
- Flatten everything by Thu 15:30; Friday NFP is after the mark and irrelevant if flat

---

## 4. Project Structure

```
HedgeMePlease/
├── plan.md
├── README.md                  # submission-facing; pipeline flowchart lives here
├── pyproject.toml             # uv-managed; hatchling build
├── .env.example               # ALPACA_API_KEY/SECRET, OPENROUTER_API_KEY, ALPACA_MCP_DIR
├── src/alpacha/
│   ├── config.py              # limits, strategy + executor params, stress cfg, calendar, clamps
│   ├── graph.py               # LangGraph StateGraph: nodes, conditional edges, checkpointing
│   ├── daemon.py              # APScheduler loop, RTH guard, graceful shutdown
│   ├── cli.py                 # status | rv | preview | scan | once | loop | flatten | panic | unhalt
│   ├── broker/
│   │   ├── mcp.py             # AlpacaMCP stdio client (envelope unwrap, chunked bars, chain paging)
│   │   └── executor.py        # mleg order ladder, confirmed cancel, close logic
│   ├── data/
│   │   └── db.py              # SQLite: trades, forecasts, rv_daily, risk_snapshots, app_state, memos
│   ├── model/
│   │   ├── volutils.py        # RV (completed days only), bipower variation, jumps, expected move
│   │   └── har.py             # enhanced HAR fit/forecast, fallback, walk-forward, error monitor
│   ├── risk/
│   │   ├── bs.py              # Black-Scholes + greeks (math.erf, no scipy)
│   │   ├── stress.py          # spot x vol shock grid over the whole book
│   │   ├── ledger.py          # positions + equity anchors + halt flag on SQLite
│   │   └── engine.py          # ladder, caps, sleeve budget, stress veto, size factor
│   ├── strategy/
│   │   ├── gates.py           # window, macro blackout, contango, IV/RV edge
│   │   └── condor.py          # chain parsing, 0.20 delta shorts, EM wings, credit floors, sizing
│   └── agents/
│       └── desk.py            # OpenRouter client, 4 agent roles, clamps, fail-open wrappers
└── tests/                     # every component: bs, stress, engine, volutils, har, gates,
                               # condor, executor pricing, db, agent parsing, full offline cycle
```

## 5. Implementation Phases

1. **Scaffolding**: branch, pyproject (langgraph, langgraph-checkpoint-sqlite, langchain-openai, mcp, apscheduler, numpy, dotenv, tzdata; pytest dev), .env.example, gitignore (env, venv, caches, state/, deploy.local.*)
2. **Deterministic core**: config, bs, stress, ledger(db), engine, volutils, har, gates, condor — with unit tests
3. **Broker layer**: AlpacaMCP client (known-good tool signatures), executor with the v2 ladder — pricing logic unit tested offline
4. **Agents**: OpenRouter client + 4 roles + clamps + tolerant JSON parsing — parsers unit tested; agents fail-open when key absent
5. **Graph + daemon + CLI**: LangGraph wiring with conditional edges and SqliteSaver; full offline cycle test against a fake broker with a synthetic chain
6. **Live smoke (needs keys)**: status, rv, preview, scan on paper account; one throwaway order accept/cancel; then go-live via `loop`
7. **Submission polish**: README (flowchart, evidence, risk table, disclosure), deploy script for Azure VM (untracked), runbook

## 6. Verification & Success Criteria

1. All unit tests pass; the offline full-cycle test walks trigger → journal with orders mocked.
2. Live smoke on paper: account/chain/forecast/preview verified; one mleg order accepted and cancel-confirmed.
3. Risk drills: simulated equity drops trigger no-new, reduce-only, and kill actions in tests.
4. Audit: every cycle writes gates, verdicts, orders, and agent notes to memos; trades reconcile against the broker.
