"""The LangGraph state machine: one invocation is one five minute cycle.

Deterministic nodes hold all money authority; agent nodes inform, choose,
veto, and narrate, each failing open to the deterministic path. Full Proposal
objects live in services.scratch (per cycle); graph state carries only
JSON-serializable summaries so checkpointing stays clean.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import time as dtime
from datetime import timedelta
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import desk
from .broker.executor import cost_to_close, submit_close, submit_open
from .config import FLATTEN_AT, SLEEVE_B, SLEEVE_B_EVENTS, STRAT, now_et
from .strategy.events import build_crush_condor
from .data.db import Db
from .model.har import best_forecast, should_demote
from .model.volutils import daily_stats
from .risk.engine import (
    AccountAction,
    budget_consumed_frac,
    check_pre_trade,
    evaluate_account,
)
from .risk.ledger import Ledger
from .strategy.condor import atm_iv, build_candidates, parse_chain, pick_far_expiry, pick_near_expiry
from .strategy.gates import evaluate_gates


@dataclass
class Services:
    broker: Any
    db: Db
    ledger: Ledger
    dry_run: bool = False
    scratch: dict = field(default_factory=dict)

    def memo(self, event: str, detail: dict) -> None:
        self.db.memo(event, detail)


class CycleState(TypedDict, total=False):
    now: str
    equity: float
    action: str
    market_open: bool
    spots: dict
    evidence: dict
    regime: dict
    gates: dict
    passing: list
    candidates: dict
    chosen: dict
    veto: dict
    verdict: dict
    executed: dict
    skip: str


async def _forecast_for(services: Services, symbol: str) -> tuple[Optional[float], str]:
    """Daily-cached RV forecast; refit on completed days only, demotion applied."""
    today = now_et().date().isoformat()
    cache_key = f"rv_forecast:{symbol}:{today}"
    cached = services.db.get_state(cache_key)
    if cached:
        return float(cached["value"]), str(cached["method"])
    bars = await services.broker.stock_bars_5min(symbol, days=130)
    stats = daily_stats(bars, now_et())
    if len(stats) > 1:
        stats = stats[1:]  # oldest chunk boundary day may be incomplete
    if len(stats) < 25:
        services.memo("rv_insufficient", {"symbol": symbol, "days": len(stats)})
        return None, "insufficient"
    services.db.upsert_rv_daily(symbol, stats)
    demoted = should_demote(services.db.forecast_vs_realized(symbol))
    value, method = best_forecast(stats, STRAT.forecast_horizon_days, demoted)
    if demoted:
        method = f"{method}(demoted)"
    services.db.set_state(cache_key, {"value": value, "method": method})
    services.db.record_forecast(symbol, STRAT.forecast_horizon_days, value, method)
    services.memo("rv_forecast", {"symbol": symbol, "forecast": round(value, 4), "method": method})
    return value, method


def build_graph(services: Services, checkpointer=None):
    ledger = services.ledger
    memo = services.memo

    async def risk_check(state: CycleState) -> CycleState:
        services.scratch.clear()
        now = now_et()
        clock = await services.broker.clock()
        equity = await services.broker.equity()
        ledger.update_equity(equity)
        action = evaluate_account(equity, ledger.hwm, ledger.day_anchor, ledger.halted)
        peak = max(ledger.hwm, equity)
        drawdown = (peak - equity) / peak if peak else 0.0
        services.db.record_risk_snapshot(equity, peak, drawdown, action.value)
        memo("cycle_start", {
            "equity": equity, "action": action.value,
            "drawdown": round(drawdown, 4),
            "budget_used": round(budget_consumed_frac(equity, ledger.hwm), 2),
        })
        return {
            "now": now.isoformat(),
            "equity": equity,
            "action": action.value,
            "market_open": bool(clock.get("is_open")),
        }

    async def flatten(state: CycleState) -> CycleState:
        if not ledger.halted:
            ledger.halt(f"kill switch at equity {state.get('equity', 0):,.0f}")
            memo("KILL_SWITCH", {"equity": state.get("equity")})
        for order in await services.broker.open_orders():
            oid = order.get("id")
            if oid:
                try:
                    await services.broker.cancel_order(oid)
                except Exception:
                    pass

        async def _close(pos):
            await submit_close(services.broker, memo, pos, "kill_switch")
            ledger.update(pos)

        open_pos = ledger.open_positions()
        if open_pos:  # concurrent: an emergency flatten must not queue behind ladders
            await asyncio.gather(*[_close(p) for p in open_pos])
        return {"skip": "kill switch"}

    async def manage(state: CycleState) -> CycleState:
        now = now_et()
        broker_syms = {p.get("symbol") for p in await services.broker.positions()}
        ledger_syms = {leg.symbol for pos in ledger.open_positions() for leg in pos.legs}
        unknown = broker_syms - ledger_syms - {None}
        if unknown:
            memo("reconcile_unknown_positions", {"symbols": sorted(unknown)})

        for pos in ledger.open_positions():
            if pos.status == "pending":
                continue
            if now >= FLATTEN_AT:
                await submit_close(services.broker, memo, pos, "contest_end_flatten")
                ledger.update(pos)
                continue
            # GLD/TLT can land on expiries before the contest flatten (no
            # Thursday listings); nothing is ever held through its own expiry
            expires_today = any(leg.expiry == now.date().isoformat() for leg in pos.legs)
            if expires_today and now.time() >= dtime(15, 15):
                await submit_close(services.broker, memo, pos, "expiry_day_close")
                ledger.update(pos)
                continue
            if pos.sleeve == "B":
                event = next((e for e in SLEEVE_B_EVENTS if e.symbol == pos.underlying), None)
                if event and now >= event.crush_exit_by:
                    await submit_close(services.broker, memo, pos, "event_crush_exit")
                    ledger.update(pos)
                    continue
            cost = await cost_to_close(services.broker, pos)
            if cost is None:
                continue
            ledger.mark_position(
                pos.position_id, round(cost, 2),
                round((pos.credit - cost) * 100 * pos.qty, 2),
            )
            if cost <= pos.credit * (1 - STRAT.profit_take_frac):
                await submit_close(services.broker, memo, pos, "profit_target_50pct")
                ledger.update(pos)
            elif cost >= pos.credit * STRAT.loss_close_mult:
                await submit_close(services.broker, memo, pos, "loss_limit")
                ledger.update(pos)
        return {}

    async def sleeve_b(state: CycleState) -> CycleState:
        """Event scheduler: sells the crush condor inside its entry window.
        Ignores Sleeve A's windows and cooldowns by design; obeys the account
        action, the risk engine, the news veto, and its own budget."""
        now = now_et()
        if not state.get("market_open") or state.get("action") != AccountAction.OK.value:
            return {}
        for event in SLEEVE_B_EVENTS:
            if not (event.crush_entry_start <= now <= event.crush_entry_end):
                continue
            status_key = f"sleeveB:{event.symbol}:crush"
            if services.db.get_state(status_key):
                continue
            # spots for the whole open book too: the stress grid inside the
            # risk check must see every existing position, not just this event
            needed = {event.symbol} | {p.underlying for p in ledger.open_positions()}
            all_spots = await services.broker.spots(sorted(needed))
            spot = all_spots.get(event.symbol, 0.0)
            payload = await services.broker.option_chain(
                event.symbol,
                expiration_date_gte=event.event_time.date().isoformat(),
                expiration_date_lte=event.post_expiry,
                strike_price_gte=spot * 0.75 if spot else None,
                strike_price_lte=spot * 1.25 if spot else None,
            )
            contracts = parse_chain(event.symbol, payload)
            proposal, diag = build_crush_condor(event, contracts, spot, now)
            memo("event_crush_candidate", diag)
            if proposal is None:
                continue
            verdict = check_pre_trade(
                ledger.open_positions(), proposal, state.get("equity", 0.0),
                ledger.hwm, ledger.day_anchor, ledger.halted, all_spots, now,
            )
            memo("event_risk_verdict", {
                "symbol": event.symbol, "approved": verdict.approved,
                "reasons": verdict.reasons, "position_id": proposal.position.position_id,
            })
            if not verdict.approved:
                continue
            try:
                headlines = await services.broker.news(event.symbol)
            except Exception:
                headlines = []
            vetoed, reason = await desk.news_veto(
                event.symbol, proposal.summary(), diag, headlines, memo
            )
            memo("event_news_veto", {"symbol": event.symbol, "veto": vetoed, "reason": reason})
            if vetoed:
                continue
            if services.dry_run:
                memo("event_dry_run_would_open", proposal.summary())
                services.db.set_state(status_key, "dry_run")
                continue
            opened = await submit_open(services.broker, memo, proposal.position)
            if opened:
                ledger.add(proposal.position, {"event": event.symbol, **diag})
                services.db.set_state(status_key, "opened")
        return {}

    async def decide_entry(state: CycleState) -> CycleState:
        now = now_et()
        if not state.get("market_open"):
            return {"skip": "market closed"}
        if state.get("action") != AccountAction.OK.value:
            return {"skip": f"account {state.get('action')}"}
        if now >= FLATTEN_AT:
            return {"skip": "past contest flatten time"}
        entries_today = services.db.conn.execute(
            "SELECT COUNT(*) AS n FROM trades WHERE substr(opened_at, 1, 10) = ? "
            "AND status != 'abandoned'",
            (now.date().isoformat(),),
        ).fetchone()["n"]
        if entries_today >= STRAT.max_entries_per_day:
            return {"skip": f"daily entry cap {STRAT.max_entries_per_day} reached"}
        last_entry = services.db.get_state("last_entry_at")
        if last_entry:
            from datetime import datetime as _dt

            elapsed_min = (now - _dt.fromisoformat(last_entry)).total_seconds() / 60.0
            if elapsed_min < STRAT.global_entry_spacing_min:
                remaining = STRAT.global_entry_spacing_min - elapsed_min
                return {"skip": f"entry spacing, {remaining:.0f}m remaining"}
        return {}

    async def gather(state: CycleState) -> CycleState:
        now = now_et()
        spots = await services.broker.spots(list(STRAT.underlyings))
        evidence: dict[str, dict] = {}
        contracts_store: dict[str, list] = {}
        for und in STRAT.underlyings:
            spot = spots.get(und, 0.0)
            if spot <= 0:
                continue
            near_payload = await services.broker.option_chain(
                und,
                expiration_date_lte=(now + timedelta(days=STRAT.near_dte_max)).date().isoformat(),
                strike_price_gte=spot * 0.88,
                strike_price_lte=spot * 1.12,
            )
            contracts = parse_chain(und, near_payload)
            contracts_store[und] = contracts
            near_expiry = pick_near_expiry(contracts, now)
            near_iv = atm_iv(contracts, near_expiry, spot) if near_expiry else None

            far_payload = await services.broker.option_chain(
                und,
                expiration_date_gte=(now + timedelta(days=STRAT.far_dte_min)).date().isoformat(),
                expiration_date_lte=(now + timedelta(days=STRAT.far_dte_max)).date().isoformat(),
                strike_price_gte=spot * 0.96,
                strike_price_lte=spot * 1.04,
            )
            far_contracts = parse_chain(und, far_payload)
            far_expiry = pick_far_expiry(far_contracts, now)
            far_iv = atm_iv(far_contracts, far_expiry, spot) if far_expiry else None

            rv, method = await _forecast_for(services, und)
            evidence[und] = {
                "near_expiry": near_expiry,
                "near_iv": near_iv,
                "far_iv": far_iv,
                "rv_forecast": rv,
                "rv_method": method,
            }
        services.scratch["contracts"] = contracts_store
        return {"spots": spots, "evidence": evidence}

    async def regime(state: CycleState) -> CycleState:
        today = now_et().date().isoformat()
        cached = services.db.get_state(f"regime:{today}")
        if cached:
            return {"regime": cached}
        view = await desk.regime_view(
            {"evidence": state.get("evidence", {}), "date": today}, memo
        )
        payload = asdict(view)
        has_evidence = any(
            (ev or {}).get("near_iv") and (ev or {}).get("rv_forecast")
            for ev in (state.get("evidence") or {}).values()
        )
        # cache only a real agent answer over real evidence; a transient LLM
        # failure or a thin premarket read must not lock defaults in all day
        if payload.get("source") == "llm" and has_evidence:
            services.db.set_state(f"regime:{today}", payload)
        memo("regime_view", payload)
        return {"regime": payload}

    async def gate_check(state: CycleState) -> CycleState:
        now = now_et()
        regime_view = state.get("regime", {})
        edge_ratio = float(regime_view.get("edge_ratio", STRAT.iv_over_rv_min_ratio))
        gates_out: dict[str, dict] = {}
        passing: list[str] = []
        for und, ev in (state.get("evidence") or {}).items():
            report = evaluate_gates(ev.get("near_iv"), ev.get("far_iv"), ev.get("rv_forecast"), now, edge_ratio)
            standdown = regime_view.get("stance") == "standdown"
            extra_fails = ["regime_standdown"] if standdown else []
            last_und_entry = services.db.get_state(f"last_entry_at:{und}")
            if last_und_entry:
                from datetime import datetime as _dt

                elapsed = (now - _dt.fromisoformat(last_und_entry)).total_seconds() / 60.0
                if elapsed < STRAT.same_underlying_cooldown_min:
                    extra_fails.append(f"underlying_cooldown_{STRAT.same_underlying_cooldown_min - elapsed:.0f}m")
            ok = report.all_pass and not extra_fails
            gates_out[und] = {
                "pass": ok,
                "failed": report.failed() + extra_fails,
                **report.details,
            }
            memo("gates", {"underlying": und, **gates_out[und]})
            if ok:
                passing.append(und)
        return {"gates": gates_out, "passing": passing}

    async def build(state: CycleState) -> CycleState:
        now = now_et()
        regime_view = state.get("regime", {})
        delta_target = float(regime_view.get("delta_target", STRAT.short_delta_target))
        contracts_store = services.scratch.get("contracts", {})
        summaries: dict[str, list] = {}
        proposals: dict[str, list] = {}
        for und in state.get("passing", []):
            cands, diag = build_candidates(
                und, contracts_store.get(und, []), state["spots"].get(und, 0.0), now, delta_target
            )
            memo("candidates", diag)
            if cands:
                proposals[und] = cands
                summaries[und] = [c.summary() for c in cands]
        services.scratch["proposals"] = proposals
        return {"candidates": summaries}

    async def propose(state: CycleState) -> CycleState:
        proposals = services.scratch.get("proposals", {})
        regime_note = str((state.get("regime") or {}).get("note", ""))
        for und in state.get("passing", []):
            cands = proposals.get(und)
            if not cands:
                continue
            idx, why = await desk.choose_candidate([c.summary() for c in cands], regime_note, memo)
            chosen = cands[idx]
            services.scratch["target"] = chosen
            services.scratch["proposer_why"] = why
            memo("proposal_chosen", {
                "underlying": und, "index": idx, "why": why,
                "position_id": chosen.position.position_id, **chosen.summary(),
            })
            return {"chosen": {**chosen.summary(), "why": why}}
        return {"skip": "no viable candidates"}

    async def veto(state: CycleState) -> CycleState:
        target = services.scratch.get("target")
        if target is None:
            return {"skip": "no target"}
        try:
            headlines = await services.broker.news(target.underlying)
        except Exception:
            headlines = []
        vetoed, reason = await desk.news_veto(
            target.underlying, target.summary(),
            (state.get("gates") or {}).get(target.underlying, {}), headlines, memo,
        )
        memo("news_veto", {
            "underlying": target.underlying, "veto": vetoed, "reason": reason,
            "position_id": target.position.position_id,
        })
        if vetoed:
            return {"veto": {"veto": True, "reason": reason}, "skip": "news veto"}
        return {"veto": {"veto": False, "reason": reason}}

    async def risk_gate(state: CycleState) -> CycleState:
        target = services.scratch.get("target")
        if target is None:
            return {"skip": "no target"}
        verdict = check_pre_trade(
            ledger.open_positions(), target, state.get("equity", 0.0),
            ledger.hwm, ledger.day_anchor, ledger.halted, state.get("spots", {}),
        )
        regime_size = float((state.get("regime") or {}).get("size_factor", 1.0))
        result = {
            "approved": verdict.approved,
            "reasons": verdict.reasons,
            "size_factor": verdict.size_factor,
            "regime_size_factor": regime_size,
        }
        memo("risk_verdict", {
            "underlying": target.underlying,
            "position_id": target.position.position_id,
            **result,
        })
        if not verdict.approved:
            return {"verdict": result, "skip": "risk engine rejected"}
        eff = verdict.size_factor * regime_size
        if eff < 0.5:
            memo("entry_skip", {"underlying": target.underlying,
                               "reason": f"combined size factor {eff:.2f} below 0.5"})
            return {"verdict": result, "skip": "size factor too small"}
        if eff < 1.0:
            # reduction floors at one contract: you cannot trade half a condor
            old_qty = target.qty
            new_qty = max(1, int(target.qty * eff))
            unit_loss = (target.width - target.credit) * 100.0
            target.qty = new_qty
            target.position.qty = new_qty
            target.max_loss = round(unit_loss * new_qty, 2)
            target.position.max_loss = target.max_loss
            scale = new_qty / old_qty
            target.net_delta_dollars *= scale
            target.net_vega_dollars *= scale
        return {"verdict": result}

    async def execute(state: CycleState) -> CycleState:
        target = services.scratch.get("target")
        if target is None:
            return {"skip": "no target"}
        entry_context = {
            "evidence": (state.get("evidence") or {}).get(target.underlying),
            "gates": (state.get("gates") or {}).get(target.underlying),
            "regime": state.get("regime"),
            "proposer_why": services.scratch.get("proposer_why"),
            "veto": state.get("veto"),
            "verdict": state.get("verdict"),
        }
        if services.dry_run:
            memo("dry_run_would_open", target.summary())
            return {"executed": {"dry_run": True, **target.summary()}}
        opened = await submit_open(services.broker, memo, target.position)
        if opened:
            ledger.add(target.position, entry_context)
            stamp = now_et().isoformat()
            services.db.set_state("last_entry_at", stamp)
            services.db.set_state(f"last_entry_at:{target.underlying}", stamp)
        return {"executed": {"opened": opened, **target.summary()}}

    async def journal(state: CycleState) -> CycleState:
        executed = state.get("executed") or {}
        if executed.get("opened") is True:
            outcome = "POSITION OPENED"
        elif executed.get("dry_run"):
            outcome = "DRY RUN ONLY, no order was sent"
        elif executed:
            outcome = "ORDER ATTEMPTED BUT NEVER FILLED, no position exists, no premium collected"
        elif state.get("skip"):
            outcome = f"NO ENTRY: {state.get('skip')}"
        else:
            outcome = "NO ENTRY ATTEMPTED"
        record = {
            "outcome": outcome,
            "action": state.get("action"),
            "skip": state.get("skip"),
            "gates": state.get("gates"),
            "chosen": state.get("chosen"),
            "verdict": state.get("verdict"),
            "executed": state.get("executed"),
            "open_positions": len(ledger.open_positions()),
        }
        note = await desk.journal_note(record, memo)
        if note:
            memo("cycle_note", {"note": note})
        memo("cycle_end", {"open_positions": record["open_positions"], "skip": state.get("skip")})
        return {}

    g = StateGraph(CycleState)
    g.add_node("risk_check", risk_check)
    g.add_node("flatten", flatten)
    g.add_node("manage", manage)
    g.add_node("sleeve_b", sleeve_b)
    g.add_node("decide_entry", decide_entry)
    g.add_node("gather", gather)
    g.add_node("regime", regime)
    g.add_node("gate_check", gate_check)
    g.add_node("build", build)
    g.add_node("propose", propose)
    g.add_node("veto", veto)
    g.add_node("risk_gate", risk_gate)
    g.add_node("execute", execute)
    g.add_node("journal", journal)

    g.add_edge(START, "risk_check")
    g.add_conditional_edges(
        "risk_check",
        lambda s: "flatten" if s.get("action") == AccountAction.KILL.value else "manage",
        {"flatten": "flatten", "manage": "manage"},
    )
    g.add_edge("flatten", "journal")
    g.add_edge("manage", "sleeve_b")
    g.add_edge("sleeve_b", "decide_entry")
    g.add_conditional_edges(
        "decide_entry",
        lambda s: "journal" if s.get("skip") else "gather",
        {"journal": "journal", "gather": "gather"},
    )
    g.add_edge("gather", "regime")
    g.add_edge("regime", "gate_check")
    g.add_conditional_edges(
        "gate_check",
        lambda s: "build" if s.get("passing") else "journal",
        {"build": "build", "journal": "journal"},
    )
    g.add_conditional_edges(
        "build",
        lambda s: "propose" if s.get("candidates") else "journal",
        {"propose": "propose", "journal": "journal"},
    )
    g.add_conditional_edges(
        "propose",
        lambda s: "journal" if s.get("skip") else "veto",
        {"journal": "journal", "veto": "veto"},
    )
    g.add_conditional_edges(
        "veto",
        lambda s: "journal" if s.get("skip") else "risk_gate",
        {"journal": "journal", "risk_gate": "risk_gate"},
    )
    g.add_conditional_edges(
        "risk_gate",
        lambda s: "execute" if not s.get("skip") else "journal",
        {"execute": "execute", "journal": "journal"},
    )
    g.add_edge("execute", "journal")
    g.add_edge("journal", END)

    return g.compile(checkpointer=checkpointer)


async def acquire_checkpointer(memo=None):
    """AsyncSqliteSaver for cycle checkpoints, or (None, None) with a logged reason."""
    from .config import CHECKPOINT_DB

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        cm = AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB))
        saver = await cm.__aenter__()
        return cm, saver
    except Exception as exc:
        if memo is not None:
            memo("checkpointer_unavailable", {"error": repr(exc)[:200]})
        return None, None


async def run_cycle(services: Services, graph=None, thread_suffix: str = "") -> dict:
    graph = graph or build_graph(services)
    thread_id = f"cycle-{now_et().strftime('%Y%m%d-%H%M%S')}{thread_suffix}"
    return await graph.ainvoke({}, config={"configurable": {"thread_id": thread_id}})
