"""The agent desk: four LLM roles over OpenRouter, all strictly fail-open.

Regime analyst  — reads the morning evidence, writes the day view, proposes
                  soft-parameter tunings that are clamped by config.CLAMPS.
Proposer        — picks one candidate from the deterministic builder's menu.
News analyst    — vetoes an approved entry over concrete catalysts only.
Journalist      — narrates the cycle for the audit trail.

No key, an API error, a timeout, or an unparseable reply all degrade to the
deterministic defaults. Agents can reduce risk or add context, never block
trading infrastructure or breach a limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from ..config import CLAMPS, STRAT, openrouter_config

_llm = None
_llm_checked = False


def get_llm():
    """Lazy ChatOpenAI pointed at OpenRouter; None when no key is configured."""
    global _llm, _llm_checked
    if _llm_checked:
        return _llm
    _llm_checked = True
    key, model = openrouter_config()
    if not key:
        return None
    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        model=model,
        temperature=0.2,
        max_tokens=500,
        timeout=25,
        max_retries=1,
    )
    return _llm


def extract_json(raw: str) -> Optional[dict]:
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def clamp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


@dataclass
class RegimeView:
    stance: str = "normal"        # normal | cautious | standdown
    edge_ratio: float = STRAT.iv_over_rv_min_ratio
    delta_target: float = STRAT.short_delta_target
    size_factor: float = 1.0
    note: str = "deterministic defaults (agent unavailable)"
    source: str = "default"       # "llm" only when the agent actually answered


def parse_regime(raw: str) -> Optional[RegimeView]:
    obj = extract_json(raw)
    if obj is None:
        return None
    stance = str(obj.get("stance", "normal")).lower()
    if stance not in ("normal", "cautious", "standdown"):
        stance = "normal"
    try:
        return RegimeView(
            stance=stance,
            edge_ratio=clamp(float(obj.get("edge_ratio", STRAT.iv_over_rv_min_ratio)), CLAMPS.edge_ratio),
            delta_target=clamp(float(obj.get("delta_target", STRAT.short_delta_target)), CLAMPS.delta_target),
            size_factor=clamp(float(obj.get("size_factor", 1.0)), CLAMPS.size_factor),
            note=str(obj.get("note", ""))[:400],
            source="llm",
        )
    except (TypeError, ValueError):
        return None


def parse_choice(raw: str, n_candidates: int) -> int:
    obj = extract_json(raw)
    if obj is None:
        return 0
    try:
        idx = int(obj.get("choice", 0))
    except (TypeError, ValueError):
        return 0
    return idx if 0 <= idx < n_candidates else 0


def parse_veto(raw: str) -> Optional[dict]:
    obj = extract_json(raw)
    if obj is None or not isinstance(obj.get("veto"), bool):
        return None
    return {"veto": obj["veto"], "reason": str(obj.get("reason", ""))[:400]}


async def _ask(system: str, user: str, memo=None, role: str = "") -> Optional[str]:
    llm = get_llm()
    if llm is None:
        return None
    from langchain_core.messages import HumanMessage, SystemMessage

    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    raw = str(content)
    if memo is not None:
        memo("llm_call", {"role": role, "input": user[:2000], "response": raw[:2000]})
    return raw


REGIME_SYSTEM = (
    "You are the regime analyst on an automated options desk that sells defined-risk "
    "iron condors on SPY and QQQ over one to three day holds. Given this morning's "
    "evidence, write the day view and tune three soft parameters. You may only move "
    "them inside these bounds, anything outside is clamped: edge_ratio in "
    f"{CLAMPS.edge_ratio}, delta_target in {CLAMPS.delta_target}, size_factor in "
    f"{CLAMPS.size_factor}. Raise edge_ratio and cut size when vol regime looks "
    "fragile; stance standdown means recommend no entries today. Reply with a single "
    'JSON object only: {"stance": "normal|cautious|standdown", "edge_ratio": x, '
    '"delta_target": x, "size_factor": x, "note": "two sentences"}.'
)


async def regime_view(evidence: dict[str, Any], memo) -> RegimeView:
    try:
        raw = await _ask(REGIME_SYSTEM, json.dumps(evidence, default=str), memo, "regime_analyst")
        if raw is None:
            return RegimeView()
        parsed = parse_regime(raw)
        if parsed is None:
            memo("regime_unparseable", {"raw": (raw or "")[:300]})
            return RegimeView()
        return parsed
    except Exception as exc:
        memo("regime_error", {"error": repr(exc)[:300]})
        return RegimeView()


PROPOSER_SYSTEM = (
    "You choose one iron condor from a menu the deterministic builder already "
    "validated. Prefer better credit per unit of width unless the regime note argues "
    "for wider wings. Reply with a single JSON object only: "
    '{"choice": <index>, "why": "one sentence"}.'
)


async def choose_candidate(
    candidates: list[dict], regime_note: str, memo
) -> tuple[int, str]:
    """Returns (index, stated reason). Defaults to the best credit-per-width."""
    default_why = "default: best credit per unit of width"
    if len(candidates) <= 1:
        return 0, default_why
    try:
        raw = await _ask(
            PROPOSER_SYSTEM,
            json.dumps({"regime_note": regime_note, "menu": candidates}, default=str),
            memo, "proposer",
        )
        if raw is None:
            return 0, default_why
        idx = parse_choice(raw, len(candidates))
        obj = extract_json(raw) or {}
        why = str(obj.get("why", ""))[:300] or default_why
        return idx, why
    except Exception as exc:
        memo("proposer_error", {"error": repr(exc)[:300]})
        return 0, default_why


VETO_SYSTEM = (
    "You are the news analyst inside an automated options desk selling defined-risk "
    "premium on index ETFs with one to three day holds. A deterministic risk engine "
    "already approved a trade; your ONLY job is to veto it if recent headlines show a "
    "concrete, dated catalyst likely to cause an outsized move before exit: an "
    "unscheduled central bank action, a major geopolitical escalation, an unexpected "
    "mega-cap event inside the window, a credit event. Routine commentary and already "
    "scheduled, already known events are NOT veto reasons. Be conservative with "
    'vetoes. Reply with a single JSON object only: {"veto": true|false, "reason": "..."}.'
)


async def news_veto(
    underlying: str, proposal_summary: dict, gates: dict, headlines: list[dict], memo
) -> tuple[bool, str]:
    try:
        lines = []
        for article in headlines[:12]:
            headline = str(article.get("headline") or "")[:200]
            created = str(article.get("created_at") or "")[:16]
            if headline:
                lines.append(f"- [{created}] {headline}")
        user = json.dumps(
            {
                "underlying": underlying,
                "proposal": proposal_summary,
                "gates": gates,
                "headlines": lines or ["(none returned)"],
            },
            default=str,
        )
        raw = await _ask(VETO_SYSTEM, user, memo, "news_analyst")
        if raw is None:
            return False, "agent unavailable"
        parsed = parse_veto(raw)
        if parsed is None:
            memo("veto_unparseable", {"underlying": underlying, "raw": (raw or "")[:300]})
            return False, "unparseable"
        return parsed["veto"], parsed["reason"]
    except Exception as exc:
        memo("veto_error", {"underlying": underlying, "error": repr(exc)[:300]})
        return False, "agent error"


EVENT_SYSTEM = (
    "You are the event analyst on an automated options desk. For one scheduled, "
    "date-verified earnings event, deterministic code has computed which phases are "
    "still viable by the clock: a run-up long strangle (bought before the print, "
    "sold before the release, harvesting the documented pre-earnings IV drift) and "
    "an IV-crush condor (sold in the final minutes before the release, covered next "
    "morning, harvesting the implied move overshoot). You decide whether to actually "
    "take each viable phase given the context: time remaining, implied move and IV "
    "level, quote quality, and headlines. You may DECLINE a viable phase; you can "
    "never enable one the clock has closed. Decline the run-up when too little "
    "drift window remains to matter; decline the crush when headlines suggest the "
    "event is postponed or unusual. Reply with a single JSON object only: "
    '{"trade_runup": true|false, "trade_crush": true|false, "note": "one sentence"}.'
)


def parse_event_view(raw: str) -> Optional[dict]:
    obj = extract_json(raw)
    if obj is None:
        return None
    if not isinstance(obj.get("trade_runup"), bool) or not isinstance(obj.get("trade_crush"), bool):
        return None
    return {
        "trade_runup": obj["trade_runup"],
        "trade_crush": obj["trade_crush"],
        "note": str(obj.get("note", ""))[:300],
    }


async def event_phase_view(context: dict[str, Any], memo) -> dict:
    """Fail-open: agent unavailable or unparseable means take viable phases."""
    default = {"trade_runup": True, "trade_crush": True, "note": "agent unavailable, defaults"}
    try:
        raw = await _ask(EVENT_SYSTEM, json.dumps(context, default=str), memo, "event_analyst")
        if raw is None:
            return default
        parsed = parse_event_view(raw)
        if parsed is None:
            memo("event_view_unparseable", {"raw": (raw or "")[:300]})
            return default
        return parsed
    except Exception as exc:
        memo("event_view_error", {"error": repr(exc)[:300]})
        return default


HEDGE_SYSTEM = (
    "You are the hedge analyst on an automated options desk whose book sells "
    "defined-risk premium (short volatility). You control one small insurance "
    "purchase: far out-of-the-money SPY puts within a fixed tiny budget, bought at "
    "most once. The reasoning you apply: the book profits from calm and takes its "
    "capped worst case in a gap, so insurance is worth owning when meaningful risk "
    "is deployed, especially ahead of known event nights (earnings after the close, "
    "major macro data), and it is cheapest when implied volatility is low, so "
    "waiting for stress to buy it defeats the purpose. Decline for now when the "
    "book is nearly empty and no event night is imminent, since the premium would "
    "bleed with little to protect. You decide WHEN, never whether the budget or "
    "structure changes; code enforces a backstop purchase before the largest event "
    "regardless. Reply with a single JSON object only: "
    '{"buy_now": true|false, "note": "one sentence of reasoning"}.'
)


def parse_hedge_view(raw: str) -> Optional[dict]:
    obj = extract_json(raw)
    if obj is None or not isinstance(obj.get("buy_now"), bool):
        return None
    return {"buy_now": obj["buy_now"], "note": str(obj.get("note", ""))[:300]}


async def hedge_view(context: dict[str, Any], memo) -> dict:
    """Fail-open to NOT buying: silence must not spend money. The code-level
    backstop guarantees protection exists before the big event night anyway."""
    default = {"buy_now": False, "note": "agent unavailable; deferring to backstop"}
    try:
        raw = await _ask(HEDGE_SYSTEM, json.dumps(context, default=str), memo, "hedge_analyst")
        if raw is None:
            return default
        parsed = parse_hedge_view(raw)
        if parsed is None:
            memo("hedge_view_unparseable", {"raw": (raw or "")[:300]})
            return default
        return parsed
    except Exception as exc:
        memo("hedge_view_error", {"error": repr(exc)[:300]})
        return default


JOURNALIST_SYSTEM = (
    "You write the audit note for one cycle of an automated options desk. Given the "
    "structured cycle record, write two or three plain sentences a judge could read: "
    "what happened and why, citing the numbers that mattered. No hype. The record's "
    "'outcome' field is the ground truth of what actually happened; your first "
    "sentence must state it faithfully. Never say premium was collected, a trade was "
    "made, or a position was taken unless outcome says POSITION OPENED. An approved "
    "or attempted trade that did not fill must be described as unfilled, with no "
    "money moved."
)


async def journal_note(cycle_record: dict[str, Any], memo) -> Optional[str]:
    try:
        raw = await _ask(JOURNALIST_SYSTEM, json.dumps(cycle_record, default=str), memo, "journalist")
        if raw is None:
            return None
        note = raw.strip()
        return note[:800] if note else None
    except Exception as exc:
        memo("journalist_error", {"error": repr(exc)[:300]})
        return None
