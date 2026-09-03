# Code Review Findings and Fixes — Aug 31, 2026

Team code review conducted on contest morning, before the first position existed.
All six findings were verified against the code and confirmed real. Chasing them
surfaced two additional bugs. Everything below was fixed, regression-tested, and
deployed to the production VM at 9:32 ET, thirteen minutes before the first
possible entry. Fix commit: 64c484e.

## 1. Critical — Zero-bid quotes blocked all position exits

`net_mid` in broker/executor.py refused any leg quoted with a zero bid. Far OTM
wings of a winning condor near expiry routinely quote 0.00 bid over a 0.01 ask,
so exactly when a position reached its profit target the pricing function
returned None and management skipped it. Review scoped this to profit-taking;
verification showed it was worse: the same path feeds submit_close, so the loss
cut, the kill-switch flatten, and the mandatory Thursday contest-end flatten
were all blocked in the winning scenario.

Fix: a zero bid under an ask of at most 0.10 is a valid quote (mid = ask/2);
a zero bid under a fatter ask is still refused as stale. Tests:
test_net_mid_tolerates_zero_bid_wings, test_net_mid_refuses_fat_ask_over_zero_bid.

## 2. High — Double-fill risk after an unconfirmed cancel

If an order ladder attempt timed out and the cancel could not be confirmed
terminal, the loop proceeded to post the next price while the old order might
still be live at the exchange: two fills, double the intended size. Adjacent
gap found during verification: reconciliation compares symbol sets, and a
double fill has identical symbols, so it would have been invisible afterward.

Fix: both the open and close ladders abort entirely when a cancel is
unconfirmed; nothing is ever requoted while a prior order may be live.
Test: test_unconfirmed_cancel_aborts_ladder proves exactly one order reaches
the broker in that scenario.

## 3. Medium — Transient LLM failure locked default regime for the day

The regime node cached whatever regime_view returned under the day key,
including the fail-open defaults produced by a transient OpenRouter error or a
thin premarket evidence read. One 429 at 9:45 would have silenced the regime
analyst until the next day.

Fix: RegimeView carries a source field; only a genuine LLM answer evaluated
over real evidence (present IVs and forecast) is cached. Failures retry
naturally on the next cycle. Guard asserted in the full-cycle test.

## 4. Medium — Emergency flatten ran serially

Kill-switch and manual flattens closed positions one at a time; three open
positions each working a multi-minute limit ladder could take many minutes to
flatten during exactly the kind of move that trips a kill switch.

Fix: all emergency closes now run concurrently via asyncio.gather, in the
graph's flatten node and the CLI flatten and panic paths.

## 5. Minor — Greeks not rescaled on size reduction

When the combined size factor reduced quantity, max loss and qty were updated
but the proposal's net delta and vega dollars kept their original values,
overstating exposure in memos and stored entry context. Risk checks were
unaffected (they had evaluated the larger values, conservatively).

Fix: delta and vega dollars scale with the quantity change.

## 6. Minor — Windows uv fallback lacked .exe

The absolute-path fallback for spawning the MCP server used a Unix-style path.
Theoretical on our stack (production VM is Linux, dev machines have uv on
PATH), fixed regardless with an os-aware executable name.

## Adjacent finds during verification

Close-ladder client order ids were reused across cycles: a close that failed
once would retry next cycle with the same ids, which the broker rejects as
duplicates, permanently wedging the position in closing state. Ids now carry a
per-attempt timestamp.

The test suite silently went online: once a real OpenRouter key existed in
.env, the "offline" full-cycle tests made live LLM calls (suite time 43s).
An autouse fixture now strips the key for every test; the suite runs in under
5 seconds, offline and free, by construction.

## Outcome

69 tests passing. Deployed to the VM at 9:32:48 ET (verified via systemd start
time, code fingerprints on disk, and the daemon restart pair in the audit
trail) with zero positions open, so no live trade was ever exposed to any of
the above.
