import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import alpaca.graph as graph_mod
from alpaca.config import SLEEVE_B, SLEEVE_B_EVENTS
from alpaca.strategy.condor import parse_chain
from alpaca.strategy.events import build_crush_condor, implied_move

from conftest import FakeBroker, synthetic_chain
from test_graph_cycle import make_services, run

ET = ZoneInfo("America/New_York")
DELL_EVENT = next(e for e in SLEEVE_B_EVENTS if e.symbol == "DELL")
CRUSH_NOW = datetime(2026, 9, 1, 15, 35, tzinfo=ET)  # inside DELL entry window


def dell_contracts(iv: float = 0.65):
    chain = synthetic_chain("DELL", spot=140.0, expiry="2026-09-04", iv=iv,
                            t_years=4 / 365, strikes=range(105, 176, 1))
    return parse_chain("DELL", chain)


def test_implied_move_from_straddle():
    move = implied_move(dell_contracts(), "2026-09-04", 140.0)
    assert move is not None
    assert 4.0 < move < 15.0  # roughly 0.8 * S * sigma * sqrt(T)


def test_crush_condor_beyond_move_and_capped():
    proposal, diag = build_crush_condor(DELL_EVENT, dell_contracts(), 140.0, CRUSH_NOW)
    assert proposal is not None, diag
    move = diag["implied_move"]
    short_put = next(l for l in proposal.legs if l.side == "sell" and l.opt_type == "put")
    short_call = next(l for l in proposal.legs if l.side == "sell" and l.opt_type == "call")
    assert short_put.strike <= 140.0 - move
    assert short_call.strike >= 140.0 + move
    assert proposal.max_loss <= SLEEVE_B.crush_max_loss
    assert proposal.position.sleeve == "B"


def test_crush_rejects_without_straddle():
    proposal, diag = build_crush_condor(DELL_EVENT, [], 140.0, CRUSH_NOW)
    assert proposal is None
    assert "implied move" in diag["reject"]


def test_graph_opens_crush_in_window_once(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    monkeypatch.setattr(graph_mod, "now_et", lambda: CRUSH_NOW)
    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    class FillingFakeBroker(FakeBroker):
        async def place_option_order(self, qty, legs, limit_price, client_order_id):
            order = {
                "id": f"ord-{len(self.placed_orders)}",
                "client_order_id": client_order_id,
                "status": "filled",
                "filled_qty": str(qty),
                "filled_avg_price": limit_price,
                "limit_price": limit_price,
                "legs": legs,
            }
            self.placed_orders.append(order)
            return order

    broker = FillingFakeBroker()
    services = make_services(tmp_path, broker=broker, dry_run=False)
    run(services)

    b_open = [p for p in services.ledger.open_positions() if p.sleeve == "B"]
    dell_crush = [p for p in b_open if p.underlying == "DELL" and "crush" in p.structure]
    assert len(dell_crush) == 1, f"the DELL crush condor must open inside its window: {b_open}"
    assert services.db.get_state("sleeveB:DELL:crush") == "opened"
    # Tuesday afternoon is also inside AVGO's run-up window; a strangle there is legitimate

    orders_before = len(broker.placed_orders)
    run(services)  # second cycle in the same window must not duplicate entries
    dell_crush_after = [
        p for p in services.ledger.open_positions()
        if p.sleeve == "B" and p.underlying == "DELL" and "crush" in p.structure
    ]
    assert len(dell_crush_after) == 1
    new_entry_orders = [
        o for o in broker.placed_orders[orders_before:]
        if o["client_order_id"].startswith("SLB-") and "-x" not in o["client_order_id"]
    ]
    assert not new_entry_orders, new_entry_orders


def test_runup_strangle_all_long_and_capped():
    from alpaca.broker.executor import is_long_structure
    from alpaca.strategy.events import build_runup_strangle

    runup_now = datetime(2026, 8, 31, 14, 50, tzinfo=ET)  # prior day, inside window
    contracts = dell_contracts(iv=0.35)  # calmer pre-event IV keeps debit under cap
    proposal, diag = build_runup_strangle(DELL_EVENT, contracts, 140.0, runup_now)
    assert proposal is not None, diag
    assert is_long_structure(proposal.position)
    assert proposal.max_loss <= SLEEVE_B.runup_max_debit
    assert proposal.position.structure == "event_runup_strangle"
    sides = {l.opt_type for l in proposal.legs}
    assert sides == {"put", "call"}


def test_runtime_viability_windows():
    monday_after_open = datetime(2026, 8, 31, 14, 50, tzinfo=ET)
    assert DELL_EVENT.runup_viable(monday_after_open)
    assert not DELL_EVENT.crush_viable(monday_after_open)
    tuesday_late = datetime(2026, 9, 1, 15, 40, tzinfo=ET)
    assert not DELL_EVENT.runup_viable(tuesday_late)  # entries end 13:00 event day
    assert DELL_EVENT.crush_viable(tuesday_late)


def test_event_view_parser_and_failopen():
    from alpaca.agents.desk import parse_event_view

    ok = parse_event_view('{"trade_runup": false, "trade_crush": true, "note": "thin window"}')
    assert ok == {"trade_runup": False, "trade_crush": True, "note": "thin window"}
    assert parse_event_view('{"trade_runup": "yes"}') is None
    assert parse_event_view("gibberish") is None


def test_debit_ladder_and_long_close_pnl(monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.broker.executor import debit_ladder_prices, submit_close, submit_open
    from alpaca.config import ExecutorConfig
    from alpaca.risk.ledger import Leg, Position

    prices = debit_ladder_prices(1.50, max_debit_price=1.54)
    assert prices[0] == 1.50 and prices[-1] <= 1.54
    assert debit_ladder_prices(1.60, max_debit_price=1.55) == []

    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    legs = [
        Leg("DELL260904P00133000", "buy", 1, 133.0, "put", "2026-09-04", 0.5, -0.3),
        Leg("DELL260904C00147000", "buy", 1, 147.0, "call", "2026-09-04", 0.5, 0.3),
    ]
    pos = Position(
        position_id="SLB-T", sleeve="B", underlying="DELL",
        structure="event_runup_strangle", legs=legs, qty=1, credit=1.50,
        width=14.0, max_loss=150.0, client_order_id="SLB-T",
        opened_at="2026-08-31T14:50:00-04:00",
    )

    class LongBroker:
        def __init__(self):
            self.orders = {}

        async def option_quotes(self, symbols):
            return {s: {"bp": 0.73, "ap": 0.77} for s in symbols}  # each leg mid 0.75

        async def place_option_order(self, qty, legs, limit_price, client_order_id):
            order = {"id": f"o{len(self.orders)}", "client_order_id": client_order_id,
                     "status": "filled", "filled_qty": str(qty),
                     "filled_avg_price": limit_price}
            self.orders[client_order_id] = order
            return order

        async def order_by_client_id(self, coid):
            return self.orders[coid]

        async def cancel_order(self, oid):
            return {"ok": True}

    broker = LongBroker()
    memos = []
    ok = asyncio.run(submit_open(broker, lambda e, d: memos.append(e), pos,
                                 max_debit_price=1.55))
    assert ok and pos.status == "open"
    assert abs(pos.credit - 1.50) < 1e-9        # debit paid at the mid
    assert abs(pos.max_loss - 150.0) < 1e-9     # long structure risks the debit

    ok = asyncio.run(submit_close(broker, lambda e, d: memos.append(e), pos, "runup_pre_print_exit"))
    assert ok and pos.status == "closed"
    # closing nets a 1.50 credit back: flat trade, zero realized
    assert abs(pos.realized_pnl - 0.0) < 1e-9


def test_graph_opens_dell_runup_on_prior_day(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    runup_now = datetime(2026, 8, 31, 14, 50, tzinfo=ET)
    monkeypatch.setattr(graph_mod, "now_et", lambda: runup_now)
    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    class CalmDellBroker(FakeBroker):
        async def option_chain(self, underlying, **kwargs):
            if underlying == "DELL":
                return synthetic_chain("DELL", 140.0, "2026-09-04", iv=0.35,
                                       t_years=4 / 365, strikes=range(105, 176, 1))
            return await super().option_chain(underlying, **kwargs)

        async def option_quotes(self, symbols):
            from alpaca.risk.bs import bs as _bs

            out = await super().option_quotes(symbols)
            for s in symbols:
                if s.startswith("DELL"):  # quotes must match the calm chain
                    strike = int(s[-8:]) / 1000.0
                    mid = max(_bs(s[10] == "C", 140.0, strike, 4 / 365, 0.35).price, 0.02)
                    out[s] = {"bp": round(mid - 0.02, 2), "ap": round(mid + 0.02, 2)}
            return out

        async def place_option_order(self, qty, legs, limit_price, client_order_id):
            order = {"id": f"ord-{len(self.placed_orders)}",
                     "client_order_id": client_order_id, "status": "filled",
                     "filled_qty": str(qty), "filled_avg_price": limit_price,
                     "limit_price": limit_price, "legs": legs}
            self.placed_orders.append(order)
            return order

    services = make_services(tmp_path, broker=CalmDellBroker(), dry_run=False)
    run(services)
    b_open = [p for p in services.ledger.open_positions() if p.sleeve == "B"]
    assert len(b_open) == 1
    assert b_open[0].structure == "event_runup_strangle"
    assert services.db.get_state("sleeveB:DELL:runup") == "opened"
    assert services.db.get_state("sleeveB:DELL:crush") is None  # not its window


def test_crush_time_exit_next_morning(tmp_path, monkeypatch):
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    class FillingFakeBroker(FakeBroker):
        async def place_option_order(self, qty, legs, limit_price, client_order_id):
            order = {
                "id": f"ord-{len(self.placed_orders)}",
                "client_order_id": client_order_id,
                "status": "filled",
                "filled_qty": str(qty),
                "filled_avg_price": limit_price,
                "limit_price": limit_price,
                "legs": legs,
            }
            self.placed_orders.append(order)
            return order

    broker = FillingFakeBroker()
    services = make_services(tmp_path, broker=broker, dry_run=False)

    monkeypatch.setattr(graph_mod, "now_et", lambda: CRUSH_NOW)
    run(services)
    assert [
        p for p in services.ledger.open_positions()
        if p.sleeve == "B" and "crush" in p.structure
    ]

    wednesday_morning = datetime(2026, 9, 2, 10, 5, tzinfo=ET)
    monkeypatch.setattr(graph_mod, "now_et", lambda: wednesday_morning)
    run(services)
    crush_open = [
        p for p in services.ledger.open_positions()
        if p.sleeve == "B" and "crush" in p.structure
    ]
    assert not crush_open, "crush condor must be covered by the morning time exit"
    closed = [
        p for p in services.ledger.all_positions()
        if p.sleeve == "B" and "crush" in p.structure
    ][0]
    assert closed.close_reason in ("event_crush_exit", "profit_target_50pct")


def test_crush_steps_put_out_when_near_call_quote_dies():
    """Sep 1 live: one side's nearest strike lost its quote, the old builder
    paired a close put with a far call and shipped -16K of delta. The builder
    must rebalance the other side instead."""
    contracts = dell_contracts()
    move = implied_move(contracts, "2026-09-04", 140.0)
    call_strikes = sorted(c.strike for c in contracts
                          if c.opt_type == "call" and c.strike >= 140.0 + move)
    put_strikes = sorted((c.strike for c in contracts
                          if c.opt_type == "put" and c.strike <= 140.0 - move), reverse=True)
    from dataclasses import replace

    contracts = [  # kill the nearest qualifying call's quote
        replace(c, bid=0.0) if c.opt_type == "call" and c.strike == call_strikes[0] else c
        for c in contracts
    ]
    proposal, diag = build_crush_condor(DELL_EVENT, contracts, 140.0, CRUSH_NOW,
                                        max_abs_delta=10_000.0)
    assert proposal is not None, diag
    short_put = next(l for l in proposal.legs if l.side == "sell" and l.opt_type == "put")
    short_call = next(l for l in proposal.legs if l.side == "sell" and l.opt_type == "call")
    assert short_call.strike >= call_strikes[1]      # dead strike unusable
    assert short_put.strike < put_strikes[0]         # put stepped out to rebalance
    assert abs(proposal.net_delta_dollars) <= 10_000.0
    assert "net_delta" in diag


def test_crush_refuses_when_no_pairing_fits_delta_cap():
    proposal, diag = build_crush_condor(DELL_EVENT, dell_contracts(), 140.0, CRUSH_NOW,
                                        max_abs_delta=1.0)
    assert proposal is None
    assert "delta-balanced" in diag["reject"]
    assert diag["rejected_pairings"].get("delta_cap", 0) > 0


def test_crush_decline_skips_cycle_but_does_not_latch(tmp_path, monkeypatch):
    """One hesitant analyst sample must not end a 25 minute crush window;
    the next cycle asks fresh. Run-up declines still latch for the day."""
    import alpaca.broker.executor as executor
    from alpaca.config import ExecutorConfig

    monkeypatch.setattr(graph_mod, "now_et", lambda: CRUSH_NOW)
    monkeypatch.setattr(
        executor, "EXEC",
        ExecutorConfig(improve_step=0.02, max_improvements=1, wait_seconds=1, poll_seconds=0),
    )

    calls = {}

    async def flaky_view(context, memo):
        sym = context["symbol"]
        calls[sym] = calls.get(sym, 0) + 1
        if sym == "DELL" and calls[sym] == 1:
            return {"trade_runup": False, "trade_crush": False, "note": "hesitant"}
        return {"trade_runup": False, "trade_crush": True, "note": "go"}

    monkeypatch.setattr(graph_mod.desk, "event_phase_view", flaky_view)

    class FillingFakeBroker(FakeBroker):
        async def place_option_order(self, qty, legs, limit_price, client_order_id):
            order = {"id": f"ord-{len(self.placed_orders)}",
                     "client_order_id": client_order_id, "status": "filled",
                     "filled_qty": str(qty), "filled_avg_price": limit_price,
                     "limit_price": limit_price, "legs": legs}
            self.placed_orders.append(order)
            return order

    services = make_services(tmp_path, broker=FillingFakeBroker(), dry_run=False)
    run(services)  # DELL crush declined this cycle
    assert services.db.get_state("sleeveB:DELL:crush") is None, "decline must not latch crush"
    assert services.db.get_state("sleeveB:DELL:view:2026-09-01") is None, "view must be dropped"
    assert services.db.get_state("sleeveB:AVGO:runup") == "declined", "runup declines still latch"
    assert not [p for p in services.ledger.open_positions() if p.underlying == "DELL"]

    run(services)  # fresh analyst approves; the trade proceeds
    assert calls["DELL"] == 2
    dell = [p for p in services.ledger.open_positions()
            if p.underlying == "DELL" and "crush" in p.structure]
    assert len(dell) == 1, "second cycle must open the crush condor"
    assert services.db.get_state("sleeveB:DELL:crush") == "opened"
