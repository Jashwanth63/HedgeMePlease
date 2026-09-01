import asyncio

import alpaca.broker.executor as executor
from alpaca.broker.executor import close_legs, ladder_prices, net_mid, open_legs, submit_open
from alpaca.config import ExecutorConfig

from test_stress import make_condor


class ZeroBidWingQuotes:
    """Shorts quoted normally; wings worthless with a zero bid, as near expiry."""

    async def option_quotes(self, symbols):
        out = {}
        for s in symbols:
            if "P00640" in s or "C00662" in s:
                out[s] = {"bp": 0.73, "ap": 0.77}
            else:
                out[s] = {"bp": 0.0, "ap": 0.02}
        return out


def test_net_mid_tolerates_zero_bid_wings():
    cost = asyncio.run(net_mid(ZeroBidWingQuotes(), make_condor(), closing=True))
    assert cost is not None
    assert abs(cost - (0.75 * 2 - 0.01 * 2)) < 1e-9


def test_net_mid_refuses_fat_ask_over_zero_bid():
    class Fat(ZeroBidWingQuotes):
        async def option_quotes(self, symbols):
            q = await super().option_quotes(symbols)
            for v in q.values():
                if v["bp"] == 0.0:
                    v["ap"] = 0.50
            return q

    assert asyncio.run(net_mid(Fat(), make_condor(), closing=True)) is None


def test_ladder_prices_concede_toward_zero():
    prices = ladder_prices(-1.20, width=5.0)
    assert prices[0] == -1.20
    assert len(prices) == executor.EXEC.max_improvements + 1
    assert prices == sorted(prices)
    assert all(-p >= 5.0 * 0.12 for p in prices)


def test_ladder_stops_at_credit_floor():
    # floor for width 5 is 0.60 credit; only steps at or above it survive
    prices = ladder_prices(-0.62, width=5.0)
    assert prices == [-0.62, -0.60]


def test_ladder_empty_when_mid_below_floor():
    assert ladder_prices(-0.40, width=5.0) == []


def test_leg_mapping_open_close():
    pos = make_condor()
    opens = open_legs(pos)
    closes = close_legs(pos)
    assert opens[0]["position_intent"] == "sell_to_open"
    assert closes[0]["side"] == "buy" and closes[0]["position_intent"] == "buy_to_close"
    assert opens[1]["position_intent"] == "buy_to_open"
    assert closes[1]["position_intent"] == "sell_to_close"


class InstantFillBroker:
    def __init__(self, fill_price: float = -1.18):
        self.fill_price = fill_price
        self.orders: dict[str, dict] = {}

    async def option_quotes(self, symbols):
        # mids sum to a 1.20 net credit for the standard test condor
        return {
            s: {"bp": 0.78, "ap": 0.82} if "P00640" in s or "C00662" in s else {"bp": 0.18, "ap": 0.22}
            for s in symbols
        }

    async def place_option_order(self, qty, legs, limit_price, client_order_id):
        order = {
            "id": f"o-{len(self.orders)}",
            "client_order_id": client_order_id,
            "status": "filled",
            "filled_qty": str(qty),
            "filled_avg_price": self.fill_price,
        }
        self.orders[client_order_id] = order
        return order

    async def order_by_client_id(self, coid):
        return self.orders[coid]

    async def cancel_order(self, order_id):
        return {"ok": True}


class NeverFillBroker(InstantFillBroker):
    async def place_option_order(self, qty, legs, limit_price, client_order_id):
        order = {
            "id": f"o-{len(self.orders)}",
            "client_order_id": client_order_id,
            "status": "accepted",
            "filled_qty": "0",
        }
        self.orders[client_order_id] = order
        return order

    async def cancel_order(self, order_id):
        for order in self.orders.values():
            if order["id"] == order_id:
                order["status"] = "canceled"
        return {"ok": True}


def fast_exec(monkeypatch):
    monkeypatch.setattr(
        executor, "EXEC", ExecutorConfig(improve_step=0.02, max_improvements=2, wait_seconds=1, poll_seconds=0)
    )


def test_submit_open_fills_and_updates_credit(monkeypatch):
    fast_exec(monkeypatch)
    pos = make_condor()
    memos = []
    ok = asyncio.run(submit_open(InstantFillBroker(), lambda e, d: memos.append(e), pos))
    assert ok
    assert pos.status == "open"
    assert abs(pos.credit - 1.18) < 1e-9
    assert "opened" in memos


class StuckCancelBroker(NeverFillBroker):
    """Order never fills AND the cancel never confirms: the danger case."""

    async def cancel_order(self, order_id):
        return {"ok": True}  # exchange never acknowledges; status stays accepted


def test_unconfirmed_cancel_aborts_ladder(monkeypatch):
    fast_exec(monkeypatch)
    broker = StuckCancelBroker()
    pos = make_condor()
    memos = []
    ok = asyncio.run(submit_open(broker, lambda e, d: memos.append(e), pos))
    assert not ok
    assert pos.status == "abandoned"
    assert len(broker.orders) == 1, "must not requote while the old order may be live"
    assert "open_abandoned" in memos
    assert "open_requote" not in memos


def test_submit_open_times_out_and_cancels(monkeypatch):
    fast_exec(monkeypatch)
    pos = make_condor()
    memos = []
    ok = asyncio.run(submit_open(NeverFillBroker(), lambda e, d: memos.append(e), pos))
    assert not ok
    assert pos.status == "abandoned"
    assert "open_abandoned" in memos
    assert memos.count("open_requote") >= 1


def test_option_order_payload_single_leg_buy():
    from alpaca.broker.mcp import option_order_payload

    leg = {"symbol": "SPY260911P00738000", "ratio_qty": "1", "side": "buy",
           "position_intent": "buy_to_open"}
    p = option_order_payload(2, [leg], 1.17, "SLC-x")
    assert p["symbol"] == "SPY260911P00738000"
    assert p["side"] == "buy"
    assert "order_class" not in p and "legs" not in p
    assert p["limit_price"] == "1.17" and p["qty"] == "2"


def test_option_order_payload_single_leg_close_price_unsigned():
    from alpaca.broker.mcp import option_order_payload

    leg = {"symbol": "SPY260911P00738000", "ratio_qty": "1", "side": "sell",
           "position_intent": "sell_to_close"}
    p = option_order_payload(1, [leg], -0.95, "SLC-x-close")
    assert p["limit_price"] == "0.95"


def test_option_order_payload_refuses_naked_short_and_bad_leg_counts():
    import pytest

    from alpaca.broker.mcp import option_order_payload

    naked = {"symbol": "SPY260911P00738000", "ratio_qty": "1", "side": "sell",
             "position_intent": "sell_to_open"}
    with pytest.raises(ValueError):
        option_order_payload(1, [naked], 1.0, "x")
    with pytest.raises(ValueError):
        option_order_payload(1, [], 1.0, "x")


def test_option_order_payload_multi_leg_keeps_signed_credit():
    from alpaca.broker.mcp import option_order_payload

    legs = [{"symbol": f"SPY260903{cp}00{k}0000", "ratio_qty": "1", "side": s,
             "position_intent": i}
            for cp, k, s, i in (("P", 63, "sell", "sell_to_open"), ("P", 62, "buy", "buy_to_open"),
                                ("C", 67, "sell", "sell_to_open"), ("C", 68, "buy", "buy_to_open"))]
    p = option_order_payload(1, legs, -1.11, "SLA-x")
    assert p["order_class"] == "mleg" and len(p["legs"]) == 4
    assert p["limit_price"] == "-1.11"
