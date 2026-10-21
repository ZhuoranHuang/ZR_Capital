import datetime as dt
from types import SimpleNamespace

from zr.journal.trade_history import Throttle, fetch


class FakeTradeCtx:
    def __init__(self):
        self.detail_calls: list[str] = []

    def history_orders(self, symbol, status, side, market, start, end):
        return [
            SimpleNamespace(
                order_id="o1", symbol="MU261023C900000.US", status="Filled", side="Buy", msg=""
            ),
            SimpleNamespace(
                order_id="o2", symbol="MU261023C900000.US", status="Rejected", side="Buy", msg=""
            ),
            SimpleNamespace(order_id="o3", symbol="MU.US", status="Filled", side="Buy", msg=""),
        ]

    def history_executions(self, symbol, start, end):
        return [
            SimpleNamespace(
                trade_id="t1", order_id="o1", symbol="MU261023C900000.US", quantity=1, price=1.0
            ),
            SimpleNamespace(trade_id="t2", order_id="o3", symbol="MU.US", quantity=10, price=900.0),
        ]

    def order_detail(self, order_id):
        self.detail_calls.append(order_id)
        if order_id == "o2":
            return SimpleNamespace(
                msg="The order amount exceeds the maximum buying power", charge_detail=None
            )
        return SimpleNamespace(
            msg="", charge_detail=SimpleNamespace(total_amount=2.04, currency="USD", items=[])
        )


def test_fetch_filters_options_dedups_and_captures_reject_reason():
    t = FakeTradeCtx()
    since = dt.datetime(2026, 5, 1, tzinfo=dt.UTC)
    until = dt.datetime(2026, 6, 30, tzinfo=dt.UTC)  # 3 段 → history 被调 3 次，结果需去重
    orders, execs, charges = fetch(
        t, since, until, options_only=True, throttle=Throttle(sleep=lambda s: None)
    )
    assert sorted(orders["order_id"]) == ["o1", "o2"]
    assert list(execs["trade_id"]) == ["t1"]
    assert orders.set_index("order_id").loc["o2", "reject_reason"].startswith("The order amount")
    assert sorted(t.detail_calls) == ["o1", "o2"]
    assert charges.iloc[0]["total_amount"] == 2.04
