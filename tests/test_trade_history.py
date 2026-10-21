import datetime as dt
from decimal import Decimal
from enum import Enum

from zr.broker.serialize import to_py
from zr.journal.trade_history import Throttle, _flatten_charges, date_chunks, is_option_symbol


def test_option_symbol_regex():
    assert is_option_symbol("SPY261021C500000.US")
    assert is_option_symbol("BRK.B261021P400000.US")
    assert not is_option_symbol("SPY.US")
    assert not is_option_symbol("spy261021c500000.US")
    assert not is_option_symbol("SPY261021C500000.HK")


def test_date_chunks_cover_range_without_overlap():
    since = dt.datetime(2026, 5, 1, tzinfo=dt.UTC)
    until = dt.datetime(2026, 7, 10, tzinfo=dt.UTC)
    chunks = list(date_chunks(since, until, days=28))
    assert chunks[0][0] == since
    assert chunks[-1][1] == until
    for (_, a_end), (b_start, _) in zip(chunks, chunks[1:], strict=False):
        assert a_end == b_start
    assert all((e - s).days <= 28 for s, e in chunks)
    assert list(date_chunks(until, until)) == []


def test_throttle_sleeps_to_keep_gap():
    slept: list[float] = []
    t = Throttle(gap_s=1.05, sleep=slept.append)
    t.wait()
    t.wait()
    assert len(slept) == 1 and 1.0 < slept[0] <= 1.05


class Side(Enum):
    Buy = 1


class _Obj:
    def __init__(self):
        self.price = Decimal("1.25")
        self.side = Side.Buy
        self.at = dt.datetime(2026, 5, 1, tzinfo=dt.UTC)
        self.items = [Decimal("2")]

    def method(self):
        return 0


def test_to_py_converts_decimal_enum_and_objects():
    out = to_py(_Obj())
    assert out == {
        "price": 1.25,
        "side": "Buy",
        "at": dt.datetime(2026, 5, 1, tzinfo=dt.UTC),
        "items": [2.0],
    }
    assert to_py(_Obj(), datetime_as_str=True)["at"] == "2026-05-01T00:00:00+00:00"


def test_flatten_charges():
    class Fee:
        def __init__(self, code, amount):
            self.code, self.name, self.amount, self.currency = code, code.lower(), amount, "USD"

    class Item:
        def __init__(self):
            self.code, self.name = "BROKER", "broker"
            self.fees = [Fee("COMM", Decimal("0.45")), Fee("PLATFORM", Decimal("0.30"))]

    class Charge:
        def __init__(self):
            self.total_amount, self.currency, self.items = Decimal("0.75"), "USD", [Item()]

    class Detail:
        charge_detail = Charge()

    rows = _flatten_charges("o1", Detail())
    assert [r["fee_code"] for r in rows] == ["COMM", "PLATFORM"]
    assert rows[0]["amount"] == 0.45 and rows[0]["total_amount"] == 0.75
    assert all(r["order_id"] == "o1" for r in rows)
