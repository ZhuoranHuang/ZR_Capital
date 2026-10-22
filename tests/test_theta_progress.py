import datetime as dt
import json

from zr.data.theta_progress import COLUMNS, TABLE_HEADER, append_row, format_row, month_summary


def test_month_summary_uses_latest_record_per_day(tmp_path):
    m = tmp_path / "options_1m" / "manifest.jsonl"
    m.parent.mkdir(parents=True)
    recs = [
        {"underlying": "SPY", "date": "2023-10-02", "error": "HTTP 474"},
        {
            "underlying": "SPY",
            "date": "2023-10-02",
            "rows_quotes": 10,
            "rows_ohlc": 9,
            "rows_oi": 2,
        },
        {"underlying": "SPY", "date": "2023-10-03", "rows_quotes": 5, "rows_ohlc": 5, "rows_oi": 1},
        {"underlying": "SPY", "date": "2023-10-04", "error": "HTTP 570"},
        {
            "underlying": "QQQ",
            "date": "2023-10-02",
            "rows_quotes": 99,
            "rows_ohlc": 99,
            "rows_oi": 9,
        },
        {"underlying": "SPY", "date": "2023-11-01", "rows_quotes": 7, "rows_ohlc": 7, "rows_oi": 1},
    ]
    m.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    d = tmp_path / "options_1m/underlying=SPY/date=2023-10-02"
    d.mkdir(parents=True)
    import pandas as pd

    pd.DataFrame({"strike": [630.0] * 60_000, "bid": [1.0] * 60_000}).to_parquet(
        d / "quotes.parquet", index=False
    )  # 真实 parquet（覆盖检查会读它）；约 0.5 MB
    s = month_summary(tmp_path, "SPY", dt.date(2023, 10, 1), dt.date(2023, 10, 31))
    # 没有正股 K → 覆盖未知（不计入 covered/uncovered）；mb 取决于 parquet 实际大小，只验 > 0
    assert {k: v for k, v in s.items() if k != "mb"} == {
        "days": 2,
        "failed": 1,
        "quotes": 15,
        "ohlc": 14,
        "oi": 3,
        "covered": 0,
        "uncovered": 0,
    }
    assert s["mb"] > 0


def test_append_row_creates_table_once_and_appends_in_order(tmp_path):
    doc = tmp_path / "INV.md"
    doc.write_text("# 数据清单\n\n## 其他小节\n\n内容\n", encoding="utf-8")
    s = {
        "days": 21,
        "failed": 0,
        "quotes": 1,
        "ohlc": 1,
        "oi": 1,
        "mb": 1.0,
        "covered": 21,
        "uncovered": 0,
    }
    append_row(doc, format_row("SPY", dt.date(2023, 10, 2), s, 600, "t1"))
    append_row(doc, format_row("SPY", dt.date(2023, 11, 1), s, 650, "t2"))
    text = doc.read_text(encoding="utf-8")
    assert text.count(TABLE_HEADER) == 1 and text.count(COLUMNS) == 1
    assert text.index("| SPY | 2023-10 |") < text.index("| SPY | 2023-11 |")
    assert "## 其他小节" in text
