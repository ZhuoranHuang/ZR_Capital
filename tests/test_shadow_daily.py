"""每日影子回放汇总（zr/research/shadow_daily.py）：同日重跑替换、按日期排序。"""

from __future__ import annotations

from zr.research.shadow_daily import COLS, upsert


def _row(day: str, k: int) -> dict:
    r = dict.fromkeys(COLS, "x")
    r.update({"日期": day, "K 数": k})
    return r


def test_upsert_replaces_same_day_and_sorts(tmp_path):
    p = tmp_path / "SHADOW_DAILY.md"
    upsert(p, _row("2026-10-22", 197))
    upsert(p, _row("2026-10-21", 36))
    df = upsert(p, _row("2026-10-22", 198))
    assert df["日期"].tolist() == ["2026-10-21", "2026-10-22"]
    assert df["K 数"].tolist() == ["36", "198"]
    assert "2026-10-22" in p.read_text(encoding="utf-8")
