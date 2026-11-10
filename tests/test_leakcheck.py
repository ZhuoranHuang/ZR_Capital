"""前视截断检验（zr/research/leakcheck.py）：复现 2026-11 的错误写法并断言能被检出；修正写法通过；
zr/features 公开函数逐一检验（教训 L-001）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zr.features.bars import et_date
from zr.research import leakcheck as lc

BARS = lc.synthetic_bars(n_days=5, seed=3)


def _day(d: pd.DataFrame) -> np.ndarray:
    return et_date(d.index)


def _minute(d: pd.DataFrame) -> np.ndarray:
    t = d.index.tz_convert(lc.ET)
    return (t.hour * 60 + t.minute - 570).to_numpy()


# ---------------- 复现错误写法 ----------------


def gap_leaky(d: pd.DataFrame) -> pd.Series:
    """错误：日级汇总用 transform 铺到分钟行后按行 shift——除当日第一行外，「昨收」其实是
    当日收盘。"""
    c = d["close"]
    prev_close = c.groupby(_day(d)).transform("last").shift(1)
    first = d["open"].groupby(_day(d)).transform("first")
    return (first / prev_close - 1).rename("gap")


def gap_fixed(d: pd.DataFrame) -> pd.Series:
    """修正：先 groupby(day) 得到日序列，按日 shift(1)，再按日期映射回分钟行；
    用当日首根开盘价（开盘即可用）。"""
    day = _day(d)
    daily = pd.DataFrame(
        {"first_open": d["open"].groupby(day).first(), "last_close": d["close"].groupby(day).last()}
    )
    gap = daily["first_open"] / daily["last_close"].shift(1) - 1
    return pd.Series(gap.reindex(day).to_numpy(), index=d.index, name="gap")


def filter15_leaky(d: pd.DataFrame) -> pd.Series:
    """错误：前 15 分钟涨幅按日 transform 铺到全天，09:31 的行也拿到了 09:44 收盘才有的值。"""
    first15 = d["close"].where(_minute(d) == 14)
    first_open = d["open"].groupby(_day(d)).transform("first")
    r = first15.groupby(_day(d)).transform("max") / first_open - 1
    return r.rename("ret15")


def filter15_fixed(d: pd.DataFrame) -> pd.Series:
    """修正：09:44 那根（第 15 根）收盘后才有值，之前为 NaN；入场须在之后。"""
    first_open = d["open"].groupby(_day(d)).transform("first")  # 开盘即可用
    c15 = d["close"].where(_minute(d) == 14)
    c15 = c15.groupby(_day(d)).ffill()  # 只向后填充（同一日内）
    return (c15 / first_open - 1).rename("ret15")


def test_leaky_gap_is_detected():
    with pytest.raises(AssertionError, match="gap"):
        lc.assert_no_lookahead_frame(gap_leaky, BARS, cuts=10)


def test_fixed_gap_first_open_passes():
    lc.assert_no_lookahead_frame(gap_fixed, BARS, cuts=10)
    g = gap_fixed(BARS)
    assert g.notna().sum() > 0 and g.groupby(_day(BARS)).nunique().max() == 1


def test_leaky_filter15_is_detected():
    with pytest.raises(AssertionError, match="ret15"):
        lc.assert_no_lookahead_frame(filter15_leaky, BARS, cuts=10)


def test_fixed_filter15_passes_and_is_nan_before_0945():
    lc.assert_no_lookahead_frame(filter15_fixed, BARS, cuts=10)
    f = filter15_fixed(BARS)
    assert f[_minute(BARS) < 14].isna().all() and f[_minute(BARS) >= 14].notna().all()


def test_series_and_frame_outputs_and_row_alignment():
    lc.assert_no_lookahead_frame(lambda d: d[["close"]].rolling(5).mean(), BARS)
    with pytest.raises(AssertionError, match="peek"):
        lc.assert_no_lookahead_frame(lambda d: d["close"].shift(-1).rename("peek"), BARS)
    with pytest.raises(ValueError, match="逐行"):
        lc.assert_no_lookahead_frame(lambda d: d["close"].iloc[::2], BARS)


def test_cut_points_fall_inside_days_and_near_open():
    first_rows = np.flatnonzero(np.r_[True, _day(BARS)[1:] != _day(BARS)[:-1]])
    ks = lc.cut_points(len(BARS), 10)  # 无时间索引：只有均匀截断点
    assert len(ks) == 10 and not set(ks) & set(first_rows)
    ks = set(lc.cut_points(len(BARS), 10, BARS.index))
    assert not ks & set(first_rows)
    # 5 天 < 10：除首日外每天开盘后的截断点全部入选
    assert {int(s) + o for s in first_rows[1:] for o in lc.OPEN_OFFSETS} <= ks


# ---------------- zr/features 公开函数 ----------------

SYN = lc.synthetic_bars()
# 已知未通过（只列出，不改原函数；见 docs/research/LEAKCHECK_FEATURES.md）。
# regime.range_score 的非因果窗口已按 D-037 修复，当前无已知未通过项。
KNOWN: dict[str, set[str]] = {}


@pytest.mark.parametrize("case", lc.synthetic_cases(), ids=lambda c: c.name)
def test_features_truncation(case):
    bad = lc.check_no_lookahead_frame(case.fn, SYN, cuts=10)
    assert set(bad) == KNOWN.get(case.name, set())


def test_every_public_feature_function_is_classified():
    covered = {f for c in lc.synthetic_cases() + lc.real_cases(lc.REPO) for f in c.covers}
    missing = [f for f in lc.public_functions() if f not in covered and f not in lc.NOT_APPLICABLE]
    assert missing == [], f"新的公开函数需要加入 leakcheck 登记或 NOT_APPLICABLE：{missing}"
    stale = [f for f in covered | set(lc.NOT_APPLICABLE) if f not in lc.public_functions()]
    assert stale == []


def _root():
    from zr.config import get_settings

    r = get_settings().zr_data_root
    ok = r is not None and (r / "intrabar" / "underlying=MU" / "date=2025-03-05").exists()
    return r if ok else None


@pytest.mark.skipif(_root() is None, reason="本地没有 MU 真实数据（ZR_DATA_ROOT）")
@pytest.mark.parametrize("i", [0, 1], ids=["option_features", "entry_features"])
def test_real_data_truncation(i):
    case = lc.real_cases(_root())[i]
    assert case.custom() == {}
