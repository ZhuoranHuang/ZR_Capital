"""开盘时段 BOLL / MACD 处理方式的锁定测试（audit/base-final 第 2 节，见 OPENING_INDICATORS.md）。

冻结的 MU 2026-10-21 1 分钟 K（tests/fixtures，长桥实拉）：
- 口径 C：含盘前盘后连续 → 09:30 那根就有完整 20 根窗口（08:52–09:30）；
- 口径 A：RTH-only 按日重置 → 前 19 根 BOLL 为空值、第 20 根（10:08）起有值，前 19 根状态只能是
  OTHER；MACD 首根收盘做种，第一根起有值（DIF = 0）；
- 口径 B：部分窗口，第一根即有值（= 收盘价）。
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zr.features.bars import ET, resample_bars
from zr.features.regime import OTHER, indicators, regimes
from zr.research import v4_decisive as v4d

FIX = Path(__file__).parent / "fixtures"


def _bars(name: str) -> pd.DataFrame:
    df = pd.read_csv(FIX / name, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return resample_bars(df.set_index("ts").sort_index(), "2min")


def _hm(b: pd.DataFrame) -> np.ndarray:
    return np.asarray(b.index.tz_convert(ET).strftime("%H:%M"))


@pytest.fixture(scope="module")
def cfg():
    return v4d.state_cfg(v4d.load_v4(), "A")


def test_convention_c_has_full_window_at_open(cfg):
    b = _bars("mu_2026_10_21_min1_ext.csv")
    x = indicators(b, cfg, daily_reset=False, warmup="nan")
    hm = _hm(b)
    i = int(np.flatnonzero(hm == "09:30")[0])
    assert np.isfinite(x["mid"].iloc[i]) and np.isfinite(x["pctb"].iloc[i])
    win = b["close"].iloc[i - 19 : i + 1]
    assert _hm(b)[i - 19] == "08:52"
    assert x["mid"].iloc[i] == pytest.approx(win.mean())
    assert x["upper"].iloc[i] == pytest.approx(win.mean() + 2 * win.std(ddof=0))


def test_convention_a_blank_first_19_bars_then_valid(cfg):
    b = _bars("mu_2026_10_21_min1_rth.csv")
    x = indicators(b, cfg, daily_reset=True, warmup="nan")
    reg = regimes(b, x, cfg, 2)
    hm = _hm(b)
    assert hm[0] == "09:30" and hm[19] == "10:08"
    assert x["mid"].iloc[:19].isna().all() and x["pctb"].iloc[:19].isna().all()
    assert np.isfinite(x["mid"].iloc[19])
    assert (reg["state"].iloc[:19] == OTHER).all()
    assert x["dif"].iloc[0] == pytest.approx(0.0) and x["dif"].iloc[:19].notna().all()


def test_convention_b_partial_window_from_first_bar(cfg):
    b = _bars("mu_2026_10_21_min1_rth.csv")
    x = indicators(b, copy.deepcopy(cfg), daily_reset=True, warmup="partial")
    assert x["mid"].iloc[0] == pytest.approx(b["close"].iloc[0])
    assert x["mid"].iloc[4] == pytest.approx(b["close"].iloc[:5].mean())


def test_v4_config_fixes_warmup_as_nan():
    v4 = v4d.load_v4()
    assert v4["primary"] == "C" and v4["conventions"]["C"]["reset"] is False
    assert v4["conventions"]["A"]["reset"] is True and v4["conventions"]["A"]["warmup"] == "nan"
