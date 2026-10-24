"""方向化入场特征（zr/features/entry_features.py）：合成数据上的截断检验、方向化、规则引用的特征名。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from zr.features import entry_features as ef
from zr.features.bars import ET

DAY = dt.date(2025, 3, 5)  # 周三
PREV = dt.date(2025, 3, 4)


def _bars(day: dt.date, start: str, end: str, rng, base: float) -> pd.DataFrame:
    idx = pd.date_range(
        pd.Timestamp(f"{day} {start}", tz=ET),
        pd.Timestamp(f"{day} {end}", tz=ET),
        freq="1min",
        inclusive="left",
    ).tz_convert("UTC")
    c = base * np.exp(np.cumsum(rng.normal(0, 5e-4, len(idx))))
    o = np.r_[base, c[:-1]]
    hi = np.maximum(o, c) * (1 + rng.uniform(0, 3e-4, len(idx)))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 3e-4, len(idx)))
    v = rng.integers(100, 2000, len(idx)).astype(float)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory) -> ef.Ctx:
    root: Path = tmp_path_factory.mktemp("zr")
    rng = np.random.default_rng(1)
    ext = pd.concat([_bars(d, "04:00", "20:00", rng, 100.0 + k) for k, d in enumerate((PREV, DAY))])
    ext.index.name = "ts"
    rth_m = ext[
        (ext.index.tz_convert(ET).time >= dt.time(9, 30))
        & (ext.index.tz_convert(ET).time < dt.time(16, 0))
    ]
    rth = {"MU": rth_m}
    for u in ef.CROSS:
        rth[u] = rth_m * 1.0
    # 秒级平价价：在 1 分钟收盘之间线性插值 + 噪声，每秒 1–3 个报价
    t0 = pd.Timestamp(f"{DAY} 09:29:50", tz=ET).value
    t1 = pd.Timestamp(f"{DAY} 16:00", tz=ET).value
    ts = np.sort(rng.integers(t0, t1, 50_000))
    S = np.interp(ts, rth_m.index.as_unit("ns").asi8, rth_m["close"].to_numpy())
    S = S * (1 + rng.normal(0, 1e-5, len(ts)))
    d = root / "intrabar" / "underlying=MU" / f"date={DAY}"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts": ts,
            "S": S,
            "strike": np.round(S),
            "expiry": DAY.isoformat(),
            "c_mid": 1.5 + rng.uniform(0, 0.1, len(ts)),
            "p_mid": 1.4 + rng.uniform(0, 0.1, len(ts)),
            "c_spread": 0.05,
            "p_spread": 0.05,
        }
    ).to_parquet(d / "S.parquet", index=False)
    return ef.Ctx(root=root, und="MU", ext=ext, rth=rth, releases=pd.DatetimeIndex([], tz="UTC"))


def test_truncation_no_lookahead(ctx):
    day = ef.load_day(ctx, DAY)
    ts = day.t[[90, 700, 5000, 12345]] + 300_000_000
    r = ef.truncation_check(ctx, DAY, ts)
    assert len(r) > 300
    assert (r["mismatch"] == 0).all(), r[r["mismatch"] > 0]


def test_directional_kinds():
    a, b = np.array([1.0, 2.0]), np.array([5.0, 6.0])
    d = np.array([1, -1])
    assert np.allclose(ef.F("signed", a).directional(d), [1.0, -2.0])
    assert np.allclose(ef.F("pair", a, b).directional(d), [1.0, 6.0])
    assert np.allclose(ef.F("pct1", np.array([0.2, 0.2])).directional(d), [0.2, 0.8])
    assert np.allclose(ef.F("pct100", np.array([30.0, 30.0])).directional(d), [30.0, 70.0])
    assert np.allclose(ef.F("none", a).directional(d), a)


def test_signed_return_mirror(ctx):
    """同一时刻看涨 / 看跌的方向化涨跌互为相反数；顺向 / 逆向极值距离互换。"""
    day = ef.load_day(ctx, DAY)
    fs = ef.day_features(day, ("sec",))
    rows = ef.grid_rows(day, day.t[[3000, 3000]])
    x = ef.sample(fs, rows, np.array([1, -1]))
    assert np.isclose(x["s_ret60_bp"].iloc[0], -x["s_ret60_bp"].iloc[1])
    assert np.isclose(x["s_ext_with15m_bp"].iloc[0], x["s_ext_against15m_bp"].iloc[1])


def test_family_of_covers_all(ctx):
    day = ef.load_day(ctx, DAY)
    fs = ef.day_features(day)
    fams = {ef.family_of(k) for k in fs}
    assert fams <= set(ef.FAMILIES)
    # 只算某一族时名字与全量一致
    for fam in ("sec", "opt", "brute"):
        part = ef.day_features(day, (fam,))
        assert set(part) == {k for k in fs if ef.family_of(k) == fam}


def test_owner_rules_reference_known_features(ctx):
    day = ef.load_day(ctx, DAY)
    names = set(ef.day_features(day))
    files = sorted(Path("config/intrabar").glob("owner_rule_*.yaml"))
    for f in files:
        r = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert r["direction"] == "both"
        assert set(r["executions"]) == {"owner", "taker"}
        for c in r["entry"]["all"]:
            assert c["feature"] == "ef"
            assert c["name"] in names, (f.name, c["name"])
