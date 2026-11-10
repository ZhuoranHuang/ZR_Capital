# ruff: noqa: E501  （中文报告行较长）
"""前视截断检验（2026-11 /tmp 特征研究前视泄漏的教训，见 RESEARCH_LOG「教训 L-001」）。

assert_no_lookahead_frame(fn, df, cuts=10)：对返回 DataFrame / Series（与输入逐行对齐）的特征函数，
在截断点 k 上把输入截到前 k 行重算，前 k 个值必须与全量结果完全一致（浮点按 rtol 1e-9 比较，
NaN 视为相等）。截断点：均匀 cuts 个并偏移 7 行（避免全部落在交易日边界上，日内泄漏只有截断在日内
时才暴露）；索引是时间时另在 cuts 个交易日的开盘后第 1 / 2 / 5 / 15 / 31 行各截一次（开盘相关特征
的泄漏只在开盘头几分钟暴露）。截断检验只能发现「用到了 t 之后的数据」；「入场早于信号可得时刻」（用信号 K
的开盘价入场、9:31 入场却用前 15 分钟涨幅过滤）属于入场时点错误，截断检验发现不了，由 CLAUDE.md 的
检查项与 scripts/run_rule.py 的入场映射约束。

`python -m zr.research.leakcheck`：对 zr/features/ 下全部公开函数逐一归类——逐行特征函数在合成
1 分钟 K（25 个交易日）上跑截断检验；需要真实数据的（期权特征、入场特征）在本地数据上跑；不是
时间序列特征的（加载器、配置、纯数学工具）列出理由。结果写 docs/research/LEAKCHECK_FEATURES.md。
未通过的函数只列出，不修改原函数。
"""

from __future__ import annotations

import datetime as dt
import importlib
import inspect
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("zr.research.leakcheck")
REPO = Path(__file__).resolve().parents[2]
OUT_MD = REPO / "docs" / "research" / "LEAKCHECK_FEATURES.md"
ET = "America/New_York"


# ---------------- 通用截断检验 ----------------


def _frame(x) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(name=x.name if x.name is not None else "value")
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError(f"特征函数须返回 DataFrame 或 Series，得到 {type(x).__name__}")


OPEN_OFFSETS = (1, 2, 5, 15, 31)  # 开盘后第几行截断：开盘相关特征的泄漏只在开盘头几分钟内暴露


def cut_points(n: int, cuts: int, index: pd.Index | None = None) -> list[int]:
    """均匀 cuts 个（偏移 7 行，避开日边界）；索引是时间时，另在 cuts 个交易日（均匀分布）的开盘后
    第 1 / 2 / 5 / 15 / 31 行各截一次（不含首日）。"""
    ks = {min(n - 1, max(2, int((i + 0.5) * n / cuts) + 7)) for i in range(cuts)}
    if isinstance(index, pd.DatetimeIndex) and len(index) == n:
        t = index.tz_convert(ET) if index.tz is not None else index
        day = np.asarray(t.date)
        starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])[1:]  # 不含首日：截到样本开头几行没有意义
        if len(starts):
            pick = starts[np.unique(np.linspace(0, len(starts) - 1, min(cuts, len(starts))).astype(int))]
            ks |= {int(s + o) for s in pick for o in OPEN_OFFSETS if s + o <= n - 1}
    return sorted(ks)


def _same(a: pd.Series, b: pd.Series) -> bool:
    if len(a) != len(b):
        return False
    av, bv = a.to_numpy(), b.to_numpy()
    try:
        af, bf = av.astype(float), bv.astype(float)
    except (TypeError, ValueError):
        na, nb = pd.isna(av), pd.isna(bv)
        return bool(((av == bv) | (na & nb)).all())
    return bool(np.isclose(af, bf, rtol=1e-9, atol=1e-12, equal_nan=True).all())


def check_no_lookahead_frame(fn: Callable, df: pd.DataFrame, cuts: int = 10) -> dict[str, list[int]]:
    """返回 {列名: [不一致的截断点 k, ...]}（空 dict = 通过）。fn(df) 必须与 df 逐行对齐。"""
    full = _frame(fn(df))
    if len(full) != len(df):
        raise ValueError(f"输出 {len(full)} 行 ≠ 输入 {len(df)} 行：不是逐行对齐的特征函数")
    bad: dict[str, list[int]] = {}
    for k in cut_points(len(df), cuts, df.index):
        try:
            part = _frame(fn(df.iloc[:k]))
        except Exception as e:  # noqa: BLE001 - 截断后报错也是「结果依赖未来数据」的一种
            bad.setdefault(f"<截断后报错 {type(e).__name__}>", []).append(k)
            continue
        if len(part) != k:
            raise ValueError(f"截断到 {k} 行后输出 {len(part)} 行")
        for c in full.columns:
            if c not in part.columns or not _same(full[c].iloc[:k], part[c]):
                bad.setdefault(str(c), []).append(k)
    return bad


def assert_no_lookahead_frame(fn: Callable, df: pd.DataFrame, cuts: int = 10) -> None:
    """截到第 k 行重算，前 k 个值必须与全量完全一致；不一致的列与截断点写进 AssertionError。"""
    bad = check_no_lookahead_frame(fn, df, cuts)
    if bad:
        detail = "；".join(f"{c}（k = {ks}）" for c, ks in sorted(bad.items()))
        raise AssertionError(f"前视：{len(bad)} 列在截断后与全量不同：{detail}")


# ---------------- 合成数据 ----------------


def synthetic_bars(n_days: int = 25, seed: int = 0, start: dt.date = dt.date(2025, 1, 6)) -> pd.DataFrame:
    """RTH 1 分钟 K（09:30 至收盘，含半日市），索引 = UTC 开盘时刻；跨日有跳空。"""
    from zr.data.calendar import session_close, trading_days

    days = trading_days(start, start + dt.timedelta(days=3 * n_days))[:n_days]
    rng = np.random.default_rng(seed)
    parts, px = [], 100.0
    for d in days:
        t0 = pd.Timestamp(dt.datetime.combine(d, dt.time(9, 30)), tz=ET)
        t1 = pd.Timestamp(dt.datetime.combine(d, session_close(d)), tz=ET)
        idx = pd.date_range(t0, t1, freq="1min", inclusive="left").tz_convert("UTC")
        px *= np.exp(rng.normal(0, 0.01))  # 隔夜跳空
        c = px * np.exp(np.cumsum(rng.normal(0, 8e-4, len(idx))))
        o = np.r_[px, c[:-1]] * (1 + rng.normal(0, 1e-4, len(idx)))
        hi = np.maximum(o, c) * (1 + rng.uniform(0, 4e-4, len(idx)))
        lo = np.minimum(o, c) * (1 - rng.uniform(0, 4e-4, len(idx)))
        v = rng.integers(100, 5000, len(idx)).astype(float)
        parts.append(pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx))
        px = c[-1]
    b = pd.concat(parts)
    b.index.name = "ts"
    return b


# ---------------- 登记：zr/features 公开函数 ----------------


@dataclass
class Case:
    name: str  # 检验名
    covers: tuple[str, ...]  # 覆盖的公开函数（模块.函数）
    fn: Callable | None = None  # bars → 逐行输出（合成数据）
    custom: Callable | None = None  # () → {列: [不一致的截断点]}（真实数据，自带截断）
    note: str = ""


def _asof(rows_close: pd.DatetimeIndex, bars: pd.DataFrame, close_times: pd.DatetimeIndex) -> pd.DataFrame:
    """把按完成时刻排列的聚合 K 映射回 1 分钟行：每行取「完成时刻 ≤ 该行收盘时刻」的最近一根。"""
    b = bars.reset_index(drop=True).assign(_t=pd.DatetimeIndex(close_times).as_unit("ns"))
    left = pd.DataFrame({"_t": pd.DatetimeIndex(rows_close).as_unit("ns")})
    m = pd.merge_asof(left, b.sort_values("_t"), on="_t", direction="backward")
    return m.drop(columns="_t").set_axis(range(len(m)))


def synthetic_cases() -> list[Case]:
    from zr.features import bars as fb
    from zr.features import calendar_features as cf
    from zr.features import channel as ch
    from zr.features import cross_asset as xa
    from zr.features import features as ff
    from zr.features import indicators as ind
    from zr.features import pullback as pb
    from zr.features import regime as rg
    from zr.features import structure as st

    def hlc(d):
        return d["high"], d["low"], d["close"]

    def cs(d):
        return d["close"]

    def resampled(d):
        b = fb.resample_bars(d, "2min")
        return _asof(d.index + pd.Timedelta(minutes=1), b, b.index + pd.Timedelta(minutes=2)).set_index(d.index)

    def vbars(d):
        b = fb.volume_bars(d, 50_000.0)
        if b.empty:
            return pd.DataFrame(index=d.index, columns=["close", "volume", "n_bars"], dtype=float)
        out = _asof(d.index + pd.Timedelta(minutes=1), b[["close", "volume", "n_bars"]], b["ts_close"] + pd.Timedelta(minutes=1))
        return out.set_index(d.index)

    def vthr(d):
        thr = fb.volume_threshold(d, lookback_days=5)
        return pd.Series(thr.reindex(fb.et_date(d.index)).to_numpy(), index=d.index, name="threshold")

    def vwap(d):
        e = d.tz_convert(ET)
        return pd.Series(ind.session_vwap(e["high"], e["low"], e["close"], e["volume"]).to_numpy(), index=d.index, name="vwap")

    def feats(d):
        return ff.compute_features(d)

    def slopes(d):
        su, sm, sl = ch.three_line_slopes(feats(d), 5)
        return pd.DataFrame({"su": su, "sm": sm, "sl": sl, "par": ch.parallelism(su, sm, sl)})

    def depth_dir(d):
        f = pb.depth_features(d)
        direction = np.where(np.arange(len(d)) % 2 == 0, 1, -1)
        return pb.directional(f, pd.Series(d.index), direction).set_index(d.index)

    cfg = rg.load_config()

    def regime_all(d):
        x = rg.indicators(d, cfg)
        reg = rg.regimes(d, x, cfg, 1)
        s = rg.state_features(d, x, reg)
        r = rg.range_score(x, d, cfg["regime"]["range"], 1)
        return pd.concat([x.add_prefix("x_"), reg.add_prefix("reg_"), s.set_axis(d.index).add_prefix("sf_"), r.rename("range_score")], axis=1)

    ref = synthetic_bars(seed=11)

    def cross(d):
        cut = d.index[-1]
        refs = {r: ref[ref.index <= cut] for r in xa.REFERENCES}
        return xa.compute_cross_asset(d, "MU", refs=refs)

    rel = pd.DatetimeIndex(pd.to_datetime(["2025-01-10 13:30", "2025-01-15 13:30", "2025-01-29 19:00"], utc=True))

    def cal(d):
        return cf.compute_calendar(d.index, releases=rel)

    return [
        Case("indicators.sma", ("indicators.sma",), lambda d: ind.sma(cs(d), 20)),
        Case("indicators.ema", ("indicators.ema",), lambda d: ind.ema(cs(d), 20)),
        Case("indicators.wilder_rsi", ("indicators.wilder_rsi",), lambda d: ind.wilder_rsi(cs(d), 14)),
        Case("indicators.macd", ("indicators.macd",), lambda d: ind.macd(cs(d), 6, 13, 6)),
        Case("indicators.bollinger", ("indicators.bollinger",), lambda d: ind.bollinger(cs(d), 20, 2.0)),
        Case("indicators.true_range", ("indicators.true_range",), lambda d: ind.true_range(*hlc(d))),
        Case("indicators.atr", ("indicators.atr",), lambda d: ind.atr(*hlc(d), 14)),
        Case("indicators.adx", ("indicators.adx",), lambda d: ind.adx(*hlc(d), 14)),
        Case("indicators.kdj", ("indicators.kdj",), lambda d: ind.kdj(*hlc(d))),
        Case("indicators.realized_vol", ("indicators.realized_vol",), lambda d: ind.realized_vol(cs(d), 30)),
        Case("indicators.session_vwap", ("indicators.session_vwap",), vwap, "索引转美东后按日重置"),
        Case("indicators.bar_slope", ("indicators.bar_slope",), lambda d: ind.bar_slope(cs(d), 5)),
        Case("indicators.cross_signal", ("indicators.cross_signal",), lambda d: ind.cross_signal(ind.ema(cs(d), 5), ind.ema(cs(d), 20))),
        Case("features.compute_features", ("features.compute_features",), feats),
        Case("channel.three_line_slopes + parallelism", ("channel.three_line_slopes", "channel.parallelism"), slopes),
        Case("channel.channel_states", ("channel.channel_states",), lambda d: ch.channel_states(d, feats(d))),
        Case("structure.compute_structure", ("structure.compute_structure",), lambda d: st.compute_structure(d, feats(d))),
        Case("structure.macd_state", ("structure.macd_state",), lambda d: st.macd_state(cs(d))),
        Case("structure.tf5_regime", ("structure.tf5_regime",), st.tf5_regime),
        Case("structure.bars_since", ("structure.bars_since",), lambda d: st.bars_since(cs(d).diff() > 0)),
        Case("pullback.depth_features + directional", ("pullback.depth_features", "pullback.directional"), depth_dir),
        Case("regime.indicators / regimes / state_features / range_score", ("regime.indicators", "regime.regimes", "regime.state_features", "regime.range_score"), regime_all,
             "1 分钟 K 当作周期 P = 1。D-037 前未通过（窗口取输入全样本的 bars_per_day，含 t 之后的行）；已改为只用当日之前已完成交易日的根数中位数、首日名义值，见 docs/research/RANGE_SCORE_FIX.md"),
        Case("cross_asset.compute_cross_asset", ("cross_asset.compute_cross_asset",), cross, "参考标的同样截到 t"),
        Case("calendar_features.compute_calendar", ("calendar_features.compute_calendar",), cal, "合成发布时刻；不含到期日（root=None）"),
        Case("bars.resample_bars（按完成时刻映射回分钟行）", ("bars.resample_bars",), resampled),
        Case("bars.volume_bars（按完成时刻映射回分钟行）", ("bars.volume_bars",), vbars),
        Case("bars.volume_threshold（按日期映射回分钟行）", ("bars.volume_threshold",), vthr, "lookback 5 天"),
    ]


def real_cases(root: Path) -> list[Case]:
    """需要本地真实数据的检验（数据不在时跳过）。"""
    from zr.features import entry_features as ef
    from zr.features import option_features as of
    from zr.features.bars import load_bars

    days = [dt.date(2025, 3, 5), dt.date(2025, 9, 10)]  # 不在 D-033 窗口内；MU 有秒级价与期权报价

    def opt_check() -> dict[str, list[int]]:
        bad: dict[str, list[int]] = {}
        for day in days:
            od = of.load_option_day(root, "MU", day)
            stock = load_bars(root, "MU", day, day)
            if od is None or stock.empty:
                raise FileNotFoundError(f"MU {day} 缺期权或正股数据")
            for q in ("quotes", "ohlc"):
                od[q] = od[q].assign(ts=pd.to_datetime(od[q]["ts"], utc=True))

            def fn(s, od=od, day=day):
                cut = s.index[-1] + pd.Timedelta(minutes=1)  # 最后一根的收盘时刻
                o = {
                    "quotes": od["quotes"][od["quotes"]["ts"] <= cut],  # t 时刻快照在 t 可得
                    "ohlc": od["ohlc"][od["ohlc"]["ts"] + pd.Timedelta(minutes=1) <= cut],  # 期权 1 分钟 bar 收盘后可得
                    "oi": od["oi"],  # 开盘前的未平仓量，全天常数
                }
                return of.day_features(s, o, day)

            for c, ks in check_no_lookahead_frame(fn, stock, cuts=10).items():
                bad.setdefault(c, []).extend(f"{day}:{k}" for k in ks)
        return bad

    def entry_check() -> dict[str, list[int]]:
        ctx = ef.Ctx.load(root, "MU")
        bad: dict[str, list[int]] = {}
        for day in days:
            t = [pd.Timestamp(dt.datetime.combine(day, dt.time(h, m, s)), tz=ET).value for h, m, s in ((9, 41, 17), (12, 3, 41), (15, 22, 9))]
            r = ef.truncation_check(ctx, day, np.array(t))
            for row in r[r["mismatch"] > 0].itertuples():
                bad.setdefault(row.feature, []).append(f"{day}×{int(row.mismatch)}")
        return bad

    return [
        Case("option_features.day_features（含 oi_level_features）", ("option_features.day_features", "option_features.oi_level_features", "option_features.compute_option_features"), custom=opt_check,
             note="MU 2025-03-05、2025-09-10 真实数据；报价按快照时刻、期权 bar 按收盘时刻、正股按收盘时刻截断；compute_option_features 逐日调用 day_features"),
        Case("entry_features 全部特征族（truncation_check）", ("entry_features.day_features", "entry_features.bar_features", "entry_features.bar_block", "entry_features.sec_features",
             "entry_features.vol_features", "entry_features.opt_features", "entry_features.optvol_features", "entry_features.cal_features", "entry_features.cross_features",
             "entry_features.brute", "entry_features.svy_features", "entry_features.load_day", "entry_features.truncation_check", "entry_features.sample", "entry_features.grid_rows"), custom=entry_check,
             note="MU 2025-03-05、2025-09-10 真实数据，每天 3 个时刻（秒级），模块自带 truncation_check（load_day(t_cut) 截断全部输入），两个方向逐特征比较"),
    ]


NOT_APPLICABLE = {
    "bars.load_bars": "数据加载器（全量读取，不是特征）",
    "bars.load_extended_bars": "数据加载器",
    "bars.et_date": "时间戳 → 美东日期的工具",
    "bars.daily_volume": "日级汇总（当日全天成交量），输出按日期索引、不逐行；在分钟行上使用必须先 shift(1) 按日（volume_threshold 已这样做并单独检验）",
    "calendar_features.expiry_days": "读取到期日表（静态日历）",
    "calendar_features.release_times": "读取定时宏观发布时刻（静态配置）",
    "channel.load_channel_config": "读取配置",
    "cross_asset.cross_asset_columns": "列名列表",
    "entry_features.family_of": "特征名 → 族名的字符串工具",
    "option_features.bs_price": "Black–Scholes 定价（纯数学）",
    "option_features.implied_vol": "隐含波动率反解（纯数学）",
    "option_features.load_option_day": "数据加载器",
    "option_features.option_days": "列出有数据的日期",
    "option_features.open_window": "按开盘价筛行权价的工具（在 day_features 检验内覆盖）",
    "option_features.strike_spacing": "行权价间距（纯数学）",
    "option_features.years_to_expiry": "到期年限（纯数学）",
    "pullback.load_config": "读取配置",
    "pullback.pullback_columns": "列名列表",
    "range_calibration.main": "离线标定脚本入口（年度打分，不产生分钟特征）",
    "range_calibration.year_scores": "离线标定（按年汇总），不产生分钟特征",
    "regime.load_config": "读取配置",
    "regime.bar_close_minute": "K 线收盘分钟的时间工具",
    "regime.bars_per_day": "整段输入的每日 K 线根数中位数（全样本统计，文档注明不得用于逐行特征）",
    "regime.nominal_bars_per_day": "名义每日根数（时段 ÷ 周期，只看第一根 K）",
    "regime.range_windows": "每行窗口根数（在 range_score 检验内覆盖）",
}


def public_functions() -> list[str]:
    import zr.features as pkg

    out = []
    for m in pkgutil.iter_modules(pkg.__path__):
        mod = importlib.import_module(f"zr.features.{m.name}")
        for n, f in inspect.getmembers(mod, inspect.isfunction):
            if not n.startswith("_") and f.__module__ == mod.__name__:
                out.append(f"{m.name}.{n}")
    return sorted(out)


# ---------------- 报告 ----------------


def run_all(root: Path | None) -> pd.DataFrame:
    df = synthetic_bars()
    rows = []
    for c in synthetic_cases():
        bad = check_no_lookahead_frame(c.fn, df, cuts=10)
        n_cols = len(_frame(c.fn(df)).columns)
        rows.append({"检验": c.name, "数据": "合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点", "列数": n_cols, "bad": bad, "note": c.note})
        log.info("%s：%s", c.name, "通过" if not bad else f"{len(bad)} 列不一致")
    for c in real_cases(root) if root is not None else []:
        try:
            bad = c.custom()
            rows.append({"检验": c.name, "数据": "真实数据", "列数": None, "bad": bad, "note": c.note})
        except FileNotFoundError as e:
            rows.append({"检验": c.name, "数据": "真实数据（缺失，未运行）", "列数": None, "bad": None, "note": f"{c.note}；{e}"})
        log.info("%s：%s", c.name, rows[-1]["bad"])
    return pd.DataFrame(rows)


def report(res: pd.DataFrame) -> str:
    from zr.ops.markdown import df_to_md

    covered = {f for c in synthetic_cases() + real_cases(Path(".")) for f in c.covers}
    pub = public_functions()
    missing = [f for f in pub if f not in covered and f not in NOT_APPLICABLE]
    t = []
    for r in res.itertuples():
        if r.bad is None:
            status, detail = "未运行", ""
        elif not r.bad:
            status, detail = "通过", ""
        else:
            status = f"**未通过（{len(r.bad)} 列）**"
            detail = "；".join(f"`{k}`（{', '.join(map(str, v))}）" for k, v in sorted(r.bad.items()))
        t.append({"检验": r.检验, "数据": r.数据, "输出列数": "" if pd.isna(r.列数) else int(r.列数), "结果": status, "不一致的列（截断点）": detail, "说明": r.note})
    na = [{"函数": k, "理由": v} for k, v in sorted(NOT_APPLICABLE.items())]
    n_fail = sum(1 for r in res.itertuples() if r.bad)
    L = [
        "# zr/features 前视截断检验",
        "",
        "`python -m zr.research.leakcheck` 生成，不手改。检验函数 `zr.research.leakcheck.assert_no_lookahead_frame`：截到第 k 行重算，前 k 个值必须与全量完全一致（rtol 1e-9，NaN 视为相等）。"
        "背景：RESEARCH_LOG 教训 L-001（2026-11 /tmp 特征研究的三处时序错误）。**未通过的函数只列出，未修改原函数。**",
        "",
        f"zr/features 公开函数 {len(pub)} 个：检验覆盖 {len([f for f in pub if f in covered])} 个，不适用 {len([f for f in pub if f in NOT_APPLICABLE])} 个，未归类 {len(missing)} 个"
        + ("（" + "、".join(missing) + "）" if missing else "") + f"。检验项 {len(res)} 个，未通过 {n_fail} 个。",
        "",
        "## 检验结果",
        "",
        df_to_md(pd.DataFrame(t)),
        "",
        "## 不适用（不是逐行时间序列特征）",
        "",
        df_to_md(pd.DataFrame(na)),
        "",
        "## 边界",
        "",
        "- 截断检验只能发现「t 时刻的值用到了 t 之后的数据」。约定：第 i 行的值在第 i 根 K 收盘时可得（包含该根的收盘价）。",
        "- 发现不了入场时点错误：用信号 K 的开盘价入场、9:31 入场却用前 15 分钟涨幅做过滤器——这些值本身没有前视，错在入场早于信息可得时刻（CLAUDE.md「前视检查」与 scripts/run_rule.py 的入场映射负责）。",
        "- 合成数据只覆盖代码路径，不覆盖真实数据里的缺口、半日市以外的异常；截断点 = 均匀 10 个（偏移 7 行落在日内）+ 10 个交易日开盘后第 1 / 2 / 5 / 15 / 31 行。",
        "- 依赖外部数据的分支（compute_calendar 的到期日标记、cross_asset 从磁盘读参考标的）未在合成数据上覆盖。",
        "",
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    from zr.config import get_settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    res = run_all(root if root is not None and Path(root).exists() else None)
    OUT_MD.write_text(report(res), encoding="utf-8")
    print(f"写入 {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
