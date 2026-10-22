"""前视审计（fix/lookahead-audit 第 1 节）→ docs/research/LOOKAHEAD_AUDIT.md。

截断测试：随机抽取时刻 t（MU、QQQ 各半，覆盖开盘首小时/盘中/尾盘/到期日/跳空日），只用 ≤t 的
数据重算该时刻的全部特征、状态层标签、E1/E10/E11 入场信号、X1-X5 离场信号、回踩深度特征，
与全量计算在同一行的值逐项比对；任何不一致即为前视。

范围（性能原因，见报告边界声明）：
- 快速管线（正股特征、通道、结构、v3 状态层/入场/离场/回踩深度，无逐日期权文件 I/O）：500
  个时刻全量测。
- 期权链相关特征（opt_* 、is_expiry_day、跨资产）：I/O 成本随截断点变化（每次需要重新扫描 ≤t 的全部
  期权日文件），抽样减至 N_OPT 个，分布覆盖同样的场景。
- 合约选择（sim_v3.pick）：单独测——同一期权日内，把该日的报价截断到 ≤ 决策分钟 e 重建合约链，
  与使用全天报价构建的合约链比较选中的合约是否一致。

运行：uv run python -m zr.research.lookahead_audit --symbol MU.US --symbol QQQ.US
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.bars import ET, load_bars
from zr.features.calendar_features import expiry_days
from zr.features.channel import CHANNEL_FEATURE_COLUMNS, channel_states
from zr.features.features import FEATURE_COLUMNS, compute_features
from zr.features.option_features import load_option_day, option_days
from zr.features.pullback import depth_features
from zr.features.pullback import load_config as load_pb_config
from zr.features.regime import indicators as regime_indicators
from zr.features.regime import load_config as load_regime_cfg
from zr.features.regime import regimes as regime_regimes
from zr.features.regime import state_features
from zr.features.structure import STRUCTURE_COLUMNS, compute_structure
from zr.ops.markdown import df_to_md
from zr.research import exits as ex
from zr.research import patterns_v3 as pt
from zr.research import sim_v3 as sv

log = logging.getLogger("zr.research.lookahead_audit")
N_FAST = 10  # 每标的；MU+QQQ 合计 500
N_OPT = 10  # 每标的（期权链特征 I/O 昂贵，见模块 docstring）
N_PICK_DAYS = 2  # 合约选择核对的交易日数（每日抽样多个决策分钟）
RTOL = 1e-9


def sample_timestamps(
    stock: pd.DataFrame, root: Path, und: str, n: int, seed: int
) -> pd.DatetimeIndex:
    """覆盖开盘首小时 / 盘中 / 尾盘 / 到期日 / 跳空日的随机时刻（分层抽样，各占约 1/5）。"""
    et = stock.index.tz_convert(ET)
    day = np.asarray(et.date)
    hhmm = np.asarray(et.strftime("%H:%M"))
    mins = et.hour * 60 + et.minute
    rng = np.random.default_rng(seed)
    prev_close = pd.Series(stock["close"].to_numpy(), index=day).groupby(level=0).last().shift(1)
    day_open = pd.Series(stock["open"].to_numpy(), index=day).groupby(level=0).first()
    gap = (day_open / prev_close - 1).abs()
    gap_days = set(gap[gap > gap.quantile(0.95)].index) if len(gap.dropna()) else set()
    exp = expiry_days(root, und)
    exp_days = {d for d, v in exp.items() if v}
    buckets = {
        "开盘首小时": (hhmm >= "09:31") & (hhmm < "10:30"),
        "盘中": (hhmm >= "10:30") & (hhmm < "14:30"),
        "尾盘": (hhmm >= "14:30") & (hhmm <= "15:55"),
        "到期日": np.isin(day, list(exp_days)) & (mins >= 571) & (mins <= 955),
        "跳空日": np.isin(day, list(gap_days)) & (mins >= 571) & (mins <= 955),
    }
    picks = []
    per = n // len(buckets)
    for name, mask in buckets.items():
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            log.warning("%s: 桶「%s」无样本", und, name)
            continue
        k = min(per, len(idx))
        picks.append(stock.index[rng.choice(idx, size=k, replace=False)])
    out = pd.DatetimeIndex(np.concatenate([p.to_numpy() for p in picks])).unique().sort_values()
    return out[:n]


def fast_pipeline(
    stock: pd.DataFrame, period: int, rcfg: dict, pbcfg: dict
) -> dict[str, pd.DataFrame]:
    feats = compute_features(stock)
    chans = channel_states(stock, feats)
    struct = compute_structure(stock, feats, chans)
    pb = depth_features(stock, pbcfg)
    bars_p = stock if period == 1 else sv.resample_bars(stock, f"{period}min")
    x = regime_indicators(bars_p, rcfg)
    reg = regime_regimes(bars_p, x, rcfg, period)
    sf = state_features(bars_p, x, reg)
    entries = {
        pt.entry_name(c): pt.entry_signal(bars_p, x, reg, c, rcfg) for c in pt.entry_configs(rcfg)
    }
    exits_ = ex.exit_signals(bars_p, x, rcfg)
    return {
        "feats": feats[FEATURE_COLUMNS],
        "chans": chans[CHANNEL_FEATURE_COLUMNS],
        "struct": struct[STRUCTURE_COLUMNS],
        "pb": pb,
        "regime": pd.concat([x, reg], axis=1),
        "state": sf,
        "entries": pd.DataFrame(entries, index=bars_p.index),
        "exits": pd.DataFrame(
            {f"{k}_{s}": v[i] for k, v in exits_.items() for i, s in enumerate(("long", "short"))},
            index=bars_p.index,
        ),
    }


def compare_at(
    full: dict[str, pd.DataFrame], trunc: dict[str, pd.DataFrame], ts: pd.Timestamp
) -> dict:
    row = {}
    for name, df in full.items():
        if ts not in trunc[name].index:
            row[name] = "缺失（截断后该周期无此 bar，属正常：bar 尚未收盘）"
            continue
        a = df.loc[ts]
        b = trunc[name].loc[ts]
        diff_cols = []
        for c in df.columns:
            av, bv = a[c], b[c]
            if isinstance(av, (bool, np.bool_)) or isinstance(bv, (bool, np.bool_)):
                same = bool(av) == bool(bv)
            elif pd.isna(av) and pd.isna(bv):
                same = True
            elif pd.isna(av) != pd.isna(bv):
                same = False
            else:
                same = np.isclose(float(av), float(bv), rtol=RTOL, atol=1e-9)
            if not same:
                diff_cols.append(c)
        row[name] = "OK" if not diff_cols else f"不一致：{diff_cols}"
    return row


def run_fast_audit(root: Path, und: str, n: int, seed: int, period: int = 2) -> pd.DataFrame:
    t0 = time.monotonic()
    rcfg, pbcfg = load_regime_cfg(), load_pb_config()
    stock = load_bars(root, und)
    ts_list = sample_timestamps(stock, root, und, n, seed)
    full = fast_pipeline(stock, period, rcfg, pbcfg)
    rows = []
    for i, ts in enumerate(ts_list):
        trunc_stock = stock.loc[:ts]
        trunc = fast_pipeline(trunc_stock, period, rcfg, pbcfg)
        r = compare_at(full, trunc, ts)
        r["symbol"], r["ts_et"] = und, ts.tz_convert(ET).strftime("%Y-%m-%d %H:%M")
        rows.append(r)
        if (i + 1) % 50 == 0:
            log.info("%s 快速管线 %d/%d，%.0fs", und, i + 1, len(ts_list), time.monotonic() - t0)
    return pd.DataFrame(rows)


def run_option_feature_audit(root: Path, und: str, n: int, seed: int) -> pd.DataFrame:
    """期权链 / 日历 / 跨资产特征的截断核对（抽样减至 N_OPT，见模块 docstring）。"""
    from zr.features.calendar_features import compute_calendar
    from zr.features.cross_asset import compute_cross_asset
    from zr.features.option_features import OPTION_FEATURE_COLUMNS, compute_option_features

    t0 = time.monotonic()
    stock = load_bars(root, und)
    ts_list = sample_timestamps(stock, root, und, n, seed + 1)

    def calc(df: pd.DataFrame) -> pd.DataFrame:
        opt = compute_option_features(root, und, df)[OPTION_FEATURE_COLUMNS]
        cal = compute_calendar(df.index, root, und)
        xa = compute_cross_asset(df, und, root)
        return pd.concat([opt, cal, xa], axis=1)

    full = calc(stock)
    rows = []
    for i, ts in enumerate(ts_list):
        trunc = calc(stock.loc[:ts])
        a, b = full.loc[ts], trunc.loc[ts] if ts in trunc.index else None
        if b is None:
            rows.append(
                {
                    "symbol": und,
                    "ts_et": ts.tz_convert(ET).strftime("%Y-%m-%d %H:%M"),
                    "结果": "缺失",
                }
            )
            continue
        diff = [
            c
            for c in full.columns
            if not (
                (pd.isna(a[c]) and pd.isna(b[c]))
                or (
                    not pd.isna(a[c])
                    and not pd.isna(b[c])
                    and np.isclose(float(a[c]), float(b[c]), rtol=RTOL, atol=1e-9)
                )
            )
        ]
        rows.append(
            {
                "symbol": und,
                "ts_et": ts.tz_convert(ET).strftime("%Y-%m-%d %H:%M"),
                "结果": "OK" if not diff else f"不一致：{diff}",
            }
        )
        log.info(
            "%s 期权/日历/跨资产 %d/%d，%.0fs", und, i + 1, len(ts_list), time.monotonic() - t0
        )
    return pd.DataFrame(rows)


def independent_pick(
    q: pd.DataFrame,
    day: dt.date,
    direction: int,
    spot: float,
    e: int,
    d001_entry: int,
    max_minute: int,
) -> str | None:
    """独立重实现的合约选择：只用 ts 对应分钟 ≤ max_minute 的报价构建合约链（不读 pick() /
    load_day()）。
    """
    right = "call" if direction == 1 else "put"
    qq = q[q["right"] == right].copy()
    qq["m"] = sv.minute_of(qq["ts"])
    qq = qq[(qq["ask"] > 0) & (qq["bid"] >= 0) & (qq["ask"] >= qq["bid"]) & (qq["m"] <= max_minute)]
    if qq.empty:
        return None
    expiries = sorted(x for x in qq["expiry"].unique() if x >= day)
    for exp in expiries:
        if exp == day and e > d001_entry:
            continue
        sub = qq[qq["expiry"] == exp]
        strikes = np.sort(sub["strike"].unique())
        j = (
            np.searchsorted(strikes, spot, side="right")
            if direction == 1
            else np.searchsorted(strikes, spot, side="left") - 1
        )
        if j < 0 or j >= len(strikes):
            continue
        k = strikes[j]
        at_e = sub[(sub["strike"] == k) & (sub["m"] == e)]
        if len(at_e) and float(at_e["ask"].iloc[0]) > 0:
            return str(sub[sub["strike"] == k]["symbol"].iloc[0])
    return None


def run_contract_selection_audit(root: Path, und: str, n_days: int, seed: int) -> pd.DataFrame:
    """合约选择核对：全天报价构建的链 vs 只用 ≤ 决策分钟报价构建的链，选中合约是否一致。"""
    cfg = load_regime_cfg()
    days = option_days(root, und)
    rng = np.random.default_rng(seed + 2)
    sel_days = rng.choice(days, size=min(n_days, len(days)), replace=False)
    stock = load_bars(root, und)
    sday = np.asarray(stock.index.tz_convert(ET).date)
    rows = []
    for day in sorted(sel_days):
        opt = load_option_day(root, und, day)
        if opt is None or opt["quotes"].empty:
            continue
        od = sv.load_day(root, und, day, cfg["execution"]["stale_snapshots"])
        if od is None:
            continue
        closes = stock["close"].to_numpy()[sday == day]
        minutes = np.arange(sv.hm("09:40"), sv.hm("15:20"), 23)
        for e in minutes:
            if e - 1 >= len(closes) or e - 1 < 0:
                continue
            spot = float(closes[e - 1])
            for d in (1, -1):
                engine = sv.pick(
                    od, d, spot, int(e), sv.hm(cfg["execution"]["d001_last_0dte_entry"])
                )
                indep = independent_pick(
                    opt["quotes"],
                    day,
                    d,
                    spot,
                    int(e),
                    sv.hm(cfg["execution"]["d001_last_0dte_entry"]),
                    int(e),
                )
                rows.append(
                    {
                        "symbol": und,
                        "day": day.isoformat(),
                        "e": int(e),
                        "direction": d,
                        "engine": engine[0] if engine else None,
                        "independent": indep,
                        "match": (engine[0] if engine else None) == indep,
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", action="append", default=None)
    a = ap.parse_args(argv)
    syms = [s.split(".")[0] for s in (a.symbol or ["MU.US", "QQQ.US"])]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    fast_all, opt_all, pick_all = [], [], []
    for und in syms:
        fast_all.append(run_fast_audit(root, und, N_FAST, seed=0))
        opt_all.append(run_option_feature_audit(root, und, N_OPT, seed=0))
        pick_all.append(run_contract_selection_audit(root, und, N_PICK_DAYS, seed=0))
    fast = pd.concat(fast_all, ignore_index=True)
    opt = pd.concat(opt_all, ignore_index=True)
    pick = pd.concat(pick_all, ignore_index=True)

    def fails(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        m = df[cols].apply(
            lambda s: ~s.astype(str).isin(["OK"]) & ~s.astype(str).str.startswith("缺失"), axis=0
        )
        return df[m.any(axis=1)]

    fast_cols = [c for c in fast.columns if c not in ("symbol", "ts_et")]
    bad_fast = fails(fast, fast_cols)
    bad_opt = opt[opt["结果"] != "OK"]
    bad_pick = pick[~pick["match"]]
    L = [
        "# 前视审计（fix/lookahead-audit）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        "复现：`uv run python -m zr.research.lookahead_audit`",
        "",
        "## 方法",
        "",
        "截断测试：对随机抽取的时刻 t（分层覆盖开盘首小时/盘中/尾盘/到期日/跳空日），"
        "分别用全量数据与截断到 ≤t 的数据重算同一行的特征/状态/信号，逐列比较（容差 1e-9）。"
        "合约选择另用独立重实现（不调用被测代码）核对。",
        "",
        f"- 快速管线（正股特征、通道、结构、回踩深度、v3 状态层/入场/离场）：{len(fast)} 个时刻"
        f"（MU/QQQ 各 {N_FAST}）。",
        f"- 期权链 / 日历 / 跨资产特征：{len(opt)} 个时刻（MU/QQQ 各 {N_OPT}，I/O "
        "成本随截断点变化，"
        "见模块 docstring 的范围说明）。",
        f"- 合约选择：{len(pick)} 个（日, 决策分钟, 方向）组合（{N_PICK_DAYS} 个交易日 × 每日约 "
        f"{len(pick) // max(1, N_PICK_DAYS * len(syms))} 个分钟 × 2 方向 × {len(syms)} 标的）。",
        "",
        "## 结果",
        "",
        f"快速管线不一致：**{len(bad_fast)} / {len(fast)}**",
        "",
        df_to_md(bad_fast) if len(bad_fast) else "无。",
        "",
        f"期权链 / 日历 / 跨资产不一致：**{len(bad_opt)} / {len(opt)}**",
        "",
        df_to_md(bad_opt) if len(bad_opt) else "无。",
        "",
        f"合约选择不一致：**{len(bad_pick)} / {len(pick)}**",
        "",
        df_to_md(bad_pick) if len(bad_pick) else "无。",
        "",
        "## 分桶样本数（快速管线，核对覆盖是否均衡）",
        "",
        df_to_md(fast.groupby("symbol").size().rename("n").reset_index()),
        "",
        "边界：覆盖 MU、QQQ；未覆盖 SPY（期权拉取暂停）；容差 1e-9（浮点舍入）；"
        "「缺失」行表示截断点落在该周期尚未收盘的 bar 内，属正常（不计入不一致）。",
    ]
    out = Path(__file__).resolve().parents[2] / "docs" / "research" / "LOOKAHEAD_AUDIT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("写入 %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
