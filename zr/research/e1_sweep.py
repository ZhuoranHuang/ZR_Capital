"""E1 回踩深度遍历（D-022，R-029）→ docs/research/E1_PULLBACK_SWEEP_<UND>.md。

网格见 config/e1_sweep.yaml：深度（pctb 6 档 / eps 6 档，二选一）× 形态 n ∈ {2, 3} × 布林周期 {10,
20, 30}
× 通道 {A, F}。信号定义见 zr/features/pullback.py。

- 事件表（正股标签，全部正股日）：每组 × 方向的事件数 / 日、ever_up_0.5_10、MFE_10 / MAE_10、
  10 分钟方向化收益均值与 t。
- 回放（真实期权，期权日）：每组 × 方向 × 持仓上限 {10, 30}，决策分钟 = 信号 bar + 1，合约与出场同
  zr/research/outcomes.py；随机对照 = 同日同笔数、同方向同持仓上限的随机决策分钟。
- 随机对照两种：全天随机分钟（real_pctile）与同日 ±15 分钟时段匹配（real_pctile_local）。
- 多重比较：预登记试验数 = 回放行数；Bonferroni 双侧 5% 的 |t| 门槛由代码算出。
运行：uv run python -m zr.research.e1_sweep --symbol MU.US
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from zr.config import get_settings
from zr.features.bars import ET, load_bars
from zr.features.channel import channel_states
from zr.features.features import compute_features
from zr.features.pullback import PullbackBase, load_config
from zr.ops.markdown import df_to_md
from zr.research import outcomes as oc
from zr.research.labels import rolling_sigma
from zr.research.labels_v2 import mfe_mae
from zr.research.replay_rule import Lookup, random_control, random_control_local, stats

log = logging.getLogger("zr.research.e1_sweep")
START, END = dt.date(2023, 10, 2), dt.date(2026, 10, 21)
H = 10


def grid(cfg: dict) -> list[dict]:
    depths = [("pctb", x) for x in cfg["pctb"]] + [("eps", x) for x in cfg["eps"]]
    return [
        {"channel": ch, "bb": p, "n": n, "kind": k, "value": v}
        for ch, p, n, (k, v) in product(cfg["channels"], cfg["bb_period"], cfg["pattern_n"], depths)
    ]


def combo_name(g: dict) -> str:
    v = f"{g['value']:.2f}" if g["kind"] == "pctb" else f"{g['value'] * 100:.2f}%"
    return f"{g['channel']}|BB{g['bb']}|n{g['n']}|{g['kind']}≤{v}"


def trial_counts(cfg: dict) -> dict:
    n_def = len(grid(cfg))
    return {
        "definitions": n_def,
        "event_cells": n_def * 2,
        "replay_rows": n_def * 2 * len(oc.CAPS),
    }


def bonferroni_t(m: int, alpha: float = 0.05) -> float:
    return float(norm.ppf(1 - alpha / (2 * m)))


def bar_labels(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """每根 bar 的多 / 空两版 H=10 正股标签（窗口 t+1..t+H，同日）。"""
    ts = pd.Series(df.index)
    sigma = rolling_sigma(df["close"], 60).to_numpy() * np.sqrt(H)
    day = np.asarray(df.index.tz_convert(ET).date)
    c = df["close"].to_numpy()
    fwd = np.full(len(c), np.nan)
    same = np.r_[day[H:] == day[:-H], np.zeros(H, dtype=bool)]
    fwd[: len(c) - H] = c[H:] / c[:-H] - 1
    fwd[~same] = np.nan
    out = {}
    for d in (1, -1):
        mfe, mae = mfe_mae(df, ts, np.full(len(df), d), H)
        ever = np.where(np.isfinite(mfe) & np.isfinite(sigma), (mfe >= 0.5 * sigma), np.nan)
        out[d] = pd.DataFrame({"mfe": mfe, "mae": mae, "ever_up": ever, "ret": fwd * d})
    return out


def event_row(lab: pd.DataFrame, pos: np.ndarray, n_days: int) -> dict:
    x = lab.iloc[pos]
    r = x["ret"].dropna()
    t = r.mean() / r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 2 and r.std() > 0 else np.nan
    return {
        "events": len(pos),
        "per_day": round(len(pos) / max(1, n_days), 2),
        "ever_up_0.5_10": round(float(x["ever_up"].mean()), 3) if len(x) else np.nan,
        "mfe_10_bp": round(float(x["mfe"].mean() * 1e4), 2) if len(x) else np.nan,
        "mae_10_bp": round(float(x["mae"].mean() * 1e4), 2) if len(x) else np.nan,
        "ret_10_bp": round(float(r.mean() * 1e4), 2) if len(r) else np.nan,
        "t_ret_10": round(float(t), 2) if np.isfinite(t) else np.nan,
    }


def run(
    root: Path, und: str, start: dt.date = START, end: dt.date = END, n_random: int = 1000
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    t0 = time.monotonic()
    cfg = load_config()
    df = load_bars(root, und, start, end)
    feats = compute_features(df)
    chans = channel_states(df, feats)
    base = PullbackBase(df, cfg["bb_period"], cfg["bb_k"], max(cfg["pattern_n"]))
    labs = bar_labels(df)
    day = np.asarray(df.index.tz_convert(ET).date)
    minute = oc.minute_index(df.index) + 1  # 决策分钟
    n_stock_days = len(set(day))
    out = oc.build(root, und, start, end)
    lk = Lookup(out) if not out.empty else None
    opt_days = set(lk.days) if lk else set()
    elig = None
    if lk:
        elig = np.zeros((len(lk.days), oc.MINUTES), dtype=bool)
        elig[:, oc.FIRST_ENTRY : oc.LAST_ENTRY + 1] = True
    in_window = (minute >= oc.FIRST_ENTRY) & (minute <= oc.LAST_ENTRY)
    on_opt = np.array([d in opt_days for d in day])
    ev_rows, rp_rows = [], []
    for g in grid(cfg):
        sig = base.signal(chans[f"chan_{g['channel']}"], g["bb"], g["kind"], g["value"], g["n"])
        name = combo_name(g)
        for d in (1, -1):
            pos = np.flatnonzero(sig == d)
            side = "long" if d == 1 else "short"
            ev_rows.append(
                {"combo": name, **g, "side": side, **event_row(labs[d], pos, n_stock_days)}
            )
            if lk is None:
                continue
            p = pos[in_window[pos] & on_opt[pos]]
            e = pd.DataFrame({"day": day[p], "minute": minute[p]})
            for cap in oc.CAPS:
                net = lk.get(e["day"], e["minute"], d, cap)
                t = e.assign(net=net).dropna(subset=["net"])
                s = {"combo": name, **g, "side": side, "cap": cap, "events_opt_days": len(e)}
                s.update(stats(t["net"], t["day"]))
                if len(t) > 2 and t["net"].std() > 0:
                    s["t_net"] = round(
                        float(t["net"].mean() / t["net"].std(ddof=1) * np.sqrt(len(t))), 2
                    )
                if len(t) >= 5:
                    s.update(random_control(lk, t, d, cap, elig, n_random))
                    s.update(random_control_local(lk, t, d, cap, n=n_random))
                rp_rows.append(s)
    evt, rp = pd.DataFrame(ev_rows), pd.DataFrame(rp_rows)
    log.info("遍历完成 %d 组，%.0fs", len(grid(cfg)), time.monotonic() - t0)
    return report(und, cfg, evt, rp, df, lk, n_random, time.monotonic() - t0), evt, rp


def marginals(rp: pd.DataFrame, cap: int) -> pd.DataFrame:
    r = rp[(rp["cap"] == cap) & (rp["n"] >= 5)]
    rows = []
    for dim in ("channel", "bb", "n", "kind", "side"):
        for v, g in r.groupby(dim):
            rows.append(
                {
                    "dim": dim,
                    "value": v,
                    "rows": len(g),
                    "median_mean_net": round(float(g["mean_net"].median()), 2),
                    "best_mean_net": round(float(g["mean_net"].max()), 2),
                    "rows_mean_net>0": int((g["mean_net"] > 0).sum()),
                    "median_real_pctile": round(float(g["real_pctile"].median()), 1),
                    "median_pctile_local": round(float(g["real_pctile_local"].median()), 1),
                }
            )
    for kind in ("pctb", "eps"):
        for v, g in r[r["kind"] == kind].groupby("value"):
            rows.append(
                {
                    "dim": f"depth:{kind}",
                    "value": v,
                    "rows": len(g),
                    "median_mean_net": round(float(g["mean_net"].median()), 2),
                    "best_mean_net": round(float(g["mean_net"].max()), 2),
                    "rows_mean_net>0": int((g["mean_net"] > 0).sum()),
                    "median_real_pctile": round(float(g["real_pctile"].median()), 1),
                    "median_pctile_local": round(float(g["real_pctile_local"].median()), 1),
                }
            )
    return pd.DataFrame(rows)


def summary(rp: pd.DataFrame, cfg: dict) -> dict:
    """供完整报告引用的要点（全部由表算出）。"""
    tc = trial_counts(cfg)
    tb = bonferroni_t(tc["replay_rows"])
    r = rp[rp["n"] >= 5] if len(rp) else rp
    if r.empty:
        return {**tc, "t_bonf": round(tb, 2), "rows_n_ge_5": 0}
    return {
        **tc,
        "t_bonf": round(tb, 2),
        "rows_n_ge_5": len(r),
        "rows_mean_net_gt_0": int((r["mean_net"] > 0).sum()),
        "rows_t_gt_bonf": int((r.get("t_net", pd.Series(dtype=float)) > tb).sum()),
        "rows_pctile_ge_95": int((r["real_pctile"] >= 95).sum()),
        "rows_pctile_local_ge_95": int((r["real_pctile_local"] >= 95).sum()),
        "best_mean_net": round(float(r["mean_net"].max()), 2),
        "best_combo": " / ".join(
            r.sort_values("mean_net", ascending=False).iloc[0][["combo", "side", "cap"]].astype(str)
        ),
        "median_mean_net": round(float(r["mean_net"].median()), 2),
    }


def report(und, cfg, evt, rp, df, lk, n_random, secs) -> str:
    tc = trial_counts(cfg)
    tb = bonferroni_t(tc["replay_rows"])
    days = sorted(set(np.asarray(df.index.tz_convert(ET).date)))
    bound_ev = (
        f"标的 {und} 正股 1 分钟 K，{days[0]}～{days[-1]} 共 {len(days)} 个交易日（RTH、"
        "交易日历过滤）；"
        "标签窗口 t+1..t+10 同日；ever_up = MFE_10 ≥ 0.5σ_10（σ = 60 分钟滚动 1 分钟收益 std）"
    )
    bound_rp = (
        f"标的 {und}；期权日 {len(lk.days) if lk else 0}"
        + (f"（{min(lk.days)}～{max(lk.days)}）" if lk else "")
        + "；决策分钟 = 信号 bar + 1，用该分钟 NBBO 买 ask、卖 bid；"
        "最近可用到期 DTE≤5、最近一档略虚值、"
        "2 张、+20% 减半、中轨结构止损、15:55 强平；费用 D-013；未覆盖：无期权数据的月份、DTE>5"
    )
    L = [
        f"# E1 回踩深度遍历（D-022，R-029）· {und}",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 耗时 {secs:.0f}s · "
        "网格 config/e1_sweep.yaml · 复现 `uv run python -m zr.research.e1_sweep --symbol "
        f"{und}.US`",
        "",
        f"试验计数（预登记）：定义 {tc['definitions']} 个（深度 12 × 形态 2 × 布林周期 3 × 通道 2）"
        "，"
        f"事件表 {tc['event_cells']} 格（× 方向），回放 {tc['replay_rows']} 行（× 方向 × 持仓上限）"
        "。"
        f"Bonferroni 双侧 5% 门槛 |t| > {tb:.2f}。",
        "",
    ]
    if len(rp):
        s = summary(rp, cfg)
        L += [
            "## 要点",
            "",
            df_to_md(pd.DataFrame([s])),
            "",
            f"边界：{bound_rp}。",
            "",
        ]
        for cap in oc.CAPS:
            L += [
                f"## 回放边际（持仓上限 {cap}，n ≥ 5 的行；按维度取中位数）",
                "",
                df_to_md(marginals(rp, cap)),
                "",
            ]
        top = rp[rp["n"] >= 30].sort_values("mean_net", ascending=False).head(15)
        L += [
            "## 回放前 15 行（n ≥ 30，按 mean_net）",
            "",
            df_to_md(top),
            "",
            f"边界：{bound_rp}。",
            "",
        ]
    L += [
        "## 事件表（全部正股日，按组 × 方向）",
        "",
        df_to_md(evt),
        "",
        f"边界：{bound_ev}。",
        "",
    ]
    if len(rp):
        L += [
            f"## 回放全表（随机对照 {n_random} 次）",
            "",
            df_to_md(rp),
            "",
            f"边界：{bound_rp}。",
            "",
        ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MU.US")
    ap.add_argument("--n-random", type=int, default=1000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    und = a.symbol.split(".")[0]
    text, _, _ = run(get_settings().zr_data_root, und, n_random=a.n_random)
    out = Path(__file__).resolve().parents[2] / "docs" / "research" / f"E1_PULLBACK_SWEEP_{und}.md"
    out.write_text(text, encoding="utf-8")
    log.info("写入 %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
