# ruff: noqa: E501  （中文报告行较长）
"""R-056：MU 噪声区突破 120 分钟变体的留出期一次性检验（预登记见 RESEARCH_LOG R-056，运行前提交）。

规则（夜间研究 PREREG 阶段 5b、开发期选出的变体，参数不改）：σ = 前 14 个交易日同一分钟 |C / 当日开盘 − 1| 的均值（按日 shift）；
UB = max(当日开盘, 前收) × (1 + 1.0 σ)，LB = min(当日开盘, 前收) × (1 − 1.0 σ)；VWAP = RTH 典型价成交量加权。
判定：10:00 起每 30 分钟至 15:30，该分钟 K 收盘 > UB 且 > VWAP → 多；< LB 且 < VWAP → 空。空仓时按下一分钟开盘 ± 1 tick 入场，
固定持有 120 分钟（第 120 根开盘出场），同一时间一个仓位。
账户层（同夜间任务阶段 6）：zr.sim.stock 逐笔调用（薄封装）——满仓复利（股数 = floor(当前权益 ÷ 含滑点入场价)，经
patch zr.sim.stock.size_shares；引擎默认 50% 名义上限）、整数股、D-013 费用、每边 1 tick、单日已实现亏损 ≥ 10% × 当日期初权益后不再开仓
（引擎不支持，封装实现）、15:50 清仓（引擎 flat）。
运行：python -m zr.research.r056_mu_trend_holdout --run（留出期只允许一次，落标记）→ --report。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import zr.sim.stock as zs
from zr.config import get_settings
from zr.research.timing.data import ACCT, FX, HOLD_A, HOLD_B_FROM, TICK, round_trip_fees, split_of
from zr.research.timing.labels import minute_grid
from zr.research.timing.stats import fmt_ci, pooled
from zr.research.timing.trend import decision_mask, noise_signal, random_dirs
from zr.sim.engine import Account

REPO = Path(__file__).resolve().parents[2]
OUT_MD = REPO / "docs" / "research" / "R056_MU_TREND_HOLDOUT.md"
UND, VM, STEP, HOLD_MIN = "MU", 1.0, 30, 120
SESSION = {"first_entry": "10:01", "last_entry": "15:49", "flat": "15:50"}
DAILY_LOSS = 0.10
N_RAND = 1000


def store(root: Path) -> Path:
    p = Path(root) / "research" / "r056"
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def full_position():
    """满仓：股数 = floor(当前权益 ÷ 入场价)，不超过 100,000 股。"""
    orig = zs.size_shares
    zs.size_shares = lambda account, entry_px, stop_px, **kw: int(
        min(np.floor(account.equity_usd / entry_px + 1e-9), 100_000)
    )
    try:
        yield
    finally:
        zs.size_shares = orig


def complete_days(g: pd.DataFrame) -> set:
    """当日 K 线到达 15:50（或提前收盘日的收盘）才计入；最后一个不完整交易日剔除。"""
    real = g[~g["filled"]]
    last = real.groupby("day")["mod"].max()
    close = g.groupby("day")["mod"].max()
    return {d for d in last.index if last[d] >= min(949, close[d])}


def simulate(
    g: pd.DataFrame,
    sig: np.ndarray,
    days: list,
    acct: str,
    long_only: bool = False,
    delay: int = 0,
    cost_mult: int = 1,
) -> pd.DataFrame:
    """逐日、逐信号调用 zr.sim.stock.simulate_day（单信号），维护权益、单一仓位与单日亏损上限。"""
    equity = ACCT[acct]
    dec = decision_mask(g["mod"].to_numpy(), STEP)
    rows = []
    pol = zs.StockPolicy(slippage_ticks=1.0 * cost_mult, tick=TICK)
    rule = {"kind": "time", "bars": HOLD_MIN}
    with full_position():
        for day in days:
            x = g[g["day"] == day]
            s0 = int(x.index[0])
            bars = pd.DataFrame(
                {
                    "ts_ns": pd.DatetimeIndex(x["ts"]).as_unit("ns").asi8,
                    **{c: x[c].to_numpy(float) for c in ("open", "high", "low", "close")},
                }
            )
            eq0, free = equity, 0
            for t in np.flatnonzero(dec[s0 : s0 + len(x)]):
                d = int(sig[s0 + t])
                if d == 0 or (long_only and d < 0):
                    continue
                i = t + 1 + delay
                if i < free or i >= len(x):
                    continue
                if eq0 - equity >= DAILY_LOSS * eq0:
                    break
                res = zs.simulate_day(
                    bars, [(i, d)], Account(equity_usd=equity), SESSION, pol, exit_rule=rule
                )
                if not res.trades:
                    continue
                tr = res.trades[0]
                fees = tr.fees_usd * cost_mult
                pnl = tr.pnl_usd - tr.fees_usd * (cost_mult - 1)
                x_row = int(np.searchsorted(bars["ts_ns"].to_numpy(), tr.exit_ts_ns, side="left"))
                rows.append(
                    {
                        "day": day,
                        "dir": d,
                        "entry_ts": tr.entry_ts_ns,
                        "exit_ts": tr.exit_ts_ns,
                        "entry": tr.entry_px,
                        "exit": tr.exit_px,
                        "qty": tr.qty,
                        "fees": fees,
                        "pnl": pnl,
                        "equity_before": equity,
                        "reason": tr.reason,
                        "hold": (tr.exit_ts_ns - tr.entry_ts_ns) / 6e10,
                        "raw_entry": tr.entry_px - d * TICK * cost_mult,
                        "raw_exit": tr.exit_px + d * TICK * cost_mult,
                        "bh_ret": (
                            bars["close"].iloc[min(x_row, len(x)) - 1]
                            if tr.exit_ts_ns > bars["ts_ns"].iloc[-1]
                            else bars["open"].iloc[x_row]
                        )
                        / bars["open"].iloc[i]
                        - 1,
                    }
                )
                equity += pnl
                free = (
                    x_row
                    if tr.exit_ts_ns == bars["ts_ns"].iloc[min(x_row, len(x) - 1)]
                    else x_row + 1
                )
    T = pd.DataFrame(rows)
    if len(T):
        T["ret_pct"] = T["pnl"] / (T["qty"] * T["entry"]) * 100
    return T


def metrics(T: pd.DataFrame, days: list, acct: str, seed: int = 0) -> dict:
    if T.empty:
        return {"n": 0}
    dd = T["day"].to_numpy()
    st = pooled(T["ret_pct"].to_numpy(), dd)
    daily = T.groupby("day")["pnl"].sum().reindex(days, fill_value=0.0)
    t_day = (
        float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))))
        if daily.std(ddof=1) > 0
        else np.nan
    )
    eq = ACCT[acct] + np.cumsum(T["pnl"].to_numpy())
    curve = np.r_[ACCT[acct], eq]
    peak = np.maximum.accumulate(curve)
    mdd = float(((curve - peak) / peak).min())
    # 随机对照：保持时刻、股数、成本，只改方向（成交价按方向重新加减滑点、费用按方向重算）
    d = T["dir"].to_numpy()
    out = {
        "n": len(T),
        "days": len(set(dd)),
        "net": st,
        "sum_usd": float(T["pnl"].sum()),
        "t_day": t_day,
        "n_days_all": len(days),
        "mdd": mdd,
        "end_equity": float(curve[-1]),
        "long_n": int((d > 0).sum()),
        "short_n": int((d < 0).sum()),
    }
    for mode in ("half", "permute"):
        R = random_dirs(d, N_RAND, mode, seed=seed)
        e_r = T["raw_entry"].to_numpy()[None, :] + R * TICK
        x_r = T["raw_exit"].to_numpy()[None, :] - R * TICK
        q = T["qty"].to_numpy()[None, :]
        f_long = round_trip_fees(
            np.broadcast_to(q, R.shape).ravel(), e_r.ravel(), x_r.ravel(), np.ones(R.size)
        ).reshape(R.shape)
        f_short = round_trip_fees(
            np.broadcast_to(q, R.shape).ravel(), e_r.ravel(), x_r.ravel(), -np.ones(R.size)
        ).reshape(R.shape)
        pnl_r = q * R * (x_r - e_r) - np.where(R > 0, f_long, f_short)
        m = (pnl_r / (q * e_r) * 100).mean(1)
        out[f"rand_{mode}_mean"], out[f"rand_{mode}_95"] = (
            float(m.mean()),
            float(np.percentile(m, 95)),
        )
    for nm, s in (("long", d > 0), ("short", d < 0)):
        out[nm] = pooled(T["ret_pct"].to_numpy()[s], dd[s], n_boot=1000)
    out["bh_hold"] = pooled(T["bh_ret"].to_numpy() * 100, dd, n_boot=1000)
    out["hold_q"] = np.percentile(T["hold"], [10, 25, 50, 75, 90]).tolist()
    out["reasons"] = T["reason"].value_counts().to_dict()
    mon = (
        T.assign(m=pd.to_datetime(T["day"].astype(str)).dt.strftime("%Y-%m"))
        .groupby("m")
        .agg(n=("pnl", "size"), usd=("pnl", "sum"), pct=("ret_pct", "mean"))
    )
    out["monthly"] = mon.round(3).reset_index().to_dict("records")
    wk = (
        T.assign(w=pd.to_datetime(T["day"].astype(str)).dt.strftime("%G-W%V"))
        .groupby("w")["pnl"]
        .sum()
    )
    if acct == "A":
        wk = wk * FX
    out["weekly"] = {
        "unit": "人民币" if acct == "A" else "美元",
        "n": len(wk),
        "median": float(wk.median()),
        "best": float(wk.max()),
        "worst": float(wk.min()),
        "loss_share": float((wk < 0).mean()),
    }
    return out


def run(root: Path, force: bool = False) -> None:
    mark = store(root) / "holdout_done.json"
    if mark.exists() and not force:
        raise SystemExit(f"留出期已运行过一次（{mark}），按预登记不得重跑")
    g = minute_grid(root, UND, dt.date(2100, 1, 1), allow_holdout=True)
    S = noise_signal(g, VM)
    sig = S["sig"].to_numpy()
    ok = complete_days(g)
    split = pd.Series(split_of(g["day"].to_numpy()), index=g.index)
    per = {}
    for sp, keys in (("dev", ("dev",)), ("val", ("val",)), ("hold", ("hold_a", "hold_b"))):
        per[sp] = sorted(
            d for d in set(g.loc[split.isin(keys), "day"]) if d in ok
        )  # 留出期两块连续复利，事后按日期分块
    excluded = sorted(set(g.loc[split.isin(["hold_a", "hold_b"]), "day"]) - ok)
    out = []
    for sp, days in per.items():
        for acct in ("A", "B"):
            for ver, kw in (
                ("both", {}),
                ("long", {"long_only": True}),
                ("cost2", {"cost_mult": 2}),
                ("delay60", {"delay": 1}),
            ):
                T = simulate(g, sig, days, acct, **kw)
                T = T.assign(split=sp, acct=acct, ver=ver)
                out.append(T)
    hold = pd.concat(out, ignore_index=True)
    hold.assign(day=hold["day"].astype(str)).to_parquet(store(root) / "trades.parquet", index=False)
    blocks = {
        "hold_a": [d for d in per["hold"] if d <= HOLD_A[1]],
        "hold_b": [d for d in per["hold"] if d >= HOLD_B_FROM],
    }
    meta = {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "days": {k: [str(v[0]), str(v[-1]), len(v)] for k, v in {**per, **blocks}.items()},
        "excluded_incomplete": [str(d) for d in excluded],
    }
    (store(root) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    mark.write_text(json.dumps(meta, ensure_ascii=False))
    print(json.dumps(meta, ensure_ascii=False))


def report(root: Path) -> str:
    from zr.ops.markdown import df_to_md

    T = pd.read_parquet(store(root) / "trades.parquet")
    T["day"] = pd.to_datetime(T["day"]).dt.date
    meta = json.loads((store(root) / "meta.json").read_text())
    g = minute_grid(root, UND, dt.date(2100, 1, 1), allow_holdout=True)
    ok = complete_days(g)
    split = split_of(g["day"].to_numpy())
    g_days = {
        sp: sorted(d for d in set(g["day"].to_numpy()[np.isin(split, keys)]) if d in ok)
        for sp, keys in (
            ("dev", ("dev",)),
            ("val", ("val",)),
            ("hold_a", ("hold_a",)),
            ("hold_b", ("hold_b",)),
            ("hold", ("hold_a", "hold_b")),
        )
    }
    res = {}
    for acct in ("A", "B"):
        for ver in ("both", "long", "cost2", "delay60"):
            for sp in ("dev", "val", "hold"):
                x = T[(T["acct"] == acct) & (T["ver"] == ver) & (T["split"] == sp)].reset_index(
                    drop=True
                )
                m = metrics(x, g_days[sp], acct)
                if sp == "hold":
                    for blk in ("hold_a", "hold_b"):
                        xb = x[np.isin(x["day"], g_days[blk])].reset_index(drop=True)
                        mb = metrics(xb, g_days[blk], acct)
                        if mb.get("n"):
                            mb["mdd"] = np.nan  # 分块不单独计算回撤（复利是连续的）
                        res[(acct, ver, blk)] = mb
                    m["sum_a"] = res[(acct, ver, "hold_a")].get("sum_usd", 0.0)
                    m["sum_b"] = res[(acct, ver, "hold_b")].get("sum_usd", 0.0)
                res[(acct, ver, sp)] = m

    def verdict(acct: str, ver: str) -> tuple[dict, bool]:
        m = res[(acct, ver, "hold")]
        if not m.get("n"):
            return {"交易笔数 ≥ 100": "0 笔"}, False
        c2, dl = res[(acct, "cost2", "hold")], res[(acct, "delay60", "hold")]
        crit = {
            "扣费后每笔均值 > 0": m["net"]["mean"] > 0,
            "按日汇总盈亏 t ≥ 2.6": m["t_day"] >= 2.6,
            "交易 ≥ 100 笔": m["n"] >= 100,
            "留出期两块都为正": m["sum_a"] > 0 and m["sum_b"] > 0,
            "成本翻倍仍 > 0": bool(c2.get("n")) and c2["net"]["mean"] > 0,
            "入场再延迟 60 秒仍 > 0": bool(dl.get("n")) and dl["net"]["mean"] > 0,
            "最大回撤 < 账户 25%": m["mdd"] > -0.25,
            "超过随机对照 (ii) 95 分位": m["net"]["mean"] > m["rand_permute_95"],
        }
        return crit, all(crit.values())

    L = [
        "# R-056：MU 噪声区突破 120 分钟变体的留出期一次性检验",
        "",
        "`python -m zr.research.r056_mu_trend_holdout`。预登记 RESEARCH_LOG R-056（运行前提交）。规则与参数照夜间研究 PREREG 阶段 5b 原样，不改。",
        f"留出期只运行一次（{meta['run_at']}）。数据：MU 长桥 1 分钟 K（RTH）；开发期 {meta['days']['dev'][0]}～{meta['days']['dev'][1]}（{meta['days']['dev'][2]} 天）、验证期 {meta['days']['val'][0]}～{meta['days']['val'][1]}（{meta['days']['val'][2]} 天）、"
        f"留出期第一块 {meta['days']['hold_a'][0]}～{meta['days']['hold_a'][1]}（{meta['days']['hold_a'][2]} 天）、第二块 {meta['days']['hold_b'][0]}～{meta['days']['hold_b'][1]}（{meta['days']['hold_b'][2]} 天）；"
        f"按预登记剔除不完整交易日：{meta['excluded_incomplete'] or '无'}。",
        "",
        "**如实记录**：(1) 该变体**未通过验证期门槛**（夜间研究：账户 B 扣费后每笔 +0.2965%，随机方向 95 分位 +0.3117%）；(2) 它是开发期 20 个变体中按账户 B 扣费后盈亏选出的一个（多重选择）；"
        "(3) 本次检验计入「留出期最多 5 次」的额度（第 1 次）。持仓上限放宽到 120 分钟经负责人同意（只对本分支）。",
        "",
    ]
    for acct in ("B", "A"):
        crit, passed = verdict(acct, "both")
        n = res[(acct, "both", "hold")].get("n", 0)
        label = "信号层通过" if acct == "B" else "1 万人民币账户可行"
        concl = "样本不足，不能判定" if n < 100 else ("**通过**" if passed else "**未通过**")
        L += [
            f"## 账户 {acct}（{'10 万美元' if acct == 'B' else '1 万人民币 ≈ 1,498.69 美元'}）：{label} → {concl}",
            "",
            "| 标准 | 结果 |",
            "|---|---|",
            *[
                f"| {k} | {'满足' if v is True else ('不满足' if v is False else v)} |"
                for k, v in crit.items()
            ],
            "",
        ]
        rows = []
        for ver, vname in (
            ("both", "多空都做"),
            ("long", "只做多"),
            ("cost2", "成本翻倍（多空）"),
            ("delay60", "入场延迟 60 秒（多空）"),
        ):
            for sp, spn in (
                ("dev", "开发期"),
                ("val", "验证期"),
                ("hold_a", "留出 06–08"),
                ("hold_b", "留出 09 以后"),
                ("hold", "留出期合计"),
            ):
                m = res[(acct, ver, sp)]
                if not m.get("n"):
                    rows.append({"版本": vname, "期": spn, "笔数 / 有交易天数": "0"})
                    continue
                rows.append(
                    {
                        "版本": vname,
                        "期": spn,
                        "笔数 / 有交易天数": f"{m['n']} / {m['days']}",
                        "扣费后每笔（按笔合并，聚类 95%；重抽样）": fmt_ci(m["net"], 3),
                        "按天等权": f"{m['net']['day_eq']:+.3f}%",
                        "胜率": f"{m['net']['win']:.3f}",
                        "合计（美元）": f"{m['sum_usd']:+,.0f}",
                        "按日盈亏 t（含无交易日）": f"{m['t_day']:+.2f}（{m['n_days_all']} 天）",
                        "最大回撤": f"{m['mdd']:.1%}"
                        if np.isfinite(m["mdd"])
                        else "（连续复利，见合计行）",
                        "随机 (i) 均值 / 95 分位": f"{m['rand_half_mean']:+.3f}% / {m['rand_half_95']:+.3f}%",
                        "随机 (ii) 均值 / 95 分位": f"{m['rand_permute_mean']:+.3f}% / {m['rand_permute_95']:+.3f}%",
                        "多 / 空 笔数": f"{m['long_n']} / {m['short_n']}",
                        "做多每笔": fmt_ci(m["long"], 3),
                        "做空每笔": fmt_ci(m["short"], 3),
                        "持仓期间买入持有": fmt_ci(m["bh_hold"], 3),
                        "持仓分钟 P10/P25/P50/P75/P90": "/".join(f"{v:.0f}" for v in m["hold_q"]),
                    }
                )
        L += [df_to_md(pd.DataFrame(rows)), ""]
        m = res[(acct, "both", "hold")]
        if m.get("n"):
            w = m["weekly"]
            L += [
                f"留出期每周盈亏（{w['unit']}，{w['n']} 周）：中位 {w['median']:+,.0f}、最好 {w['best']:+,.0f}、最差 {w['worst']:+,.0f}、亏损周占比 {w['loss_share']:.0%}。",
                "",
                "按月（留出期，多空都做）：",
                "",
                df_to_md(
                    pd.DataFrame(m["monthly"]).rename(
                        columns={
                            "m": "月",
                            "n": "笔数",
                            "usd": "盈亏（美元）",
                            "pct": "每笔均值（%）",
                        }
                    )
                ),
                "",
            ]
    L += [
        "## 边界",
        "",
        "- 只有 MU 一个标的、一个变体；1 分钟 K 收盘确认、下一分钟开盘 ± 1 tick 成交，未建模排队与盘口深度；做空未计借券费与保证金。",
        "- 账户层：满仓复利、整数股、D-013 费用；15:50 清仓（夜间研究的开发 / 验证结果按收盘清仓、固定权益，本表按本次设定重算，与夜间数字不完全相同）。",
        "- 随机对照 (i) 每笔方向独立 ±1 各半；(ii) 保持策略多空笔数、只打乱方向分配（扣除 MU 单边漂移）——通过标准以 (ii) 为准。股数、时刻、费用按实际成交保持。",
        "- 「按日盈亏 t」用该期全部完整交易日（无交易日记 0）。成本翻倍 = 费用 × 2 且每边滑点 2 tick。入场延迟 60 秒 = 信号后第二分钟开盘入场，出场规则不变。",
        "- 留出期结论只针对这一个预先选定的变体；不调参、不重跑。",
        "",
    ]
    txt = "\n".join(L)
    OUT_MD.write_text(txt, encoding="utf-8")
    return txt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    root = get_settings().zr_data_root
    if a.run:
        run(root)
    if a.report:
        report(root)
        print(f"写入 {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
