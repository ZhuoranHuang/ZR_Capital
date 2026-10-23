"""限价单成交模型校准——按挂单段（feat/fill-model A 节）。

订单表（orders.parquet）只存最终价格；改价历史在 order_detail.history
（VPS 上 scripts/diag_pull_order_history.py 拉取，$ZR_DATA_ROOT/diag/order_history.parquet，
时间为 naive UTC）。每个 New / Replaced 事件开始一个「挂单段」（新的挂单价与等待时长），
段在下一个 New / Replaced、或订单终结（Filled / Canceled / Expired / Rejected）时结束。
成交按成交时刻归入所在段。

样本：fill_model.order_sample（D-030 入选持仓段涉及的全部订单 + 同期全部 Canceled / Expired
期权限价单），只用限价单（LO）；LIT / MIT（触价单，触发前不在盘口）与市价单 MO 单独计数、
不进拟合。行情：ThetaData 逐笔 NBBO + OPRA 成交（fill_model pull 缓存的
diag/owner_orders/{oid}_quotes|trades）。段只取 RTH 内部分（盘前提交的从 09:30 起算）。

每段记录：挂单位置（相对买一 / 卖一 / 中间价，tick 与 spread 倍数）、是否在段内成交、
首笔成交耗时、部分成交、首笔成交后 10/30/60 秒中间价变化（按方向，负 = 逆向选择）、
订单撤单 / 过期后 30/60/300 秒中间价变化（按方向，正 = 错过的有利走势）；并用
zr/backtest/fill_limit 的挂单判定（卖一 ≤ 挂单价、OPRA 成交越过挂单价、恰好等于挂单价时
以概率 q 成交）逐段模拟，标定 q。

运行：uv run python -m zr.research.fill_segments [--cached]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.backtest.fill_limit import Book, _rest
from zr.config import get_settings
from zr.execution.ticks import tick_size
from zr.journal.analyze_history import load
from zr.research import fill_model as fm

log = logging.getLogger("zr.research.fill_segments")
BOUNDARY = ("New", "Replaced")
TERMINAL = ("Filled", "Canceled", "Expired", "Rejected")
DELTAS_FILL = (10, 30, 60)
DELTAS_CANCEL = (30, 60, 300)
HORIZONS = (2, 5, 10, 30, 60, 300)
Q_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
N_SEEDS = 20


def load_history(root: Path) -> pd.DataFrame:
    h = pd.read_parquet(root / "diag" / "order_history.parquet")
    h["order_id"] = h["order_id"].astype(str)
    h["ts"] = pd.to_datetime(h["time"].astype(str)).dt.tz_localize("UTC")
    order = {"New": 0, "Replaced": 1, "PartialFilled": 2, "Filled": 3}
    h["_o"] = h["status"].map(order).fillna(4)
    return h.sort_values(["order_id", "ts", "_o"], kind="stable").reset_index(drop=True)


def _ns(x) -> np.ndarray:
    """时间 → int64 纳秒（pandas 默认微秒精度，astype("int64") 会得到微秒）。"""
    return pd.DatetimeIndex(pd.to_datetime(x, utc=True)).as_unit("ns").asi8


def _asof(q: pd.DataFrame, ns: int):
    i = int(np.searchsorted(q["ns"].to_numpy(), ns, side="right")) - 1
    return q.iloc[max(i, 0)]


def _mid(r) -> float:
    return float((r["bid"] + r["ask"]) / 2)


def _drop_own(t: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """去掉负责人自己的成交在 OPRA 里的那一笔（同价、同量、±2 s 内，每笔成交匹配一次）。"""
    if t.empty or fills.empty:
        return t
    tns = _ns(t["ts"])
    keep = np.ones(len(t), bool)
    for ts, px, qty in zip(_ns(fills["ts"]), fills["price"], fills["quantity"], strict=True):
        m = np.flatnonzero(
            keep
            & (np.abs(tns - ts) <= 2 * 10**9)
            & np.isclose(t["price"].to_numpy(float), float(px))
            & (t["size"].to_numpy(float) == float(qty))
        )
        if len(m):
            keep[m[np.argmin(np.abs(tns[m] - ts))]] = False
    return t[keep]


def segments(root: Path) -> pd.DataFrame:
    s = fm.order_sample(root)
    s["order_id"] = s["order_id"].astype(str)
    h = load_history(root)
    _, ex, _ = load(root)
    ex = ex.assign(ts=fm._utc(ex["ts_et"], naive_is_utc=False), order_id=ex["order_id"].astype(str))
    rows = []
    for _, o in s.iterrows():
        base = {k: o[k] for k in ("order_id", "group", "symbol", "und", "side", "type", "status")}
        hh = h[h["order_id"] == o["order_id"]]
        if o["type"] != "LO":
            rows.append({**base, "skip": f"非限价单（{o['type']}）"})
            continue
        if hh.empty:
            rows.append({**base, "skip": "无改价历史"})
            continue
        w = fm.window(o)
        d = fm._load_ticks(root, o["order_id"])
        if w is None or d is None:
            rows.append({**base, "skip": "盘外" if w is None else "无有效报价"})
            continue
        q, t = d
        book = Book.from_frames(q, t if len(t) else None)
        fills0 = ex[ex["order_id"] == o["order_id"]]
        book_x = Book.from_frames(q, _drop_own(t, fills0) if len(t) else None)
        sgn = 1 if o["side"] == "Buy" else -1
        open_ns = (w[0] + fm.PRE).value
        bnd = hh[hh["status"].isin(BOUNDARY)]
        term = hh[hh["status"].isin(TERMINAL)]
        end_ns = term["ts"].iloc[-1].value if len(term) else (w[1] - fm.POST).value
        fills = ex[ex["order_id"] == o["order_id"]].sort_values("ts")
        starts = _ns(bnd["ts"]).tolist()
        fns = _ns(fills["ts"])
        for k, (st, px, qty) in enumerate(
            zip(starts, bnd["price"].astype(float), bnd["quantity"].astype(float), strict=True)
        ):
            en = starts[k + 1] if k + 1 < len(starts) else end_ns
            kind = (
                "replaced"
                if k + 1 < len(starts)
                else (term["status"].iloc[-1].lower() if len(term) else "open")
            )
            st_c = max(st, open_ns)
            if en <= st_c:
                continue
            a0 = _asof(q, st_c)
            mid0, spr = _mid(a0), float(a0["ask"] - a0["bid"])
            tick = float(tick_size(o["symbol"], mid0))
            fseg = fills[(fns >= st) & (fns < en + (1 if kind == "filled" else 0) * 10**9)]
            prev = _asof(q, st_c - 30 * 10**9)
            rec = {
                **base,
                "skip": "",
                "seg": k,
                "n_seg": len(starts),
                "start_ns": st_c,
                "end_ns": en,
                "end_kind": kind,
                "live_s": (en - st_c) / 1e9,
                "limit": px,
                "qty": qty,
                "bid0": float(a0["bid"]),
                "ask0": float(a0["ask"]),
                "mid0": mid0,
                "spread_ticks": spr / tick,
                "tick": tick,
                "offset_ticks": sgn * (px - mid0) / tick,
                "offset_spread": sgn * (px - mid0) / spr if spr > 0 else np.nan,
                "own_ticks": sgn * (px - (a0["bid"] if sgn == 1 else a0["ask"])) / tick,
                "marketable": bool(px >= a0["ask"] if sgn == 1 else px <= a0["bid"]),
                "mid_dir30_ticks": sgn * (mid0 - _mid(prev)) / tick,
                "filled": len(fseg) > 0,
                "fill_qty": float(fseg["quantity"].sum()),
            }
            rec["partial"] = rec["filled"] and rec["fill_qty"] < qty
            if rec["filled"]:
                f0 = int(_ns(fseg["ts"])[0])
                rec["fill_s"] = max((f0 - st_c) / 1e9, 0.0)
                mf = _mid(_asof(q, f0))
                for dl in DELTAS_FILL:
                    m2 = _mid(_asof(q, f0 + dl * 10**9))
                    rec[f"adv_{dl}s_ticks"] = sgn * (m2 - mf) / tick
                    rec[f"adv_{dl}s_pct"] = sgn * (m2 / mf - 1) * 100
            if kind in ("canceled", "expired"):
                mc = _mid(_asof(q, en))
                for dl in DELTAS_CANCEL:
                    m2 = _mid(_asof(q, en + dl * 10**9))
                    rec[f"miss_{dl}s_ticks"] = sgn * (m2 - mc) / tick
                    rec[f"miss_{dl}s_pct"] = sgn * (m2 / mc - 1) * 100
            en_sim = en + (10**9 if kind == "filled" else 0)
            # 模拟：同一挂单价、同一段时长，q 取网格，每个 q 平均 N_SEEDS 个种子
            for qp in Q_GRID:
                hit, tfirst = [], []
                for seed in range(N_SEEDS if 0 < qp < 1 else 1):
                    rng = np.random.default_rng(seed)
                    # 改价历史的时刻截到整秒：成交段的终点 +1 s，否则会截掉同一秒内的 OPRA 成交
                    f, _ = _rest(book, sgn, px, st_c, en_sim, qty, rng, qp)
                    hit.append(bool(f))
                    tfirst.append((f[0][0] - st_c) / 1e9 if f else np.nan)
                rec[f"sim_fill_q{qp}"] = float(np.mean(hit))
                rec[f"sim_fill_s_q{qp}"] = float(np.nanmedian(tfirst)) if any(hit) else np.nan
                hx = [
                    bool(
                        _rest(book_x, sgn, px, st_c, en_sim, qty, np.random.default_rng(sd), qp)[0]
                    )
                    for sd in range(N_SEEDS if 0 < qp < 1 else 1)
                ]
                rec[f"simx_fill_q{qp}"] = float(np.mean(hx))
            rows.append(rec)
    return pd.DataFrame(rows)


def fit(seg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """成交概率（按段、按等待时长 T）：样本 = 存活 ≥ T 或 T 内成交的段，y = T 内成交。"""
    from sklearn.linear_model import LogisticRegression

    g = seg[(seg["skip"] == "") & seg["offset_spread"].notna()].copy()
    g["filled"] = g["filled"].astype(bool)
    rows = []
    for T in HORIZONS:
        yin = g["filled"] & (g["fill_s"] <= T)
        m = yin | (g["live_s"] >= T)
        for i in g.index[m]:
            r = g.loc[i]
            rows.append(
                {
                    "T": T,
                    "y": int(yin.loc[i]),
                    "offset_spread": r["offset_spread"],
                    "logT": np.log(T),
                    "mid_dir30_ticks": r["mid_dir30_ticks"],
                    "spread_ticks": r["spread_ticks"],
                }
            )
    d = pd.DataFrame(rows)
    cols = ["offset_spread", "logT", "mid_dir30_ticks", "spread_ticks"]
    mdl = LogisticRegression(C=1.0, max_iter=1000).fit(d[cols].to_numpy(float), d["y"])
    coef = pd.DataFrame(
        {
            "特征": [
                "截距",
                "挂单位置 ÷ spread",
                "log 等待秒数",
                "最近 30 秒中间价方向（tick）",
                "spread（tick）",
            ],
            "系数": np.r_[mdl.intercept_, mdl.coef_[0]].round(3),
        }
    )
    d["p_hat"] = mdl.predict_proba(d[cols].to_numpy(float))[:, 1]
    d["位置"] = pd.cut(
        d["offset_spread"],
        [-np.inf, -0.5, -0.25, 0.0, 0.25, 0.5, np.inf],
        labels=[
            "≤ −0.5（己方最优价及以内）",
            "(−0.5, −0.25]",
            "(−0.25, 0]（中间价下方）",
            "(0, 0.25]",
            "(0.25, 0.5)",
            "≥ 0.5（对手价及以外）",
        ],
    )
    tab = (
        d.groupby(["T", "位置"], observed=True)
        .agg(段数=("y", "size"), 实际成交率=("y", "mean"), 模型成交率=("p_hat", "mean"))
        .round(3)
        .reset_index()
    )
    return coef, tab


def calibrate_q(seg: pd.DataFrame) -> pd.DataFrame:
    g = seg[seg["skip"] == ""].copy()
    g["filled"] = g["filled"].astype(bool)
    rows = []
    for qp in Q_GRID:
        p = g[f"sim_fill_q{qp}"]
        y = g["filled"].astype(float)
        both = g["filled"] & (p > 0)
        rows.append(
            {
                "q": qp,
                "段数": len(g),
                "实际成交段": int(y.sum()),
                "模拟成交段（期望）": round(float(p.sum()), 1),
                "逐段 Brier": round(float(((p - y) ** 2).mean()), 4),
                "实际成交 → 模拟也成交（期望）": round(float(p[g["filled"]].mean()), 3),
                "实际未成交 → 模拟成交（期望）": round(float(p[~g["filled"]].mean()), 3),
                "去掉自己成交后：实际成交 → 模拟成交（期望）": round(
                    float(g.loc[g["filled"], f"simx_fill_q{qp}"].mean()), 3
                ),
                "首笔成交耗时：模拟 − 实际（中位数 s）": round(
                    float((g.loc[both, f"sim_fill_s_q{qp}"] - g.loc[both, "fill_s"]).median()), 1
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    import sys

    cache = root / "research" / "fill_model" / "owner_segments.parquet"
    seg = pd.read_parquet(cache) if "--cached" in sys.argv and cache.exists() else segments(root)
    out = root / "research" / "fill_model"
    out.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(out / "owner_segments.parquet")
    print(seg["skip"].value_counts().to_dict())
    print(calibrate_q(seg).to_string())
    coef, tab = fit(seg)
    print(coef.to_string())
    report(root, seg)
    return 0


# ---------------- 报告 ----------------

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "FILL_MODEL_CALIBRATION.md"
Q_CAL = 0.9  # 逐段 Brier 最小的 q（见 calibrate_q），写入 zr/backtest/fill_limit.LimitRule 默认值


def _med(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return "—"
    return (
        f"{x.median():.2f}（P25 {x.quantile(0.25):.2f} / P75 {x.quantile(0.75):.2f}，n={len(x)}）"
    )


def report(root: Path, seg: pd.DataFrame) -> None:
    from zr.ops.markdown import df_to_md

    h = load_history(root)
    s = fm.order_sample(root)
    s["order_id"] = s["order_id"].astype(str)
    hs = h[h["order_id"].isin(s["order_id"])]
    g = seg[seg["skip"] == ""].copy()
    g["filled"] = g["filled"].astype(bool)
    g["位置"] = pd.cut(
        g["offset_spread"],
        [-np.inf, -0.5, -0.25, 0.0, 0.25, 0.5, np.inf],
        labels=[
            "≤ −0.5（己方最优价及以内）",
            "(−0.5, −0.25]",
            "(−0.25, 0]",
            "(0, 0.25]",
            "(0.25, 0.5)",
            "≥ 0.5（对手价及以外）",
        ],
    )
    cq = calibrate_q(seg)
    coef, tab = fit(seg)
    pos = (
        g.groupby("位置", observed=True)
        .agg(
            段数=("filled", "size"),
            段内成交率=("filled", "mean"),
            成交耗时中位数_s=("fill_s", "median"),
            段时长中位数_s=("live_s", "median"),
        )
        .round(3)
        .reset_index()
    )
    grp = (
        g.groupby(["group", "side"])
        .agg(
            段数=("filled", "size"),
            订单数=("order_id", "nunique"),
            改价段=("end_kind", lambda x: int((x == "replaced").sum())),
            段内成交=("filled", "sum"),
            挂单位置中位数_spread=("offset_spread", "median"),
            挂单位置中位数_tick=("offset_ticks", "median"),
            spread中位数_tick=("spread_ticks", "median"),
        )
        .round(3)
        .reset_index()
    )
    f = g[g["filled"]]
    adv = pd.DataFrame(
        [
            {
                "窗口": f"成交后 {d} s",
                "中间价变化（tick，按方向）": _med(f[f"adv_{d}s_ticks"]),
                "中间价变化（%）": _med(f[f"adv_{d}s_pct"]),
                "不利（< 0）占比": round(float((f[f"adv_{d}s_ticks"] < 0).mean()), 3),
                "均值 %": round(float(f[f"adv_{d}s_pct"].mean()), 3),
            }
            for d in DELTAS_FILL
        ]
    )
    c = g[g["end_kind"].isin(["canceled", "expired"])]
    miss = pd.DataFrame(
        [
            {
                "窗口": f"撤单 / 过期后 {d} s",
                "中间价变化（tick，按方向）": _med(c[f"miss_{d}s_ticks"]),
                "中间价变化（%）": _med(c[f"miss_{d}s_pct"]),
                "有利（> 0，错过）占比": round(float((c[f"miss_{d}s_ticks"] > 0).mean()), 3),
                "均值 %": round(float(c[f"miss_{d}s_pct"].mean()), 3),
            }
            for d in DELTAS_CANCEL
        ]
    )
    split = hs[hs["status"] == "PartialFilled"]["order_id"].nunique()

    def n_kind(k: str) -> int:
        return int((g["end_kind"] == k).sum())

    lines = [
        "# 限价单成交模型校准（按挂单段）",
        "",
        "`uv run python -m zr.research.fill_segments` 生成（行情缓存由 "
        "`uv run python -m zr.research.fill_model pull` 拉取；改价历史由 VPS 上 "
        "`scripts/diag_pull_order_history.py` 拉取）。",
        "",
        "## 样本",
        "",
        f"- 改价历史：{len(h)} 行 / {h['order_id'].nunique()} 单，其中改价（Replaced）记录 "
        f"{int((h['status'] == 'Replaced').sum())} 条；样本订单内 {len(hs)} 行，改价 "
        f"{int((hs['status'] == 'Replaced').sum())} 条、涉及 "
        f"{hs[hs['status'] == 'Replaced']['order_id'].nunique()} 单。",
        f"- 订单样本 {len(s)} 单（D-030 入选持仓段涉及 {int((s['group'] == 'd030').sum())}、同期"
        "撤单 "
        f"{int((s['group'] == 'canceled').sum())}、过期 {int((s['group'] == 'expired').sum())}）。"
        "不进拟合："
        + "，".join(f"{k} {v} 单" for k, v in seg["skip"].value_counts().items() if k)
        + "。",
        f"- 挂单段 {len(g)} 个（{g['order_id'].nunique()} 单）：以改价结束 "
        f"{n_kind('replaced')}、成交结束 {n_kind('filled')}"
        "、"
        f"撤单 {n_kind('canceled')}、过期 {n_kind('expired')}"
        "。"
        f"分笔成交的订单 {split} 单，全部在同一秒内成交完，没有在改价 / 撤单时留下部分成交。",
        "- 时间：改价历史为整秒（naive UTC）；ThetaData 报价 / OPRA 成交为毫秒。",
        "",
        "## 挂单位置（按组、方向）",
        "",
        "位置 = 方向 ×（挂单价 − 段开始时中间价）÷ spread：−0.5 = 己方最优价（买一 / 卖一），0 "
        "= 中间价，"
        "+0.5 = 对手价。",
        "",
        df_to_md(grp),
        "",
        "## 成交率与耗时（按挂单位置）",
        "",
        df_to_md(pos),
        "",
        "按等待时长 T 的成交率（样本 = 存活 ≥ T 或 T 内成交的段）与逻辑回归（描述用）：",
        "",
        df_to_md(tab),
        "",
        df_to_md(coef),
        "",
        "## 逆向选择（成交后中间价，按方向；负 = 对自己不利）",
        "",
        df_to_md(adv),
        "",
        "## 错过的机会（撤单 / 过期后中间价，按方向；正 = 价格朝原本想要的方向走了）",
        "",
        df_to_md(miss),
        "",
        "## 模拟器标定（zr/backtest/fill_limit 的挂单判定，逐段重放）",
        "",
        "每段用负责人的实际挂单价与实际段时长，按「卖一 ≤ 挂单价（买，按卖一量）→ 成交；OPRA 成"
        "交价越过"
        "挂单价 → 成交；OPRA 成交价恰好等于挂单价 → 以概率 q 成交」判定，q 取网格、每个 q 20 个"
        "随机种子"
        "取期望。",
        "",
        df_to_md(cq.assign(**{"逐段 Brier": cq["逐段 Brier"].map("{:.4f}".format)})),
        "",
        f"取 **q = {Q_CAL}**（逐段 Brier 最小）。读法与局限：",
        "",
        "- OPRA 数据里包含负责人自己的成交，「实际成交 → 模拟也成交」因此偏乐观（q = 1 时为 100"
        "%）；"
        "去掉与负责人成交同价、同量、±2 s 内的那一笔后，同一 q 下只有约 "
        f"{cq.loc[cq['q'] == Q_CAL, '去掉自己成交后：实际成交 → 模拟成交（期望）'].item():.0%} 的"
        "成交段被"
        "市场其余成交解释。真实的反事实介于两者之间：没有负责人的挂单时，吃掉它的那笔对手单会以"
        "更差的价格"
        "成交，在模拟里同样算「越过挂单价」，但这笔成交在数据里并不存在。",
        "- 「实际未成交 → 模拟成交」是没有自我成交污染的指标："
        f"q = {Q_CAL} 时 {cq.loc[cq['q'] == Q_CAL, '实际未成交 → 模拟成交（期望）'].item():.1%}。",
        "",
        "## 边界",
        "",
        f"- 负责人 2026-06-09 ~ 2026-08-19 的期权限价单（MU 为主），{len(g)} 个挂单段；"
        "样本小，逻辑回归系数只作描述，不外推到其他标的或流动性环境。",
        "- 不含盘外提交并撤掉的订单、触价单（LIT / MIT）与市价单。",
        "- 排队位置无法观测，q 是把「恰好在挂单价成交」时我方是否轮到的概率合成一个常数。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
