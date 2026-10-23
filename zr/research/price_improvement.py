"""价格改善校准（R-046）→ docs/research/PRICE_IMPROVEMENT.md、
config/price_improvement.yaml（脚本生成）。

样本：负责人 D-030 入选持仓段的全部订单（fill_model.order_sample 的 d030 组，MU / DRAM / META）。
按挂单段（改价历史的 New / Replaced，整秒、naive UTC）：
- 可立即成交：市价单（MO）、触价单（MIT，取触发生效的 New）、以及到达当秒即可成交的限价单
  （买单挂单价 ≥ 当秒卖一、卖单 ≤ 当秒买一）。「当秒 NBBO」= 该秒内最后一个 ThetaData 逐笔报价
  （该秒内无更新则取之前最近的）；另报该秒开始时的报价作对照。改善量 = 按方向（买：卖一 − 成交价；
  卖：成交价 − 买一），以 tick 与价差比例计；成交 = 该段开始后 2 秒内的成交（按成交量加权）。
- 挂单后成交的限价单：成交价在成交当秒 NBBO 中的位置（0 = 己方最优价，1 = 对手价）与相对
  挂单价的改善。
分组：订单类型、价差宽度（tick）、权利金档位（中间价，美元）、DTE（日历日）。
模型参数（config/price_improvement.yaml）：按「类别（market = MO / MIT / 吃价差；limit = 可立即
成交的限价单）× 价差档」存改善量的价差比例经验值（样本 < 5 的格子退回同类别全部样本，再退回全部）。
运行：uv run python -m zr.research.price_improvement
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.config import get_settings
from zr.execution.ticks import tick_size
from zr.journal.analyze_history import load
from zr.ops.markdown import df_to_md
from zr.research import fill_model as fm
from zr.research import fill_segments as fs

log = logging.getLogger("zr.research.price_improvement")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "PRICE_IMPROVEMENT.md"
CFG = REPO / "config" / "price_improvement.yaml"
NS = 1_000_000_000
SPREAD_BINS = [0, 5, 10, 20, 10_000]
SPREAD_LABELS = ["≤5", "6–10", "11–20", ">20"]
PREM_BINS = [0, 5, 15, 30, 10_000]
PREM_LABELS = ["<5", "5–15", "15–30", "≥30"]
MIN_N = 5


def spread_bucket(spread_ticks: float) -> str:
    i = int(np.searchsorted(SPREAD_BINS, spread_ticks, side="left")) - 1
    return SPREAD_LABELS[min(max(i, 0), len(SPREAD_LABELS) - 1)]


def _q_at(q: pd.DataFrame, ns: int):
    i = int(np.searchsorted(q["ns"].to_numpy(), ns, side="right")) - 1
    return q.iloc[i] if i >= 0 else None


def samples(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = fm.order_sample(root)
    s = s[s["group"] == "d030"].copy()
    s["order_id"] = s["order_id"].astype(str)
    h = fs.load_history(root)
    _, ex, _ = load(root)
    ex = ex.assign(ts=fm._utc(ex["ts_et"], naive_is_utc=False), order_id=ex["order_id"].astype(str))
    imm, rest = [], []
    for _, o in s.iterrows():
        d = fm._load_ticks(root, o["order_id"])
        hh = h[h["order_id"] == o["order_id"]]
        bnd = hh[hh["status"].isin(fs.BOUNDARY)]
        f = ex[ex["order_id"] == o["order_id"]]
        if d is None or bnd.empty or f.empty:
            continue
        q, _ = d
        sgn = 1 if o["side"] == "Buy" else -1
        w = fm.window(o)
        open_ns = (w[0] + fm.PRE).value if w else 0
        st = fs._ns(bnd["ts"])
        fns = fs._ns(f["ts"])
        dte = (o["expiry"] - o["day"]).days
        for k, (t0, px) in enumerate(zip(st, bnd["price"].astype(float), strict=True)):
            t0 = max(int(t0), open_ns)
            t_end = int(st[k + 1]) if k + 1 < len(st) else int(fns.max()) + NS
            qe, qs = _q_at(q, t0 + NS - 1), _q_at(q, t0)
            if qe is None:
                continue
            bid, ask = float(qe["bid"]), float(qe["ask"])
            mid, spr = (bid + ask) / 2, ask - bid
            tick = float(tick_size(o["symbol"], mid))
            kind = o["type"] if o["type"] in ("MO", "MIT") else "LO"
            marketable = kind != "LO" or (px >= ask if sgn == 1 else px <= bid)
            seg_f = f[(fns >= t0) & (fns < t_end)]
            if seg_f.empty:
                continue
            fpx = float((seg_f["price"] * seg_f["quantity"]).sum() / seg_f["quantity"].sum())
            base = {
                "order_id": o["order_id"],
                "symbol": o["symbol"],
                "und": o["und"],
                "type": kind,
                "side": o["side"],
                "dte": dte,
                "mid": mid,
                "spread_ticks": spr / tick,
                "tick": tick,
                "fill_px": fpx,
                "limit": px,
            }
            imm_f = seg_f[fs._ns(seg_f["ts"]) <= t0 + 2 * NS]
            if marketable and len(imm_f):
                fpx_i = float((imm_f["price"] * imm_f["quantity"]).sum() / imm_f["quantity"].sum())
                opp = ask if sgn == 1 else bid
                impr = sgn * (opp - fpx_i)
                opp_s = float(qs["ask"] if sgn == 1 else qs["bid"]) if qs is not None else np.nan
                imm.append(
                    {
                        **base,
                        "fill_px": fpx_i,
                        "improve_ticks": impr / tick,
                        "improve_frac": impr / spr if spr > 0 else np.nan,
                        "improve_ticks_start_quote": sgn * (opp_s - fpx_i) / tick,
                    }
                )
            elif not marketable:
                tf = int(fs._ns(seg_f["ts"]).min())
                qf = _q_at(q, tf + NS - 1)
                if qf is None:
                    continue
                b2, a2 = float(qf["bid"]), float(qf["ask"])
                sp2 = a2 - b2
                pos = ((fpx - b2) if sgn == 1 else (a2 - fpx)) / sp2 if sp2 > 0 else np.nan
                rest.append(
                    {
                        **base,
                        "pos_in_spread": pos,
                        "vs_limit_ticks": sgn * (px - fpx) / tick,
                        "inside": b2 < fpx < a2,
                    }
                )
    imm, rest = pd.DataFrame(imm), pd.DataFrame(rest)
    for d in (imm, rest):
        if len(d):
            d["spread_bucket"] = [spread_bucket(x) for x in d["spread_ticks"]]
            d["prem_bucket"] = pd.cut(d["mid"], PREM_BINS, labels=PREM_LABELS, right=False).astype(
                str
            )
            d["dte_bucket"] = np.where(d["dte"] <= 0, "0", np.where(d["dte"] == 1, "1", "≥2"))
            d["kind"] = np.where(d["type"].isin(["MO", "MIT"]), "market", "limit")
    return imm, rest


def _dist(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return {"n": 0}
    return {
        "n": len(x),
        "中位数": round(float(x.median()), 3),
        "P25": round(float(x.quantile(0.25)), 3),
        "P75": round(float(x.quantile(0.75)), 3),
        "均值": round(float(x.mean()), 3),
        ">0 占比": round(float((x > 0).mean()), 3),
    }


def table(d: pd.DataFrame, by: str, col: str) -> pd.DataFrame:
    rows = [{by: k, **_dist(g[col])} for k, g in d.groupby(by)]
    return pd.DataFrame(rows + [{by: "全部", **_dist(d[col])}])


def model_params(imm: pd.DataFrame) -> dict:
    """{kind: {spread_bucket: [价差比例…]}}，样本 < MIN_N 的格子不写（取值时退回）。"""
    out = {
        "min_n": MIN_N,
        "spread_bins": SPREAD_BINS,
        "spread_labels": SPREAD_LABELS,
        "all": [],
        "by_kind": {},
    }
    v = imm["improve_frac"].dropna()
    out["all"] = [round(float(x), 4) for x in v]
    for kind, g in imm.groupby("kind"):
        gv = g["improve_frac"].dropna()
        ent = {"all": [round(float(x), 4) for x in gv], "by_spread": {}}
        for b, gg in g.groupby("spread_bucket"):
            vv = gg["improve_frac"].dropna()
            if len(vv) >= MIN_N:
                ent["by_spread"][b] = [round(float(x), 4) for x in vv]
        out["by_kind"][kind] = ent
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    imm, rest = samples(root)
    store = root / "research" / "price_improvement"
    store.mkdir(parents=True, exist_ok=True)
    imm.to_parquet(store / "immediate.parquet", index=False)
    rest.to_parquet(store / "resting.parquet", index=False)
    params = model_params(imm)
    CFG.write_text(
        "# 由 `uv run python -m zr.research.price_improvement` 生成，勿手改（R-046 价格改善模型"
        "参数：\n"
        "# 改善量 = 价差 × 比例；比例按 类别 × 价差档 抽样，样本 < min_n 的格子退回同类别、再退"
        "回全部）\n" + yaml.safe_dump(params, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    inside = float(rest["inside"].mean()) if len(rest) else float("nan")
    lines = [
        "# 价格改善校准（R-046）",
        "",
        "`uv run python -m zr.research.price_improvement`。样本：负责人 D-030 入选持仓段的订单"
        f"（2026-06-09～2026-08-19）；可立即成交 {len(imm)} 段（市价单 / 触价单 / 到达当秒即可成"
        "交的限价单），"
        f"挂单后成交的限价单 {len(rest)} 段。改善量按方向：正 = 比当秒对手价好。",
        "",
        "## 可立即成交的订单：成交价相对当秒对手价的改善（tick）",
        "",
    ]
    for by, name in (
        ("type", "订单类型"),
        ("spread_bucket", "价差宽度（tick）"),
        ("prem_bucket", "权利金（中间价，美元）"),
        ("dte_bucket", "DTE"),
    ):
        lines += [f"### 按{name}", "", df_to_md(table(imm, by, "improve_ticks")), ""]
    lines += [
        "## 改善量占价差比例（模型抽样用）",
        "",
        df_to_md(table(imm, "kind", "improve_frac")),
        "",
        df_to_md(
            pd.concat(
                [
                    table(g, "spread_bucket", "improve_frac").assign(类别=k)
                    for k, g in imm.groupby("kind")
                ]
            )
        ),
        "",
        "对照：用该秒开始时的报价计算的改善（tick）：",
        "",
        df_to_md(table(imm, "type", "improve_ticks_start_quote")),
        "",
        "## 挂单后成交的限价单：成交价在当时报价中的位置",
        "",
        "位置：0 = 己方最优价（买一 / 卖一），1 = 对手价；相对挂单价的改善（tick，正 = 比挂单价"
        "好）。",
        "",
        df_to_md(table(rest, "type", "pos_in_spread")),
        "",
        df_to_md(table(rest, "type", "vs_limit_ticks")),
        "",
        f"成交价严格在买一与卖一之间的：{inside:.0%}。",
        "",
        "## 边界",
        "",
        "- 负责人 2026-06-09～2026-08-19 D-030 样本；改价历史整秒，「当秒」报价取该秒内最后一个；"
        "样本小，分组只作描述。",
        "- 改善量可能为负（成交价比当秒对手价差），来自该秒内报价变化或成交延迟，原样保留。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(table(imm, "type", "improve_ticks").to_string())
    print(table(imm, "kind", "improve_frac").to_string())
    print(table(rest, "type", "pos_in_spread").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
