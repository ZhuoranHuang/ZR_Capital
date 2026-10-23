"""R-043 第 1 节：负责人出场习惯（D-030 入选持仓段，盈利 / 亏损分开）
→ docs/research/EXIT_HABITS.md。

每个持仓段：张数；平仓分几张委托卖出（次数、每次占比）；卖出时权利金相对买入均价的涨跌
（按卖出成交价与卖出时刻 tick 中间价各算一次）；持仓时长；第一笔卖出前正股 2 分钟 K
（口径 C，含盘前盘后连续，MU 才有）：之前连续逆向收盘根数（看涨 收盘 < 开盘，看跌镜像）、%b、
MACD(6,13,6) 柱是否缩短（按方向）/ 变号、收盘是否在中轨逆向一侧、是否跌破前一更高低点
（X5 水位）；持仓期间期权最大有利偏移 MFE（tick 中间价相对买入均价）与第一笔卖出时已回吐
MFE 的比例。行情：diag/v4_ticks（ThetaData 逐笔 NBBO）。
运行：uv run python -m zr.research.exit_habits
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.regime import indicators
from zr.journal.intraday_list import build as build_list
from zr.ops.markdown import df_to_md
from zr.research import fill_model as fm
from zr.research import v4_decisive as v4d
from zr.research import v4_limit as vl
from zr.research.exits import _swing_level

log = logging.getLogger("zr.research.exit_habits")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "EXIT_HABITS.md"


def _mid_at(bk, ns: int) -> float:
    i = bk.asof(ns)
    return float(bk.mid(i)) if i is not None else np.nan


def bar_context(bars: pd.DataFrame, x: pd.DataFrame, lv: dict, t_ns: int, d: int) -> dict:
    """t 之前最后一根已收盘 2 分钟 K 的情况（d = +1 看涨、−1 看跌）。"""
    close_ns = v4d.ns(bars.index + pd.Timedelta(minutes=2))
    i = int(np.searchsorted(close_ns, t_ns, side="right")) - 1
    if i < 1:
        return {}
    o, c = bars["open"].to_numpy(float), bars["close"].to_numpy(float)
    adverse = (c < o) if d == 1 else (c > o)
    run = 0
    k = i
    while k >= 0 and adverse[k]:
        run += 1
        k -= 1
    h = x["hist"].to_numpy()
    mid, pb = x["mid"].to_numpy(), x["pctb"].to_numpy()
    lvl = lv[d][i]
    return {
        "adverse_run": run,
        "pctb": float(pb[i]),
        "hist_shrink": bool(d * h[i] < d * h[i - 1]),
        "hist_flip": bool(np.sign(h[i]) != np.sign(h[i - 1]) and d * h[i] < 0),
        "beyond_mid": bool(d * (c[i] - mid[i]) < 0),
        "broke_swing": bool(np.isfinite(lvl) and d * (c[i] - lvl) < 0),
    }


def episodes(root: Path) -> pd.DataFrame:
    ep = build_list(root)
    sel = ep[ep["category"] == "入选"].reset_index(drop=True)
    _, ex, _ = fm.load(root)
    ex = ex.assign(ts=fm._utc(ex["ts_et"], naive_is_utc=False), order_id=ex["order_id"].astype(str))
    v4 = v4d.load_v4()
    conv = v4["conventions"]["C"]
    cfg = v4d.state_cfg(v4, "C")
    bars = v4d.build_bars(root, "MU", conv, max(sel["open_ts"]).date())
    x = indicators(bars, cfg, daily_reset=False, warmup="nan")
    lv = {1: _swing_level(bars, True), -1: _swing_level(bars, False)}
    rows = []
    for k, r in sel.iterrows():
        sym = r["symbol"]
        und = fm.parse_contract(sym)["und"]
        d = 1 if r["right"] == "C" else -1
        f = ex[
            (ex["symbol"] == sym)
            & (ex["ts"] >= r["open_ts"].tz_convert("UTC"))
            & (ex["ts"] <= r["close_ts"].tz_convert("UTC"))
        ]
        b, s = f[f["side"] == "Buy"], f[f["side"] == "Sell"]
        qty = float(b["quantity"].sum())
        px0 = float((b["price"] * b["quantity"]).sum() / qty)
        te = int(v4d.ns(pd.DatetimeIndex([b["ts"].max()]))[0])
        so = (
            s.groupby("order_id")
            .agg(ts=("ts", "min"), q=("quantity", "sum"), px=("price", lambda p: np.nan))
            .sort_values("ts")
        )
        for oid in so.index:
            g = s[s["order_id"] == oid]
            so.loc[oid, "px"] = float((g["price"] * g["quantity"]).sum() / g["quantity"].sum())
        t1 = int(v4d.ns(pd.DatetimeIndex([so["ts"].iloc[0]]))[0])
        bk = vl.load_book(root, und, sym, r["open_ts"].date())
        mid1 = _mid_at(bk, t1) if bk is not None else np.nan
        mfe = np.nan
        if bk is not None:
            j0, j1 = (
                int(np.searchsorted(bk.q_ns, te)),
                int(np.searchsorted(bk.q_ns, t1, side="right")),
            )
            if j1 > j0:
                mfe = float(((bk.bid[j0:j1] + bk.ask[j0:j1]) / 2).max() / px0 - 1) * 100
        sell_chg = (float(so["px"].iloc[0]) / px0 - 1) * 100
        mid_chg = (mid1 / px0 - 1) * 100
        rec = {
            "episode": k,
            "und": und,
            "symbol": sym,
            "dir": "call" if d == 1 else "put",
            "open_et": r["open_ts"].strftime("%Y-%m-%d %H:%M:%S"),
            "qty": qty,
            "n_sells": len(so),
            "first_sell_share": round(float(so["q"].iloc[0]) / qty, 3),
            "single_exit": len(so) == 1,
            "sell_chg_pct": sell_chg,
            "sell_mid_chg_pct": mid_chg,
            "hold_min_first": (t1 - te) / 60e9,
            "hold_min": float(r["hold_min"]),
            "pnl": float(r["pnl_net"]),
            "win": float(r["pnl_net"]) > 0,
            "mfe_pct": mfe,
            "giveback": (mfe - mid_chg) / mfe if np.isfinite(mfe) and mfe > 0 else np.nan,
        }
        if und == "MU":
            rec.update(bar_context(bars, x, lv, t1, d))
        rows.append(rec)
    return pd.DataFrame(rows)


def _q(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return "—"
    q = x.quantile([0.25, 0.5, 0.75]).round(2).tolist()
    return f"中位 {q[1]}（P25 {q[0]} / P75 {q[2]}，n={len(x)}）"


def summary(e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for name, g in (("全部", e), ("盈利", e[e["win"]]), ("亏损", e[~e["win"]])):
        m = g[g["und"] == "MU"]
        rows.append(
            {
                "组": name,
                "持仓段": len(g),
                "一次全卖占比": round(float(g["single_exit"].mean()), 3),
                "分批时卖出次数": _q(g.loc[~g["single_exit"], "n_sells"]),
                "首次卖出占比": _q(g.loc[~g["single_exit"], "first_sell_share"]),
                "首次卖出 成交价涨跌 %": _q(g["sell_chg_pct"]),
                "首次卖出 中间价涨跌 %": _q(g["sell_mid_chg_pct"]),
                "到首次卖出（分钟）": _q(g["hold_min_first"]),
                "MFE %": _q(g["mfe_pct"]),
                "回吐 MFE 比例": _q(g["giveback"]),
                **{
                    f"卖前连续逆向 ≥{n} 根（MU）": round(float((m["adverse_run"] >= n).mean()), 3)
                    for n in (1, 2, 3)
                },
                "卖前柱缩短（MU）": round(float(m["hist_shrink"].mean()), 3),
                "卖前柱变号（MU）": round(float(m["hist_flip"].mean()), 3),
                "卖前收在中轨逆向侧（MU）": round(float(m["beyond_mid"].mean()), 3),
                "卖前跌破更高低点（MU）": round(float(m["broke_swing"].mean()), 3),
                "卖前 %b（MU）": _q(m["pctb"]),
            }
        )
    t = pd.DataFrame(rows).set_index("组").T.reset_index().rename(columns={"index": "项"})
    return t, e


def candidates(e: pd.DataFrame) -> list[dict]:
    """从汇总提炼 x4 候选（参数全部由数据算出，规则形式事先固定）：
    c1 止盈：中间价达到盈利单首次卖出中间价涨幅的中位数即全卖；
    c2 逆向 K：连续 N 根逆向收盘即全卖，N = 盈利与亏损单卖前逆向根数的中位数（至少 1）；
    c3 止损：中间价跌到亏损单首次卖出中间价跌幅的中位数即全卖。
    三条合为一组出场（任一先到），另加 30 分钟上限与强平。"""
    w, lo = e[e["win"]], e[~e["win"]]
    tp = float(np.nanmedian(w["sell_mid_chg_pct"]))
    sl = float(np.nanmedian(lo["sell_mid_chg_pct"]))
    n = int(max(1, round(float(np.nanmedian(e.loc[e["und"] == "MU", "adverse_run"])))))
    return [
        {"id": "c1_tp", "tp_pct": round(tp, 1)},
        {"id": "c2_adverse", "n": n},
        {"id": "c3_sl", "sl_pct": round(sl, 1)},
    ]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    e = episodes(root)
    store = root / "research" / "exit_habits"
    store.mkdir(parents=True, exist_ok=True)
    e.to_parquet(store / "episodes.parquet", index=False)
    t, _ = summary(e)
    cand = candidates(e)
    cols = [
        "open_et",
        "und",
        "symbol",
        "dir",
        "qty",
        "n_sells",
        "first_sell_share",
        "sell_chg_pct",
        "sell_mid_chg_pct",
        "hold_min_first",
        "mfe_pct",
        "giveback",
        "adverse_run",
        "pctb",
        "hist_shrink",
        "hist_flip",
        "beyond_mid",
        "broke_swing",
        "pnl",
    ]
    lines = [
        "# 负责人出场习惯（R-043 第 1 节）",
        "",
        "`uv run python -m zr.research.exit_habits`。样本：D-030 入选 51 段（2026-06-09～2026-0"
        "8-19；"
        "正股 K 线指标只有 MU）。卖出时刻 = 第一张平仓委托的首笔成交；涨跌相对买入均价。",
        "",
        "## 汇总（盈利 / 亏损分开）",
        "",
        df_to_md(t),
        "",
        "## 出场规则候选（x4，参数由上表算出，形式事先固定）",
        "",
        df_to_md(pd.DataFrame(cand)),
        "",
        "c1：中间价达到 +tp_pct 全卖；c2：正股 2 分钟 K 连续 n 根逆向收盘全卖；c3：中间价跌到 s"
        "l_pct 全卖；"
        "x4 = c1 + c2 + c3 任一先到，另加 30 分钟上限与强平（写入 RESEARCH_LOG R-043 后再运行）。",
        "",
        "## 逐段",
        "",
        df_to_md(e[[c for c in cols if c in e]].round(3)),
        "",
        "## 边界",
        "",
        "- 51 段，绝大多数为 MU；样本小，分布只作描述。MFE 用 tick 中间价，tick 只覆盖持仓段前后"
        "的拉取窗口。",
        "- 规则候选是对负责人行为的描述性提炼，不等于负责人的真实决策过程。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    import yaml

    (REPO / "config" / "v4_exits_x4.yaml").write_text(
        "# R-043 x4 候选（zr/research/exit_habits.py 生成，勿手改）\n"
        + yaml.safe_dump({"x4": cand}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(t.to_string())
    print(cand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
