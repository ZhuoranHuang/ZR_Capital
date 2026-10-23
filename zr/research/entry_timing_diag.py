"""诊断（不做新研究）：负责人 D-030 日内单腿买入委托的提交时刻 vs 2 分钟 K 与 V4 信号；
看跌看涨平价反推正股价的精度 → docs/research/ENTRY_TIMING_DIAG.md。

1. 每张买入委托（D-030 入选 51 段的开仓委托）提交时刻在所属 2 分钟 K（UTC 整 2 分钟网格，口径 C）
   中的秒数（0–120）。
2. 提交时刻「正在形成的 2 分钟 K」：BOLL(20,2) / MACD(6,13,6) 用此前已完成的 2 分钟 K（口径 C：
   含盘前盘后连续、长桥原始 K）加一个临时收盘价计算——临时价 (a) 提交前最后一根完整 1 分钟 K 的收盘，
   (b) 提交时刻同到期日同行权价看涨 / 看跌中间价按平价反推的正股价 S = C − P + K·e^(−rT)（r = 0.04，
   到期 16:00 ET，不计股息与提前行权）。报告临时 %b、柱相对上一根已完成 K 的方向，以及上一根已完成 K
   本身的 %b、柱方向、状态。正股 K 只有 MU（DRAM、META 只报第 1 项）。
3. V4 H1–H3（口径 C，R-033 参数）在提交前后 4 分钟内收盘的信号 K：时间差、方向是否与所买期权一致
   （看涨 = 多、看跌 = 空），是否满足护栏。
4. 平价反推精度：每张买入委托提交前 30 分钟至提交后 5 分钟，同合约对的 ThetaData 逐笔 NBBO；
   在每根长桥 MU 1 分钟 K 收盘时刻（标签 + 60 秒）取两条腿此前最后一个报价，反推价 − 1 分钟收盘价。
运行：uv run python -m zr.research.entry_timing_diag
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.data.thetadata import lb_symbol
from zr.features.bars import ET, load_extended_bars
from zr.features.regime import STATE_NAMES, indicators, regimes
from zr.ops.markdown import df_to_md
from zr.research import fill_model as fm
from zr.research import owner_limit as ol
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.entry_timing_diag")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "ENTRY_TIMING_DIAG.md"
R = 0.04
P2 = 120_000_000_000
PRE, POST = pd.Timedelta(minutes=30), pd.Timedelta(minutes=5)
SIG_WIN = 240_000_000_000


def buy_orders(root: Path) -> pd.DataFrame:
    sel, orders = ol.episode_orders(root)
    s = fm.order_sample(root)
    s["order_id"] = s["order_id"].astype(str)
    b = orders[orders["side"] == "Buy"].merge(
        s[["order_id", "submit", "expiry", "strike", "right", "und", "type", "limit"]],
        on="order_id",
        suffixes=("", "_o"),
    )
    b["submit_raw_ns"] = v4d.ns(pd.DatetimeIndex(b["submit"]))
    b["dir"] = np.where(b["right"] == "call", 1, -1)
    b["open_et"] = [sel.loc[k, "open_ts"].strftime("%Y-%m-%d %H:%M:%S") for k in b["episode"]]
    b["pnl"] = [float(sel.loc[k, "pnl_net"]) for k in b["episode"]]
    return b.sort_values("submit_raw_ns").reset_index(drop=True)


def pair_windows(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, o in b.iterrows():
        sub = pd.Timestamp(o["submit_raw_ns"], tz="UTC")
        day = sub.tz_convert(ET).date()
        for right in ("call", "put"):
            rows.append(
                {
                    "und": o["und"],
                    "symbol": lb_symbol(o["und"], o["expiry"], right, o["strike"]),
                    "day": day,
                    "start": sub - PRE,
                    "end": sub + POST,
                    "expiry": o["expiry"],
                    "strike": o["strike"],
                    "right": right,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["symbol", "day", "start"])


def _quotes(root: Path, und: str, sym: str, day) -> tuple[np.ndarray, np.ndarray] | None:
    tk = v4d.load_ticks(root, und, sym, day)
    if tk is None or not len(tk.ts):
        return None
    return tk.ts, tk.mid


def _asof(arr: tuple, ns: int, max_age_s: float = 10.0) -> float:
    ts, mid = arr
    i = int(np.searchsorted(ts, ns, side="right")) - 1
    if i < 0 or (ns - ts[i]) > max_age_s * 1e9:
        return math.nan
    return float(mid[i])


def parity_s(root, o, ns: int, cache: dict) -> float:
    day = pd.Timestamp(ns, tz="UTC").tz_convert(ET).date()
    legs = []
    for right in ("call", "put"):
        sym = lb_symbol(o["und"], o["expiry"], right, o["strike"])
        if (sym, day) not in cache:
            cache[(sym, day)] = _quotes(root, o["und"], sym, day)
        q = cache[(sym, day)]
        legs.append(math.nan if q is None else _asof(q, ns))
    exp_close = pd.Timestamp(f"{o['expiry']} 16:00", tz=ET).value
    t = max((exp_close - ns) / (365 * 24 * 3600e9), 0.0)
    return legs[0] - legs[1] + float(o["strike"]) * math.exp(-R * t)


def forming(
    closes: np.ndarray, ef: float, es: float, dea: float, hist_prev: float, p: float, v4
) -> dict:
    """已完成 K 的收盘（最后 19 根）+ 临时价 p → %b；EMA 状态 + p → 临时柱及相对上一根的方向。"""
    if not np.isfinite(p) or len(closes) < 19:
        return {"pctb": math.nan, "hist_dir": math.nan}
    w = np.r_[closes[-19:], p]
    mid, sd = w.mean(), w.std(ddof=0)
    up, lo = mid + 2 * sd, mid - 2 * sd
    m = v4["macd"]
    af, a_s, ag = (2 / (m[k] + 1) for k in ("fast", "slow", "signal"))
    ef2, es2 = af * p + (1 - af) * ef, a_s * p + (1 - a_s) * es
    dif = ef2 - es2
    dea2 = ag * dif + (1 - ag) * dea
    h = dif - dea2
    return {
        "pctb": (p - lo) / (up - lo) if up > lo else math.nan,
        "hist_dir": float(np.sign(h - hist_prev)),
        "hist": h,
    }


def run(root: Path) -> dict:
    v4 = v4d.load_v4()
    b = buy_orders(root)
    w = pair_windows(b)
    for und, g in w.groupby("und"):
        log.info("%s 合约对 tick：%s", und, v4d.pull_ticks(root, und, g.drop(columns="und")))
    conv = v4["conventions"]["C"]
    cfg = v4d.state_cfg(v4, "C")
    end = max(pd.Timestamp(x, tz="UTC").tz_convert(ET).date() for x in b["submit_raw_ns"])
    bars = v4d.build_bars(root, "MU", conv, end)
    x = indicators(bars, cfg, daily_reset=False, warmup="nan")
    reg = regimes(bars, x, cfg, v4["period"])
    sig = v4d.hypothesis_signals(bars, x, reg, cfg, v4)
    days = {pd.Timestamp(n, tz="UTC").tz_convert(ET).date() for n in b["submit_raw_ns"]}
    time_ok, ok_l, ok_s = v4d.guard_mask(bars, reg, v4, days)
    bns = v4d.ns(bars.index)
    cns = bns + P2
    close = bars["close"].to_numpy(float)
    m = v4["macd"]
    ef = bars["close"].ewm(span=m["fast"], adjust=False).mean().to_numpy()
    es = bars["close"].ewm(span=m["slow"], adjust=False).mean().to_numpy()
    dif = ef - es
    dea = pd.Series(dif).ewm(span=m["signal"], adjust=False).mean().to_numpy()
    hist = dif - dea
    one = load_extended_bars(root, "MU", min(days), end, clean=False)
    one_close_ns = v4d.ns(one.index) + 60_000_000_000
    one_c = one["close"].to_numpy(float)
    cache: dict = {}
    rows = []
    for _, o in b.iterrows():
        sub = int(o["submit_raw_ns"])
        rec = {
            "open_et": o["open_et"],
            "submit_et": pd.Timestamp(sub, tz="UTC").tz_convert(ET).strftime("%Y-%m-%d %H:%M:%S"),
            "und": o["und"],
            "symbol": o["symbol"],
            "dir": int(o["dir"]),
            "type": o["type"],
            "pnl": o["pnl"],
            "sec_in_bar": (sub % P2) / 1e9,
        }
        if o["und"] == "MU":
            b0 = sub - sub % P2
            i = int(np.searchsorted(bns, b0, side="left")) - 1  # 最后一根已完成 K
            if i >= 20:
                j = int(np.searchsorted(one_close_ns, sub, side="right")) - 1
                p1 = float(one_c[j]) if j >= 0 else math.nan
                pp = parity_s(root, o, sub, cache)
                d = int(o["dir"])
                fa = forming(close[: i + 1], ef[i], es[i], dea[i], hist[i], p1, v4)
                fb = forming(close[: i + 1], ef[i], es[i], dea[i], hist[i], pp, v4)
                rec.update(
                    {
                        "px_1m": p1,
                        "px_parity": pp,
                        "pctb_form_1m": fa["pctb"],
                        "pctb_form_parity": fb["pctb"],
                        "hist_dir_form_1m_aligned": fa["hist_dir"] * d
                        if np.isfinite(fa["hist_dir"])
                        else math.nan,
                        "hist_dir_form_parity_aligned": fb["hist_dir"] * d
                        if np.isfinite(fb["hist_dir"])
                        else math.nan,
                        "pctb_last": float(x["pctb"].iloc[i]),
                        "hist_dir_last_aligned": float(np.sign(hist[i] - hist[i - 1])) * d,
                        "state_last": STATE_NAMES[int(reg["state"].iloc[i])],
                        "state_dir_last": int(reg["dir"].iloc[i]),
                    }
                )
                near = np.flatnonzero(np.abs(cns - sub) <= SIG_WIN)
                hits = []
                for k in near:
                    for h in ("H1", "H2", "H3"):
                        if sig[h][k] != 0:
                            ok = bool(time_ok[k] and (ok_l[k] if sig[h][k] == 1 else ok_s[k]))
                            hits.append(
                                {
                                    "hyp": h,
                                    "dt_s": (cns[k] - sub) / 1e9,
                                    "same_dir": int(sig[h][k]) == d,
                                    "guard_ok": ok,
                                }
                            )
                rec["sig_n"] = len(hits)
                rec["sig_before_same"] = sum(1 for z in hits if z["dt_s"] <= 0 and z["same_dir"])
                rec["sig_after_same"] = sum(1 for z in hits if z["dt_s"] > 0 and z["same_dir"])
                rec["sig_opposite"] = sum(1 for z in hits if not z["same_dir"])
                rec["sig_detail"] = "; ".join(
                    f"{z['hyp']} {z['dt_s']:+.0f}s {'同向' if z['same_dir'] else '反向'}"
                    f"{'' if z['guard_ok'] else '（护栏不允许）'}"
                    for z in hits
                )
        rows.append(rec)
    d = pd.DataFrame(rows)
    # 4. 平价反推精度
    prow = []
    for _, o in b[b["und"] == "MU"].iterrows():
        sub = int(o["submit_raw_ns"])
        lo, hi = sub - int(PRE.value), sub + int(POST.value)
        for k in np.flatnonzero((one_close_ns > lo) & (one_close_ns <= hi)):
            t = int(one_close_ns[k])
            et = pd.Timestamp(t, tz="UTC").tz_convert(ET)
            if not ("09:31" <= et.strftime("%H:%M") <= "16:00"):
                continue
            s_hat = parity_s(root, o, t - 1, cache)
            if np.isfinite(s_hat):
                prow.append(
                    {
                        "order": o["order_id"],
                        "t": t,
                        "close": float(one_c[k]),
                        "s_hat": s_hat,
                        "err": s_hat - float(one_c[k]),
                        "err_bp": (s_hat / float(one_c[k]) - 1) * 1e4,
                        "moneyness_pct": (float(o["strike"]) / float(one_c[k]) - 1) * 100,
                        "dte": (o["expiry"] - et.date()).days,
                    }
                )
    par = pd.DataFrame(prow).drop_duplicates(["t", "s_hat"])
    return {"orders": d, "parity": par, "n_episodes": int(b["episode"].nunique())}


def _q(x) -> str:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if not len(x):
        return "—"
    q = x.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist()
    return f"P10 {q[0]} / P25 {q[1]} / 中位 {q[2]} / P75 {q[3]} / P90 {q[4]}（n={len(x)}）"


def write(res: dict) -> None:
    d, par = res["orders"], res["parity"]
    mu = d[d["und"] == "MU"]
    pos = pd.cut(d["sec_in_bar"], [0, 30, 60, 90, 120], right=False, include_lowest=True)
    pos_t = d.groupby(pos, observed=False).size().rename("委托数").reset_index()
    pos_t["占比"] = (pos_t["委托数"] / len(d)).round(3)
    pos10 = d.groupby(
        pd.cut(d["sec_in_bar"], range(0, 130, 10), right=False), observed=False
    ).size()

    def share(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        return f"{(s > 0).mean():.0%}（n={len(s)}）" if len(s) else "—"

    ind = pd.DataFrame(
        [
            {"量": "临时 %b（1 分钟收盘作当前价）", "分布": _q(mu["pctb_form_1m"])},
            {"量": "临时 %b（平价反推价作当前价）", "分布": _q(mu["pctb_form_parity"])},
            {"量": "上一根已完成 K 的 %b", "分布": _q(mu["pctb_last"])},
            {
                "量": "按方向的临时 %b：看涨 = %b，看跌 = 1 − %b（1 分钟）",
                "分布": _q(np.where(mu["dir"] == 1, mu["pctb_form_1m"], 1 - mu["pctb_form_1m"])),
            },
            {
                "量": "临时柱方向与所买方向一致（1 分钟）",
                "分布": share(mu["hist_dir_form_1m_aligned"]),
            },
            {
                "量": "临时柱方向与所买方向一致（平价）",
                "分布": share(mu["hist_dir_form_parity_aligned"]),
            },
            {
                "量": "上一根已完成 K 柱方向与所买方向一致",
                "分布": share(mu["hist_dir_last_aligned"]),
            },
        ]
    )
    st = mu.groupby(["state_last", "state_dir_last"]).size().rename("委托数").reset_index()
    sig = pd.DataFrame(
        [
            {"项": "MU 买入委托", "值": len(mu)},
            {"项": "提交前后 4 分钟内有 H1–H3 信号 K 收盘", "值": int((mu["sig_n"] > 0).sum())},
            {"项": "其中提交前（含同时）有同向信号", "值": int((mu["sig_before_same"] > 0).sum())},
            {
                "项": "其中提交后才有同向信号",
                "值": int(((mu["sig_after_same"] > 0) & (mu["sig_before_same"] == 0)).sum()),
            },
            {
                "项": "4 分钟内只有反向信号",
                "值": int(
                    (
                        (mu["sig_opposite"] > 0)
                        & (mu["sig_before_same"] + mu["sig_after_same"] == 0)
                    ).sum()
                ),
            },
        ]
    )
    par_t = pd.DataFrame(
        [
            {"量": "反推价 − 1 分钟收盘（美元）", "分布": _q(par["err"])},
            {"量": "|误差|（美元）", "分布": _q(par["err"].abs())},
            {"量": "误差（bp）", "分布": _q(par["err_bp"])},
            {"量": "|误差|（bp）", "分布": _q(par["err_bp"].abs())},
        ]
    )
    if len(par):
        mb = par.assign(价内外=pd.cut(par["moneyness_pct"].abs(), [0, 0.5, 1, 2, 100], right=False))
        by_m = (
            mb.groupby("价内外", observed=True)["err_bp"]
            .agg(样本="size", 中位误差bp="median", 中位绝对误差bp=lambda x: x.abs().median())
            .round(2)
            .reset_index()
        )
        by_d = (
            par.groupby("dte")["err_bp"]
            .agg(样本="size", 中位误差bp="median", 中位绝对误差bp=lambda x: x.abs().median())
            .round(2)
            .reset_index()
        )
    cols = [
        "submit_et",
        "symbol",
        "type",
        "sec_in_bar",
        "px_1m",
        "px_parity",
        "pctb_form_1m",
        "pctb_form_parity",
        "pctb_last",
        "hist_dir_form_1m_aligned",
        "hist_dir_last_aligned",
        "state_last",
        "state_dir_last",
        "sig_detail",
        "pnl",
    ]
    lines = [
        "# 负责人入场时刻诊断（D-030 日内单腿买入委托）",
        "",
        "`uv run python -m zr.research.entry_timing_diag`。诊断，不做新研究、不入任何检验族。"
        f"样本：D-030 入选 {res['n_episodes']} 段的全部开仓买入委托 {len(d)} 张"
        f"（MU {len(mu)} 张；DRAM / META 无本地正股 K，只报第 1 项）。时间为委托提交时刻（盘前提"
        "交保留原时刻）。",
        "",
        "## 1. 提交时刻在所属 2 分钟 K 中的位置（秒，0 = K 开盘）",
        "",
        df_to_md(pos_t.astype({"sec_in_bar": str})),
        "",
        "10 秒分档："
        + "，".join(f"{int(i.left)}–{int(i.right)} s {n}" for i, n in pos10.items())
        + "。",
        "",
        f"分布：{_q(d['sec_in_bar'])}。",
        "",
        "## 2. 提交时刻正在形成的 2 分钟 K（MU，口径 C）",
        "",
        "临时 %b / 柱 = 此前已完成的 2 分钟 K + 临时收盘价。柱方向「一致」= 看涨时柱比上一根大、"
        "看跌时比上一根小。",
        "",
        df_to_md(ind),
        "",
        "上一根已完成 K 的状态（方向 +1 / −1 / 0）：",
        "",
        df_to_md(st),
        "",
        "## 3. V4 H1–H3 在提交前后 4 分钟内的信号（MU，口径 C，K 收盘时刻）",
        "",
        df_to_md(sig),
        "",
        "## 4. 平价反推正股价的精度（MU，长桥 1 分钟 K 收盘时刻）",
        "",
        "合约对 = 负责人所买合约的同到期日、同行权价看涨 + 看跌；报价取 1 分钟 K 收盘时刻之前最"
        "后一个 ThetaData"
        " 逐笔 NBBO（超过 10 秒未更新的不用）；S = C_mid − P_mid + K·e^(−rT)。",
        "",
        df_to_md(par_t),
        "",
    ]
    if len(par):
        lines += [
            "按行权价相对正股的距离：",
            "",
            df_to_md(by_m),
            "",
            "按 DTE：",
            "",
            df_to_md(by_d),
            "",
        ]
    lines += [
        "## 逐张委托",
        "",
        df_to_md(d[[c for c in cols if c in d]].round(3)),
        "",
        "## 边界",
        "",
        "- 负责人 2026-06-09～2026-08-19 的 D-030 样本；指标只算 MU；口径 C（含盘前盘后连续），"
        "App 开盘段数值未核实（STATUS 0f）。",
        "- 平价反推用单一合约对、欧式平价、不计股息与提前行权；1 分钟收盘是该分钟最后一笔成交价"
        "，与 NBBO 中点本身有约半个价差的差。",
        "- 委托提交时刻 ≠ 负责人做决定的时刻（可能更早）。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    res = run(root)
    store = root / "research" / "entry_timing"
    store.mkdir(parents=True, exist_ok=True)
    res["orders"].to_parquet(store / "orders.parquet", index=False)
    res["parity"].to_parquet(store / "parity.parquet", index=False)
    write(res)
    log.info("写入 %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
