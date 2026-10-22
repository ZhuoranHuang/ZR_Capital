"""负责人交易史：只保留「日内单腿」作为策略对标样本（D-029）→ docs/research/INTRADAY_SINGLE_LEG.md。

1. 筛选（从严，同时满足才入选）：同一交易日开平；单腿；该笔持仓期间及入场前后 5 分钟内同一标的
   没有反方向期权持仓（排除跨式、宽跨式、对冲）；DTE ≤ 7。样本单位沿用 157 笔 FIFO 配对
   （zr/journal/analyze_history.fifo_round_trips；一次入场分批平仓会拆成多行）。
2. 入选样本逐笔复盘：状态（入场前最后一根已完成的 2 分钟 K）、形态、MACD、带宽、量比、MA 位置、
   时段、DTE、行权价距离、tick 级入场价相对 ask、持仓期间最大有利/不利偏移、出场价相对 bid、
   扣费后盈亏。口径 A（D-027，负责人指定为底座）+ 口径 C（D-028 连续口径）对照。
3. 规律：状态 × 顺逆 × 形态汇总；盈亏单特征对比；深度 ≤3 决策树规则；护栏证据。
   决策树只用入场前就能算出的 K 线特征（不用 DTE、行权价、成交价、持仓时长等执行结果），
   这样提取的规则才能在 K 线上回放（第 4 节，R-035）。
运行：uv run python -m zr.research.intraday_single_leg [--no-pull]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

from zr.config import get_settings
from zr.data.thetadata import ThetaClient
from zr.features.bars import ET
from zr.features.regime import CHANNEL, IGNITION, RANGE, STATE_NAMES, _consec, indicators, regimes
from zr.journal.analyze_history import fifo_round_trips, load
from zr.ops.markdown import df_to_md
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.intraday_single_leg")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "INTRADAY_SINGLE_LEG.md"
RULES_OUT = REPO / "config" / "v4_rules_d030.yaml"  # R-035（D-029 样本）的规则保留在 v4_rules.yaml
NEAR = pd.Timedelta(minutes=5)
MAX_DTE = 7
# 决策树可用的 K 线特征（方向化：多头原样、空头镜像），入场前可知
TREE_FEATURES = [
    "state_same",
    "state_counter",
    "state_range",
    "state_channel_same",
    "ign_outer_k",
    "pb_pctb",
    "pb_mid_hw",
    "breakout",
    "hist_n",
    "hist_chg_n",
    "dif_gt_dea",
    "dif_gt_0",
    "bw",
    "bw_expand",
    "vol_ratio",
    "ma5_pos",
    "ma10_pos",
    "minute",
]


# ---------------- 1. 筛选 ----------------


def d030_sample(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D-030 入选样本（持仓段为单位）→ 复盘所需列；另返回全部持仓段（含分类）供报告引用。"""
    from zr.journal.intraday_list import build

    ep = build(root)
    s = ep[ep["category"] == "入选"].copy()
    out = pd.DataFrame(
        {
            "und": s["und"],
            "symbol": s["symbol"],
            "entry_ts": s["open_ts"].dt.tz_convert("UTC"),
            "exit_ts": s["close_ts"].dt.tz_convert("UTC"),
            "dir": np.where(s["right"] == "C", 1, -1),
            "right": s["right"],
            "strike": s["strike"],
            "expiry": s["expiry"],
            "dte": s["dte"],
            "qty": s["qty"],
            "hold_min": s["hold_min"],
            "pnl_net": s["pnl_net"],
            "entry_px": s["entry_px"],
            "exit_px": s["exit_px"],
            "lots": s["lots"],
        }
    )
    return out.reset_index(drop=True), ep


def round_trips(root: Path) -> pd.DataFrame:
    _, ex, ch = load(root)
    rt, _ = fifo_round_trips(ex, ch)
    rt = rt.reset_index(drop=True)
    rt["entry_ts"] = pd.to_datetime(rt["entry_et"].astype(str), utc=True)
    rt["exit_ts"] = pd.to_datetime(rt["exit_et"].astype(str), utc=True)
    rt["dir"] = np.where(rt["right"] == "C", 1, -1)
    return rt


def screen(rt: pd.DataFrame) -> pd.DataFrame:
    """逐笔判定入选 / 剔除及原因。"""
    out = rt.copy()
    e_day = out["entry_ts"].dt.tz_convert(ET).dt.date
    x_day = out["exit_ts"].dt.tz_convert(ET).dt.date
    reasons = [[] for _ in range(len(out))]
    for i, r in out.iterrows():
        if e_day[i] != x_day[i]:
            reasons[i].append("过夜")
        if r["dte"] > MAX_DTE:
            reasons[i].append("长期合约")
        if r["direction"] != "long":
            reasons[i].append("其他（非买入开仓）")
        w0, w1 = r["entry_ts"] - NEAR, max(r["exit_ts"], r["entry_ts"] + NEAR)
        opp = out[
            (out["und"] == r["und"])
            & (out["right"] != r["right"])
            & (out["entry_ts"] <= w1)
            & (out["exit_ts"] >= w0)
        ]
        if len(opp):
            near = opp[(opp["entry_ts"] - r["entry_ts"]).abs() <= NEAR]
            if len(near):
                kind = "跨式" if (near["strike"] == r["strike"]).any() else "宽跨式"
                reasons[i].append(f"疑似跨式（{kind}，{len(near)} 笔反向腿 ±5 分钟内开仓）")
            if len(opp) > len(near):
                reasons[i].append(f"对冲（持仓期间同标的有 {len(opp) - len(near)} 笔反向持仓）")
    out["reasons"] = ["；".join(x) for x in reasons]
    out["selected"] = [len(x) == 0 for x in reasons]
    out["primary_reason"] = [x[0].split("（")[0] if x else "" for x in reasons]
    return out


# ---------------- 2. 逐笔复盘 ----------------


def bar_features(b: pd.DataFrame, x: pd.DataFrame, reg: pd.DataFrame, reset: bool) -> dict:
    """每根 2 分钟 K 的入场前特征（未方向化的原始量 + 按方向的版本在 orient 里算）。"""
    o, h, lo, c, v = (b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    day = np.asarray(b.index.tz_convert(ET).date)
    pctb = x["pctb"].to_numpy()
    with np.errstate(invalid="ignore"):
        up_run, dn_run = _consec(pctb > 1), _consec(pctb < 0)
    L = np.minimum(np.r_[np.nan, lo[:-1]], lo)
    H = np.maximum(np.r_[np.nan, h[:-1]], h)
    hh = pd.Series(h).shift(1).rolling(20, min_periods=20).max().to_numpy()
    ll = pd.Series(lo).shift(1).rolling(20, min_periods=20).min().to_numpy()
    vr = v / pd.Series(v).shift(1).rolling(20, min_periods=20).mean().to_numpy()
    cs = pd.Series(c, index=b.index)
    if reset:
        g = cs.groupby(day)
        ma5 = g.transform(lambda s: s.rolling(5, min_periods=5).mean()).to_numpy()
        ma10 = g.transform(lambda s: s.rolling(10, min_periods=10).mean()).to_numpy()
    else:
        ma5 = cs.rolling(5, min_periods=5).mean().to_numpy()
        ma10 = cs.rolling(10, min_periods=10).mean().to_numpy()
    st = reg["state"].to_numpy()
    prev_range = np.r_[False, st[:-1] == RANGE] & np.r_[False, day[1:] == day[:-1]]
    et = b.index.tz_convert(ET)
    return {
        "state": st,
        "sdir": reg["dir"].to_numpy(),
        "up_run": up_run,
        "dn_run": dn_run,
        "L": L,
        "H": H,
        "mid": x["mid"].to_numpy(),
        "hw": x["hw"].to_numpy(),
        "lower": x["lower"].to_numpy(),
        "hh": hh,
        "ll": ll,
        "prev_range": prev_range,
        "hist": x["hist"].to_numpy(),
        "dif": x["dif"].to_numpy(),
        "dea": x["dea"].to_numpy(),
        "bw": x["bw"].to_numpy(),
        "close": c,
        "vr": vr,
        "ma5": ma5,
        "ma10": ma10,
        "minute": (et.hour * 60 + et.minute - 570).to_numpy() + 2,
        "day": day,
    }


def orient(f: dict, i: int, d: int) -> dict:
    """第 i 根 K 收盘时、方向 d 的特征（空头镜像，使规则对两个方向通用）。"""
    st, sd = int(f["state"][i]), int(f["sdir"][i])
    c, mid, hw = f["close"][i], f["mid"][i], f["hw"][i]
    with np.errstate(invalid="ignore", divide="ignore"):
        if d == 1:
            pb_pctb = (f["L"][i] - f["lower"][i]) / (2 * hw)
            pb_mid = (f["L"][i] - mid) / hw
            brk = bool(f["prev_range"][i] and c > f["hh"][i])
            k = int(f["up_run"][i])
        else:
            pb_pctb = 1 - (f["H"][i] - f["lower"][i]) / (2 * hw)
            pb_mid = (mid - f["H"][i]) / hw
            brk = bool(f["prev_range"][i] and c < f["ll"][i])
            k = int(f["dn_run"][i])
        same = bool(i) and f["day"][i - 1] == f["day"][i]
        hist_prev = f["hist"][i - 1] if same else np.nan
        bw_prev = f["bw"][i - 1] if same else np.nan
        return {
            "state_name": STATE_NAMES[st],
            "state_rel": ("同向" if sd == d else "反向") if sd else "",
            "state_same": int(sd == d and st in (IGNITION, CHANNEL)),
            "state_counter": int(sd == -d and st in (IGNITION, CHANNEL)),
            "state_range": int(st == RANGE),
            "state_channel_same": int(st == CHANNEL and sd == d),
            "ign_outer_k": k if (st == IGNITION and sd == d) else 0,
            "pb_pctb": pb_pctb,
            "pb_mid_hw": pb_mid,
            "breakout": int(brk),
            "hist_n": d * f["hist"][i] / c,
            "hist_chg_n": d * (f["hist"][i] - hist_prev) / c,
            "dif_gt_dea": int(d * (f["dif"][i] - f["dea"][i]) > 0),
            "dif_gt_0": int(d * f["dif"][i] > 0),
            "bw": f["bw"][i],
            "bw_expand": f["bw"][i] / bw_prev - 1,
            "vol_ratio": f["vr"][i],
            "ma5_pos": d * (c - f["ma5"][i]) / c,
            "ma10_pos": d * (c - f["ma10"][i]) / c,
            "minute": int(f["minute"][i]),
            "spot": c,
        }


def pattern_of(o: dict) -> str:
    if o["ign_outer_k"] > 0:
        return f"顺点火第 {o['ign_outer_k']} 根"
    if o["state_channel_same"] and np.isfinite(o["pb_pctb"]):
        return "通道回踩"
    if o["breakout"]:
        return "突破"
    return "无法归类"


def trade_ticks(client, root: Path, r) -> pd.DataFrame | None:
    """该笔合约 [入场 − 5 分钟, 出场 + 5 分钟] 的逐笔 NBBO（缓存）。"""
    d = r["entry_ts"].tz_convert(ET).date()
    p = (
        root
        / "diag"
        / "owner_ticks"
        / f"date={d.isoformat()}"
        / f"{r['symbol']}_{r['entry_ts'].value}_{r['exit_ts'].value}.parquet"
    )
    if p.exists():
        return pd.read_parquet(p)
    if client is None:
        return None
    s = (r["entry_ts"] - NEAR).tz_convert(ET)
    e = (r["exit_ts"] + NEAR).tz_convert(ET)
    try:
        df = client.history_quote(
            r["und"],
            r["expiry"],
            d,
            interval="tick",
            strike=f"{r['strike']:.3f}",
            right="call" if r["right"] == "C" else "put",
            start_time=s.strftime("%H:%M:%S.000"),
            end_time=e.strftime("%H:%M:%S.000"),
        )
    except Exception as exc:  # noqa: BLE001 - 单笔失败记录后继续
        log.warning("tick 失败 %s：%s", r["symbol"], exc)
        return None
    df = df[[c for c in ("ts", "bid", "ask") if c in df.columns]]
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def tick_review(tk: pd.DataFrame | None, r) -> dict:
    if tk is None or tk.empty:
        return {}
    tk = tk[(tk["ask"] > 0) & (tk["bid"] >= 0) & (tk["ask"] >= tk["bid"])].sort_values("ts")
    ts = pd.to_datetime(tk["ts"], utc=True)
    at = lambda t: tk[ts <= t].iloc[-1] if (ts <= t).any() else None  # noqa: E731 - 当时 NBBO
    a0, b1 = at(r["entry_ts"]), at(r["exit_ts"])
    hold = tk[(ts >= r["entry_ts"]) & (ts <= r["exit_ts"])]
    out = {}
    if a0 is not None:
        out["entry_vs_ask_pct"] = (r["entry_px"] / a0["ask"] - 1) * 100
    if b1 is not None and b1["bid"] > 0:
        out["exit_vs_bid_pct"] = (r["exit_px"] / b1["bid"] - 1) * 100
    if len(hold):
        out["mfe_pct"] = (hold["bid"].max() / r["entry_px"] - 1) * 100
        out["mae_pct"] = (hold["bid"].min() / r["entry_px"] - 1) * 100
    return out


def review(root: Path, sel: pd.DataFrame, v4: dict, pull: bool) -> pd.DataFrame:
    client = ThetaClient(concurrency=1) if pull else None
    rows = []
    state = {}
    for conv in ("A", "C"):
        for und in sorted(sel["und"].unique()):
            c = v4["conventions"][conv]
            try:
                b = v4d.build_bars(root, und, c, dt.date(2026, 10, 21))
            except FileNotFoundError:
                continue
            cfg = v4d.state_cfg(v4, conv)
            x = indicators(b, cfg, daily_reset=c["reset"], warmup=c["warmup"])
            reg = regimes(b, x, cfg, v4["period"])
            state[(conv, und)] = (b, bar_features(b, x, reg, c["reset"]))
    for i, r in sel.iterrows():
        base = {
            "row": i,
            "und": r["und"],
            "symbol": r["symbol"],
            "dir": int(r["dir"]),
            "entry": r["entry_ts"].tz_convert(ET).strftime("%Y-%m-%d %H:%M:%S"),
            "dte": int(r["dte"]),
            "qty": r["qty"],
            "hold_min": round(float(r["hold_min"]), 1),
            "pnl_net": round(float(r["pnl_net"]), 2),
            "pnl_pct": round(float(r["pnl_net"] / (r["entry_px"] * 100 * r["qty"]) * 100), 2),
            "before_first_bar": bool(r["entry_ts"].tz_convert(ET).time() < dt.time(9, 32)),
            "lots": int(r.get("lots", 1)),
        }
        for conv in ("A", "C"):
            if (conv, r["und"]) not in state:
                continue
            b, f = state[(conv, r["und"])]
            close = b.index + pd.Timedelta(minutes=2)
            same_day = (
                np.asarray(b.index.tz_convert(ET).date) == r["entry_ts"].tz_convert(ET).date()
            )
            k = np.flatnonzero(same_day & (close <= r["entry_ts"]))
            if not len(k):
                base[f"{conv}_state"] = "未覆盖（首根 K 收盘前）"
                continue
            o = orient(f, int(k[-1]), int(r["dir"]))
            base[f"{conv}_state"] = o["state_name"] + (
                f" {o['state_rel']}" if o["state_rel"] else ""
            )
            base[f"{conv}_pattern"] = pattern_of(o)
            if conv == "A":
                base.update({kk: o[kk] for kk in TREE_FEATURES})
                base["strike_dist_pct"] = r["dir"] * (r["strike"] - o["spot"]) / o["spot"] * 100
        base.update(tick_review(trade_ticks(client, root, r), r))
        rows.append(base)
    return pd.DataFrame(rows)


# ---------------- 3. 规律 ----------------


def positions(t: pd.DataFrame) -> pd.DataFrame:
    """FIFO 配对 → 入场决策（同一合约同一入场时刻的分批平仓合并；盈亏相加）。规律与规则在这一层做，
    否则一次入场拆成的多行会被当成多个独立样本（89 行只对应 54 次入场）。"""
    t = t.assign(pos=t["symbol"] + "@" + t["entry"], prem=t["pnl_net"] / (t["pnl_pct"] / 100))
    first = [
        c
        for c in t.columns
        if c
        not in (
            "pnl_net",
            "pnl_pct",
            "qty",
            "hold_min",
            "mfe_pct",
            "mae_pct",
            "exit_vs_bid_pct",
            "prem",
            "row",
        )
    ]
    g = t.groupby("pos", sort=False)
    out = g[first].first()
    out["lots"] = g.size()
    out["qty"] = g["qty"].sum()
    out["pnl_net"] = g["pnl_net"].sum().round(2)
    out["pnl_pct"] = (out["pnl_net"] / g["prem"].sum() * 100).round(2)
    out["hold_min"] = g["hold_min"].max()
    out["mfe_pct"] = g["mfe_pct"].max()
    out["mae_pct"] = g["mae_pct"].min()
    out["exit_vs_bid_pct"] = g["exit_vs_bid_pct"].mean()
    return out.reset_index(drop=True)


def summarize(t: pd.DataFrame, conv: str) -> pd.DataFrame:
    g = t.groupby([f"{conv}_state", f"{conv}_pattern"], dropna=False)["pnl_net"]
    wins = t.assign(w=t["pnl_net"].where(t["pnl_net"] > 0), l=t["pnl_net"].where(t["pnl_net"] <= 0))
    gw = wins.groupby([f"{conv}_state", f"{conv}_pattern"], dropna=False)
    lots = t.groupby([f"{conv}_state", f"{conv}_pattern"], dropna=False)["lots"].sum()
    return pd.DataFrame(
        {
            "入场数": g.size(),
            "FIFO 行数": lots,
            "胜率": g.apply(lambda s: round(float((s > 0).mean()), 2)),
            "盈亏比": (gw["w"].mean() / -gw["l"].mean()).round(2),
            "合计 USD": g.sum().round(0),
            "均值 USD": g.mean().round(1),
        }
    ).reset_index()


def win_loss_compare(t: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in TREE_FEATURES
        + [
            "dte",
            "strike_dist_pct",
            "hold_min",
            "entry_vs_ask_pct",
            "exit_vs_bid_pct",
            "mfe_pct",
            "mae_pct",
        ]
        if c in t
    ]
    w, lo = t[t["pnl_net"] > 0], t[t["pnl_net"] <= 0]
    rows = []
    for c in cols:
        a, b = w[c].dropna().astype(float), lo[c].dropna().astype(float)
        p = stats.mannwhitneyu(a, b).pvalue if len(a) >= 3 and len(b) >= 3 else np.nan
        rows.append(
            {
                "特征": c,
                "盈利单中位": round(float(a.median()), 4) if len(a) else np.nan,
                "亏损单中位": round(float(b.median()), 4) if len(b) else np.nan,
                "Mann–Whitney p（描述性）": round(float(p), 4) if np.isfinite(p) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("Mann–Whitney p（描述性）")


def extract_rules(t: pd.DataFrame, max_rules: int = 5, min_leaf: int = 5) -> list[dict]:
    d = t.dropna(subset=["state_same"]).reset_index(drop=True)
    X = np.nan_to_num(d[TREE_FEATURES].astype(float).to_numpy(), nan=-999.0)
    y = (d["pnl_net"] > 0).astype(int).to_numpy()
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=min_leaf, random_state=0).fit(X, y)
    tr = tree.tree_
    leaf = tree.apply(X)
    rules = []

    def walk(node, path):
        if tr.children_left[node] == -1:
            m = leaf == node
            if m.sum() and y[m].mean() > 0.5:
                rules.append(
                    {
                        "conditions": [[c, op, float(v)] for c, op, v in path],
                        "n": int(m.sum()),
                        "lots": int(d.loc[m, "lots"].sum()),
                        "days": int(d.loc[m, "entry"].str[:10].nunique()),
                        "n_nan": int(
                            d.loc[m, sorted({c for c, _, _ in path})].isna().any(axis=1).sum()
                        ),
                        "win_rate": round(float(y[m].mean()), 3),
                        "sum_usd": round(float(d.loc[m, "pnl_net"].sum()), 2),
                    }
                )
            return
        c, v = TREE_FEATURES[tr.feature[node]], tr.threshold[node]
        walk(tr.children_left[node], path + [(c, "<=", v)])
        walk(tr.children_right[node], path + [(c, ">", v)])

    walk(0, [])
    rules.sort(key=lambda r: (-r["win_rate"], -r["n"]))
    return rules[:max_rules]


def guard_evidence(t: pd.DataFrame, conv: str) -> pd.DataFrame:
    s = t[f"{conv}_state"].fillna("")
    groups = {
        "逆 IGNITION 入场": s.str.startswith("IGNITION 反向"),
        "首根 K 收盘前入场（< 09:32）": t["before_first_bar"],
        "RANGE 中入场": s.str.startswith("RANGE"),
        "其余": ~(
            s.str.startswith("IGNITION 反向") | t["before_first_bar"] | s.str.startswith("RANGE")
        ),
    }
    rows = []
    for k, m in groups.items():
        x = t.loc[m, "pnl_net"]
        rows.append(
            {
                "组": k,
                "入场数": int(m.sum()),
                "FIFO 行数": int(t.loc[m, "lots"].sum()),
                "交易日数": int(t.loc[m, "entry"].str[:10].nunique()),
                "胜率": round(float((x > 0).mean()), 2) if len(x) else np.nan,
                "合计 USD": round(float(x.sum()), 0),
                "均值 USD": round(float(x.mean()), 1) if len(x) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rule_text(r: dict) -> str:
    return " 且 ".join(f"{c} {op} {v:.4g}" for c, op, v in r["conditions"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    sel, ep = d030_sample(root)
    t = review(root, sel, v4, pull=not args.no_pull)
    pos = t  # D-030 的单位已是持仓段（分批建仓 / 卖出已合并），不再按 FIFO 行聚合
    cov = pos[pos["A_state"].notna()] if "A_state" in pos else pos.iloc[0:0]
    rules = extract_rules(cov) if len(cov) >= 10 else []
    RULES_OUT.write_text(
        yaml.safe_dump(
            {
                "source": "zr/research/intraday_single_leg.py（口径 A，D-030 入选日内单腿持仓段）",
                "rules": [{"id": f"R{i + 1}", **r} for i, r in enumerate(rules)],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cnt = ep["category"].value_counts().to_dict()
    L = [
        "# 负责人交易史：日内单腿复盘与规律（D-030 样本）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 复现 "
        "`uv run python -m zr.research.intraday_single_leg`",
        "",
        "## 1. 样本",
        "",
        "按负责人指令（D-030），只用 `docs/research/INTRADAY_SINGLE_LEG_LIST.md` 的入选样本"
        f"（持仓段为单位）：全部 {len(ep)} 段，入选 {cnt.get('入选', 0)}、剔除 "
        f"{cnt.get('剔除', 0)}、反手重叠（待确认）{cnt.get('待确认', 0)}、"
        f"其他 {cnt.get('其他', 0)}。入选按标的：{sel['und'].value_counts().to_dict()}，"
        f"合计扣费后 {sel['pnl_net'].sum():.2f} USD。"
        "此前 D-029 版本（FIFO 行为单位、DTE ≤ 7）已被取代，见 git 历史。",
        "",
        "## 2. 逐笔复盘（口径 A = D-027；C = 含盘前盘后连续口径对照）",
        "",
        "状态取入场前最后一根已完成的 2 分钟 K；pb_pctb / pb_mid_hw = 回踩极值的 %b 与距中轨"
        "（带半宽倍数，空头镜像）；hist_n = MACD(6,13,6) 柱 ÷ 收盘（方向化）；"
        "entry_vs_ask_pct = 成交价相对当时 ask；mfe/mae = 持仓期间 bid 相对入场价的最大有利/"
        "不利偏移；exit_vs_bid_pct = 出场价相对当时 bid。只有 MU、QQQ 有正股 K 线。",
        "",
        df_to_md(t.round(4)),
        "",
        "## 3. 规律",
        "",
        f"以下按持仓段统计（{len(pos)} 段；「FIFO 行数」= 该段包含的分批卖出配对行）。",
        "",
        "### 按状态 × 顺逆 × 形态（口径 A）",
        "",
        df_to_md(summarize(cov, "A")) if len(cov) else "无",
        "",
        "### 同上（口径 C 对照）",
        "",
        df_to_md(summarize(cov, "C")) if len(cov) and "C_state" in cov else "无",
        "",
        "### 盈利单 vs 亏损单（每个特征的中位数与秩和检验，描述性，未做多重比较校正）",
        "",
        df_to_md(win_loss_compare(cov)) if len(cov) else "无",
        "",
        "### 深度 ≤3 决策树：「盈利单长什么样」（只用入场前 K 线特征；口径 A；叶子 ≥5 次入场；"
        "缺失值按 −999 参与比较）",
        "",
        df_to_md(
            pd.DataFrame(
                [
                    {
                        "规则": f"R{i + 1}",
                        "条件": rule_text(r),
                        "入场数": r["n"],
                        "FIFO 行数": r["lots"],
                        "交易日数": r["days"],
                        "其中依赖缺失值": r["n_nan"],
                        "胜率": r["win_rate"],
                        "合计 USD": r["sum_usd"],
                    }
                    for i, r in enumerate(rules)
                ]
            )
        )
        if rules
        else "无胜率 > 0.5 的叶子。",
        "",
        "这些规则是在同一批样本上拟合出来的描述，样本内胜率必然偏高；已冻结到 "
        "`config/v4_rules_d030.yaml`，在两年 MU 数据上的独立回放见 V4_DECISIVE.md（R-037）。",
        "",
        "### 护栏证据（入选样本，口径 A）",
        "",
        df_to_md(guard_evidence(cov, "A")) if len(cov) else "无",
        "",
        "口径 C：",
        "",
        df_to_md(guard_evidence(cov, "C")) if len(cov) and "C_state" in cov else "无",
        "",
        "## 边界",
        "",
        "- 样本：负责人 2026-06-09～2026-08-18 的期权往返（FIFO 配对）中按上述条件入选者；"
        "只有 MU、QQQ 可做状态复盘。手工交易，样本小，全部为描述性。",
        "- 「单腿」按长桥订单（每单一个合约）+ ±5 分钟内无同标的反向腿判定；同方向多合约同时开仓"
        "不算多腿。DTE 为交易史记录的日历日 DTE。",
        "- tick 为 ThetaData 逐笔 NBBO；成交价相对 ask/bid 用成交时刻之前最近一笔报价。",
        "",
    ]
    OUT.write_text("\n".join(L), encoding="utf-8")
    store = root / "research" / "v4"
    store.mkdir(parents=True, exist_ok=True)
    t.to_parquet(store / "owner_intraday_review.parquet", index=False)
    print(f"D-030 入选 {len(sel)} 段（可复盘 {len(cov)}）；规则 {len(rules)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------- 4. 规则向量化（R-035 回放用；与 orient 逐值一致） ----------------


def oriented_arrays(f: dict, d: int) -> dict[str, np.ndarray]:
    st, sd, c = f["state"], f["sdir"], f["close"]
    mid, hw = f["mid"], f["hw"]
    same = np.r_[False, f["day"][1:] == f["day"][:-1]]
    hist_prev = np.where(same, np.r_[np.nan, f["hist"][:-1]], np.nan)
    bw_prev = np.where(same, np.r_[np.nan, f["bw"][:-1]], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        if d == 1:
            pb_pctb = (f["L"] - f["lower"]) / (2 * hw)
            pb_mid = (f["L"] - mid) / hw
            brk = f["prev_range"] & (c > f["hh"])
            k = f["up_run"]
        else:
            pb_pctb = 1 - (f["H"] - f["lower"]) / (2 * hw)
            pb_mid = (mid - f["H"]) / hw
            brk = f["prev_range"] & (c < f["ll"])
            k = f["dn_run"]
        trend = np.isin(st, (IGNITION, CHANNEL))
        return {
            "state_same": ((sd == d) & trend).astype(float),
            "state_counter": ((sd == -d) & trend).astype(float),
            "state_range": (st == RANGE).astype(float),
            "state_channel_same": ((st == CHANNEL) & (sd == d)).astype(float),
            "ign_outer_k": np.where((st == IGNITION) & (sd == d), k, 0).astype(float),
            "pb_pctb": pb_pctb,
            "pb_mid_hw": pb_mid,
            "breakout": brk.astype(float),
            "hist_n": d * f["hist"] / c,
            "hist_chg_n": d * (f["hist"] - hist_prev) / c,
            "dif_gt_dea": (d * (f["dif"] - f["dea"]) > 0).astype(float),
            "dif_gt_0": (d * f["dif"] > 0).astype(float),
            "bw": f["bw"],
            "bw_expand": f["bw"] / bw_prev - 1,
            "vol_ratio": f["vr"],
            "ma5_pos": d * (c - f["ma5"]) / c,
            "ma10_pos": d * (c - f["ma10"]) / c,
            "minute": f["minute"].astype(float),
        }


def rule_mask(arr: dict[str, np.ndarray], conditions: list) -> np.ndarray:
    """规则 → 布尔数组；缺失值按 −999 参与比较（与提取时的决策树一致）。"""
    m = np.ones(len(next(iter(arr.values()))), bool)
    for col, op, thr in conditions:
        v = np.nan_to_num(arr[col].astype(float), nan=-999.0)
        m &= (v <= thr) if op == "<=" else (v > thr)
    return m
