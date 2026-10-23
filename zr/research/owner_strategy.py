"""负责人挂单策略的提取与参数化（成交模型校准修正版，替代 R-039 的「固定挂单价 × 固定 T」）。

1. 逐单特征（features）：以委托提交时刻为决策时刻（盘前提交取 09:30），记录提交时买一 / 卖一 /
   中间价 / 最新成交价（OPRA，提交前最后一笔），初始挂单价相对这四者的 tick 差（按方向：正 = 更靠近
   对手价）、提交至成交或撤单 / 过期的时长、改价次数、每次改价的 tick 幅度（按方向）与间隔、
   成交时刻相对提交时刻的正股价格变化（MU、QQQ 有 1 分钟 K；其他标的不覆盖）。
2. 策略参数（extract）：按买（开仓）/ 卖（平仓）分别给出初始挂单位置（相对中间价，spread 倍数）的
   经验分布、首次改价前等待时长、改价间隔与幅度、改价次数、委托总时长（等待时长，截到 5–120 秒）
   的经验分布、撤单前市场走向。写 config/owner_strategy.yaml（脚本生成，勿手改）。
样本：zr/research/calib_sample.classify 纳入的订单（D-030 日内单腿）中的限价单（LO）；
数据：改价历史（diag/order_history.parquet）+ ThetaData 缓存。
运行：uv run python -m zr.research.owner_strategy
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.config import get_settings
from zr.execution.ticks import tick_size
from zr.features.bars import load_bars
from zr.journal.analyze_history import load
from zr.research import fill_model as fm
from zr.research import fill_segments as fs

log = logging.getLogger("zr.research.owner_strategy")
REPO = Path(__file__).resolve().parents[2]
STRAT = REPO / "config" / "owner_strategy.yaml"
WAIT_CLIP = (5.0, 120.0)
Q_CAL = 0.9  # 挂单段逐段重放 Brier 最小（zr/research/fill_segments.calibrate_q，校准样本）
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _last_trade(t: pd.DataFrame, ns: int) -> float:
    if t is None or t.empty:
        return np.nan
    tn = fs._ns(t["ts"])
    i = int(np.searchsorted(tn, ns, side="right")) - 1
    return float(t["price"].iloc[i]) if i >= 0 else np.nan


def _und_px(bars: dict, und: str, ns: int) -> float:
    b = bars.get(und)
    if b is None:
        return np.nan
    i = int(np.searchsorted(b[0], ns, side="right")) - 1
    return float(b[1][i]) if i >= 0 else np.nan


def features(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from zr.research.calib_sample import classify

    s = classify(root)
    s = s[s["calib"]]
    h = fs.load_history(root)
    _, ex, _ = load(root)
    ex = ex.assign(ts=fm._utc(ex["ts_et"], naive_is_utc=False), order_id=ex["order_id"].astype(str))
    bars = {}
    for u in ("MU", "QQQ"):
        b = load_bars(root, u)
        bars[u] = (
            pd.DatetimeIndex(b.index).as_unit("ns").asi8 + 60_000_000_000,  # 1 分钟 K 收盘时刻
            b["close"].to_numpy(float),
        )
    rows, amends = [], []
    for _, o in s[s["type"] == "LO"].iterrows():
        w = fm.window(o)
        d = fm._load_ticks(root, o["order_id"])
        hh = h[h["order_id"] == o["order_id"]]
        if w is None or d is None or hh.empty:
            continue
        q, t = d
        sgn = 1 if o["side"] == "Buy" else -1
        open_ns = (w[0] + fm.PRE).value
        bnd = hh[hh["status"].isin(fs.BOUNDARY)]
        term = hh[hh["status"].isin(fs.TERMINAL)]
        st = fs._ns(bnd["ts"])
        sub = max(int(st[0]), open_ns)
        a0 = fs._asof(q, sub)
        bid, ask = float(a0["bid"]), float(a0["ask"])
        mid, spr = (bid + ask) / 2, ask - bid
        tick = float(tick_size(o["symbol"], mid))
        last = _last_trade(t, sub)
        px0 = float(bnd["price"].iloc[0])
        fills = ex[ex["order_id"] == o["order_id"]]
        end_ns = int(fs._ns(term["ts"])[-1]) if len(term) else (w[1] - fm.POST).value
        f0 = int(fs._ns(fills["ts"]).min()) if len(fills) else None
        u0 = _und_px(bars, o["und"], sub)
        uf = _und_px(bars, o["und"], f0) if f0 else np.nan
        dirn = 1 if o["right"] == "call" else -1
        rec = {
            "order_id": o["order_id"],
            "group": o["group"],
            "symbol": o["symbol"],
            "und": o["und"],
            "side": o["side"],
            "status": o["status"],
            "premarket_submit": int(st[0]) < open_ns,
            "submit_ns": sub,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "last": last,
            "spread_ticks": spr / tick,
            "tick": tick,
            "limit0": px0,
            # 按方向：正 = 更靠近对手价（买单更高、卖单更低）
            "vs_bid_ticks": sgn * (px0 - bid) / tick,
            "vs_ask_ticks": sgn * (px0 - ask) / tick,
            "vs_mid_ticks": sgn * (px0 - mid) / tick,
            "vs_last_ticks": sgn * (px0 - last) / tick if np.isfinite(last) else np.nan,
            "k0_spread": sgn * (px0 - mid) / spr if spr > 0 else np.nan,
            "filled": len(fills) > 0,
            "fill_s": (f0 - sub) / 1e9 if f0 else np.nan,
            "life_s": (end_ns - sub) / 1e9,
            "n_amend": int((bnd["status"] == "Replaced").sum()),
            "first_amend_s": (int(st[1]) - sub) / 1e9 if len(st) > 1 else np.nan,
            "fill_px": float((fills["price"] * fills["quantity"]).sum() / fills["quantity"].sum())
            if len(fills)
            else np.nan,
            "und_chg_pct": (uf / u0 - 1) * 100 if np.isfinite(u0) and np.isfinite(uf) else np.nan,
            "und_chg_dir_pct": dirn * (uf / u0 - 1) * 100
            if np.isfinite(u0) and np.isfinite(uf)
            else np.nan,
        }
        # 撤单 / 过期：撤单时中间价相对提交时（按方向，正 = 价格朝对手方向跑开，对买单 = 涨了）
        if not rec["filled"]:
            ac = fs._asof(q, end_ns)
            rec["away_at_cancel_ticks"] = sgn * (((ac["bid"] + ac["ask"]) / 2) - mid) / tick
        rows.append(rec)
        prices = bnd["price"].astype(float).to_numpy()
        for k in range(1, len(st)):
            if int(st[k]) < open_ns:  # 盘前改价不计
                continue
            amends.append(
                {
                    "order_id": o["order_id"],
                    "side": o["side"],
                    "k": k,
                    "interval_s": (int(st[k]) - max(int(st[k - 1]), open_ns)) / 1e9,
                    "since_submit_s": (int(st[k]) - sub) / 1e9,
                    "step_ticks": sgn * (prices[k] - prices[k - 1]) / tick,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(amends)


def _q(x: pd.Series) -> list[float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return [round(float(x.quantile(p)), 4) for p in QS] if len(x) else []


def chase_fit(root: Path) -> dict:
    """改价幅度 ~ 上次挂单以来中间价朝对手方向的移动（tick），最小二乘（买卖合并；挂单段来自
    zr/research/fill_segments 的校准样本输出）。"""
    s = pd.read_parquet(root / "research" / "fill_model" / "owner_segments.parquet")
    s = s[s["skip"] == ""].sort_values(["order_id", "seg"])
    sg = np.where(s["side"] == "Buy", 1, -1)
    s["move"] = sg * (s["mid0"] - s.groupby("order_id")["mid0"].shift()) / s["tick"]
    s["step"] = sg * (s["limit"] - s.groupby("order_id")["limit"].shift()) / s["tick"]
    lt = s[s["seg"] > 0].dropna(subset=["move", "step"])
    b, a = np.polyfit(lt["move"].to_numpy(float), lt["step"].to_numpy(float), 1)
    return {
        "chase_a": round(float(a), 4),
        "chase_b": round(float(b), 4),
        "chase_n": int(len(lt)),
        "chase_corr": round(float(lt["move"].corr(lt["step"])), 3),
    }


def extract(f: pd.DataFrame, a: pd.DataFrame, chase: dict | None = None) -> dict:
    """参数化策略（按方向）。经验分布用分位点表示；模拟时按事件逐单抽样（固定种子）。"""
    out = {"quantiles": list(QS), "wait_clip_s": list(WAIT_CLIP)}
    for side in ("Buy", "Sell"):
        g = f[(f["side"] == side) & ~f["premarket_submit"]]
        aa = a[(a["side"] == side) & (a["interval_s"] > 0)]
        life = g["life_s"].clip(*WAIT_CLIP)
        out[side] = {
            "n_orders": int(len(g)),
            "k0_spread": [
                round(float(x), 4) for x in g["k0_spread"].dropna()
            ],  # 经验分布（全部值）
            "k0_spread_quantiles": _q(g["k0_spread"]),
            "wait_s": [round(float(x), 2) for x in life],  # 委托总时长（截到 5–120 秒）的经验分布
            "wait_s_quantiles": _q(life),
            "amend_interval_s": float(aa["interval_s"].median()) if len(aa) else None,
            "amend_step_ticks": float(aa["step_ticks"].median()) if len(aa) else None,
            "amend_toward_frac": round(float((aa["step_ticks"] > 0).mean()), 3)
            if len(aa)
            else None,
            "amend_count_quantiles": _q(g["n_amend"]),
            "fill_rate": round(float(g["filled"].mean()), 3),
            # 撤单条件：撤单 / 过期时中间价相对提交时朝对手方向跑开的 tick 数（中位数）
            "cancel_away_ticks": float(g.loc[~g["filled"], "away_at_cancel_ticks"].median())
            if (~g["filled"]).any()
            else None,
            "n_cancel": int((~g["filled"]).sum()),
            "amend_counts": [int(x) for x in g["n_amend"]],
            **({k: chase[k] for k in ("chase_a", "chase_b")} if chase else {}),
        }
    if chase:
        out["chase_fit"] = chase
    return out


def write_strategy(params: dict) -> None:
    head = (
        "# 由 `uv run python -m zr.research.owner_strategy` 生成，勿手改。负责人挂单策略的参数化"
        "（2026-06-09～2026-08-19 期权限价单，盘前提交的单子不进参数）。\n"
    )
    STRAT.write_text(
        head + yaml.safe_dump(params, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    f, a = features(root)
    store = root / "research" / "fill_model"
    store.mkdir(parents=True, exist_ok=True)
    f.to_parquet(store / "owner_order_strategy.parquet", index=False)
    a.to_parquet(store / "owner_amendments.parquet", index=False)
    p = extract(f, a, chase_fit(root))
    write_strategy(p)
    from zr.backtest.owner_fill import OwnerStrategy

    strat = OwnerStrategy.from_yaml(STRAT, queue_fill_prob=Q_CAL)
    v = validate(root, f, strat)
    v.to_parquet(store / "owner_strategy_validation.parquet", index=False)
    print(validation_table(v).T.to_string())
    report(root)
    return 0


# ---------------- 验证：从负责人提交时刻重放全部校准样本订单 ----------------

N_SEEDS = 50


def validate(root: Path, f: pd.DataFrame, strat) -> pd.DataFrame:
    """每单 N_SEEDS 次抽样（委托在提交时刻到达，lag 0；一律按入场口径——等满或跑开即撤单）。
    另跑一次「该单实际参数」：k0、等待时长（截到 5–120 s）、改价次数取该单真实值。"""
    import dataclasses

    from zr.backtest.owner_fill import execute_owner
    from zr.research.owner_limit import order_book

    st0 = dataclasses.replace(strat, lag_s=0.0)
    rows = []
    for _, o in f.iterrows():
        bk = order_book(root, [o["order_id"]])
        if bk is None:
            continue
        side = 1 if o["side"] == "Buy" else -1
        sims = []
        for sd in range(N_SEEDS):
            r = execute_owner(
                bk,
                side,
                1.0,
                int(o["submit_ns"]),
                st0,
                o["symbol"],
                np.random.default_rng(sd),
                "entry",
            )
            if r is not None:
                sims.append(r)
        own = execute_owner(
            bk,
            side,
            1.0,
            int(o["submit_ns"]),
            st0,
            o["symbol"],
            np.random.default_rng(0),
            "entry",
            draw={
                "k0": o["k0_spread"],
                "wait_s": float(np.clip(o["life_s"] + (1.0 if o["filled"] else 0.0), *WAIT_CLIP)),
                "n_amend": int(o["n_amend"]),
            },
        )
        filled = [r for r in sims if r.filled_qty > 0]
        sgn = side
        rows.append(
            {
                "order_id": o["order_id"],
                "side": o["side"],
                "actual_filled": bool(o["filled"]),
                "actual_fill_s": o["fill_s"],
                "actual_px": o["fill_px"],
                "tick": o["tick"],
                "sim_fill_prob": len(filled) / len(sims) if sims else np.nan,
                "sim_fill_s": float(
                    np.median([(r.first_fill_ns - o["submit_ns"]) / 1e9 for r in filled])
                )
                if filled
                else np.nan,
                "sim_px": float(np.mean([r.avg_px for r in filled])) if filled else np.nan,
                "away_cancel_frac": float(np.mean([r.end_reason == "away" for r in sims]))
                if sims
                else np.nan,
                "own_filled": bool(own is not None and own.filled_qty > 0),
                "own_px": own.avg_px if own is not None and own.filled_qty > 0 else np.nan,
                "own_fill_s": (own.first_fill_ns - o["submit_ns"]) / 1e9
                if own is not None and own.filled_qty > 0
                else np.nan,
            }
        )
    v = pd.DataFrame(rows)
    v["px_bias_ticks"] = (
        np.where(v["side"] == "Buy", 1, -1) * (v["sim_px"] - v["actual_px"]) / v["tick"]
    )
    v["px_bias_pct"] = (
        np.where(v["side"] == "Buy", 1, -1) * (v["sim_px"] / v["actual_px"] - 1) * 100
    )
    v["own_bias_ticks"] = (
        np.where(v["side"] == "Buy", 1, -1) * (v["own_px"] - v["actual_px"]) / v["tick"]
    )
    v["time_bias_s"] = v["sim_fill_s"] - v["actual_fill_s"]
    v["own_time_bias_s"] = v["own_fill_s"] - v["actual_fill_s"]
    _ = sgn
    return v


def validation_table(v: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, g in [
        ("全部", v),
        ("买（开仓）", v[v["side"] == "Buy"]),
        ("卖（平仓）", v[v["side"] == "Sell"]),
    ]:
        a, c = g[g["actual_filled"]], g[~g["actual_filled"]]
        rows.append(
            {
                "方向": side,
                "订单": len(g),
                "实际成交率": round(float(g["actual_filled"].mean()), 3),
                "模拟成交率（抽样期望）": round(float(g["sim_fill_prob"].mean()), 3),
                "模拟成交率（该单实际参数）": round(float(g["own_filled"].mean()), 3),
                "实际成交单：模拟也成交（抽样）": round(float(a["sim_fill_prob"].mean()), 3)
                if len(a)
                else np.nan,
                "实际撤单 / 过期：模拟也未成交（抽样）": round(
                    float(1 - c["sim_fill_prob"].mean()), 3
                )
                if len(c)
                else np.nan,
                "实际撤单 / 过期：模拟也未成交（实际参数）": round(
                    float(1 - c["own_filled"].mean()), 3
                )
                if len(c)
                else np.nan,
                "成交价偏差中位数 tick（抽样，正 = 模拟更差）": round(
                    float(a["px_bias_ticks"].median()), 2
                ),
                "成交价偏差中位数 %（抽样）": round(float(a["px_bias_pct"].median()), 3),
                "成交价偏差中位数 tick（实际参数）": round(float(a["own_bias_ticks"].median()), 2),
                "成交耗时偏差中位数 s（抽样）": round(float(a["time_bias_s"].median()), 1),
                "成交耗时偏差中位数 s（实际参数）": round(float(a["own_time_bias_s"].median()), 1),
            }
        )
    return pd.DataFrame(rows)


# ---------------- 报告 ----------------

OUT = REPO / "docs" / "research" / "FILL_MODEL_CALIBRATION.md"


def _dist(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return "—"
    q = x.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist()
    return f"P10 {q[0]} / P25 {q[1]} / 中位 {q[2]} / P75 {q[3]} / P90 {q[4]}（n={len(x)}）"


def report(root: Path) -> None:
    from zr.ops.markdown import df_to_md
    from zr.research.calib_sample import classify

    store = root / "research" / "fill_model"
    f = pd.read_parquet(store / "owner_order_strategy.parquet")
    a = pd.read_parquet(store / "owner_amendments.parquet")
    v = pd.read_parquet(store / "owner_strategy_validation.parquet")
    seg = pd.read_parquet(store / "owner_segments.parquet")
    p = yaml.safe_load(STRAT.read_text(encoding="utf-8"))
    h = fs.load_history(root)
    cl = classify(root)
    cls = cl.groupby(["group", "side", "class"]).size().reset_index(name="订单数")
    unk = int(cl["class"].str.startswith("无法判断").sum())
    feat_rows = []
    for side, g in (("买（开仓）", f[f["side"] == "Buy"]), ("卖（平仓）", f[f["side"] == "Sell"])):
        for col, name in (
            ("vs_bid_ticks", "挂单价 − 买一（tick，按方向）"),
            ("vs_ask_ticks", "挂单价 − 卖一（tick，按方向）"),
            ("vs_mid_ticks", "挂单价 − 中间价（tick，按方向）"),
            ("vs_last_ticks", "挂单价 − 最新成交价（tick，按方向）"),
            ("k0_spread", "挂单位置（相对中间价，spread 倍数）"),
            ("spread_ticks", "提交时 spread（tick）"),
            ("fill_s", "提交 → 首笔成交（秒，已成交）"),
            ("life_s", "提交 → 成交完 / 撤单 / 过期（秒）"),
            ("n_amend", "改价次数"),
            ("first_amend_s", "提交 → 首次改价（秒）"),
            ("und_chg_pct", "提交 → 首笔成交 正股变化 %"),
            ("und_chg_dir_pct", "同上，按期权方向（看涨 +、看跌 −）"),
        ):
            feat_rows.append({"方向": side, "量": name, "分布": _dist(g[col])})
    for side, g in (("买（开仓）", a[a["side"] == "Buy"]), ("卖（平仓）", a[a["side"] == "Sell"])):
        feat_rows.append({"方向": side, "量": "改价间隔（秒）", "分布": _dist(g["interval_s"])})
        feat_rows.append(
            {
                "方向": side,
                "量": "改价幅度（tick，正 = 向对手方向）",
                "分布": _dist(g["step_ticks"]),
            }
        )
    c = f[~f["filled"]]
    feat_rows.append(
        {
            "方向": "全部",
            "量": "撤单 / 过期时中间价跑开（tick，按方向）",
            "分布": _dist(c["away_at_cancel_ticks"]),
        }
    )
    fillpos = (
        f.assign(位置=pd.cut(f["k0_spread"], [-99, -2, -1, -0.5, 0, 0.5, 99]))
        .groupby(["side", "位置"], observed=True)["filled"]
        .agg(订单数="size", 成交率="mean")
        .round(3)
        .reset_index()
    )
    strat_rows = []
    for side in ("Buy", "Sell"):
        x = p[side]
        strat_rows.append(
            {
                "方向": "买（开仓）" if side == "Buy" else "卖（平仓）",
                "订单数": x["n_orders"],
                "初始位置 k0 分位（P10/25/50/75/90）": x["k0_spread_quantiles"],
                "等待时长 W 分位（截到 5–120 s）": x["wait_s_quantiles"],
                "改价次数分位": x["amend_count_quantiles"],
                "改价间隔中位数 s": x["amend_interval_s"],
                "改价幅度（a + b × 跑开 tick）": f"{x.get('chase_a')} + {x.get('chase_b')} × 跑开",
                "撤单：跑开 ≥ tick": x["cancel_away_ticks"],
                "实际成交率": x["fill_rate"],
            }
        )
    cq = fs.calibrate_q(seg)
    kb, ks = p["Buy"]["k0_spread_quantiles"], p["Sell"]["k0_spread_quantiles"]
    lines = [
        "# 成交模型校准：负责人挂单策略",
        "",
        "复现：`uv run python -m zr.research.fill_segments` → `uv run python -m zr.research.own"
        "er_strategy`"
        "（行情缓存 `zr.research.fill_model pull`；改价历史在 VPS 上由 `scripts/diag_pull_order"
        "_history.py` 拉取）。"
        "本版取代 R-039 的「固定挂单价 × 固定 T」（已作废）。",
        "",
        "## 1. 样本（只用 D-030 日内单腿的订单）",
        "",
        f"- 改价历史：{len(h)} 行 / {h['order_id'].nunique()} 单，改价（Replaced）记录 "
        f"{int((h['status'] == 'Replaced').sum())} 条（2026-10-22 VPS 拉取，无需重拉）。",
        f"- 校准样本 {int(cl['calib'].sum())} 单纳入、{int((~cl['calib']).sum())} 单不纳入，其中"
        "**无法判断归属的撤单 "
        f"{unk} 单**（单列，不纳入）。判定规则见 `zr/research/calib_sample.py`（长期合约 = 提交日"
        " DTE > 7，沿用 D-029）。",
        f"- 纳入的订单中限价单（LO）进入特征与策略：{len(f)} 单（市价单与触价单不进，盘前提交的 "
        f"{int(f['premarket_submit'].sum())} 单以 09:30 为决策时刻、不进策略参数）。",
        "",
        df_to_md(cls),
        "",
        "## 2. 逐单特征（决策时刻 = 委托提交时刻）",
        "",
        "tick 差按方向：正 = 比参照价更靠近对手价（买单更高、卖单更低）。最新成交价 = 提交前最后"
        "一笔 OPRA 成交。"
        "正股变化只覆盖 MU、QQQ（本地有 1 分钟 K）。",
        "",
        df_to_md(pd.DataFrame(feat_rows)),
        "",
        "按初始挂单位置的最终成交率：",
        "",
        df_to_md(fillpos),
        "",
        "## 3. 负责人挂单策略（参数化，`config/owner_strategy.yaml`，脚本生成）",
        "",
        df_to_md(pd.DataFrame(strat_rows)),
        "",
        f"改价「追价」拟合（买卖合并，{p['chase_fit']['chase_n']} 次改价）：幅度 = "
        f"{p['chase_fit']['chase_a']} + {p['chase_fit']['chase_b']} × 上次挂单以来中间价朝对手方"
        "向移动的 tick，"
        f"相关系数 {p['chase_fit']['chase_corr']}。",
        "",
        f"读法：负责人开仓买单的初始挂单位置中位数为中间价 {kb[2]:+.2f} 个 spread"
        f"（P25 {kb[1]:+.2f}，−0.5 即买一），平仓卖单中位数 {ks[2]:+.2f}；"
        f"开仓买单最终成交率 {p['Buy']['fill_rate']:.0%}、平仓卖单 {p['Sell']['fill_rate']:.0%}。",
        "",
        "模拟时（`zr/backtest/owner_fill.py`）：每单从上述经验分布抽样 k0、等待时长 W（截到 5–1"
        "20 秒）、改价次数；"
        "每隔改价间隔中位数改一次价（追价幅度按拟合）；中间价跑开 ≥ 撤单阈值或等满 W 秒即结束——"
        "入场放弃（错过），"
        "离场吃买一；止损与强平直接吃买一。成交判定：卖一 ≤ 挂单价、OPRA 成交越过挂单价、恰好等"
        "于时以 q 成交。",
        "",
        "## 4. 成交判定参数 q（挂单段逐段重放，同一校准样本）",
        "",
        df_to_md(cq.assign(**{"逐段 Brier": cq["逐段 Brier"].map("{:.4f}".format)})),
        "",
        f"取 q = {Q_CAL}（Brier 最小）。OPRA 数据含负责人自己的成交，「实际成交 → 模拟也成交」偏"
        "乐观；"
        "去掉自己那一笔后的值是下界。",
        "",
        "## 5. 验证：从负责人提交时刻重放全部校准样本限价单",
        "",
        f"每单抽样 {N_SEEDS} 次（委托在提交时刻到达；一律按入场口径——等满或跑开即撤单），"
        "并用「该单实际参数」（实际 k0、实际时长截到 5–120 秒、实际改价次数）各跑一次。",
        "",
        df_to_md(validation_table(v).T.reset_index().rename(columns={"index": "指标"})),
        "",
        "结论：",
        "",
        "- 成交判定本身（同一挂单价与时长）与真实记录一致：实际参数下，两边都成交的单子成交价与"
        "耗时偏差中位数为 0；"
        "实际撤单的单子在模拟中也全部未成交。",
        "- 与情境无关的抽样策略成交率远低于负责人（见上表）：负责人把单子挂在哪里、挂多久，与当"
        "时行情相关"
        "（挂得远的单子常常等到价格过来），这部分信息无法由独立抽样复现。截到 120 秒也减少了成交"
        f"（负责人 120 秒内成交的比例为 {float(((f['fill_s'] <= 120) & f['filled']).mean()):.1%}"
        "，全部时长为 "
        f"{float(f['filled'].mean()):.1%}）。",
        "",
        "## 边界",
        "",
        "- 负责人 2026-06-09～2026-08-19 的期权限价单，D-030 日内单腿口径；117 单，以 MU 为主；"
        "不外推到其他标的、"
        "其他时期或其他人。",
        "- 改价历史时刻为整秒；排队位置不可观测（q 合成）；ThetaData 盘口量只在负责人订单缓存里"
        "有（V4 缓存无量）。",
        "- 策略参数在同一样本上提取与验证（样本内）。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
