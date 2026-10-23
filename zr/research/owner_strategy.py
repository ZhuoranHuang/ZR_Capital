"""负责人挂单策略的提取与参数化（成交模型校准修正版，替代 R-039 的「固定挂单价 × 固定 T」）。

1. 逐单特征（features）：以委托提交时刻为决策时刻（盘前提交取 09:30），记录提交时买一 / 卖一 /
   中间价 / 最新成交价（OPRA，提交前最后一笔），初始挂单价相对这四者的 tick 差（按方向：正 = 更靠近
   对手价）、提交至成交或撤单 / 过期的时长、改价次数、每次改价的 tick 幅度（按方向）与间隔、
   成交时刻相对提交时刻的正股价格变化（MU、QQQ 有 1 分钟 K；其他标的不覆盖）。
2. 策略参数（extract）：按买（开仓）/ 卖（平仓）分别给出初始挂单位置（相对中间价，spread 倍数）的
   经验分布、首次改价前等待时长、改价间隔与幅度、改价次数、委托总时长（等待时长，截到 5–120 秒）
   的经验分布、撤单前市场走向。写 config/owner_strategy.yaml（脚本生成，勿手改）。
数据：fill_model.order_sample 的限价单（LO）+ 改价历史（diag/order_history.parquet）+ ThetaData 缓存。
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
    s = fm.order_sample(root)
    s["order_id"] = s["order_id"].astype(str)
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
            amends.append(
                {
                    "order_id": o["order_id"],
                    "side": o["side"],
                    "k": k,
                    "interval_s": (int(st[k]) - max(int(st[k - 1]), open_ns)) / 1e9,
                    "step_ticks": sgn * (prices[k] - prices[k - 1]) / tick,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(amends)


def _q(x: pd.Series) -> list[float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return [round(float(x.quantile(p)), 4) for p in QS] if len(x) else []


def extract(f: pd.DataFrame, a: pd.DataFrame) -> dict:
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
        }
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
    p = extract(f, a)
    write_strategy(p)
    for side in ("Buy", "Sell"):
        print(side, {k: v for k, v in p[side].items() if k not in ("k0_spread", "wait_s")})
    print(f.groupby(["side", "status"]).size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
