"""QQQ 期权回放偏差排查（METHOD_V3 第 0 节）→ docs/research/QQQ_ARTIFACT.md。

在抽样交易日上按 v2 规则（zr/research/outcomes.py：最近可用到期日、略虚值一档、2 张、+20% 减半、
中轨结构止损、持仓上限、15:55 强平）逐分钟、两个方向重算「随机分钟入场」的扣费后均值，
逐项打开 / 关闭可疑口径，看哪一项造成「两方向都盈利」：

- V0 原口径（与 outcomes.py 相同）
- stop_lag：结构止损按 1 分钟正股 bar j 收盘判定（j+1 才知道），原口径却用 bid[j]（j 时刻快照）成交
  → 改为触发后第一个快照 bid[j+1]
- zero_bid：原口径过滤掉 bid=0 的快照，出场时向前回退到更早的正 bid → 改为 bid=0 视为可成交价 0
- entry_lag：入场 ask 用决策时刻后第一个快照 + 2 秒（1 分钟快照下 = m+1），原口径用 ask[m]
- tp_lag：+20% 减半在快照 j 判定，原口径同一快照 bid[j] 成交 → 改为 bid[j+1]
- ALL：以上全部

另外检查：合约是否选成实值（成交时刻正股价 vs 行权价）、DTE 分布、陈旧报价比例（与前 3
个快照完全相同）、
带宽外无合约的分钟比例；随机抽 10 笔打印合约、报价与时间戳。
运行：uv run python -m zr.research.qqq_artifact
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.bars import ET, load_bars
from zr.features.features import compute_features
from zr.features.option_features import load_option_day, option_days
from zr.ops.markdown import df_to_md
from zr.research import outcomes as oc

log = logging.getLogger("zr.research.qqq_artifact")
VARIANTS = {
    "V0 原口径": {},
    "stop_lag": {"stop_lag": True},
    "zero_bid": {"zero_bid": True},
    "entry_lag": {"entry_lag": True},
    "tp_lag": {"tp_lag": True},
    "ALL": {"stop_lag": True, "zero_bid": True, "entry_lag": True, "tp_lag": True},
}


@dataclass
class DayQ:
    day: dt.date
    close: np.ndarray
    mid: np.ndarray
    expiries: list
    chains: dict
    bid: dict  # 过滤 bid=0 后（原口径）
    bid0: dict  # 保留 bid=0
    ask: dict
    stale: dict
    stamp: np.ndarray  # 分钟 → 快照时间（UTC）


def build(stock: pd.DataFrame, mid: pd.Series, opt: dict, day: dt.date) -> DayQ | None:
    idx = oc.minute_index(stock.index)
    ok = (idx >= 0) & (idx < oc.MINUTES)
    close = np.full(oc.MINUTES, np.nan)
    midv = np.full(oc.MINUTES, np.nan)
    close[idx[ok]] = stock["close"].to_numpy()[ok]
    midv[idx[ok]] = mid.to_numpy()[ok]
    q = opt["quotes"]
    if q.empty:
        return None
    q = q[(q["ask"] > 0) & (q["ask"] >= q["bid"]) & (q["bid"] >= 0)].copy()
    q["m"] = oc.minute_index(q["ts"])
    q = q[(q["m"] >= 0) & (q["m"] < oc.MINUTES)]
    bid, bid0, ask, stale = {}, {}, {}, {}
    for sym, g in q.groupby("symbol", sort=False):
        m = g["m"].to_numpy()
        b = np.full(oc.MINUTES, np.nan)
        a = np.full(oc.MINUTES, np.nan)
        b[m] = g["bid"].to_numpy(float)
        a[m] = g["ask"].to_numpy(float)
        bid0[sym] = b.copy()
        bb = b.copy()
        bb[bb <= 0] = np.nan
        bid[sym] = bb
        aa = a.copy()
        aa[~np.isfinite(bb)] = np.nan  # 原口径：bid=0 的整行被过滤
        ask[sym] = aa
        key = (
            g[["bid", "ask", "bid_size", "ask_size"]].astype(float).to_numpy()
            if "bid_size" in g
            else g[["bid", "ask"]].astype(float).to_numpy()
        )
        same = np.zeros(oc.MINUTES, dtype=bool)
        st = np.r_[False, (key[1:] == key[:-1]).all(axis=1)]
        run = np.zeros(len(st), dtype=int)
        for i in range(1, len(st)):
            run[i] = run[i - 1] + 1 if st[i] else 0
        same[m] = run >= 3
        stale[sym] = same
    con = q[["symbol", "expiry", "strike", "right"]].drop_duplicates("symbol")
    expiries = sorted(e for e in con["expiry"].unique() if 0 <= (e - day).days <= oc.MAX_DTE)
    chains = {
        (e, r): g.sort_values("strike")[["strike", "symbol"]].reset_index(drop=True)
        for (e, r), g in con.groupby(["expiry", "right"])
        if e in expiries
    }
    t0 = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).tz_convert("UTC")
    stamp = np.array([t0 + pd.Timedelta(minutes=i) for i in range(oc.MINUTES)])
    return DayQ(day, close, midv, expiries, chains, bid, bid0, ask, stale, stamp)


def pick(d: DayQ, direction: int, spot: float, m_ask: int):
    right = "call" if direction == 1 else "put"
    for e in d.expiries:
        ch = d.chains.get((e, right))
        if ch is None or ch.empty:
            continue
        s = ch["strike"].to_numpy()
        j = (
            np.searchsorted(s, spot, side="right")
            if direction == 1
            else (np.searchsorted(s, spot, side="left") - 1)
        )
        if j < 0 or j >= len(s):
            continue
        sym = ch["symbol"].iloc[j]
        if np.isfinite(d.ask[sym][m_ask]):
            return sym, e, float(s[j])
    return None


def simulate(d: DayQ, m: int, direction: int, sym: str, cap: int, v: dict) -> dict | None:
    e_m = m + 1 if v.get("entry_lag") else m
    if e_m > oc.LAST_ENTRY + 1:
        return None
    ask0 = d.ask[sym][e_m]
    if not np.isfinite(ask0):
        return None
    bids = d.bid0[sym] if v.get("zero_bid") else d.bid[sym]
    last = min(e_m + cap, oc.FORCE_EXIT)
    exit_m, reason = last, "cap"
    proceeds, fees, remaining, half = 0.0, oc.fee(oc.QTY, "Buy", ask0), oc.QTY, False
    for j in range(e_m + 1, last + 1):
        b = bids[j]
        if np.isfinite(b) and not half and b >= ask0 * (1 + oc.TAKE_PROFIT):
            fill_j = j + 1 if v.get("tp_lag") else j
            bf = bids[min(fill_j, oc.FORCE_EXIT)]
            if np.isfinite(bf):
                proceeds += bf * 100
                fees += oc.fee(1, "Sell", bf) if bf > 0 else 0.0
                remaining, half = remaining - 1, True
        c, mm = d.close[j], d.mid[j]
        if (
            np.isfinite(c)
            and np.isfinite(mm)
            and ((direction == 1 and c < mm) or (direction == -1 and c > mm))
        ):
            exit_m = min(j + 1, oc.FORCE_EXIT) if v.get("stop_lag") else j
            reason = "struct_stop"
            break
    j = exit_m
    if v.get("zero_bid"):
        while j <= oc.FORCE_EXIT and not np.isfinite(bids[j]):  # 向后找第一个快照（不回看）
            j += 1
        if j > oc.FORCE_EXIT:
            return None
    else:
        while j > e_m and not np.isfinite(bids[j]):  # 原口径：向前回退
            j -= 1
    b = bids[j]
    if not np.isfinite(b):
        return None
    if remaining:
        proceeds += b * 100 * remaining
        fees += oc.fee(remaining, "Sell", b) if b > 0 else 0.0
    net = proceeds - ask0 * 100 * oc.QTY - fees
    return {"net": net, "entry_m": e_m, "exit_m": j, "ask0": ask0, "exit_bid": b, "reason": reason}


def run_symbol(root: Path, und: str, every: int, seed: int = 0) -> tuple[pd.DataFrame, dict, list]:
    days = option_days(root, und)[::every]
    stock = load_bars(root, und)
    mid = compute_features(stock)["bb_mid"]
    sday = np.asarray(stock.index.tz_convert(ET).date)
    rows, checks, samples = (
        [],
        {
            "entries": 0,
            "itm_at_fill": 0,
            "stale_entry": 0,
            "no_contract": 0,
            "minutes": 0,
            "dte": [],
        },
        [],
    )
    rng = np.random.default_rng(seed)
    t0 = time.monotonic()
    for k, day in enumerate(days):
        opt = load_option_day(root, und, day)
        msk = sday == day
        if opt is None or not msk.any():
            continue
        d = build(stock[msk], mid[msk], opt, day)
        if d is None:
            continue
        for m in range(oc.FIRST_ENTRY, oc.LAST_ENTRY + 1):
            spot = d.close[m - 1]
            if not np.isfinite(spot):
                continue
            for direction in (1, -1):
                checks["minutes"] += 1
                pk = pick(d, direction, spot, m)
                if pk is None:
                    checks["no_contract"] += 1
                    continue
                sym, exp, strike = pk
                checks["entries"] += 1
                checks["dte"].append((exp - day).days)
                fill_spot = d.close[m] if np.isfinite(d.close[m]) else spot
                checks["itm_at_fill"] += int(
                    (direction == 1 and strike <= fill_spot)
                    or (direction == -1 and strike >= fill_spot)
                )
                checks["stale_entry"] += int(d.stale[sym][m])
                for cap in oc.CAPS:
                    for name, v in VARIANTS.items():
                        r = simulate(d, m, direction, sym, cap, v)
                        if r is None:
                            continue
                        rows.append((day.year, direction, cap, name, r["net"], r["reason"]))
                        if (
                            name == "ALL"
                            and cap == 30
                            and rng.random() < 2e-4
                            and len(samples) < 10
                        ):
                            samples.append(
                                {
                                    "day": day.isoformat(),
                                    "symbol": sym,
                                    "dir": direction,
                                    "spot_decision": round(spot, 2),
                                    "strike": strike,
                                    "entry_snapshot_et": pd.Timestamp(d.stamp[r["entry_m"]])
                                    .tz_convert(ET)
                                    .strftime("%H:%M:%S"),
                                    "ask_entry": r["ask0"],
                                    "bid_at_entry": d.bid0[sym][r["entry_m"]],
                                    "exit_snapshot_et": pd.Timestamp(d.stamp[r["exit_m"]])
                                    .tz_convert(ET)
                                    .strftime("%H:%M:%S"),
                                    "bid_exit": r["exit_bid"],
                                    "reason": r["reason"],
                                    "net_usd": round(r["net"], 2),
                                }
                            )
        if (k + 1) % 20 == 0:
            log.info("%s %d/%d 天 %.0fs", und, k + 1, len(days), time.monotonic() - t0)
    df = pd.DataFrame(rows, columns=["year", "direction", "cap", "variant", "net", "reason"])
    return df, checks, samples


def tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = df.pivot_table(
        index=["variant", "cap"], columns="direction", values="net", aggfunc="mean"
    ).round(2)
    main.columns = [f"dir {c:+d}" for c in main.columns]
    main["n"] = df.groupby(["variant", "cap"]).size()
    year = (
        df[df["cap"] == 30]
        .pivot_table(index=["variant", "year"], columns="direction", values="net", aggfunc="mean")
        .round(2)
    )
    year.columns = [f"dir {c:+d}" for c in year.columns]
    reason = (
        df[(df["cap"] == 30) & df["variant"].isin(["V0 原口径", "ALL"])]
        .pivot_table(index=["variant", "reason"], values="net", aggfunc=["count", "mean"])
        .round(2)
    )
    reason.columns = ["n", "mean_net"]
    return main.reset_index(), year.reset_index(), reason.reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5, help="每隔几个期权日抽 1 天")
    ap.add_argument("--out", default="docs/research/QQQ_ARTIFACT.md")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    L = [
        "# QQQ 期权回放偏差排查（METHOD_V3 第 0 节）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 脚本 "
        f"`zr/research/qqq_artifact.py`（每 {a.every} 个期权日抽 1 天）",
        "",
    ]
    L += [
        "## 机制",
        "",
        "结构止损按正股 1 分钟 bar j 的收盘判定（bar j 覆盖 [j, j+1)，收盘在 j+1 才知道），"
        "原口径却用"
        "时间戳 j 的期权快照 bid[j] 成交——即在触发止损的那一分钟**开始之前**的价格离场，"
        "提前躲掉了这一分钟的不利变动。多空两个方向都受益；QQQ 点差小、费用占比低，"
        "这一分钟的前视收益"
        "足以把随机入场推成正值；MU 点差大，仍为负但被低估约一半。修复：一切由 bar 收盘触发的出场，"
        "用触发后第一个快照（j+1）成交；入场用决策时刻后第一个快照 + 2 秒（m+1）；bid=0 "
        "视为可成交价 0，"
        "不回看更早的 bid；止盈在快照 j 判定、j+1 成交。v3 引擎 `zr/research/sim_v3.py` "
        "采用全部修复口径。",
        "",
        "**影响**：方法论 v2 第一轮（docs/research/METHOD_V2_ROUND1*.md、RESEARCH_LOG "
        "R-017～R-029）"
        "中所有真实期权盈亏数字都含此前视偏差（MU 同样偏高），只能作相对比较参考，"
        "不能作绝对值结论。",
        "",
    ]
    for und in ("QQQ", "MU"):
        df, ck, samples = run_symbol(root, und, a.every)
        main_t, year_t, reason_t = tables(df)
        g = df[df["cap"] == 30].groupby(["variant", "direction"])["net"].mean().unstack()
        delta = (g - g.loc["V0 原口径"]).drop(index="V0 原口径").round(2)
        delta.columns = [f"Δ dir {c:+d}" for c in delta.columns]
        worst = delta.abs().sum(axis=1).idxmax()
        L += [
            f"### {und} 结论（由下表算出）",
            "",
            f"持仓上限 30 时各口径相对原口径的变化（USD / 笔）；影响最大的一项是 **{worst}**：",
            "",
            df_to_md(delta.reset_index()),
            "",
        ]
        dte = pd.Series(ck["dte"]).value_counts(normalize=True).sort_index().round(3)
        L += [
            f"## {und} 明细",
            "",
            "随机分钟入场（每个决策分钟 × 两方向全部计入，等价于随机入场期望），扣费后 USD / 笔（2 "
            "张）：",
            "",
            df_to_md(main_t),
            "",
            "按年（持仓上限 30）：",
            "",
            df_to_md(year_t),
            "",
            "出场原因（持仓上限 30）：",
            "",
            df_to_md(reason_t),
            "",
            f"检查：分钟 × 方向 {ck['minutes']}，无合约 {ck['no_contract']}"
            f"（{ck['no_contract'] / max(1, ck['minutes']):.2%}）；入场 {ck['entries']}，"
            f"成交时刻已实值 {ck['itm_at_fill'] / max(1, ck['entries']):.2%}，"
            "入场快照陈旧（与前 3 个快照完全相同）"
            f"{ck['stale_entry'] / max(1, ck['entries']):.2%}；"
            f"DTE 分布 {dte.to_dict()}。",
            "",
            "随机 10 笔（ALL 口径、持仓上限 30）：",
            "",
            df_to_md(pd.DataFrame(samples)),
            "",
        ]
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    log.info("写入 %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
