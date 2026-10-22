"""直接回放负责人的规则（METHOD_V2 规格 2，R-017；负责人 2026-10-22 补充 1/2）。

- 事件：E1 收紧版——通道内回踩中轨：|low−中轨|/中轨 ≤ ε（0.05%）且收盘回到中轨同侧，确认 1 根
  （触及后下一根收盘仍在中轨同侧）。通道用 A（MACD 三条件）与 F（布林三线同号平行、持续、回撤受限）
  ，
  zr/features/channel.py。做多 = 上升通道买 call，做空 = 下降通道买 put，分开统计。
- 合约与出场：zr/research/outcomes.py（最近可用到期日 DTE≤5、最近一档略虚值、2 张、+20% 减半、
  中轨结构止损、持仓上限 10/30 分钟、买 ask 卖 bid、D-013）。实际 DTE 记录并分层。
- 变体：排除 09:30–10:00 决策。
- 随机对照：同日同笔数、同方向同持仓上限，决策分钟在全天合格分钟里随机，1,000 次 → 真实合计分位。
- 通道三步（补充 1）：(1) 通道内任意分钟入场 vs 通道外；(2) 通道内回踩入场 vs 通道内随机分钟入场；
  (3) 按通道质量（持续根数、斜率、平行度、最大回撤）三分位分层。
- DTE 0–1 专表（补充 2）：MU 出现周一/周三到期合约之后的日子 vs 之前的周四/周五。
输出：docs/research/REPLAY_RULE_<UND>.md；交易明细 ZR_DATA_ROOT/research/v2/replay_<UND>.parquet
运行：uv run python -m zr.research.replay_rule --symbol MU.US
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.bars import load_bars
from zr.features.channel import channel_states
from zr.features.features import compute_features
from zr.ops.markdown import df_to_md
from zr.research import outcomes as oc

log = logging.getLogger("zr.research.replay_rule")
ET = ZoneInfo("America/New_York")
EPS, CONFIRM = 0.0005, 1
CHANNELS = ("A", "F")
N_RANDOM = 1000
OPEN_EXCL = 31  # 决策分钟 < 31（10:01 前）视为 09:30–10:00
SESSIONS = [
    ("09:30–10:00", 1, 31),
    ("10:00–11:30", 31, 121),
    ("11:30–14:00", 121, 271),
    ("14:00–15:30", 271, 361),
]


def pullback_events(
    df: pd.DataFrame, feats: pd.DataFrame, chan: pd.Series, eps: float = EPS, confirm: int = CONFIRM
) -> pd.Series:
    """通道内回踩中轨（+1 多 / −1 空 / 0）。只用 ≤t 的数据；确认 1 根 = t−1 触及、t
    收盘仍在中轨同侧。
    """
    mid = feats["bb_mid"]
    touch_lo = (df["low"] - mid).abs() / mid <= eps
    touch_hi = (df["high"] - mid).abs() / mid <= eps
    long = (chan == 1) & touch_lo & (df["close"] > mid)
    short = (chan == -1) & touch_hi & (df["close"] < mid)
    if confirm:
        long = long.shift(1, fill_value=False) & (df["close"] > mid) & (chan == 1)
        short = short.shift(1, fill_value=False) & (df["close"] < mid) & (chan == -1)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def stats(net: pd.Series, days: pd.Series | None = None) -> dict:
    if len(net) == 0:
        return {"n": 0}
    wins, losses = net[net > 0], net[net < 0]
    cum = net.cumsum()
    out = {
        "n": len(net),
        "win_rate": round(float((net > 0).mean()), 3),
        "payoff": round(float(wins.mean() / -losses.mean()), 2)
        if len(wins) and len(losses)
        else np.nan,
        "mean_net": round(float(net.mean()), 2),
        "sum_net": round(float(net.sum()), 2),
        "max_dd": round(float((cum - cum.cummax()).min()), 2),
    }
    if days is not None:
        out["per_day"] = round(len(net) / max(1, days.nunique()), 2)
    return out


def session_of(minute: pd.Series) -> pd.Series:
    out = pd.Series("other", index=minute.index)
    for name, lo, hi in SESSIONS:
        out[(minute >= lo) & (minute < hi)] = name
    return out


class Lookup:
    """outcomes 表按 (方向, 持仓上限) 拆成 天 × 分钟 的 net 矩阵，供查表与随机抽样。"""

    def __init__(self, out: pd.DataFrame):
        self.days = sorted(out["day"].unique())
        self.day_ix = {d: i for i, d in enumerate(self.days)}
        self.net: dict[tuple[int, int], np.ndarray] = {}
        self.dte: dict[int, np.ndarray] = {}
        for (direction, cap), g in out.groupby(["direction", "cap"]):
            m = np.full((len(self.days), oc.MINUTES), np.nan)
            m[g["day"].map(self.day_ix).to_numpy(), g["minute"].to_numpy()] = g["net"].to_numpy()
            self.net[(direction, cap)] = m
            if cap == oc.CAPS[0]:
                t = np.full((len(self.days), oc.MINUTES), np.nan)
                t[g["day"].map(self.day_ix).to_numpy(), g["minute"].to_numpy()] = g[
                    "dte"
                ].to_numpy()
                self.dte[direction] = t

    def get(self, days: pd.Series, minutes: pd.Series, direction: int, cap: int) -> np.ndarray:
        di = days.map(self.day_ix)
        ok = di.notna().to_numpy()
        res = np.full(len(days), np.nan)
        res[ok] = self.net[(direction, cap)][di[ok].astype(int).to_numpy(), minutes.to_numpy()[ok]]
        return res


def random_control(
    lk: Lookup,
    real: pd.DataFrame,
    direction: int,
    cap: int,
    eligible: np.ndarray,
    n: int = N_RANDOM,
    seed: int = 0,
) -> dict:
    """eligible：天 × 分钟 布尔矩阵（合格入场分钟）。每次每天抽与真实相同笔数。"""
    rng = np.random.default_rng(seed)
    net = lk.net[(direction, cap)]
    counts = real.groupby("day").size()
    totals = np.zeros(n)
    for day, k in counts.items():
        i = lk.day_ix.get(day)
        if i is None:
            continue
        cand = np.flatnonzero(eligible[i] & np.isfinite(net[i]))
        if len(cand) == 0:
            continue
        draws = net[i][cand[rng.integers(0, len(cand), size=(n, int(k)))]]
        totals += draws.sum(axis=1)
    real_total = float(real["net"].sum())
    return {
        "rand_p05": round(float(np.percentile(totals, 5)), 1),
        "rand_p50": round(float(np.percentile(totals, 50)), 1),
        "rand_p95": round(float(np.percentile(totals, 95)), 1),
        "real_pctile": round(float((totals <= real_total).mean() * 100), 1),
    }


def minute_matrix(
    days: list, sday: np.ndarray, minute: np.ndarray, values: np.ndarray
) -> np.ndarray:
    ix = {d: i for i, d in enumerate(days)}
    m = np.zeros((len(days), oc.MINUTES), dtype=values.dtype)
    di = np.array([ix.get(d, -1) for d in sday])
    ok = (di >= 0) & (minute >= 0) & (minute < oc.MINUTES)
    m[di[ok], minute[ok]] = values[ok]
    return m


def run(
    root: Path, und: str, start: dt.date, end: dt.date, n_random: int
) -> tuple[str, pd.DataFrame]:
    t0 = time.monotonic()
    out = oc.build(root, und, start, end)
    if out.empty:
        return f"# 规则直接回放 {und}\n\n无期权数据（待 {und} 全量）。\n", pd.DataFrame()
    lk = Lookup(out)
    stock = load_bars(root, und, start, end)
    feats = compute_features(stock)
    chans = channel_states(stock, feats)
    sday = np.asarray(stock.index.tz_convert(ET).date)
    bar_min = oc.minute_index(stock.index)  # 触发 bar 的分钟；决策分钟 = +1
    stock_days = sorted(set(sday))
    # 覆盖矩阵
    cov = pd.DataFrame({"day": stock_days})
    cov["month"] = pd.to_datetime(cov["day"]).dt.to_period("M").astype(str)
    cov["opt"] = cov["day"].isin(set(lk.days))
    covm = (
        cov.groupby("month")
        .agg(stock_days=("day", "size"), option_days=("opt", "sum"))
        .reset_index()
    )

    trades, summary = [], []
    eligible_all = np.zeros((len(lk.days), oc.MINUTES), dtype=bool)
    eligible_all[:, oc.FIRST_ENTRY : oc.LAST_ENTRY + 1] = True
    elig_ex = eligible_all.copy()
    elig_ex[:, :OPEN_EXCL] = False
    for ch in CHANNELS:
        sig = pullback_events(stock, feats, chans[f"chan_{ch}"])
        ev = pd.DataFrame(
            {"day": sday, "minute": bar_min + 1, "direction": sig.to_numpy(), "ts": stock.index}
        )
        ev = ev[
            (ev["direction"] != 0)
            & (ev["minute"] >= oc.FIRST_ENTRY)
            & (ev["minute"] <= oc.LAST_ENTRY)
        ]
        ev = ev[ev["day"].isin(set(lk.days))]
        q = chans.loc[
            ev["ts"],
            [f"chan_{ch}_len", f"chan_{ch}_slope", f"chan_{ch}_parallel", f"chan_{ch}_maxdd"],
        ]
        ev[["q_len", "q_slope", "q_parallel", "q_maxdd"]] = q.to_numpy()
        for direction, cap, ex in product((1, -1), oc.CAPS, (False, True)):
            e = ev[ev["direction"] == direction]
            if ex:
                e = e[e["minute"] >= OPEN_EXCL]
            net = lk.get(e["day"], e["minute"], direction, cap)
            dte = (
                lk.dte[direction][
                    e["day"].map(lk.day_ix).astype(int).to_numpy(), e["minute"].to_numpy()
                ]
                if len(e)
                else []
            )
            t = e.assign(net=net, dte=dte, cap=cap, channel=ch, ex0930=ex).dropna(subset=["net"])
            side = "long" if direction == 1 else "short"
            name = f"{ch}|{side}|cap={cap}|{'ex0930' if ex else 'all'}"
            s = {
                "config": name,
                "events": len(e),
                "no_contract": int(np.isnan(net).sum()),
                **stats(t["net"], t["day"]),
            }
            if len(t) >= 5:
                s.update(
                    random_control(lk, t, direction, cap, elig_ex if ex else eligible_all, n_random)
                )
            summary.append(s)
            trades.append(t.assign(config=name))
    tr = pd.concat(trades, ignore_index=True)
    summ = pd.DataFrame(summary)
    tr["session"] = session_of(tr["minute"])
    tr["weekday"] = pd.to_datetime(tr["day"]).dt.day_name().str[:3]
    tr["d001_violation"] = (tr["dte"] == 0) & (tr["minute"] > oc.D001_LAST_0DTE_ENTRY)

    # Mon/Wed 到期时代
    exps = pd.Series(sorted(out["expiry"].unique()))
    monwed = exps[pd.to_datetime(exps).dt.weekday.isin([0, 2])]
    era_start = monwed.min() if len(monwed) else None

    base = tr[(tr["ex0930"] == False)]  # noqa: E712
    lines = [
        f"# 规则直接回放 {und}（METHOD_V2 规格 2，R-017）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · {start}～{end} · "
        f"耗时 {time.monotonic() - t0:.0f}s · 随机对照 {n_random} 次/组",
        f"规则：E1 收紧版（ε={EPS:.2%}、确认 {CONFIRM} 根）× 通道 {'/'.join(CHANNELS)}；"
        "合约与出场见 "
        "zr/research/outcomes.py（最近可用到期日 DTE≤5、最近一档略虚值、2 张、+20% 减半、"
        "中轨结构止损、"
        "持仓上限 10/30 分钟、买 ask 卖 bid、D-013 费用）。",
        "",
        "## 覆盖矩阵",
        "",
        df_to_md(covm),
        "",
        f"期权日 {len(lk.days)} / 正股交易日 {len(stock_days)}；只在期权日上回放。",
        "",
        "## 16 组配置（通道 × 方向 × 持仓上限 × 是否排除 09:30–10:00）",
        "",
        "no_contract = 决策分钟没有可买合约（DTE≤5 内都无 ask）的事件数；real_pctile = 真实合计在 "
        "1,000 次同日同笔数随机入场中的分位。",
        "",
        df_to_md(summ),
        "",
    ]
    for dim, label in (
        ("dte", "实际 DTE"),
        ("session", "入场时段"),
        ("weekday", "星期"),
        ("d001_violation", "D-001 违例（0DTE 且 10:15 后）"),
    ):
        g = (
            base.groupby(["channel", "direction", "cap", dim])
            .apply(lambda x: pd.Series(stats(x["net"], x["day"])))
            .reset_index()
        )
        lines += [f"## 按{label}分层（不排除开盘；各组合分别统计）", "", df_to_md(g), ""]
    # DTE 0–1 专表
    if era_start is not None:
        d01 = base[base["dte"].isin([0, 1])].copy()
        d01["era"] = np.where(
            pd.to_datetime(d01["day"]) >= pd.Timestamp(era_start),
            f"{era_start} 起（含周一/周三到期）",
            "之前（仅周五到期）",
        )
        pre = d01[(d01["era"].str.startswith("之前")) & d01["weekday"].isin(["Thu", "Fri"])]
        post = d01[~d01["era"].str.startswith("之前")]
        g1 = (
            post.groupby(["channel", "direction", "cap", "dte"])
            .apply(lambda x: pd.Series(stats(x["net"], x["day"])))
            .reset_index()
        )
        g2 = (
            pre.groupby(["channel", "direction", "cap", "dte", "weekday"])
            .apply(lambda x: pd.Series(stats(x["net"], x["day"])))
            .reset_index()
        )
        lines += [
            "## DTE 0–1 专表",
            "",
            f"{und} 首个周一/周三到期日：{era_start}。之前只有周五到期，DTE 0–1 只出现在周四（DTE "
            "1）与周五（DTE 0）。",
            "",
            "### 含周一/周三到期合约之后",
            "",
            df_to_md(g1),
            "",
            "### 之前的周四 / 周五",
            "",
            df_to_md(g2),
            "",
        ]
    # 通道三步
    lines += ["## 通道三步（补充 1；持仓上限 30、不排除开盘）", ""]
    for ch in CHANNELS:
        st = chans[f"chan_{ch}"].to_numpy()
        stm = minute_matrix(lk.days, sday, bar_min + 1, st.astype(np.int8))  # 决策分钟上的通道状态
        rows = []
        for direction in (1, -1):
            net = lk.net[(direction, 30)]
            inside = stm == direction
            inside[:, : oc.FIRST_ENTRY] = False
            inside[:, oc.LAST_ENTRY + 1 :] = False
            outside = stm == 0
            outside[:, : oc.FIRST_ENTRY] = False
            outside[:, oc.LAST_ENTRY + 1 :] = False
            for nm, msk in (("通道内任意分钟（同向）", inside), ("通道外任意分钟", outside)):
                v = net[msk & np.isfinite(net)]
                rows.append(
                    {"channel": ch, "direction": direction, "group": nm, **stats(pd.Series(v))}
                )
            pb = tr[
                (tr["channel"] == ch)
                & (tr["direction"] == direction)
                & (tr["cap"] == 30)
                & ~tr["ex0930"].astype(bool)
            ]
            rows.append(
                {
                    "channel": ch,
                    "direction": direction,
                    "group": "通道内回踩入场（E1）",
                    **stats(pb["net"]),
                }
            )
            if len(pb) >= 5:
                rc = random_control(lk, pb, direction, 30, inside, n_random, seed=1)
                rows[-1].update(
                    {
                        "vs_inside_random_pctile": rc["real_pctile"],
                        "inside_rand_p50": rc["rand_p50"],
                    }
                )
        lines += [
            f"### 通道 {ch}：(1) 通道内 vs 通道外；(2) 回踩 vs 通道内随机",
            "",
            df_to_md(pd.DataFrame(rows)),
            "",
        ]
        pb = tr[(tr["channel"] == ch) & (tr["cap"] == 30) & (tr["ex0930"] == False)].copy()  # noqa: E712
        qrows = []
        for qc in ("q_len", "q_slope", "q_parallel", "q_maxdd"):
            if pb[qc].nunique() < 3:
                continue
            pb["bucket"] = pd.qcut(pb[qc].rank(method="first"), 3, labels=["低", "中", "高"])
            for (direction, b), g in pb.groupby(["direction", "bucket"], observed=True):
                qrows.append(
                    {
                        "channel": ch,
                        "quality": qc,
                        "direction": direction,
                        "tercile": b,
                        "range": f"{g[qc].min():.4g}–{g[qc].max():.4g}",
                        **stats(g["net"]),
                    }
                )
        lines += [
            f"### 通道 {ch}：(3) 按通道质量三分位（回踩入场）",
            "",
            df_to_md(pd.DataFrame(qrows)),
            "",
        ]
    lines += [
        "## 边界",
        "",
        f"边界：标的 {und}；{start}～{end} 中有期权数据的 {len(lk.days)} 个交易日（RTH、"
        "交易日历过滤）；"
        f"事件 {int(summ['events'].sum())} 个（含重复计入各配置）；费用 D-013 + 买 ask 卖 bid；"
        "对齐：决策分钟 = 触发 bar 后一分钟，"
        "用该分钟时间戳的 NBBO；未覆盖：无期权数据的月份（见覆盖矩阵）、DTE>5 的情形、ask "
        "缺失的分钟；"
        "随机对照只随机入场分钟，合约与出场规则不变。",
        "",
        "## 复现",
        "",
        "```bash",
        f"uv run python -m zr.research.outcomes --symbol {und}.US",
        f"uv run python -m zr.research.replay_rule --symbol {und}.US",
        "```",
        "",
    ]
    return "\n".join(lines), tr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="MU.US")
    p.add_argument("--start", default="2023-10-02")
    p.add_argument("--end", default="2026-10-21")
    p.add_argument("--n-random", type=int, default=N_RANDOM)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    und = a.symbol.split(".")[0]
    text, tr = run(
        root, und, dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end), a.n_random
    )
    store = root / "research" / "v2"
    store.mkdir(parents=True, exist_ok=True)
    if len(tr):
        tr.drop(columns=["ts"]).to_parquet(store / f"replay_{und}.parquet", index=False)
    out = Path(a.out) if a.out else Path("docs/research") / f"REPLAY_RULE_{und}.md"
    out.write_text(text, encoding="utf-8")
    log.info("写入 %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
