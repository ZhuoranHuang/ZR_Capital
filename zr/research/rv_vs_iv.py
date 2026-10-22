"""RV − IV（R-015）：用 ThetaData ATM 报价反推开盘隐含波动率，与随后实现波动比较。

- IV：每个交易日 09:31 ET 的 ATM（最接近开盘价的行权价）call 与 put 中间价，
  Black-Scholes 反推（brentq），
  取 call/put 平均；合约选当日到期（0DTE），没有则次一到期（1DTE）；T 按日历时间到到期日 16:00 ET，
  无风险利率取 4%（假设，对 0–1DTE 影响可忽略）。
- RV：随后 1 小时 / 全天 1 分钟对数收益 std，年化 √(252×390)；分时段 RV 同法。
运行：uv run python -m zr.research.rv_vs_iv --underlying SPY → docs/research/RV_VS_IV.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from zr.config import get_settings
from zr.features.bars import load_bars
from zr.ops.markdown import df_to_md

ET = ZoneInfo("America/New_York")
R = 0.04
ANN = math.sqrt(252 * 390)
SESSIONS = [
    ("09:30–10:00", 570, 600),
    ("10:00–11:30", 600, 690),
    ("11:30–14:00", 690, 840),
    ("14:00–15:30", 840, 930),
    ("15:30–16:00", 930, 960),
]


def bs_price(S, K, T, sigma, right):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if right == "call" else (K - S))
    d1 = (math.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == "call":
        return S * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)
    return K * math.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, right) -> float:
    intrinsic = max(0.0, (S - K) if right == "call" else (K - S))
    if price <= intrinsic + 1e-6 or T <= 0:
        return float("nan")
    try:
        return brentq(lambda s: bs_price(S, K, T, s, right) - price, 1e-4, 5.0, maxiter=200)
    except ValueError:
        return float("nan")


def day_iv(quotes: pd.DataFrame, day: dt.date, spot: float, at_minute: int = 571) -> dict | None:
    q = quotes[
        quotes["ts"].dt.tz_convert(ET).dt.hour * 60 + quotes["ts"].dt.tz_convert(ET).dt.minute
        == at_minute
    ]
    if q.empty:
        return None
    exps = sorted(q["expiry"].unique())
    exp = next((e for e in exps if e >= day), None)
    if exp is None:
        return None
    q = q[q["expiry"] == exp]
    k_atm = q.iloc[(q["strike"] - spot).abs().argsort()[:1]]["strike"].iloc[0]
    q = q[q["strike"] == k_atm]
    expiry_dt = dt.datetime.combine(exp, dt.time(16, 0), tzinfo=ET)
    now = dt.datetime.combine(day, dt.time(at_minute // 60, at_minute % 60), tzinfo=ET)
    T = (expiry_dt - now).total_seconds() / (365 * 24 * 3600)
    ivs = {}
    for r in ("call", "put"):
        row = q[q["right"] == r]
        if row.empty:
            continue
        bid, ask = float(row["bid"].iloc[0]), float(row["ask"].iloc[0])
        if bid <= 0 or ask <= 0:
            continue
        ivs[r] = implied_vol((bid + ask) / 2, spot, float(k_atm), T, r)
    if not ivs:
        return None
    return {
        "day": day,
        "expiry": exp,
        "dte": (exp - day).days,
        "strike": float(k_atm),
        "spot": spot,
        "iv_call": ivs.get("call"),
        "iv_put": ivs.get("put"),
        "iv": float(np.nanmean(list(ivs.values()))),
    }


def realized(df_day: pd.DataFrame) -> dict:
    et = df_day.index.tz_convert(ET)
    mod = et.hour * 60 + et.minute
    r = np.log(df_day["close"]).diff()
    out = {
        "rv_day": float(r.std(ddof=1) * ANN),
        "rv_1h": float(r[(mod > 570) & (mod <= 630)].std(ddof=1) * ANN),
    }
    for name, lo, hi in SESSIONS:
        seg = r[(mod >= lo) & (mod < hi)]
        out[f"rv_{name}"] = float(seg.std(ddof=1) * ANN) if len(seg) > 5 else float("nan")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--underlying", default="SPY")
    parser.add_argument("--out", default="docs/research/RV_VS_IV.md")
    args = parser.parse_args(argv)
    root = get_settings().zr_data_root
    und = args.underlying
    bars = load_bars(root, und)
    days_dir = sorted((root / "options_1m" / f"underlying={und}").glob("date=*"))
    rows = []
    for d in days_dir:
        day = dt.date.fromisoformat(d.name[5:])
        qp = d / "quotes.parquet"
        if not qp.exists():
            continue
        quotes = pd.read_parquet(qp)
        if quotes.empty:
            continue
        bday = bars[bars.index.tz_convert(ET).date == day]
        if bday.empty:
            continue
        spot = float(bday["open"].iloc[0])
        iv = day_iv(quotes, day, spot)
        if iv is None:
            continue
        rows.append({**iv, **realized(bday)})
    tab = pd.DataFrame(rows)
    if tab.empty:
        Path(args.out).write_text("# RV − IV\n\n没有可用的期权报价数据。\n", encoding="utf-8")
        print("no data")
        return 0
    tab["rv_day_minus_iv"] = tab["rv_day"] - tab["iv"]
    tab["rv_1h_minus_iv"] = tab["rv_1h"] - tab["iv"]
    sess = pd.DataFrame({name: tab[f"rv_{name}"] - tab["iv"] for name, _, _ in SESSIONS})
    summ = pd.DataFrame(
        {
            "指标": ["RV_day − IV", "RV_1h − IV"] + [f"RV_{n} − IV" for n, _, _ in SESSIONS],
            "均值": [tab["rv_day_minus_iv"].mean(), tab["rv_1h_minus_iv"].mean()]
            + [sess[n].mean() for n, _, _ in SESSIONS],
            "中位数": [tab["rv_day_minus_iv"].median(), tab["rv_1h_minus_iv"].median()]
            + [sess[n].median() for n, _, _ in SESSIONS],
            "RV>IV 占比": [(tab["rv_day_minus_iv"] > 0).mean(), (tab["rv_1h_minus_iv"] > 0).mean()]
            + [(sess[n] > 0).mean() for n, _, _ in SESSIONS],
            "n": [tab["rv_day_minus_iv"].notna().sum(), tab["rv_1h_minus_iv"].notna().sum()]
            + [sess[n].notna().sum() for n, _, _ in SESSIONS],
        }
    ).round(4)
    show = tab[
        [
            "day",
            "dte",
            "strike",
            "spot",
            "iv",
            "rv_1h",
            "rv_day",
            "rv_1h_minus_iv",
            "rv_day_minus_iv",
        ]
    ].round(4)
    lines = [
        f"# RV − IV（R-015）{und}",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        f"{len(tab)} 个交易日 · "
        "IV = 09:31 ET ATM call/put 中间价 BS 反推均值（r=4%，T 到到期日 16:00 ET）；"
        "RV 年化 √(252×390)。",
        "",
        "## 汇总（年化波动率差）",
        "",
        df_to_md(summ),
        "",
        "## 按日",
        "",
        df_to_md(show),
        "",
        "## 复现",
        "",
        "```bash",
        f"uv run python -m zr.research.rv_vs_iv --underlying {und}",
        "```",
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {args.out}：{len(tab)} 天")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
