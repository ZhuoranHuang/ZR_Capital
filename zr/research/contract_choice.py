"""合约选择对比（audit/base-final 第 6 节）：D-030 入选样本的实际合约 vs V4 规则合约。

每个持仓段在负责人实际开仓时刻：
- 正股价 = 开仓时刻之前最后一根完整 1 分钟 K 的收盘（长桥 RTH 1 分钟 K）；
- 规则合约 = v4_decisive.pick_contract（DTE 日历日 0/1 最近到期，10:15 后排除 0DTE，略虚值 1 档），
  期权链 = 当日 options_1m（ThetaData）；
- 档数 = 同一到期日的上市行权价里，正股价与行权价之间隔几档（虚值为正、实值为负；
  第一档虚值 = 1）；百分比 = 方向 ×（行权价 / 正股价 − 1）；
- delta = Black-Scholes（r = 0.04，到期 16:00 ET），IV 由开仓那分钟 NBBO 中间价反推
  （options_1m 的 t 时刻报价）。
只覆盖有期权链数据的标的（MU）；DRAM、META 只列实际合约。

运行：uv run python -m zr.research.contract_choice
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from zr.config import get_settings
from zr.features.bars import ET, load_bars
from zr.features.option_features import R, implied_vol
from zr.research import intraday_single_leg as isl
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.contract_choice")


def bs_delta(s: float, k: float, t: float, sigma: float, right: str) -> float:
    if not (np.isfinite(sigma) and sigma > 0 and t > 0):
        return float("nan")
    d1 = (math.log(s / k) + (R + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return float(norm.cdf(d1) if right == "call" else norm.cdf(d1) - 1)


def years(ts: pd.Timestamp, expiry) -> float:
    close = pd.Timestamp(f"{expiry} 16:00", tz=ET)
    return (close - ts).total_seconds() / (365 * 24 * 3600)


def steps(strikes: np.ndarray, spot: float, k: float, d: int) -> int:
    """虚值档数：第一档虚值 = 1；平值以内（越过正股价）为 0、−1…"""
    s = np.sort(strikes)
    if d == 1:
        otm = s[s > spot]
        return (
            int(np.searchsorted(otm, k, side="left") + 1)
            if k > spot
            else -int(((s <= spot) & (s > k)).sum())
        )
    otm = s[s < spot][::-1]
    return (
        int(np.searchsorted(-otm, -k, side="left") + 1)
        if k < spot
        else -int(((s >= spot) & (s < k)).sum())
    )


def quote_mid(q: pd.DataFrame, sym: str, ts: pd.Timestamp) -> float:
    g = q[(q["symbol"] == sym) & (q["ts"] <= ts) & (q["ask"] > 0) & (q["bid"] > 0)]
    if g.empty:
        return float("nan")
    r = g.iloc[-1]
    return float((r["bid"] + r["ask"]) / 2)


def run(root: Path) -> pd.DataFrame:
    v4 = v4d.load_v4()
    s, _ = isl.d030_sample(root)
    bars = {"MU": load_bars(root, "MU")}
    rows = []
    for _, e in s.iterrows():
        ts = e["entry_ts"]
        et = ts.tz_convert(ET)
        day = et.date()
        d = int(e["dir"])
        right = "call" if d == 1 else "put"
        row = {
            "开仓（ET）": et.strftime("%Y-%m-%d %H:%M:%S"),
            "标的": e["und"],
            "方向": "多" if d == 1 else "空",
            "实际合约": e["symbol"],
            "实际 DTE": int(e["dte"]),
            "盈亏": float(e["pnl_net"]),
        }
        p = root / "options_1m" / f"underlying={e['und']}" / f"date={day}" / "quotes.parquet"
        if e["und"] not in bars or not p.exists():
            rows.append(row)
            continue
        b = bars[e["und"]]
        prior = b[b.index <= ts.floor("min") - pd.Timedelta(minutes=1)]
        spot = float(prior["close"].iloc[-1])
        q = pd.read_parquet(p)
        q["ts"] = pd.to_datetime(q["ts"], utc=True)
        q["expiry"] = pd.to_datetime(q["expiry"]).dt.date
        chain = v4d.day_chain(root, e["und"], day)
        pk = v4d.pick_contract(chain, day, d, spot, et.time(), v4)
        exp_a = pd.Timestamp(e["expiry"]).date()
        ks_a = q.loc[(q["expiry"] == exp_a) & (q["right"] == right), "strike"].unique()
        m_a = quote_mid(q, e["symbol"], ts.floor("min"))
        iv_a = implied_vol(m_a, spot, float(e["strike"]), years(ts, exp_a), right)
        row.update(
            {
                "正股价": spot,
                "实际行权价": float(e["strike"]),
                "实际档数": steps(ks_a, spot, float(e["strike"]), d) if len(ks_a) else np.nan,
                "实际 %": d * (float(e["strike"]) / spot - 1) * 100,
                "实际 delta": bs_delta(spot, float(e["strike"]), years(ts, exp_a), iv_a, right),
            }
        )
        if pk is None:
            dtes = sorted({(x - day).days for (x, rr) in chain if rr == right})
            late = et.time() > v4d._hm(v4["execution"]["d001_last_0dte_entry"])
            row["规则无合约原因"] = (
                "10:15 后排除 0DTE，且无 1DTE"
                if late and 0 in dtes and 1 not in dtes
                else "当日无 0/1DTE 到期"
            )
        else:
            ks_r = chain[(pk["expiry"], right)][0]
            m_r = quote_mid(q, pk["symbol"], ts.floor("min"))
            iv_r = implied_vol(m_r, spot, pk["strike"], years(ts, pk["expiry"]), right)
            row.update(
                {
                    "规则合约": pk["symbol"],
                    "规则 DTE": pk["dte"],
                    "规则行权价": pk["strike"],
                    "规则档数": steps(ks_r, spot, pk["strike"], d),
                    "规则 %": d * (pk["strike"] / spot - 1) * 100,
                    "规则 delta": bs_delta(
                        spot, pk["strike"], years(ts, pk["expiry"]), iv_r, right
                    ),
                    "同一合约": pk["symbol"] == e["symbol"],
                    "同一到期日": pk["expiry"] == exp_a,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summary(t: pd.DataFrame) -> pd.DataFrame:
    m = t[t["规则合约"].notna()] if "规则合约" in t else t.iloc[:0]
    med = lambda c: float(np.nanmedian(m[c])) if len(m) else np.nan  # noqa: E731
    return pd.DataFrame(
        [
            {"项": "持仓段（D-030 入选）", "值": len(t)},
            {"项": "有期权链、可比（MU）", "值": len(m)},
            {
                "项": "规则无合约：10:15 后排除 0DTE 且无 1DTE",
                "值": int((t.get("规则无合约原因") == "10:15 后排除 0DTE，且无 1DTE").sum()),
            },
            {
                "项": "规则无合约：当日无 0/1DTE 到期",
                "值": int((t.get("规则无合约原因") == "当日无 0/1DTE 到期").sum()),
            },
            {"项": "实际合约 = 规则合约", "值": int(m["同一合约"].sum())},
            {"项": "到期日相同", "值": int(m["同一到期日"].sum())},
            {"项": "实际 DTE 中位数", "值": med("实际 DTE")},
            {"项": "规则 DTE 中位数", "值": med("规则 DTE")},
            {"项": "实际档数中位数（虚值为正）", "值": med("实际档数")},
            {"项": "实际档数 ≤ 0（平值 / 实值）笔数", "值": int((m["实际档数"] <= 0).sum())},
            {"项": "实际 % 中位数", "值": round(med("实际 %"), 3)},
            {"项": "规则 % 中位数", "值": round(med("规则 %"), 3)},
            {
                "项": "实际 |delta| 中位数",
                "值": round(float(np.nanmedian(m["实际 delta"].abs())), 3),
            },
            {
                "项": "规则 |delta| 中位数",
                "值": round(float(np.nanmedian(m["规则 delta"].abs())), 3),
            },
        ]
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    t = run(root)
    out = root / "research" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    t.to_parquet(out / "contract_choice.parquet")
    print(summary(t).to_string())
    print(pd.crosstab(t["实际 DTE"], t.get("规则 DTE")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
