"""护栏证据复核（feat/guardrail-assistant）→ docs/research/GUARDRAIL_EVIDENCE.md。

在负责人 D-030 日内单腿入选样本上，按每个持仓段的真实开仓时刻，取开仓前最后一根**已收盘**的
2 分钟 K 的状态（口径 C 主、A 对照；`v4_decisive.build_bars` + `regime.indicators/regimes`，
与 V4 相同），把持仓段分到各护栏类别，报告笔数、胜率、扣费后盈亏。护栏只在证据写进本报告后启用。

类别（一笔可同时属于多类）：
- 首根前：开仓早于 09:32 ET（第一根 RTH 2 分钟 K 收盘前）；
- 逆点火：状态 IGNITION 且方向与交易方向相反；
- RANGE：状态 RANGE；
- 0DTE 10:15 后：DTE = 0 且开仓晚于 10:15 ET；
- 其余：不属于以上任何一类。
边界：样本 2026-06-09～2026-08-19，51 段（MU 48、DRAM 2、META 1；只有 MU 有本地 K 线）。

运行：uv run python -m zr.research.guardrail_evidence
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.bars import ET
from zr.features.regime import STATE_NAMES, indicators, regimes
from zr.ops.markdown import df_to_md
from zr.research import intraday_single_leg as isl
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.guardrail_evidence")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "GUARDRAIL_EVIDENCE.md"
CATS = ("首根前", "逆点火", "RANGE", "0DTE 10:15 后", "其余")


def classify(root: Path, conv_key: str) -> pd.DataFrame:
    v4 = v4d.load_v4()
    s, _ = isl.d030_sample(root)
    s = s[s["und"] == "MU"].reset_index(drop=True)
    conv, cfg = v4["conventions"][conv_key], v4d.state_cfg(v4, conv_key)
    end = max(s["entry_ts"]).tz_convert(ET).date()
    b = v4d.build_bars(root, "MU", conv, end)
    x = indicators(b, cfg, daily_reset=conv["reset"], warmup=conv["warmup"])
    reg = regimes(b, x, cfg, v4["period"])
    close = v4d.ns(b.index + pd.Timedelta(minutes=v4["period"]))
    rows = []
    for _, e in s.iterrows():
        t = int(e["entry_ts"].value)
        i = int(np.searchsorted(close, t, side="right")) - 1  # 开仓前最后一根已收盘 K
        et = e["entry_ts"].tz_convert(ET)
        st, sd = int(reg["state"].iloc[i]), int(reg["dir"].iloc[i])
        d = int(e["dir"])
        cats = []
        if et.time() < dt.time(9, 32):
            cats.append("首根前")
        if STATE_NAMES[st] == "IGNITION" and sd == -d:
            cats.append("逆点火")
        if STATE_NAMES[st] == "RANGE":
            cats.append("RANGE")
        if int(e["dte"]) == 0 and et.time() > dt.time(10, 15):
            cats.append("0DTE 10:15 后")
        rows.append(
            {
                "开仓（ET）": et.strftime("%Y-%m-%d %H:%M:%S"),
                "方向": "多" if d == 1 else "空",
                "合约": e["symbol"],
                "DTE": int(e["dte"]),
                "状态": STATE_NAMES[st],
                "状态方向": sd,
                "状态 K（ET）": b.index[i].tz_convert(ET).strftime("%H:%M"),
                "类别": "、".join(cats) if cats else "其余",
                "盈亏": float(e["pnl_net"]),
            }
        )
    return pd.DataFrame(rows)


def summary(t: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in CATS:
        m = t["类别"].str.contains(c) if c != "其余" else t["类别"] == "其余"
        g = t[m]
        out.append(
            {
                "类别": c,
                "笔数": len(g),
                "胜率": round(float((g["盈亏"] > 0).mean()), 3) if len(g) else np.nan,
                "盈亏合计": round(float(g["盈亏"].sum()), 2),
                "每笔均值": round(float(g["盈亏"].mean()), 2) if len(g) else np.nan,
            }
        )
    out.append(
        {
            "类别": "全部（MU）",
            "笔数": len(t),
            "胜率": round(float((t["盈亏"] > 0).mean()), 3),
            "盈亏合计": round(float(t["盈亏"].sum()), 2),
            "每笔均值": round(float(t["盈亏"].mean()), 2),
        }
    )
    return pd.DataFrame(out)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    tc, ta = classify(root, "C"), classify(root, "A")
    sc, sa = summary(tc), summary(ta)
    lines = [
        "# 护栏证据复核（D-030 日内单腿样本）",
        "",
        "`uv run python -m zr.research.guardrail_evidence` 生成。每个持仓段取真实开仓前最后一根",
        "已收盘 2 分钟 K 的状态（与 V4 相同的状态层；口径 C 主、A 对照）。盈亏 = 对账单扣费后。",
        "",
        "## 口径 C（主）",
        "",
        df_to_md(sc),
        "",
        "## 口径 A（对照）",
        "",
        df_to_md(sa),
        "",
        "## 读法与启用决定",
        "",
        "- 证据只来自负责人 48 段 MU 交易（2026-06-09～2026-08-19），各类别笔数很少；",
        "  护栏只作**警告 / 提示**，不据此禁止负责人开仓",
        "  （日 / 周亏损线除外，那是风控而非择时证据）。",
        "- 逆点火、首根前：警告；RANGE：提示；0DTE 10:15 后：提示 + 建议仓位减半",
        "  （D-001 的 10:15 限制按 feat/fill-model 的敏感性结果另定）。",
        "- 未覆盖：DRAM、META（3 段，无本地 K 线）；样本期外的行为。",
        "",
        "## 逐笔（口径 C）",
        "",
        df_to_md(tc),
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(sc.to_string())
    print(sa.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
