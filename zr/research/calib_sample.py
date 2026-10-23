"""成交模型校准样本（负责人 2026-10-22 补充指令）：只用 D-030 日内单腿的订单。

纳入：
- D-030 入选持仓段对应的全部买入与卖出订单（fill_model.order_sample 的 d030 组）；
- 同期、同标的、属于日内单腿操作的撤单 / 过期期权限价单（未成交）。判定（实现方口径，逐单列出理由）：
  · 长期合约（提交日 DTE > 7，沿用 D-029 的界线）→ 不纳入；
  · 卖单：提交时刻落在同合约某持仓段内——该段为 D-030 入选 → 纳入（平仓挂单）；该段为过夜 / 跨式或
    对冲 / 卖出开仓 / 反手重叠 → 不纳入；不在任何持仓段内（卖出开仓尝试等）→ 无法判断；
  · 买单：提交前后 5 分钟（[提交 − 5 min, 结束 + 5 min]）同标的没有任何持仓 → 纳入；
    只有同合约、在撤单之后才开仓的 D-030 入选持仓段（撤单后改挂成交，同一操作）→ 纳入；
    与过夜 / 跨式或对冲 / 卖出开仓持仓重叠 → 不纳入；其他（同标的另有入选持仓，可能是加仓、反手或
    跨式的一腿）→ 无法判断。
无法判断的不纳入校准，单列笔数。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from zr.journal.intraday_list import build as build_list
from zr.research import fill_model as fm

MARGIN = pd.Timedelta(minutes=5)
LONG_DTE = 7


def classify(root: Path) -> pd.DataFrame:
    s = fm.order_sample(root)
    s["order_id"] = s["order_id"].astype(str)
    ep = build_list(root)
    ep = ep.assign(
        t0=ep["open_ts"].dt.tz_convert("UTC"),
        t1=ep["interval_end"].dt.tz_convert("UTC"),
        und=[x.split("2")[0] if isinstance(x, str) else x for x in ep["und"]],
    )
    out = []
    for _, o in s.iterrows():
        if o["group"] == "d030":
            out.append({"order_id": o["order_id"], "calib": True, "class": "D-030 入选持仓段订单"})
            continue
        ts, te = o["submit"], o["end"] if pd.notna(o["end"]) else o["submit"]
        dte = (o["expiry"] - o["day"]).days
        if dte > LONG_DTE:
            out.append(
                {
                    "order_id": o["order_id"],
                    "calib": False,
                    "class": f"不纳入：长期合约（DTE {dte}）",
                }
            )
            continue
        same = ep[ep["und"] == o["und"]]
        if o["side"] == "Sell":
            hold = same[(same["symbol"] == o["symbol"]) & (same["t0"] <= ts) & (same["t1"] >= ts)]
            if hold.empty:
                c = (False, "无法判断：卖单不在任何持仓段内")
            elif (hold["category"] == "入选").all():
                c = (True, "纳入：入选持仓段的平仓挂单")
            else:
                c = (
                    False,
                    f"不纳入：{hold['category'].iloc[0]}持仓（{hold['reason'].iloc[0]}）的卖单",
                )
        else:
            near = same[(same["t1"] >= ts - MARGIN) & (same["t0"] <= te + MARGIN)]
            repost = near[
                (near["symbol"] == o["symbol"]) & (near["category"] == "入选") & (near["t0"] >= te)
            ]
            other = near.drop(repost.index)
            if near.empty:
                c = (True, "纳入：前后 5 分钟同标的无持仓")
            elif other.empty:
                c = (True, "纳入：撤单后同合约改挂成交（同一操作）")
            elif (other["category"] != "入选").any():
                k = other[other["category"] != "入选"].iloc[0]
                c = (False, f"不纳入：与{k['category']}持仓（{k['reason']}）重叠")
            else:
                c = (False, "无法判断：同标的另有入选持仓（加仓 / 反手 / 跨式一腿）")
        out.append({"order_id": o["order_id"], "calib": c[0], "class": c[1]})
    return s.merge(pd.DataFrame(out), on="order_id")
