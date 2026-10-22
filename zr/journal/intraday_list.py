"""负责人交易史：日内单腿清单（D-028 最终版筛选规则，替代此前所有版本）。

输出 docs/research/INTRADAY_SINGLE_LEG_LIST.md。

规则（负责人 2026-10-22）：
- 单位 = 持仓段：同一合约从空仓到再次空仓（分批建仓、部分成交、分批卖出按 FIFO 配对后合并）。
- 保留：同一交易日开平的单腿期权交易，不限到期日；同标的 call 与 put 的持仓区间不重叠
  （先平后开的反手两笔都保留）。
- 剔除：过夜；同标的 call 与 put 持仓区间重叠，且重叠 ≥ 5 分钟或两腿开仓时间相差 ≤ 2 分钟
  （视为跨式 / 对冲）。
- 待确认（不计入也不剔除）：有重叠但不满足剔除条件——「反手重叠」，由负责人逐笔判断。
- 卖出开仓（先卖后买或没有买入平仓）：交易史里没有平仓成交（可能到期失效或被行权），盈亏缺结算
  数据，单列，不计入；其持仓区间（到平仓成交或到期日 16:00）仍参与 call / put 重叠判定。
运行：uv run python -m zr.journal.intraday_list
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.journal.analyze_history import fifo_round_trips, load
from zr.ops.markdown import df_to_md

ET = "America/New_York"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "INTRADAY_SINGLE_LEG_LIST.md"
OVERLAP_MIN = pd.Timedelta(minutes=5)
OPEN_GAP = pd.Timedelta(minutes=2)
_SYM = r"^(?P<und>[A-Z.\-]+?)(?P<exp>\d{6})(?P<right>[CP])(?P<strike>\d+)\.US$"


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), utc=True).dt.tz_convert(ET)


def episodes(ex: pd.DataFrame) -> pd.DataFrame:
    """逐合约按时间累计净持仓：0 → 非 0 开一段，回到 0 结束；末尾仍非 0 记为未平仓。"""
    e = ex.assign(ts=_ts(ex["ts_et"]), q=np.where(ex["side"] == "Buy", 1, -1) * ex["quantity"])
    e = e.sort_values(["symbol", "ts"], kind="stable")
    rows = []
    for sym, g in e.groupby("symbol", sort=False):
        pos, cur = 0.0, None
        for _, r in g.iterrows():
            if pos == 0:
                cur = {
                    "symbol": sym,
                    "side": "long" if r["q"] > 0 else "short",
                    "open_ts": r["ts"],
                    "buy_q": 0.0,
                    "buy_v": 0.0,
                    "sell_q": 0.0,
                    "sell_v": 0.0,
                    "fills": 0,
                }
            pos += r["q"]
            k = "buy" if r["q"] > 0 else "sell"
            cur[f"{k}_q"] += abs(r["q"])
            cur[f"{k}_v"] += abs(r["q"]) * r["price"]
            cur["fills"] += 1
            if abs(pos) < 1e-9:
                pos = 0.0
                rows.append({**cur, "close_ts": r["ts"], "closed": True})
                cur = None
        if cur is not None:
            rows.append({**cur, "close_ts": pd.NaT, "closed": False})
    ep = pd.DataFrame(rows)
    m = ep["symbol"].str.extract(_SYM)
    ep["und"] = m["und"]
    ep["right"] = m["right"]
    ep["strike"] = m["strike"].astype(float) / 1000
    ep["expiry"] = pd.to_datetime(m["exp"], format="%y%m%d").dt.date
    ep["dte"] = [(x - t.date()).days for x, t in zip(ep["expiry"], ep["open_ts"], strict=True)]
    # 卖出开仓没有平仓成交时，持仓区间算到到期日 16:00（只用于重叠判定）
    exp_close = pd.to_datetime(
        [dt.datetime.combine(x, dt.time(16, 0)) for x in ep["expiry"]]
    ).tz_localize(ET)
    ep["interval_end"] = ep["close_ts"].where(ep["closed"], exp_close)
    return ep.reset_index(drop=True)


def attach_pnl(ep: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    """把 FIFO 配对行（只有买入开仓—卖出平仓）归到所属持仓段，盈亏与费用相加。"""
    rt = rt.assign(entry_ts=_ts(rt["entry_et"]), exit_ts=_ts(rt["exit_et"]))
    ep = ep.copy()
    ep["lots"], ep["pnl_net"], ep["fees"] = 0, np.nan, np.nan
    for i, r in ep[ep["side"] == "long"].iterrows():
        m = (
            (rt["symbol"] == r["symbol"])
            & (rt["entry_ts"] >= r["open_ts"])
            & (rt["entry_ts"] <= (r["close_ts"] if r["closed"] else rt["entry_ts"].max()))
        )
        ep.loc[i, "lots"] = int(m.sum())
        ep.loc[i, "pnl_net"] = round(float(rt.loc[m, "pnl_net"].sum()), 2)
        ep.loc[i, "fees"] = round(float(rt.loc[m, "fees_actual"].sum()), 2)
    return ep


def classify(ep: pd.DataFrame, include_short: bool = True) -> pd.DataFrame:
    ep = ep.copy()
    ep["category"], ep["reason"], ep["partner"] = "", "", ""
    for i, r in ep.iterrows():
        if r["side"] == "short":
            ep.loc[i, ["category", "reason"]] = [
                "其他",
                "卖出开仓（多为同行权价 call / put 成对卖出，疑似卖出跨式；交易史无平仓成交）",
            ]
            continue
        if not r["closed"]:
            ep.loc[i, ["category", "reason"]] = ["其他", "买入后无平仓成交（未平仓或到期）"]
            continue
        if r["close_ts"].date() != r["open_ts"].date():
            ep.loc[i, ["category", "reason"]] = ["剔除", "过夜"]
            continue
        opp = ep[(ep["und"] == r["und"]) & (ep["right"] != r["right"])]
        if not include_short:
            opp = opp[opp["side"] == "long"]
        hard, soft = [], []
        for _, o in opp.iterrows():
            lo, hi = max(r["open_ts"], o["open_ts"]), min(r["interval_end"], o["interval_end"])
            if hi <= lo:
                continue
            ov, gap = hi - lo, abs(r["open_ts"] - o["open_ts"])
            desc = (
                f"{o['symbol']}（{'卖出开仓' if o['side'] == 'short' else '买入'}）"
                f"{o['open_ts'].strftime('%m-%d %H:%M:%S')}–"
                f"{o['interval_end'].strftime('%m-%d %H:%M:%S')}，重叠 "
                f"{ov.total_seconds() / 60:.1f} 分钟，开仓相差 {gap.total_seconds() / 60:.1f} 分钟"
            )
            (hard if (ov >= OVERLAP_MIN or gap <= OPEN_GAP) else soft).append(desc)
        if hard:
            ep.loc[i, ["category", "reason", "partner"]] = ["剔除", "跨式或对冲", "；".join(hard)]
        elif soft:
            ep.loc[i, ["category", "reason", "partner"]] = ["待确认", "反手重叠", "；".join(soft)]
        else:
            ep.loc[i, "category"] = "入选"
    return ep


def build(root: Path, include_short: bool = True) -> pd.DataFrame:
    _, ex, ch = load(root)
    rt, _ = fifo_round_trips(ex, ch)
    ep = classify(attach_pnl(episodes(ex), rt), include_short)
    ep["entry_px"] = (ep["buy_v"] / ep["buy_q"]).round(4)
    ep["exit_px"] = (ep["sell_v"] / ep["sell_q"]).round(4)
    ep["qty"] = ep["buy_q"]
    ep["hold_min"] = ((ep["close_ts"] - ep["open_ts"]).dt.total_seconds() / 60).round(1)
    return ep.sort_values("open_ts").reset_index(drop=True)


def _list(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": t["open_ts"].dt.strftime("%Y-%m-%d"),
            "入场（ET）": t["open_ts"].dt.strftime("%H:%M:%S"),
            "出场（ET）": t["close_ts"].dt.strftime("%m-%d %H:%M:%S"),
            "标的": t["und"],
            "call/put": t["right"],
            "行权价": t["strike"],
            "到期日": t["expiry"],
            "DTE": t["dte"],
            "张数": t["qty"],
            "入场价": t["entry_px"],
            "出场价": t["exit_px"],
            "持仓分钟": t["hold_min"],
            "扣费后 USD": t["pnl_net"],
        }
    )


def by_und(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for u, g in t.groupby("und"):
        w, lo = g["pnl_net"][g["pnl_net"] > 0], g["pnl_net"][g["pnl_net"] <= 0]
        rows.append(
            {
                "标的": u,
                "笔数": len(g),
                "张数": int(g["qty"].sum()),
                "胜率": round(float((g["pnl_net"] > 0).mean()), 3),
                "盈亏比": round(float(w.mean() / -lo.mean()), 2)
                if len(w) and len(lo) and lo.mean() < 0
                else np.nan,
                "扣费后合计 USD": round(float(g["pnl_net"].sum()), 2),
                "首笔": g["open_ts"].min().strftime("%Y-%m-%d"),
                "末笔": g["open_ts"].max().strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    root = get_settings().zr_data_root
    _, ex, ch = load(root)
    rt, _ = fifo_round_trips(ex, ch)
    ep = build(root)
    sel, rej, pend, oth = (ep[ep["category"] == k] for k in ("入选", "剔除", "待确认", "其他"))
    month = sel.assign(m=sel["open_ts"].dt.strftime("%Y-%m")).pivot_table(
        index="m", columns="und", values="pnl_net", aggfunc=["count", "sum"], fill_value=0
    )
    month.columns = [f"{'笔数' if a == 'count' else '扣费后 USD'} {b}" for a, b in month.columns]
    rej_tab = pd.DataFrame(
        {
            "合约": rej["symbol"],
            "开仓（ET）": rej["open_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "平仓（ET）": rej["close_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "张数": rej["qty"],
            "扣费后 USD": rej["pnl_net"],
            "原因": rej["reason"],
            "另一腿（开–平）": rej["partner"],
        }
    )
    pend_tab = pd.DataFrame(
        {
            "合约": pend["symbol"],
            "开仓（ET）": pend["open_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "平仓（ET）": pend["close_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "张数": pend["qty"],
            "扣费后 USD": pend["pnl_net"],
            "另一腿（开–平、重叠时长）": pend["partner"],
        }
    )
    oth_tab = pd.DataFrame(
        {
            "合约": oth["symbol"],
            "方向": oth["side"],
            "首笔成交（ET）": oth["open_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "成交笔数": oth["fills"],
            "卖出张数": oth["sell_q"],
            "买入张数": oth["buy_q"],
            "原因": oth["reason"],
        }
    )
    first, last = ex["ts_et"].min(), ex["ts_et"].max()
    alt = build(root, include_short=False)
    alt_tab = alt["category"].value_counts().rename_axis("类别").rename("持仓段").reset_index()
    j = ep[["symbol", "open_ts", "category"]].merge(
        alt[["symbol", "open_ts", "category"]], on=["symbol", "open_ts"], suffixes=("", "_alt")
    )
    diff = j[j["category"] != j["category_alt"]]
    diff_tab = pd.DataFrame(
        {
            "合约": diff["symbol"],
            "开仓（ET）": diff["open_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "字面口径": diff["category"],
            "不含卖出开仓腿": diff["category_alt"],
        }
    )
    lots_in = int(ep.loc[ep["category"] == "入选", "lots"].sum())
    L = [
        "# 负责人交易史：日内单腿清单（D-028 最终版筛选规则）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 复现 "
        "`uv run python -m zr.journal.intraday_list`",
        "",
        f"数据：本地交易史（`zr/journal/trade_history.py` 导出），{len(ex)} 笔成交，"
        f"{first} ～ {last}（ET）。是否为账户全部历史未验证——只能保证是导出范围内的全部。",
        "",
        "规则：单位 = 持仓段（同一合约从空仓到再空仓，分批建仓 / 分批卖出合并为一笔）；"
        "保留同日开平的"
        "单腿期权交易，不限到期日；剔除过夜，以及同标的 call 与 put 持仓区间重叠且（重叠 ≥ 5 分钟或"
        "两腿开仓相差 ≤ 2 分钟）；有重叠但不满足剔除条件的列为「反手重叠」待负责人判断。",
        "",
        "## 6. 笔数核对（先列，便于对账）",
        "",
        df_to_md(
            pd.DataFrame(
                [
                    {"类别": "入选", "持仓段": len(sel), "其中 FIFO 配对行": lots_in},
                    {
                        "类别": "剔除",
                        "持仓段": len(rej),
                        "其中 FIFO 配对行": int(rej["lots"].sum()),
                    },
                    {
                        "类别": "待确认（反手重叠）",
                        "持仓段": len(pend),
                        "其中 FIFO 配对行": int(pend["lots"].sum()),
                    },
                    {
                        "类别": "其他（卖出开仓 / 无平仓成交）",
                        "持仓段": len(oth),
                        "其中 FIFO 配对行": int(oth["lots"].sum()),
                    },
                    {"类别": "合计", "持仓段": len(ep), "其中 FIFO 配对行": int(ep["lots"].sum())},
                ]
            )
        ),
        "",
        f"FIFO 配对行总数 {len(rt)}（`fifo_round_trips`，只配买入开仓—卖出平仓）；上表合计 "
        f"{int(ep['lots'].sum())}，{'一致' if int(ep['lots'].sum()) == len(rt) else '不一致'}。"
        f"另有 {int((ep['side'] == 'short').sum())} 个卖出开仓持仓段（"
        f"{int(ep.loc[ep['side'] == 'short', 'fills'].sum())} 笔成交）在 FIFO 配对之外，"
        "交易史里没有对应的买入平仓（多为当日或次日到期，可能到期失效或被行权），盈亏需结算数据，"
        "未计入任何类别的盈亏。",
        "",
        "### 卖出开仓的腿是否参与 call / put 重叠判定的影响",
        "",
        "按规则字面，卖出开仓的腿也是 call / put 持仓（区间算到到期日 16:00），本清单采用此口径。"
        "若不让这些腿参与判定（只看买入开仓的腿之间是否重叠），分类变为：",
        "",
        df_to_md(alt_tab),
        "",
        "两种口径分类不同的持仓段：",
        "",
        df_to_md(diff_tab) if len(diff_tab) else "无。",
        "",
        "请负责人确认采用哪一种（STATUS 待确认）。另：MU260812C860000.US 的卖出成交时间记录为 "
        "08-11 08:10:04 ET（委托 04:20 ET 提交），早于期权开盘 09:30，原因未核实。",
        "",
        "## 1. 入选清单（从最早一笔到最后一笔）",
        "",
        df_to_md(_list(sel)),
        "",
        "## 2. 按标的汇总（入选）",
        "",
        df_to_md(by_und(sel)),
        "",
        "## 3. 按月份 × 标的（入选）",
        "",
        df_to_md(month.reset_index().rename(columns={"m": "月份"})),
        "",
        "## 4. 剔除清单",
        "",
        df_to_md(rej_tab),
        "",
        "## 5. 反手重叠（待负责人逐笔判断，不计入也不剔除）",
        "",
        df_to_md(pend_tab) if len(pend) else "无。",
        "",
        "## 其他：卖出开仓 / 无平仓成交",
        "",
        df_to_md(oth_tab) if len(oth) else "无。",
        "",
        "## 边界",
        "",
        "- 盈亏为扣费后实际成交（FIFO 配对，费用按订单实际收费分摊）；DTE 仅作描述。",
        "- 卖出开仓的持仓区间在没有平仓成交时算到到期日 16:00 ET，只用于 call / put 重叠判定。",
        "- 交易史范围之外的订单未覆盖。",
        "",
    ]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(
        f"持仓段 {len(ep)}：入选 {len(sel)}、剔除 {len(rej)}、待确认 {len(pend)}、其他 {len(oth)}；"
        f"FIFO 行 {int(ep['lots'].sum())}/{len(rt)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
