"""交易史分析：FIFO 配对往返、分布统计、被拒 / 撤销订单 → docs/TRADE_HISTORY_REPORT.md。

运行：uv run python -m zr.journal.analyze_history
输入：ZR_DATA_ROOT/journal/{orders,executions,charges}.parquet（trade_history.py 产出）。
时间：SDK 时间戳为 naive UTC，报告统一转 America/New_York。
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.ops.markdown import df_to_md

ET = "America/New_York"
SYM_RE = re.compile(r"^(?P<und>[A-Z.\-]+?)(?P<exp>\d{6})(?P<right>[CP])(?P<strike>\d+)\.US$")


def parse_symbol(sym: str) -> dict:
    m = SYM_RE.match(sym)
    if not m:
        return {"und": sym, "expiry": pd.NaT, "right": None, "strike": None}
    return {
        "und": m["und"],
        "expiry": dt.datetime.strptime(m["exp"], "%y%m%d").date(),
        "right": m["right"],
        "strike": int(m["strike"]) / 1000,
    }


def _to_et(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s)
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC")
    return t.dt.tz_convert(ET)


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    j = root / "journal"
    o = pd.read_parquet(j / "orders.parquet")
    e = pd.read_parquet(j / "executions.parquet")
    c = pd.read_parquet(j / "charges.parquet")
    o["submitted_et"] = _to_et(o["submitted_at"])
    e["ts_et"] = _to_et(e["trade_done_at"])
    e = e.merge(o[["order_id", "side", "order_type"]], on="order_id", how="left")
    return o, e, c


def fifo_round_trips(
    execs: pd.DataFrame, charges: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按合约 FIFO 配对。返回 (往返, 未配对单边)。费用按张数从订单实际费用与公式费用各自摊。"""
    actual_total = charges.drop_duplicates("order_id").set_index("order_id")["total_amount"]
    order_qty = execs.groupby("order_id")["quantity"].sum()
    order_vwap = execs.groupby("order_id").apply(
        lambda g: float((g["price"] * g["quantity"]).sum() / g["quantity"].sum())
    )
    order_side = execs.groupby("order_id")["side"].first()
    formula_total = {
        oid: float(
            option_order_fees(int(order_qty[oid]), order_side[oid], order_vwap[oid])["total"]
        )
        for oid in order_qty.index
    }

    def fee_share(oid: str, qty: float) -> tuple[float, float]:
        share = qty / float(order_qty[oid])
        return float(actual_total.get(oid, 0.0)) * share, formula_total[oid] * share

    trips, open_lots = [], defaultdict(deque)
    for _, e in execs.sort_values("ts_et").iterrows():
        sym, side, qty, px = e["symbol"], e["side"], float(e["quantity"]), float(e["price"])
        sign = 1 if side == "Buy" else -1
        lots = open_lots[sym]
        remaining = qty
        while remaining > 0 and lots and lots[0]["sign"] == -sign:
            lot = lots[0]
            q = min(remaining, lot["qty"])
            entry_fee_a, entry_fee_f = fee_share(lot["order_id"], q)
            exit_fee_a, exit_fee_f = fee_share(e["order_id"], q)
            gross = (px - lot["px"]) * 100 * q * lot["sign"]
            trips.append(
                {
                    "symbol": sym,
                    **parse_symbol(sym),
                    "direction": "long" if lot["sign"] == 1 else "short",
                    "entry_et": lot["ts"],
                    "exit_et": e["ts_et"],
                    "hold_min": (e["ts_et"] - lot["ts"]).total_seconds() / 60,
                    "qty": q,
                    "entry_px": lot["px"],
                    "exit_px": px,
                    "pnl_gross": gross,
                    "fees_actual": entry_fee_a + exit_fee_a,
                    "fees_formula": entry_fee_f + exit_fee_f,
                }
            )
            lot["qty"] -= q
            remaining -= q
            if lot["qty"] <= 1e-9:
                lots.popleft()
        if remaining > 0:
            lots.append(
                {
                    "sign": sign,
                    "qty": remaining,
                    "px": px,
                    "ts": e["ts_et"],
                    "order_id": e["order_id"],
                }
            )
    trips_df = pd.DataFrame(trips)
    if len(trips_df):
        trips_df["pnl_net"] = trips_df["pnl_gross"] - trips_df["fees_actual"]
        trips_df["dte"] = (
            pd.to_datetime(trips_df["expiry"])
            - trips_df["entry_et"].dt.tz_localize(None).dt.normalize()
        ).dt.days
    unmatched = pd.DataFrame(
        [
            {
                "symbol": s,
                "side": "Buy" if lot["sign"] == 1 else "Sell",
                "qty": lot["qty"],
                "px": lot["px"],
                "ts_et": lot["ts"],
            }
            for s, lots in open_lots.items()
            for lot in lots
        ]
    )
    return trips_df, unmatched


def _bucket_30min(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%H:") + (ts.dt.minute // 30 * 30).astype(str).str.zfill(2)


def _hold_bucket(m: pd.Series) -> pd.Series:
    bins = [0, 5, 15, 30, 60, 120, 240, 1e9]
    labels = ["≤5m", "5–15m", "15–30m", "30–60m", "1–2h", "2–4h", ">4h"]
    return pd.cut(m, bins=bins, labels=labels, right=True, include_lowest=True)


def _stats(g: pd.DataFrame) -> pd.Series:
    wins = g.loc[g["pnl_net"] > 0, "pnl_net"]
    losses = g.loc[g["pnl_net"] < 0, "pnl_net"]
    return pd.Series(
        {
            "trips": len(g),
            "contracts": g["qty"].sum(),
            "win_rate": round(len(wins) / len(g), 3) if len(g) else float("nan"),
            "avg_win": round(wins.mean(), 2) if len(wins) else float("nan"),
            "avg_loss": round(losses.mean(), 2) if len(losses) else float("nan"),
            "payoff": round(wins.mean() / -losses.mean(), 2)
            if len(wins) and len(losses)
            else float("nan"),
            "avg_pnl_net": round(g["pnl_net"].mean(), 2),
            "sum_pnl_net": round(g["pnl_net"].sum(), 2),
        }
    )


def build_report(root: Path) -> str:
    orders, execs, charges = load(root)
    trips, unmatched = fifo_round_trips(execs, charges)
    md = df_to_md
    out = [
        "# 交易史报告",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        f"数据：{root / 'journal'}",
        f"订单 {len(orders)}，成交 {len(execs)} 笔，FIFO 往返 {len(trips)} 笔，"
        f"未配对单边 {len(unmatched)} 笔（不计入统计）。",
        "费用：`fees_actual` 为订单实际费用按张摊到往返；"
        "`fees_formula` 为 `zr/costs/fees.py` 公式。",
        "",
        "## 总览",
        "",
        md(_stats(trips).to_frame("全部").T.reset_index(drop=True)),
        f"\n费用合计：实际 {trips['fees_actual'].sum():.2f}，"
        f"公式 {trips['fees_formula'].sum():.2f}，"
        f"差 {trips['fees_actual'].sum() - trips['fees_formula'].sum():.2f} USD",
        "",
        "## 按标的",
        "",
        md(trips.groupby("und").apply(_stats).reset_index()),
        "",
        "## 按入场 DTE",
        "",
        md(trips.groupby("dte").apply(_stats).reset_index()),
        "",
        "## 按入场时段（ET，30 分钟桶）",
        "",
        md(
            trips.groupby(_bucket_30min(trips["entry_et"]))
            .apply(_stats)
            .reset_index()
            .rename(columns={"entry_et": "bucket"})
        ),
        "",
        "## 按持仓时长",
        "",
        md(
            trips.groupby(_hold_bucket(trips["hold_min"]), observed=True)
            .apply(_stats)
            .reset_index()
            .rename(columns={"hold_min": "bucket"})
        ),
        "",
    ]
    cols = [
        "symbol",
        "direction",
        "entry_et",
        "exit_et",
        "hold_min",
        "qty",
        "entry_px",
        "exit_px",
        "pnl_net",
    ]
    fmt = trips[cols].copy()
    fmt["entry_et"] = fmt["entry_et"].dt.strftime("%Y-%m-%d %H:%M")
    fmt["exit_et"] = fmt["exit_et"].dt.strftime("%Y-%m-%d %H:%M")
    fmt["hold_min"] = fmt["hold_min"].round(0)
    fmt["pnl_net"] = fmt["pnl_net"].round(2)
    out += [
        "## 最大盈利前 10 笔",
        "",
        md(fmt.nlargest(10, "pnl_net")),
        "",
        "## 最大亏损前 10 笔",
        "",
        md(fmt.nsmallest(10, "pnl_net")),
        "",
    ]
    if len(unmatched):
        u = unmatched.copy()
        u["ts_et"] = u["ts_et"].dt.strftime("%Y-%m-%d %H:%M")
        out += ["## 未配对单边", "", md(u), ""]

    rej = orders[orders["status"] == "Rejected"].copy()
    rej["info"] = rej["symbol"].map(parse_symbol)
    rej["expiry"] = rej["info"].map(lambda d: d["expiry"])
    rej["dte"] = (
        pd.to_datetime(rej["expiry"]) - rej["submitted_et"].dt.tz_localize(None).dt.normalize()
    ).dt.days
    rej["time_et"] = rej["submitted_et"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # reject_reason 由 trade_history.py 从 order_detail 取；旧数据没有这列时回退到 msg
    rej["reason"] = rej["reject_reason"] if "reject_reason" in rej else rej["msg"]
    rej["reason"] = rej["reason"].fillna("").replace("", "(空)")
    rej["after_11_on_expiry"] = (rej["dte"] == 0) & (rej["submitted_et"].dt.hour >= 11)
    rtab = rej[
        ["time_et", "symbol", "side", "quantity", "dte", "after_11_on_expiry", "reason"]
    ].sort_values("time_et")
    out += [
        "## 被拒订单",
        "",
        f"{len(rej)} 笔；到期日 11:00 ET 之后：{int(rej['after_11_on_expiry'].sum())} 笔；"
        f"DTE=0：{int((rej['dte'] == 0).sum())} 笔。",
        "",
        md(rtab),
        "",
        "### 拒绝原因计数",
        "",
        md(rej["reason"].value_counts().rename_axis("reason").reset_index(name="count")),
        "",
    ]
    can = orders[orders["status"] == "Canceled"]
    out += [
        "## 撤销订单按时段（ET，30 分钟桶）",
        "",
        md(
            _bucket_30min(can["submitted_et"])
            .value_counts()
            .sort_index()
            .rename_axis("bucket")
            .reset_index(name="count")
        ),
        "",
        "## 复现",
        "",
        "```bash",
        "bash deploy/sync_from_vps.sh        # 拉取 journal parquet",
        "uv run python -m zr.journal.analyze_history",
        "```",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/TRADE_HISTORY_REPORT.md")
    args = parser.parse_args(argv)
    text = build_report(get_settings().zr_data_root)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
