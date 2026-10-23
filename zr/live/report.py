"""日终报告 docs/live/DAILY_YYYY-MM-DD.md（由当日 trades_YYYY-MM-DD.jsonl 与助手状态生成）。

内容：入场笔数、护栏触发、离场原因、滑点（入场价相对当时卖一；离场成交相对下单时买一）、
纸面盈亏，以及「若无系统接管」的估算：负责人自己在账户里卖出的按其成交价，否则按 15:50 前
最后一个买一持有到收盘。全部是纸面结果，不是券商成交。
"""

from __future__ import annotations

import datetime as dt
import math
from collections import Counter
from pathlib import Path

from zr.live.core import Assistant, buy_fee, et_of, sell_fee


def no_takeover_pnl(asst: Assistant, pos) -> float:
    """负责人自己的离场（若在账户里看到卖出）；否则按 15:50 那一刻的买一持有到清仓时刻
    （助手在 15:50 前退出时用最后已知买一）。"""
    cost = pos.entry_px * pos.qty * 100 + buy_fee(pos.qty, pos.entry_px)
    if pos.owner_sells:
        sold = sum(q for _, _, q in pos.owner_sells)
        cash = sum(px * q * 100 - sell_fee(q, px) for _, px, q in pos.owner_sells)
        left = max(pos.qty - sold, 0)
    else:
        cash, left = 0.0, pos.qty
    q = (asst.quotes_at_flat or asst.quotes).get(pos.symbol)
    bid = q[1] if q else math.nan
    if left and math.isfinite(bid):
        cash += bid * left * 100 - sell_fee(left, bid)
    return cash - cost


def build(asst: Assistant, day: dt.date) -> str:
    pos = list(asst.positions.values())
    levels = Counter((a.level, a.code) for a in asst.alerts if a.code != "状态变化")
    reasons: Counter = Counter()
    exit_slip, entry_slip = [], []
    rows = []
    for p in pos:
        for o in p.orders:
            if o.fills:
                reasons[o.reason] += o.qty - o.left
                if math.isfinite(o.bid_at_create):
                    exit_slip.append(o.avg_px - o.bid_at_create)
        if math.isfinite(p.ask0):
            entry_slip.append(p.entry_px - p.ask0)
        sys_pnl = p.realized() + p.unrealized(asst.quotes.get(p.symbol, (0, math.nan))[1])
        rows.append(
            "| {} | {} | {} | {:.2f} | {} | {} | {:.2f} | {:.2f} |".format(
                et_of(p.entry_ns).strftime("%H:%M:%S"),
                p.symbol,
                p.qty,
                p.entry_px,
                "、".join(
                    a.code for a in asst.alerts if a.symbol == p.symbol and a.ns == p.entry_ns
                )
                or "—",
                "、".join(sorted({o.reason for o in p.orders})) or "—",
                sys_pnl,
                no_takeover_pnl(asst, p),
            )
        )
    tot_sys = sum(
        p.realized() + p.unrealized(asst.quotes.get(p.symbol, (0, math.nan))[1]) for p in pos
    )
    tot_no = sum(no_takeover_pnl(asst, p) for p in pos)
    mean = lambda x: sum(x) / len(x) if x else math.nan  # noqa: E731
    lines = [
        f"# 纸面护栏助手日报 {day.isoformat()}",
        "",
        "全部为纸面结果（`zr/live`，不向券商提交订单）；口径 "
        f"{asst.cfg['convention']}；纸面权益 {asst.equity:.0f} USD（暂定）。",
        "",
        f"- 入场笔数：{len(pos)}",
        f"- 系统接管纸面盈亏：{tot_sys:.2f} USD；若无系统接管（估算）：{tot_no:.2f} USD",
        f"- 入场价 − 当时卖一（均值）：{mean(entry_slip):.4f}；离场成交 − 下单时买一（均值）："
        f"{mean(exit_slip):.4f}",
        f"- kill switch：{'是' if asst.killed else '否'}",
        "",
        "## 护栏 / 风控触发",
        "",
        "| 级别 | 项 | 次数 |",
        "|---|---|---|",
        *[f"| {lv} | {code} | {n} |" for (lv, code), n in sorted(levels.items())],
        "",
        "## 离场原因（成交张数）",
        "",
        "| 原因 | 张数 |",
        "|---|---|",
        *[f"| {r} | {n} |" for r, n in reasons.most_common()],
        "",
        "## 逐笔",
        "",
        "| 入场（ET） | 合约 | 张数 | 入场价 | 护栏 | 系统离场原因 | 系统盈亏 | 若无接管 |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "「若无接管」：负责人在账户里自己卖出的按其成交价；",
        "否则按 15:50 的买一持有（估算，未计滑点）。",
        "",
    ]
    return "\n".join(lines)


def write(asst: Assistant, day: dt.date, out_dirs: list[Path]) -> list[Path]:
    text = build(asst, day)
    paths = []
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"DAILY_{day.isoformat()}.md"
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths
