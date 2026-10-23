"""每日影子回放与「批量 vs 实时」对账（E 节）：录制器收盘后在 VPS 自动运行，结果按日累积到
$ZR_DATA_ROOT/reports/SHADOW_DAILY.md（仓库 docs/research/SHADOW_DAILY.md 是合并时的拷贝；
VPS 上 deploy.sh 会 stash 工作区改动，所以累积文件放在数据目录）。

步骤：zr.data.official_bars 增量更新官方 2 分钟 K → zr.research.shadow_replay 回放当日
（起点 = 录制器第一笔 MU 成交后的第一个完整 2 分钟分箱，终点 = 当日收盘）
→ 汇总一行（同日重跑则替换）。
运行：uv run python -m zr.research.shadow_daily [--day YYYY-MM-DD] [--no-update]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.data.calendar import is_trading_day
from zr.features.bars import ET
from zr.ops.markdown import df_to_md
from zr.research import shadow_replay as sr

log = logging.getLogger("zr.research.shadow_daily")
COLS = [
    "日期",
    "回放起点 ET",
    "K 数",
    "OHLC=官方",
    "close=官方",
    "量=官方",
    "状态一致",
    "H1 影子/批量/不一致",
    "H2 影子/批量/不一致",
    "H3 影子/批量/不一致",
    "收线延迟中位 s",
    "收线延迟最大 s",
    "迟到成交",
    "交易 批量/影子/配对",
    "逐笔一致",
    "净值 批量",
    "净值 影子",
]


def summarize(day: dt.date, r: dict) -> dict:
    j = r["bars"]
    px = np.logical_and.reduce(
        [np.isclose(j[c], j[f"{c}_o"]) for c in ("open", "high", "low", "close")]
    )
    lag = (j["completed_ns"] - (j.index.astype("int64") + 120_000_000_000)) / 1e9
    row = {
        "日期": str(day),
        "回放起点 ET": j.index.min().tz_convert(ET).strftime("%H:%M"),
        "K 数": len(j),
        "OHLC=官方": int(px.sum()),
        "close=官方": int(np.isclose(j["close"], j["close_o"]).sum()),
        "量=官方": int((j["volume"] == j["volume_o"]).sum()),
        "状态一致": int((j["state"] == j["state_batch"]).sum()),
        "收线延迟中位 s": round(float(lag.median()), 3),
        "收线延迟最大 s": round(float(lag.max()), 3),
        "迟到成交": int(r["late"]),
    }
    for h in ("H1", "H2", "H3"):
        row[f"{h} 影子/批量/不一致"] = (
            f"{int((j[h] != 0).sum())}/{int((j[f'{h}_batch'] != 0).sum())}/"
            f"{int((j[h] != j[f'{h}_batch']).sum())}"
        )
    t = r["trades"]
    if len(t):
        both = t["_merge"] == "both"
        same = (
            both
            & (t["symbol_batch"] == t["symbol_shadow"])
            & (t["entry_ns_batch"] == t["entry_ns_shadow"])
            & (t["exit_ns_batch"] == t["exit_ns_shadow"])
            & np.isclose(t["net_usd_batch"], t["net_usd_shadow"], atol=0.005)
        )
        row["交易 批量/影子/配对"] = (
            f"{int((t['_merge'] != 'right_only').sum())}/{int((t['_merge'] != 'left_only').sum())}/"
            f"{int(both.sum())}"
        )
        row["逐笔一致"] = int(same.sum())
        row["净值 批量"] = round(float(t["net_usd_batch"].sum()), 2)
        row["净值 影子"] = round(float(t["net_usd_shadow"].sum()), 2)
    else:
        row.update(
            {"交易 批量/影子/配对": "0/0/0", "逐笔一致": 0, "净值 批量": 0.0, "净值 影子": 0.0}
        )
    return row


def upsert(path: Path, row: dict) -> pd.DataFrame:
    csv = path.with_suffix(".csv")
    old = pd.read_csv(csv, dtype=str) if csv.exists() else pd.DataFrame(columns=COLS)
    old = old[old["日期"] != row["日期"]]
    df = pd.concat([old, pd.DataFrame([{k: str(v) for k, v in row.items()}])], ignore_index=True)
    df = df.sort_values("日期").reset_index(drop=True)[COLS]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    path.write_text(
        "\n".join(
            [
                "# 每日影子回放（实时增量路径 vs 批量回测）",
                "",
                "`uv run python -m zr.research.shadow_daily`（VPS，systemd zr-shadow-daily.time"
                "r，录制器收盘后）。"
                "口径见 docs/research/BASE_FINAL_AUDIT.md 第 5 节：MU，口径 C，状态从 2026-06-0"
                "1 起用官方 2 分钟 K 灌入；"
                "影子 = 录制器逐笔成交 → 增量引擎，信号在收线时刻决策；批量 = 官方 2 分钟 K 向量"
                "化；成交都用录制器盘口。",
                "",
                df_to_md(df),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None)
    ap.add_argument("--no-update", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    day = dt.date.fromisoformat(args.day) if args.day else dt.datetime.now(ET).date()
    if not is_trading_day(day):
        log.info("%s 不是交易日，跳过", day)
        return 0
    if not (root / "recorder" / f"date={day}" / "underlying=MU").exists():
        log.warning("%s 没有录制器数据", day)
        return 1
    if not args.no_update:
        from zr.data import official_bars

        official_bars.main(["--pages", "2"])
    res = sr.run(root, [day])
    row = summarize(day, res[day])
    out = root / "reports" / "SHADOW_DAILY.md"
    upsert(out, row)
    log.info("%s：%s → %s", day, row, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
