"""对比 ThetaData 1 分钟 NBBO 与长桥录制器 Depth 推送：同一合约、同一分钟的 bid/ask 是否一致。

运行：uv run python -m zr.data.theta_vs_recorder --date 2026-10-21 --underlying SPY
前提：options_1m/underlying=SPY/date=2026-10-21/quotes.parquet 与
      recorder/date=2026-10-21/underlying=SPY/depth_*.parquet 都在本地。
录制器侧：每个 symbol 每分钟取该分钟内最后一条 Depth（bids[0]/asks[0]），
与 Theta 该分钟末次 NBBO 对齐。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from zr.config import get_settings
from zr.ops.markdown import df_to_md


def recorder_minute_quotes(root: Path, day: dt.date, und: str) -> pd.DataFrame:
    d = root / "recorder" / f"date={day.isoformat()}" / f"underlying={und}"
    files = sorted(d.glob("depth_*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["symbol"].str.contains(r"\d{6}[CP]\d+\.US$", regex=True)]
    rows = []
    for r in df.itertuples(index=False):
        p = json.loads(r.payload)
        bids, asks = p.get("bids") or [], p.get("asks") or []
        rows.append(
            {
                "symbol": r.symbol,
                "recv": pd.Timestamp(r.recv_ts_utc_ns, unit="ns", tz="UTC"),
                "bid_rec": bids[0]["price"] if bids else None,
                "ask_rec": asks[0]["price"] if asks else None,
                "bid_size_rec": bids[0]["volume"] if bids else None,
                "ask_size_rec": asks[0]["volume"] if asks else None,
            }
        )
    q = pd.DataFrame(rows).dropna(subset=["bid_rec", "ask_rec"])
    q["minute"] = q["recv"].dt.floor("min")
    return q.sort_values("recv").groupby(["symbol", "minute"]).last().reset_index()


def theta_minute_quotes(root: Path, day: dt.date, und: str) -> pd.DataFrame:
    p = root / "options_1m" / f"underlying={und}" / f"date={day.isoformat()}" / "quotes.parquet"
    if not p.exists():
        return pd.DataFrame()
    t = pd.read_parquet(p)
    # Theta 1m 的时间戳是区间时刻，其报价为该分钟内末次 NBBO；与录制器"该分钟最后一条"同义
    t["minute"] = t["ts"].dt.floor("min")
    return t[["symbol", "minute", "bid", "ask", "bid_size", "ask_size"]]


def compare(rec: pd.DataFrame, theta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    m = rec.merge(theta, on=["symbol", "minute"], how="inner")
    if m.empty:
        return m, {"matched_minutes": 0}
    m["bid_diff"] = (m["bid_rec"] - m["bid"]).round(4)
    m["ask_diff"] = (m["ask_rec"] - m["ask"]).round(4)
    stats = {
        "recorder_symbol_minutes": len(rec),
        "theta_symbol_minutes": len(theta),
        "matched_minutes": len(m),
        "bid_equal_pct": round(100 * (m["bid_diff"] == 0).mean(), 2),
        "ask_equal_pct": round(100 * (m["ask_diff"] == 0).mean(), 2),
        "both_equal_pct": round(100 * ((m["bid_diff"] == 0) & (m["ask_diff"] == 0)).mean(), 2),
        "bid_abs_diff_mean": round(m["bid_diff"].abs().mean(), 4),
        "ask_abs_diff_mean": round(m["ask_diff"].abs().mean(), 4),
        "bid_abs_diff_p95": round(m["bid_diff"].abs().quantile(0.95), 4),
        "ask_abs_diff_p95": round(m["ask_diff"].abs().quantile(0.95), 4),
        "within_1_tick_pct": round(
            100 * ((m["bid_diff"].abs() <= 0.01) & (m["ask_diff"].abs() <= 0.01)).mean(), 2
        ),
    }
    return m, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--underlying", default="SPY")
    parser.add_argument("--out", default=None, help="markdown 输出路径（默认打印）")
    args = parser.parse_args(argv)
    root = get_settings().zr_data_root
    day = dt.date.fromisoformat(args.date)
    rec = recorder_minute_quotes(root, day, args.underlying)
    theta = theta_minute_quotes(root, day, args.underlying)
    m, stats = compare(rec, theta)
    lines = [f"### Theta vs 录制器 {args.underlying} {day}", "", df_to_md(pd.DataFrame([stats]))]
    if len(m):
        worst = m.reindex(
            m[["bid_diff", "ask_diff"]].abs().max(axis=1).sort_values(ascending=False).index
        ).head(10)
        worst = worst[
            ["symbol", "minute", "bid_rec", "bid", "ask_rec", "ask", "bid_diff", "ask_diff"]
        ]
        lines += ["", "差异最大的 10 条：", "", df_to_md(worst)]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
