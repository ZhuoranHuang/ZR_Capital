"""ThetaData 全量拉取进度：按 manifest 汇总一个月的结果，
追加到 docs/DATA_INVENTORY.md 的进度表。

由 scripts/theta_full_pull.sh 每完成一个月调用一次：
  uv run python -m zr.data.theta_progress --underlying SPY \
      --start 2023-10-02 --end 2023-10-31 --seconds 812 --rc 0
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from zr.config import get_settings
from zr.data.calendar import is_trading_day

TABLE_HEADER = "## 全量拉取进度（scripts/theta_full_pull.sh 自动追加）"
COLUMNS = (
    "| 标的 | 月份 | 交易日 | 失败日 | 覆盖/未覆盖 | quotes 行 | ohlc 行 | oi 行 | 拉取秒数 "
    "| parquet MB | 完成时间 (ET) |"
)
SEP = "|---|---|---|---|---|---|---|---|---|---|---|"


def month_summary(root: Path, und: str, start: dt.date, end: dt.date) -> dict:
    """一个月的汇总：交易日/失败日按 manifest（同一天以最后一条非补拉记录为准）；
    行数与大小直接读 parquet 元数据（补拉会追加记录，不能只看最后一条）。"""
    import pyarrow.parquet as pq

    manifest = root / "options_1m" / "manifest.jsonl"
    latest: dict[str, dict] = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("underlying") != und or r.get("supplement"):
                continue
            d = dt.date.fromisoformat(str(r["date"]))
            if start <= d <= end and is_trading_day(d):  # 休市日记录（已删目录）不计
                latest[d.isoformat()] = r
    ok = [d for d, r in latest.items() if "error" not in r]
    from zr.data.thetadata import coverage_report

    cov = coverage_report(root, und, [dt.date.fromisoformat(d) for d in latest])
    n_cov = int((cov["covered"] == True).sum()) if len(cov) else 0  # noqa: E712
    n_unc = int((cov["covered"] == False).sum()) if len(cov) else 0  # noqa: E712
    rows = {"quotes": 0, "ohlc": 0, "oi": 0}
    size = 0
    for d in latest:
        p = root / "options_1m" / f"underlying={und}" / f"date={d}"
        if not p.is_dir():
            continue
        for name in rows:
            f = p / f"{name}.parquet"
            if f.exists():
                rows[name] += pq.read_metadata(f).num_rows
                size += f.stat().st_size
    return {
        "days": len(ok),
        "failed": len(latest) - len(ok),
        "quotes": rows["quotes"],
        "ohlc": rows["ohlc"],
        "oi": rows["oi"],
        "mb": round(size / 1e6, 2),
        "covered": n_cov,
        "uncovered": n_unc,
    }


def format_row(und: str, start: dt.date, s: dict, seconds: int, finished: str) -> str:
    return (
        f"| {und} | {start.strftime('%Y-%m')} | {s['days']} | {s['failed']} | "
        f"{s.get('covered', 0)}/{s.get('uncovered', 0)} | {s['quotes']} | "
        f"{s['ohlc']} | {s['oi']} | {seconds} | {s['mb']} | {finished} |"
    )


def append_row(doc: Path, row: str) -> None:
    text = doc.read_text(encoding="utf-8") if doc.exists() else "# 数据清单\n"
    if TABLE_HEADER not in text:
        text = text.rstrip("\n") + f"\n\n{TABLE_HEADER}\n\n{COLUMNS}\n{SEP}\n"
    lines = text.rstrip("\n").split("\n")
    # 插入到进度表最后一行之后（表可能不在文末）
    start = lines.index(TABLE_HEADER)
    i = start + 1
    while i < len(lines) and (lines[i].startswith("|") or lines[i] == ""):
        if lines[i] == "" and i > start + 3:
            break
        i += 1
    lines.insert(i, row)
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--underlying", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--seconds", type=int, default=0)
    p.add_argument("--rc", type=int, default=0)
    p.add_argument("--doc", default="docs/DATA_INVENTORY.md")
    a = p.parse_args(argv)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    s = month_summary(get_settings().zr_data_root, a.underlying, start, end)
    finished = (
        dt.datetime.now(dt.UTC).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M")
    )
    if a.rc != 0:
        finished += f"（rc={a.rc}）"
    append_row(Path(a.doc), format_row(a.underlying, start, s, a.seconds, finished))
    print(format_row(a.underlying, start, s, a.seconds, finished))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
