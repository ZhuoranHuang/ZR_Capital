# ruff: noqa: E501
"""数据盘点（只用标准库，Mac 与 VPS 都能跑）：扫描数据根，每个数据集一行——大小、文件数、路径里出现的最早 / 最晚日期。

数据集 = 数据根下第一层目录；带 underlying= 分区的再按标的拆一层。日期取自路径中的 YYYY-MM-DD（没有则取 year=YYYY）。
运行：python -m zr.data.inventory [--root 路径] [--json]；VPS：ssh zr python3 - --root ~/zr_data --json < zr/data/inventory.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

DATE = re.compile(r"(20\d\d-\d\d-\d\d)")
YEAR = re.compile(r"year=(20\d\d)")


def scan(root: Path, skip: tuple[str, ...] = ()) -> list[dict]:
    out: dict[str, dict] = {}
    for top in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in skip):
        for dp, _, files in os.walk(top):
            rel = os.path.relpath(dp, root)
            parts = rel.split(os.sep)
            key = top.name
            # 只在「数据集/underlying=X」或「数据集/子集/underlying=X」时按标的拆；录制器等按日期分区的不拆
            for depth in (1, 2):
                if len(parts) > depth and parts[depth].startswith("underlying="):
                    if all(not x.startswith("date=") for x in parts[1:depth]):
                        key = "/".join(parts[: depth + 1])
                    break
            r = out.setdefault(
                key, {"dataset": key, "bytes": 0, "files": 0, "first": None, "last": None}
            )
            for f in files:
                if f.startswith("."):
                    continue
                path = os.path.join(rel, f)
                try:
                    r["bytes"] += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    continue
                r["files"] += 1
                ds = DATE.findall(path) or [f"{y}" for y in YEAR.findall(path)]
                for d in ds:
                    r["first"] = d if r["first"] is None or d < r["first"] else r["first"]
                    r["last"] = d if r["last"] is None or d > r["last"] else r["last"]
    return [r for r in out.values() if r["files"]]


def to_md(rows: list[dict]) -> str:
    L = ["| 数据集 | 大小 | 文件数 | 最早 | 最晚 |", "|---|---|---|---|---|"]
    for r in rows:
        b = r["bytes"]
        size = f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"
        L.append(
            f"| `{r['dataset']}` | {size} | {r['files']:,} | {r['first'] or '—'} | {r['last'] or '—'} |"
        )
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.root:
        root = Path(os.path.expanduser(a.root))
    else:
        from zr.config import get_settings

        root = Path(get_settings().zr_data_root)
    rows = scan(root, tuple(a.skip))
    print(json.dumps(rows, ensure_ascii=False) if a.json else to_md(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
