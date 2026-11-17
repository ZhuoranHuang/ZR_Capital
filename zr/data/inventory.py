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


def replace_section(doc: Path, heading_prefix: str, text: str) -> None:
    """docs 里以 heading_prefix 开头的二级标题整节替换为 text；没有就追加到末尾。"""
    body = doc.read_text(encoding="utf-8")
    if heading_prefix in body:
        i = body.index(heading_prefix)
        j = body.find("\n## ", i + 1)
        body = body[:i] + text.rstrip("\n") + "\n" + (body[j:] if j >= 0 else "")
    else:
        body = body.rstrip("\n") + "\n\n" + text.rstrip("\n") + "\n"
    doc.write_text(body, encoding="utf-8")


def overview_section(root: Path, vps_json: str | None = None) -> str:
    """Mac 数据根总览；给了 VPS 的 --json 输出时，另列 VPS 上有而 Mac 没有（或更旧）的数据集。"""
    import datetime as dt

    rows = scan(root)
    L = [
        f"## 数据总览（Mac 外接盘 `$ZR_DATA_ROOT`；`zr/data/inventory.py --write-doc`，{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC 生成）",
        "",
        "数据集 = 数据根下第一层目录（带 underlying= 分区的按标的拆开）；最早 / 最晚取自路径里的日期或年份。",
        "",
        to_md(rows),
        "",
    ]
    if vps_json:
        mac = {r["dataset"]: r for r in rows}
        vps = json.loads(Path(vps_json).read_text(encoding="utf-8"))
        only = [
            r
            for r in vps
            if r["dataset"] not in mac
            or mac[r["dataset"]]["files"] < r["files"]
            or (mac[r["dataset"]]["last"] or "") < (r["last"] or "")
        ]
        L += [
            f"VPS `~/zr_data` 共 {sum(r['bytes'] for r in vps) / 1e9:.2f} GB、{len(vps)} 个数据集；其中 Mac 没有或更旧的：",
            "",
            to_md(only) if only else "（无）",
            "",
        ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument(
        "--write-doc", action="store_true", help="写入 docs/DATA_INVENTORY.md 的「数据总览」小节"
    )
    ap.add_argument("--vps-json", help="VPS 上 --json 的输出文件，用于列出只在 VPS 的数据")
    a = ap.parse_args(argv)
    if a.root:
        root = Path(os.path.expanduser(a.root))
    else:
        from zr.config import get_settings

        root = Path(get_settings().zr_data_root)
    if a.write_doc:
        doc = Path(__file__).resolve().parents[2] / "docs" / "DATA_INVENTORY.md"
        replace_section(doc, "## 数据总览", overview_section(root, a.vps_json))
        print(doc)
        return 0
    rows = scan(root, tuple(a.skip))
    print(json.dumps(rows, ensure_ascii=False) if a.json else to_md(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
