# ruff: noqa: E501  （中文说明较长）
"""Massive SIP 1 分钟 K（D-038：MU、QQQ、SPY、SMH、AMD 的历史分钟 K 以此为准）。

- consolidate：把 VPS pass1 的逐日文件 $ZR_DATA_ROOT/sip/massive/extra1m/YYYY/<日>.parquet 合并成
  $ZR_DATA_ROOT/bars_1m_sip/underlying=<代码>/year=<年>/bars.parquet（与长桥 bars_1m 同布局；ts = UTC K 线开始时刻；
  session ∈ Pre / Intraday / Post，Intraday = 09:30 至当日收盘（提前收盘日按交易所日程）；只有有成交的分钟才有行，
  不补占位 K；不复权）。
- load_sip_1m：读取。
- inventory：Massive 数据清单（markdown），写入 docs/DATA_INVENTORY.md 的「Massive」小节。
运行：python -m zr.data.sip_bars --consolidate | --inventory
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

SYMS = ("MU", "QQQ", "SPY", "SMH", "AMD")


def sip_dir(root: Path) -> Path:
    return Path(root) / "bars_1m_sip"


def consolidate(root: Path) -> dict:
    src = Path(root) / "sip" / "massive" / "extra1m"
    by_year: dict[int, list[Path]] = {}
    for p in sorted(src.glob("*/*.parquet")):
        by_year.setdefault(int(p.parent.name), []).append(p)
    out = {}
    for y, files in sorted(by_year.items()):
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        d = d.drop_duplicates(["symbol", "ts"]).sort_values(["symbol", "ts"])
        for s, g in d.groupby("symbol"):
            p = sip_dir(root) / f"underlying={s}" / f"year={y}" / "bars.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            g.reset_index(drop=True).to_parquet(tmp, index=False)
            tmp.replace(p)
            out[(s, y)] = len(g)
    return {f"{s} {y}": n for (s, y), n in out.items()}


def load_sip_1m(
    root: Path,
    symbol: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    rth_only: bool = True,
) -> pd.DataFrame:
    files = sorted((sip_dir(root) / f"underlying={symbol}").glob("year=*/bars.parquet"))
    d = (
        pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if files
        else pd.DataFrame()
    )
    if d.empty:
        return d
    day = d["ts"].dt.tz_convert("America/New_York").dt.date
    keep = pd.Series(True, index=d.index)
    if start:
        keep &= day >= start
    if end:
        keep &= day <= end
    if rth_only:
        keep &= d["session"] == "Intraday"
    return d[keep].reset_index(drop=True)


def inventory(root: Path) -> str:
    from zr.data.massive_archive import MANIFEST, raw_root, read_manifest

    root = Path(root)
    L = [
        f"## Massive（SIP，D-038；`zr/data/sip_bars.py --inventory`，{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC 在 Mac 生成）",
        "",
    ]
    man = read_manifest(raw_root(root) / MANIFEST)
    ver = raw_root(root) / "VERIFIED_mac.txt"
    nver = len(ver.read_text(encoding="utf-8").split()) if ver.exists() else 0
    for pre in ("minute_aggs_v1", "day_aggs_v1"):
        ks = sorted(k for k in man if f"/{pre}/" in k)
        if ks:
            L.append(
                f"- 原始文件 `massive_raw/us_stocks_sip/{pre}/`：{len(ks):,} 个，{ks[0].split('/')[-1][:10]}～{ks[-1].split('/')[-1][:10]}，"
                f"{sum(man[k][0] for k in ks) / 1e9:.2f} GB（sha256 清单 `massive_raw/MANIFEST_vps.tsv`）"
            )
    L.append(
        f"- Mac 外接盘核对通过（大小与 sha256 与 VPS 清单一致）：{nver:,} 个文件（`massive_raw/VERIFIED_mac.txt`）；核对通过且逐日处理完成的已从 VPS 删除。"
    )
    L += [
        "",
        "SIP 1 分钟 K（`bars_1m_sip/`，含盘前盘后，不复权，只有有成交的分钟）：",
        "",
        "| 标的 | 最早 (UTC) | 最晚 (UTC) | 行数 | 常规时段行数 | 交易日 | 常规时段缺根的天数（缺根合计） | 按年 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    from zr.data.calendar import session_close

    gaps = []
    for s in SYMS:
        files = sorted((sip_dir(root) / f"underlying={s}").glob("year=*/bars.parquet"))
        if not files:
            continue
        parts = {
            int(f.parent.name.split("=")[1]): pd.read_parquet(f, columns=["ts", "session"])
            for f in files
        }
        d = pd.concat(parts.values())
        rth = d[d["session"] == "Intraday"]
        per = rth.groupby(rth["ts"].dt.tz_convert("America/New_York").dt.date).size()
        exp = pd.Series(
            {x: session_close(x).hour * 60 + session_close(x).minute - 570 for x in per.index}
        )
        short = (exp - per)[per != exp]
        if len(short) and len(short) <= 3:
            gaps.append(f"{s}：" + "、".join(f"{k} 缺 {int(v)} 根" for k, v in short.items()))
        L.append(
            f"| {s} | {d['ts'].min()} | {d['ts'].max()} | {len(d):,} | {len(rth):,} | {per.size:,} | {len(short)}（{int(short.sum())}） | "
            + ", ".join(f"{y}: {len(v):,}" for y, v in parts.items())
            + " |"
        )
    L += [
        "",
        "常规时段应有根数按交易所日程（提前收盘日 13:00）计算；缺根 = 该分钟在源文件里没有行（没有成交，或源数据缺）。"
        + ("缺根天数 ≤ 3 的标的逐日列出：" + "；".join(gaps) + "。" if gaps else ""),
    ]
    m = root / "sip" / "massive"
    L += ["", "R-057 逐日产出（`sip/massive/`）：", ""]
    for name, desc in (
        ("daily", "日聚合 + 常规时段聚合 + 开盘区间，每日一个文件"),
        ("bars1", "每日 RelVol 前 50 的常规时段 1 分钟 K"),
        ("cand_out", "基础版候选的成交结果"),
        ("extra1m", "五只标的全时段 1 分钟 K（逐日，已合并到 bars_1m_sip）"),
    ):
        fs = sorted((m / name).glob("*/*.parquet"))
        if fs:
            L.append(
                f"- `{name}/`：{len(fs):,} 个交易日（{fs[0].stem}～{fs[-1].stem}），{sum(f.stat().st_size for f in fs) / 1e6:,.0f} MB——{desc}"
            )
    pit = sorted((m / "ref" / "pit").glob("*.parquet"))
    if pit:
        L.append(
            f"- `ref/`：XNYS / XNAS 普通股代码表（含已退市）、拆股表、每周时点快照 {len(pit)} 个（{pit[0].stem}～{pit[-1].stem}）"
        )
    return "\n".join(L) + "\n"


def write_inventory(root: Path, doc: Path) -> None:
    """把 inventory() 的输出写进 docs/DATA_INVENTORY.md：已有「## Massive（SIP」小节就整节替换，否则追加到末尾。"""
    text = doc.read_text(encoding="utf-8")
    sec = inventory(root)
    key = "## Massive（SIP"
    if key in text:
        i = text.index(key)
        j = text.find("\n## ", i + 1)
        text = text[:i] + sec + (text[j + 1 :] if j >= 0 else "")
    else:
        text = text.rstrip("\n") + "\n\n" + sec
    doc.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from zr.config import get_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument(
        "--write-inventory", action="store_true", help="写入 docs/DATA_INVENTORY.md 的 Massive 小节"
    )
    a = ap.parse_args(argv)
    root = get_settings().zr_data_root
    if a.consolidate:
        print(consolidate(root))
    if a.inventory:
        print(inventory(root))
    if a.write_inventory:
        doc = Path(__file__).resolve().parents[2] / "docs" / "DATA_INVENTORY.md"
        write_inventory(root, doc)
        print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
