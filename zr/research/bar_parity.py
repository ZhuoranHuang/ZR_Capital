"""K 线一致性（audit/base-final 第 1 节）：长桥官方 2 / 5 分钟 K vs 本地 1 分钟 K 合成，逐根比对。

官方 K 由 scripts/diag_pull_candles.py 在 VPS 拉取（RTH：TradeSessions.Intraday 2 / 5 分钟；
含盘前盘后：TradeSessions.All 2 分钟，口径 C 用），放在 $ZR_DATA_ROOT/diag/official_bars/。
本地合成 = zr.features.bars.resample_bars（RTH 用 load_bars；全时段用
load_extended_bars(clean=False)，与口径 A / C 实际使用的数据一致）。比对范围 = 两边都有数据的
交易日（本地最后一个分箱可能不完整，不计）。

逐根判定：时间标签两边都存在、O/H/L/C 完全相等（容差 1e-6）、成交量完全相等。
不一致按原因归类：只一边有、缺 1 分钟 K（分箱内 1 分钟根数 < 周期）、价格不同、成交量不同。

运行：uv run python -m zr.research.bar_parity
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.bars import ET, et_date, load_bars, load_extended_bars, resample_bars

log = logging.getLogger("zr.research.bar_parity")
PX = ("open", "high", "low", "close")
CASES = (  # (标的, 周期分钟, 是否全时段)
    ("MU", 2, False),
    ("MU", 5, False),
    ("QQQ", 2, False),
    ("QQQ", 5, False),
    ("MU", 2, True),
    ("QQQ", 2, True),
)


def official_path(root: Path, und: str, period: int, full: bool) -> Path:
    return root / "diag" / "official_bars" / f"{und}_{period}m{'_all' if full else ''}.parquet"


def load_official(root: Path, und: str, period: int, full: bool) -> pd.DataFrame | None:
    p = official_path(root, und, period, full)
    if not p.exists():
        return None
    o = pd.read_parquet(p)
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    o = o.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    return o


def synth(one_min: pd.DataFrame, period: int) -> pd.DataFrame:
    """1 分钟 → period 分钟，附分箱内 1 分钟根数 n1。"""
    b = resample_bars(one_min, f"{period}min")
    n = (
        one_min["close"].groupby(one_min.index.floor(f"{period}min")).count()
    )  # 分箱锚 UTC 零点，与 resample_bars(origin="start_day") 相同
    b["n1"] = n.reindex(b.index).fillna(0).astype(int)
    return b


def compare(local: pd.DataFrame, off: pd.DataFrame, period: int) -> pd.DataFrame:
    """逐根对齐（outer join），只保留两边共同覆盖的交易日；返回每根的判定。"""
    common = sorted(set(et_date(local.index)) & set(et_date(off.index)))
    if not common:
        return pd.DataFrame()
    last_local = local.index.max()  # 本地最后一个分箱可能不完整
    lo = local[np.isin(et_date(local.index), common) & (local.index < last_local)]
    of = off[np.isin(et_date(off.index), common) & (off.index < last_local)]
    j = lo[[*PX, "volume", "n1"]].join(of[[*PX, "volume"]], rsuffix="_o", how="outer")
    j["day"] = et_date(j.index)
    has_l, has_o = j["open"].notna(), j["open_o"].notna()
    px_eq = np.logical_and.reduce([np.isclose(j[c], j[f"{c}_o"], atol=1e-6) for c in PX])
    vol_eq = j["volume"] == j["volume_o"]
    cause = np.select(
        [
            has_l & ~has_o,
            ~has_l & has_o,
            px_eq & vol_eq,
            j["n1"] < period,
            ~px_eq,
        ],
        ["only_local", "only_official", "match", "missing_1m", "price"],
        default="volume",
    )
    j["cause"] = cause
    return j


def summarize(j: pd.DataFrame, und: str, period: int, full: bool) -> dict:
    c = j["cause"].value_counts()
    days = j.groupby("day")["cause"].apply(lambda s: (s != "match").any())
    both = j["open"].notna() & j["open_o"].notna()
    return {
        "标的": und,
        "周期": f"{period} 分钟",
        "时段": "含盘前盘后" if full else "RTH",
        "日期": f"{j['day'].min()} ~ {j['day'].max()}",
        "交易日": int(j["day"].nunique()),
        "比对 K 数": len(j),
        "完全一致": int(c.get("match", 0)),
        "一致率": round(float(c.get("match", 0)) / len(j), 6),
        "仅本地": int(c.get("only_local", 0)),
        "仅官方": int(c.get("only_official", 0)),
        "缺 1 分钟 K": int(c.get("missing_1m", 0)),
        "价格不同": int(c.get("price", 0)),
        "仅成交量不同": int(c.get("volume", 0)),
        "close 一致率（两边都有）": round(
            float(np.isclose(j.loc[both, "close"], j.loc[both, "close_o"], atol=1e-6).mean()), 6
        ),
        "有不一致的交易日": int(days.sum()),
    }


def run(root: Path) -> tuple[pd.DataFrame, dict[tuple, pd.DataFrame]]:
    rows, detail = [], {}
    cache: dict[tuple, pd.DataFrame] = {}
    for und, period, full in CASES:
        off = load_official(root, und, period, full)
        if off is None:
            log.warning("缺官方 K：%s", official_path(root, und, period, full))
            continue
        key = (und, full)
        if key not in cache:
            cache[key] = (
                load_extended_bars(root, und, clean=False) if full else load_bars(root, und)
            )
        j = compare(synth(cache[key], period), off, period)
        if j.empty:
            continue
        detail[(und, period, full)] = j
        rows.append(summarize(j, und, period, full))
        log.info("%s", rows[-1])
    return pd.DataFrame(rows), detail


def mismatch_examples(j: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    bad = j[j["cause"] != "match"].copy()
    bad["et"] = bad.index.tz_convert(ET).strftime("%Y-%m-%d %H:%M")
    cols = ["et", "cause", "n1", *PX, "volume", *(f"{c}_o" for c in PX), "volume_o"]
    return bad[cols].head(k)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    summary, detail = run(root)
    out = root / "research" / "bar_parity"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    for (und, period, full), j in detail.items():
        j[j["cause"] != "match"].to_parquet(
            out / f"mismatch_{und}_{period}m{'_all' if full else ''}.parquet"
        )
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
