"""1 分钟 K 质量检查（只看 RTH，按交易日历）：

- `bar_range_gt_5atr`：单根振幅 (high−low) > 5 × ATR(14)（ATR 用前一根为止的值，无前视）
- `wick_spike`：影线远离实体 > 5 × ATR 且收回（疑似坏成交，只影响 high/low）
- `zero_volume_rth`：交易日 RTH 内 volume = 0 的 bar
- 日级 `day_range_outlier`：当日振幅 (max high − min low)/open > 4 × 前 20 个交易日振幅中位数

这些是「标记」，不是删除：大振幅日多数是真实行情（例：MU 2024-06-20，157→142.92，
连续下跌、成交量大，判断为真实——未与外部来源核对）。
`clean_wicks` 只对 wick_spike 的 bar 把 high/low 收回到 max/min(open, close)，
用于覆盖检查的当日高低点。
运行：uv run python -m zr.data.bar_quality --symbols MU.US SPY.US QQQ.US SMH.US
      → 写 DATA_INVENTORY 小节
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from zr.features import indicators as ind

ET = ZoneInfo("America/New_York")
K_ATR = 5.0
DAY_K = 4.0
SECTION = "## 正股 1 分钟 K 质量（zr/data/bar_quality.py 自动生成）"


def _same_minute_baseline(x: pd.Series, days: int = 20) -> pd.Series:
    """同一分钟在前 `days` 个交易日的中位数（只用之前的交易日，无前视）。"""
    et = x.index.tz_convert(ET)
    key = pd.MultiIndex.from_arrays([et.date, et.hour * 60 + et.minute])
    piv = pd.Series(x.to_numpy(), index=key).unstack(level=1).sort_index()
    base = piv.shift(1).rolling(days, min_periods=5).median()
    return pd.Series(base.stack(future_stack=True).reindex(key).to_numpy(), index=x.index)


def flag_bars(df: pd.DataFrame) -> pd.DataFrame:
    """df：RTH bar（索引 ts UTC，升序）。返回同索引的布尔标记。

    阈值同时要求超过 K×ATR(14，前一根为止) 与 K×「同一分钟前 20 日振幅中位数」：
    开盘那一分钟天然振幅大，只用隔夜带过来的 ATR 会把几乎每个 09:30 都标成异常。
    """
    atr_prev = ind.atr(df["high"], df["low"], df["close"], 14).shift(1)
    rng = df["high"] - df["low"]
    base = _same_minute_baseline(rng)
    body_hi = df[["open", "close"]].max(axis=1)
    body_lo = df[["open", "close"]].min(axis=1)
    wick = pd.concat([df["high"] - body_hi, body_lo - df["low"]], axis=1).max(axis=1)
    out = pd.DataFrame(index=df.index)
    out["bar_range_gt_5atr"] = (rng > K_ATR * atr_prev) & (rng > K_ATR * base)
    out["wick_spike"] = (wick > K_ATR * atr_prev) & (wick > K_ATR * base)
    out["zero_volume_rth"] = df["volume"] <= 0
    return out.fillna(False)


def clean_wicks(df: pd.DataFrame, flags: pd.DataFrame | None = None) -> pd.DataFrame:
    flags = flag_bars(df) if flags is None else flags
    out = df.copy()
    m = flags["wick_spike"].to_numpy()
    out.loc[m, "high"] = df.loc[m, ["open", "close"]].max(axis=1)
    out.loc[m, "low"] = df.loc[m, ["open", "close"]].min(axis=1)
    return out


def day_ranges(df: pd.DataFrame) -> pd.DataFrame:
    day = df.index.tz_convert(ET).date
    g = df.groupby(day)
    d = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    )
    d["range_pct"] = (d["high"] - d["low"]) / d["open"]
    base = d["range_pct"].shift(1).rolling(20, min_periods=10).median()
    d["range_vs_med20"] = d["range_pct"] / base
    d["day_range_outlier"] = d["range_vs_med20"] > DAY_K
    return d


def report(root: Path, symbols: list[str]) -> str:
    from zr.features.bars import load_bars

    lines = [
        SECTION,
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        "只看 RTH（交易日历过滤）。",
        "",
    ]
    summ = []
    detail = []
    for sym in symbols:
        und = sym.split(".")[0]
        try:
            raw = load_bars(root, und, rth_only=False)
            df = load_bars(root, und)
        except FileNotFoundError:
            summ.append({"symbol": und, "note": "无本地 K 线"})
            continue
        f = flag_bars(df)
        dr = day_ranges(df)
        dropped = len(raw) - len(df)
        summ.append(
            {
                "symbol": und,
                "rth_bars": len(df),
                "dropped_non_rth_or_holiday": dropped,
                "bar_range_gt_5atr": int(f["bar_range_gt_5atr"].sum()),
                "wick_spike": int(f["wick_spike"].sum()),
                "zero_volume_rth": int(f["zero_volume_rth"].sum()),
                "day_range_outlier_days": int(dr["day_range_outlier"].sum()),
            }
        )
        top = df[f["wick_spike"]].assign(symbol=und, kind="wick_spike").head(10)
        detail.append(top)
        out_days = (
            dr[dr["day_range_outlier"]]
            .assign(symbol=und)
            .reset_index()
            .rename(columns={"index": "date"})
        )
        if len(out_days):
            detail.append(
                out_days[
                    [
                        "symbol",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "range_pct",
                        "range_vs_med20",
                    ]
                ]
            )
    from zr.ops.markdown import df_to_md

    lines += [df_to_md(pd.DataFrame(summ)), ""]
    lines += [
        "dropped_non_rth_or_holiday = 原始 session=Intraday 记录中被交易日历剔除的行"
        "（全天休市日的 390 根零成交 bar、"
        "提前收盘日 13:00 后的 bar）。",
        "",
        "### 标记明细（wick_spike 前 10 根 + 全部日级振幅异常日）",
        "",
    ]
    for d in detail:
        if len(d):
            d = d.copy()
            for c in d.columns:
                if pd.api.types.is_float_dtype(d[c]):
                    d[c] = d[c].round(4)
            if "ts" in d.index.names:
                d = d.reset_index()
                d["ts"] = d["ts"].dt.tz_convert(ET).dt.strftime("%Y-%m-%d %H:%M")
            lines += [df_to_md(d), ""]
    return "\n".join(lines) + "\n"


def write_section(doc: Path, text: str) -> None:
    s = doc.read_text(encoding="utf-8") if doc.exists() else "# 数据清单\n"
    if SECTION in s:
        start = s.index(SECTION)
        nxt = s.find("\n## ", start + len(SECTION))
        s = s[:start] + text + (s[nxt + 1 :] if nxt != -1 else "")
    else:
        s = s.rstrip("\n") + "\n\n" + text
    doc.write_text(s, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from zr.config import get_settings

    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["MU.US", "SPY.US", "QQQ.US", "SMH.US"])
    p.add_argument("--doc", default="docs/DATA_INVENTORY.md")
    a = p.parse_args(argv)
    text = report(get_settings().zr_data_root, a.symbols)
    write_section(Path(a.doc), text)
    print(text[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
