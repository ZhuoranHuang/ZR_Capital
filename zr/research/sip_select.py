# ruff: noqa: E501  （中文注释较长）
"""R-057 Massive 版选股（Mac 上批量运行，输入为 VPS pass1 / meta 的产出，经同步任务传回）。

- 标的池（预登记修订 (2)）：当日分钟文件有成交（日文件有该代码），且该代码在 reference tickers 的时点快照中为
  XNYS / XNAS 普通股。快照按周取（每周第一个交易日，date= 该日）；周内上市的新股下周才进池（新股本就需 14 个交易日历史，
  不影响合格），周内退市的股票当周仍在池中（退市后没有成交，不影响），周内类型变化忽略。另抽 12 个交易日做逐日快照核对。
- 合格（补充，2026-11-14）：Massive 日聚合（d_open / d_high / d_low / d_close / d_volume）→ zr.research.sip_orb.daily_eligibility，
  拆股日取自拆股接口（split_to ÷ split_from 超出 ×1.2 / ÷1.2），交易日序号 s … s+14 剔除。
- 开盘区间、RelVol、方向（修订 (5)）：09:30–09:34 五根 1 分钟 K（pass1 的 or_*）→ zr.research.sip_orb.or_stats。
- 选股：zr.research.sip_orb.select（合格 ∩ RelVol ≥ 1 ∩ 非十字星，RelVol 降序、同值按代码）取前 50。
- 基础版候选：池内 ∩ 合格 ∩ 非十字星（不加 RelVol 条件）。

输出 $ZR_DATA_ROOT/sip/massive/derived/：candidates.parquet（池内有开盘区间的全部股票日）、pass2_rows.parquet（上传到 VPS 供 pass2）、
pool_check.json（逐日快照核对）。
运行：python -m zr.research.sip_select
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zr.research import sip_orb as so

COLS = [
    "date",
    "ticker",
    "d_open",
    "d_high",
    "d_low",
    "d_close",
    "d_volume",
    "or_open",
    "or_high",
    "or_low",
    "or_close",
    "or_volume",
    "or_bars",
]


def mroot(root: Path | None = None) -> Path:
    if root is None:
        from zr.config import get_settings

        root = get_settings().zr_data_root
    return Path(root) / "sip" / "massive"


def load_daily(r: Path, tickers: set | None = None) -> pd.DataFrame:
    parts = []
    for f in sorted((r / "daily").glob("*/*.parquet")):
        x = pd.read_parquet(f, columns=COLS)
        if tickers is not None:
            x = x[x["ticker"].isin(tickers)]
        parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d


def split_table(splits: pd.DataFrame) -> dict[str, list]:
    """拆股接口 → {代码: [执行日]}，只保留比例超出 ×1.2 / ÷1.2 的（与原规则的阈值相同）。"""
    s = splits.dropna(subset=["split_from", "split_to"]).copy()
    ratio = s["split_to"].astype(float) / s["split_from"].astype(float)
    s = s[(ratio > so.SPLIT_BAND) | (ratio < 1 / so.SPLIT_BAND)]
    s["execution_date"] = pd.to_datetime(s["execution_date"]).dt.date
    return {t: sorted(g["execution_date"]) for t, g in s.groupby("ticker")}


def pool_membership(pit_dir: Path, days: list) -> pd.DataFrame:
    """每个交易日 → 该日所在周的快照（快照日 ≤ 当日中最近的一个）里的普通股。返回 (date, ticker)。"""
    snaps = sorted(pit_dir.glob("*.parquet"))
    sd = [pd.Timestamp(p.stem).date() for p in snaps]
    rows = []
    cache: dict = {}
    for d in days:
        k = int(np.searchsorted(np.array(sd, dtype=object), d, side="right")) - 1
        if k < 0:
            continue
        if k not in cache:
            cache.clear()
            cache[k] = pd.read_parquet(snaps[k])["ticker"].dropna().unique()
        rows.append(pd.DataFrame({"date": d, "ticker": cache[k]}))
    return pd.concat(rows, ignore_index=True)


LOOKBACK_ROWS = so.LOOKBACK
GAP_DAYS = 30  # 同一代码相邻两个有日线的交易日相隔超过 30 个自然日 → 视为新标的（代码复用或长期停牌后恢复），历史从断档后重新累计


def _segment_features(g: pd.DataFrame, split_dates) -> pd.DataFrame:
    first = g["date"].iloc[0]
    e = so.daily_eligibility(
        g.rename(
            columns={
                "d_open": "open",
                "d_high": "high",
                "d_low": "low",
                "d_close": "close",
                "d_volume": "volume",
            }
        ),
        # 只计本段历史之内的拆股：段首之前的拆股即使在 15 个交易日内，也落在 14 日预热期里，不需要再剔除
        split_dates=[x for x in (split_dates or []) if x >= first],
    )
    has_or = g["or_bars"].fillna(0) > 0
    o = g.loc[has_or, ["date", "or_open", "or_high", "or_low", "or_close", "or_volume"]].rename(
        columns={"or_volume": "or_vol"}
    )
    o = so.or_stats(o)
    return e.merge(o, on="date", how="left")


def ticker_features(g: pd.DataFrame, split_dates=None) -> pd.DataFrame:
    """一个代码的日线（按日期升序）→ 合格标记与开盘区间特征（每行 = 一个交易日；只用当日开盘前已知的信息 + 当日 09:30–09:34）。
    预登记「代码变更视为新标的（需重新累计 14 日）」：相邻两个有日线的交易日相隔超过 GAP_DAYS 个自然日就切段，
    ATR、14 日均量、RelVol 与拆股序号都只用本段之内的数据（断档只看当日之前的日期，不引入前视）。"""
    g = g.sort_values("date").reset_index(drop=True)
    gap = pd.to_datetime(g["date"]).diff().dt.days
    seg = (gap > GAP_DAYS).cumsum().to_numpy()
    if seg[-1] == 0:
        out = _segment_features(g, split_dates)
        out["segment"] = 0
        return out
    parts = []
    for k, x in g.groupby(seg, sort=True):
        f = _segment_features(x.reset_index(drop=True), split_dates)
        f["segment"] = int(k)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def build(r: Path) -> dict:
    ref = r / "ref"
    pit_tickers = set()
    for p in (ref / "pit").glob("*.parquet"):
        pit_tickers |= set(pd.read_parquet(p)["ticker"].dropna())
    d = load_daily(r, pit_tickers)
    n_no_day = int(d["d_open"].isna().sum())
    d = d[d["d_open"].notna()]  # 只有分钟 K、日文件没有该代码的行（极少）不参与
    days = sorted(d["date"].unique())
    sp = split_table(pd.read_parquet(ref / "splits.parquet"))
    parts = []
    for t, g in d.groupby("ticker", sort=True):
        f = ticker_features(g, sp.get(t))
        f["symbol"] = t
        parts.append(f)
    F = pd.concat(parts, ignore_index=True)
    pool = pool_membership(ref / "pit", days).rename(columns={"ticker": "symbol"})
    pool["in_pool"] = True
    F = F.merge(pool, on=["date", "symbol"], how="left")
    F["in_pool"] = F["in_pool"].fillna(False).astype(bool)
    F["traded"] = F["open"].notna()
    C = F[F["in_pool"] & F["traded"] & F["or_open"].notna()].copy()
    C["eligible"] = C["eligible"].astype(bool)
    top50 = so.select(C, 50)
    top50["rank"] = top50.groupby("date").cumcount() + 1
    C = C.merge(top50[["date", "symbol", "rank"]], on=["date", "symbol"], how="left")
    C["rank"] = C["rank"].fillna(0).astype(int)
    C["base"] = C["eligible"] & (C["dir"] != 0)
    out = r / "derived"
    out.mkdir(parents=True, exist_ok=True)
    C.to_parquet(out / "candidates.parquet", index=False)
    P = C[C["base"]][["date", "symbol", "dir", "or_high", "or_low", "atr_prev", "rank", "base"]]
    P.to_parquet(out / "pass2_rows.parquet", index=False)
    # 逐日快照核对
    chk = {}
    for p in sorted((ref / "pit" / "daily_check").glob("*.parquet")):
        day = pd.Timestamp(p.stem).date()
        dly = set(pd.read_parquet(p)["ticker"].dropna())
        wk = set(pool.loc[pool["date"] == day, "symbol"])
        traded = set(d.loc[d["date"] == day, "ticker"])
        chk[str(day)] = {
            "daily_snapshot": len(dly),
            "weekly_snapshot": len(wk),
            "traded_in_daily_not_weekly": sorted((dly - wk) & traded),
            "traded_in_weekly_not_daily": sorted((wk - dly) & traded),
        }
    (out / "pool_check.json").write_text(
        json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {
        "days": len(days),
        "first_day": str(days[0]),
        "last_day": str(days[-1]),
        "pool_rows": int(C.shape[0]),
        "eligible_rows": int(C["eligible"].sum()),
        "base_rows": int(C["base"].sum()),
        "top50_rows": int((C["rank"] > 0).sum()),
        "pass2_rows": len(P),
        "rows_without_day_agg_dropped": n_no_day,
    }


def gap_sensitivity(r: Path, prefix: str = "candidates_prefix.parquet") -> dict:
    """断档阈值敏感性：用修正前的选股结果（derived/candidates_prefix.parquet），数「断档后前 14 行」里有多少个标的日
    曾被判合格 / 进入前 50 / 前 20，阈值取 10、20、30、60、120 个自然日。写 derived/gap_sensitivity.json。"""
    pit_tickers = set()
    for p in (r / "ref" / "pit").glob("*.parquet"):
        pit_tickers |= set(pd.read_parquet(p)["ticker"].dropna())
    d = load_daily(r, pit_tickers)
    d = d[d["d_open"].notna()].sort_values(["ticker", "date"])
    d["gap"] = (
        pd.to_datetime(d["date"]) - pd.to_datetime(d.groupby("ticker")["date"].shift(1))
    ).dt.days
    d["k"] = d.groupby("ticker").cumcount()
    C = pd.read_parquet(r / "derived" / prefix)
    C["date"] = pd.to_datetime(C["date"]).dt.date
    out = {}
    for G in (10, 20, 30, 60, 120):
        g = d[d["gap"] > G][["ticker", "k"]].rename(columns={"k": "k0"})
        m = d[["ticker", "date", "k"]].merge(g, on="ticker")
        m = m[(m["k"] >= m["k0"]) & (m["k"] < m["k0"] + LOOKBACK_ROWS)].drop_duplicates(
            ["ticker", "date"]
        )
        x = C.merge(
            m.rename(columns={"ticker": "symbol"})[["symbol", "date"]], on=["symbol", "date"]
        )
        t20 = x[(x["rank"] > 0) & (x["rank"] <= 20)]
        out[str(G)] = {
            "events": len(g),
            "tickers": int(g["ticker"].nunique()),
            "eligible_rows": int(x["eligible"].sum()),
            "top50_rows": int((x["rank"] > 0).sum()),
            "top20_rows": len(t20),
            "top20_symbols": sorted(set(t20["symbol"])),
        }
    (r / "derived" / "gap_sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="默认 $ZR_DATA_ROOT")
    ap.add_argument(
        "--gap-sensitivity",
        action="store_true",
        help="只算断档阈值敏感性（需要 derived/candidates_prefix.parquet）",
    )
    a = ap.parse_args(argv)
    if a.gap_sensitivity:
        print(
            json.dumps(gap_sensitivity(mroot(Path(a.root) if a.root else None)), ensure_ascii=False)
        )
        return 0
    print(json.dumps(build(mroot(Path(a.root) if a.root else None)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
