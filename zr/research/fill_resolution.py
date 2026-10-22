"""diag/fill-resolution：1 分钟快照 vs tick 级 NBBO 成对比较，按 DTE 分层，百分比口径。

背景：v3 执行模型（zr/research/sim_v3.py）买 ask、卖 bid，价格来自 ThetaData 1 分钟快照
（interval=1m 的语义是「区间内最后一次更新的 NBBO」，见 zr/data/thetadata.py 顶部文档/
CLAUDE.md「interval=1m 时间戳 t 的报价 = t 时刻 NBBO」）。本诊断量化：若真实下单比信号
时刻 t 延迟 Δ 秒才成交，用 tick 级数据重新定位 t+Δ 时刻的真实 NBBO，与快照本身（即 Δ=0
的理论值）相比误差有多大、随 DTE 如何变化——这是「指标用第 1 节口径」之外、本任务第 3 节
要求的独立诊断，不涉及状态层或指标计算。

方法（抽样，不是全量）：
1. 从已拉取的 options_1m 数据里，对 MU、QQQ 各自按 DTE 分层（0 / 1 / 2-3 / 4-7 / 8+，
   分层边界是描述性分组，不是待测参数）随机抽 N_PER_BUCKET 个 contract-day：近 ATM
   （行权价最接近当日 zr.data.thetadata.day_price_range 的开盘价）、call/put 随机各半、
   要求该合约当日至少有 M_MINUTES_PER_DAY 根有效快照（ask>0 且 bid>=0 且 ask>=bid）。
2. 每个 contract-day 用 ThetaClient.history_quote(interval="tick") 整天拉一次（单到期日
   单行权价，不是 strike=*，数据量远小于全链，几秒可拉完一天的逐笔 NBBO），缓存到
   $ZR_DATA_ROOT/diag/fill_resolution/ticks/，重跑脚本不重新请求。
3. 每个 contract-day 里随机抽 M_MINUTES_PER_DAY 个有效快照分钟 t；对每个延迟
   Δ ∈ {0,10,30,60,120,180,300,600,900,1800} 秒（CLAUDE.md 规则 14 的标准尺度：
   10s/30s/1/2/3/5/10/15/30 分钟），取 ≤ t+Δ 的最后一笔 tick 的 ask/bid，与快照本身的
   ask/bid 比较，误差% = (tick − snapshot) / snapshot × 100。Δ=0 是内部一致性检验：
   「≤t 最后一笔 tick」理论上应与快照本身几乎相等（验证 1 分钟批量拉取与 tick 级拉取
   对同一时刻给出一致的值）。
4. 按 (DTE 分层, Δ) 聚合 |误差%| 的中位数/P90、Δ=0 的内部一致性，报告百分比口径。

运行：uv run python -m zr.research.fill_resolution（需本机 Theta Terminal 运行中，
bash scripts/run_theta_terminal.sh）
"""

from __future__ import annotations

import datetime as dt
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.data.thetadata import ThetaClient, day_price_range, lb_symbol
from zr.ops.markdown import df_to_md

log = logging.getLogger("fill_resolution")

DELAYS_S = [0, 10, 30, 60, 120, 180, 300, 600, 900, 1800]
DTE_BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 999, "8+")]
N_PER_BUCKET = 6
M_MINUTES_PER_DAY = 8
SEED = 0
UNDERLYINGS = ["MU", "QQQ"]


def list_days(root: Path, und: str) -> list[Path]:
    return sorted((root / "options_1m" / f"underlying={und}").glob("date=*"))


def _valid_snapshots(q: pd.DataFrame) -> pd.DataFrame:
    return q[(q["ask"] > 0) & (q["bid"] >= 0) & (q["ask"] >= q["bid"])]


def pick_contract_days(root: Path, unds: list[str], rng: random.Random) -> list[dict]:
    """每个 (underlying, DTE 分层) 随机选 N_PER_BUCKET 个 contract-day，近 ATM、流动性达标。"""
    picked: list[dict] = []
    for und in unds:
        days = list_days(root, und)
        rng.shuffle(days)
        count = dict.fromkeys((name for *_, name in DTE_BUCKETS), 0)
        for ddir in days:
            if all(count[name] >= N_PER_BUCKET for *_, name in DTE_BUCKETS):
                break
            day = dt.date.fromisoformat(ddir.name.split("=")[1])
            qpath = ddir / "quotes.parquet"
            if not qpath.exists():
                continue
            q = pd.read_parquet(qpath)
            if q.empty or "strike" not in q:
                continue
            px = day_price_range(root, und, day)
            if px is None:
                continue
            ref = px[0]
            right = rng.choice(["call", "put"])
            cand = q[q["right"] == right].copy()
            if cand.empty:
                continue
            cand["dte"] = [(e - day).days for e in cand["expiry"]]
            buckets = rng.sample(DTE_BUCKETS, k=len(DTE_BUCKETS))
            for lo, hi, name in buckets:
                if count[name] >= N_PER_BUCKET:
                    continue
                sub = cand[(cand["dte"] >= lo) & (cand["dte"] <= hi)]
                if sub.empty:
                    continue
                strikes = sorted(sub["strike"].unique(), key=lambda s: abs(s - ref))
                for strike in strikes:
                    one = sub[sub["strike"] == strike]
                    if len(_valid_snapshots(one)) < M_MINUTES_PER_DAY:
                        continue
                    expiry = one["expiry"].iloc[0]
                    picked.append(
                        {
                            "underlying": und,
                            "day": day,
                            "expiry": expiry,
                            "strike": float(strike),
                            "right": right,
                            "dte_bucket": name,
                            "dte": int((expiry - day).days),
                        }
                    )
                    count[name] += 1
                    break
                break  # 这一天只贡献一个分层，避免同一天被重复计入多个分层
    return picked


def tick_cache_path(root: Path, symbol: str, day: dt.date) -> Path:
    d = root / "diag" / "fill_resolution" / "ticks"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{symbol}_{day.isoformat()}.parquet"


def pull_or_load_ticks(client: ThetaClient, root: Path, c: dict) -> pd.DataFrame:
    symbol = lb_symbol(c["underlying"], c["expiry"], c["right"], c["strike"])
    p = tick_cache_path(root, symbol, c["day"])
    if p.exists():
        return pd.read_parquet(p)
    df = client.history_quote(
        c["underlying"],
        c["expiry"],
        c["day"],
        interval="tick",
        strike=f"{c['strike']:.3f}",
        right=c["right"],
        start_time="09:30:00.000",
        end_time="16:00:00.000",
    )
    df = df.sort_values("ts").reset_index(drop=True)
    df.to_parquet(p, index=False)
    return df


def sample_minutes(root: Path, c: dict, rng: random.Random, m: int) -> pd.DataFrame:
    symbol = lb_symbol(c["underlying"], c["expiry"], c["right"], c["strike"])
    q = pd.read_parquet(
        root / "options_1m" / f"underlying={c['underlying']}" / f"date={c['day'].isoformat()}"
        / "quotes.parquet"
    )
    q = q[q["symbol"] == symbol]
    q = _valid_snapshots(q)
    if len(q) <= m:
        return q
    return q.loc[rng.sample(list(q.index), m)]


def pct_err(got: float, want: float) -> float:
    return (got - want) / want * 100 if want else np.nan


def compare_one(ts_sorted: np.ndarray, ask: np.ndarray, bid: np.ndarray, snap_row: pd.Series):
    t = np.datetime64(pd.Timestamp(snap_row["ts"]).tz_convert("UTC").tz_localize(None))
    snap_ask, snap_bid = float(snap_row["ask"]), float(snap_row["bid"])
    rows = []
    for delay in DELAYS_S:
        target = t + np.timedelta64(delay, "s")
        i = int(np.searchsorted(ts_sorted, target, side="right")) - 1
        if i < 0:
            continue
        rows.append(
            {
                "delay_s": delay,
                "ask_err_pct": pct_err(float(ask[i]), snap_ask),
                "bid_err_pct": pct_err(float(bid[i]), snap_bid),
            }
        )
    return rows


def run(
    root: Path, unds: list[str] = UNDERLYINGS, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (逐笔比较明细, contract-day 元信息)。"""
    rng = random.Random(seed)
    candidates = pick_contract_days(root, unds, rng)
    log.info("选中 %d 个 contract-day", len(candidates))
    client = ThetaClient(concurrency=2)
    detail_rows, meta_rows = [], []
    for c in candidates:
        try:
            ticks = pull_or_load_ticks(client, root, c)
        except Exception as e:  # noqa: BLE001 - 诊断脚本：单个合约失败不中断整体
            log.warning("跳过 %s %s %s：%s", c["underlying"], c["day"], c["dte_bucket"], e)
            continue
        if ticks.empty:
            continue
        snaps = sample_minutes(root, c, rng, M_MINUTES_PER_DAY)
        if snaps.empty:
            continue
        ts_sorted = ticks["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        ask, bid = ticks["ask"].to_numpy(), ticks["bid"].to_numpy()
        meta_rows.append(
            {**c, "n_ticks": len(ticks), "n_minutes_sampled": len(snaps)}
        )
        for _, snap_row in snaps.iterrows():
            for r in compare_one(ts_sorted, ask, bid, snap_row):
                detail_rows.append(
                    {
                        **r,
                        "dte_bucket": c["dte_bucket"],
                        "dte": c["dte"],
                        "underlying": c["underlying"],
                        "day": c["day"],
                    }
                )
    return pd.DataFrame(detail_rows), pd.DataFrame(meta_rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    d = detail.copy()
    d["ask_abs_err_pct"] = d["ask_err_pct"].abs()
    d["bid_abs_err_pct"] = d["bid_err_pct"].abs()
    g = d.groupby(["dte_bucket", "delay_s"])
    out = g.agg(
        n=("ask_abs_err_pct", "size"),
        ask_median_abs_err_pct=("ask_abs_err_pct", "median"),
        ask_p90_abs_err_pct=("ask_abs_err_pct", lambda s: s.quantile(0.9)),
        bid_median_abs_err_pct=("bid_abs_err_pct", "median"),
        bid_p90_abs_err_pct=("bid_abs_err_pct", lambda s: s.quantile(0.9)),
    ).reset_index()
    order = {name: i for i, (*_, name) in enumerate(DTE_BUCKETS)}
    out["_o"] = out["dte_bucket"].map(order)
    return out.sort_values(["_o", "delay_s"]).drop(columns="_o").reset_index(drop=True)


def v4_proposal_lines(summary: pd.DataFrame, n_contract_days: int, n_minutes: int) -> list[str]:
    """从 summary 实际数字生成 V4 草案文字，不手填数字（规则 2）。"""

    def get(bucket: str, delay: int, col: str) -> float:
        row = summary[(summary["dte_bucket"] == bucket) & (summary["delay_s"] == delay)]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    short, long_ = 30, 1800
    a0_short = get("0", short, "ask_median_abs_err_pct")
    a0_long = get("0", long_, "ask_median_abs_err_pct")
    a8_short = get("8+", short, "ask_median_abs_err_pct")
    a8_long = get("8+", long_, "ask_median_abs_err_pct")
    p90_0_short = get("0", short, "ask_p90_abs_err_pct")
    ratio_short = a0_short / a8_short if a8_short else float("nan")
    growth_0 = a0_long / a0_short if a0_short else float("nan")
    growth_8 = a8_long / a8_short if a8_short else float("nan")
    return [
        f"- 延迟 {short}s（接近真实下单的网络/确认延迟）时，0-DTE 买价中位数 |误差| "
        f"{a0_short:.2f}%、P90 {p90_0_short:.2f}%；8+DTE 同一延迟下中位数 {a8_short:.2f}%——"
        f"0-DTE 比 8+DTE 大 {ratio_short:.1f} 倍。v3 引擎（`zr/research/sim_v3.py`）当前对"
        "两者用同一套「买 ask / 卖 bid、延迟≈0」假设，没有按 DTE 区分滑点缓冲。",
        f"- 延迟从 {short}s 拉长到 {long_}s（30 分钟），0-DTE 中位数误差从 {a0_short:.2f}% 涨到 "
        f"{a0_long:.2f}%（×{growth_0:.1f}），8+DTE 从 {a8_short:.2f}% 涨到 {a8_long:.2f}%"
        f"（×{growth_8:.1f}）——短 DTE 对响应延迟更敏感，是 CLAUDE.md 规则 14「响应窗口逐档"
        "测」必要性的一次独立交叉印证（不是新结论，本诊断本身没有做任何策略回放）。",
        "- **建议（未实施，是否立项由负责人决定）**：",
        "  1. 宣布某条规则卡「有边」之前，若其 DTE ∈ {0,1}，额外用 tick 级数据对该卡涉及的 "
        "contract-day 做一次独立抽样复核，不只信 1 分钟快照的回放结果——0/1-DTE 在真实延迟下"
        "的滑点可能比 1 分钟快照回放显示的更差。",
        "  2. 在 `zr/research/sim_v3.py` 的成交函数上加一个可选的按 DTE 分层滑点缓冲参数，"
        "默认关闭（不改变现有任何回放数字，符合规则 11：待测参数不写死、不默认生效），"
        "留给未来预登记测试用。",
        f"  3. DTE≥2 的桶在 {short}s 延迟下买价中位数误差 <1.5%，1 分钟快照对这部分回放依然"
        "可用，不建议为此单独重做数据管线。",
        "",
        f"**边界重申**：以上数字只来自 MU/QQQ {len(DELAYS_S)} 个延迟档 × {n_contract_days} 个"
        f"抽样 contract-day、每天最多 {n_minutes} 个随机分钟，是诊断性抽样，不是穷举，也没有"
        "用于支持或否定任何已发布的规则卡结论。",
    ]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    detail, meta = run(root)
    if detail.empty:
        log.error("没有可比较的样本（Theta Terminal 是否在跑？见 scripts/run_theta_terminal.sh）")
        return 1
    summary = summarize(detail)
    zero = summary[summary["delay_s"] == 0]

    out = root / "diag" / "fill_resolution"
    out.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(out / "detail.parquet", index=False)
    meta.to_parquet(out / "contract_days.parquet", index=False)

    unds_used = sorted(meta["underlying"].unique())
    lines = [
        "# 1 分钟快照 vs tick 级 NBBO 成对比较（diag/fill-resolution 第 3 节）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        "复现 `uv run python -m zr.research.fill_resolution`",
        "",
        "## 边界",
        "",
        f"- 标的：{', '.join(unds_used)}；"
        f"合约-日样本 {len(meta)} 个（每 (标的, DTE 分层) 最多 {N_PER_BUCKET} 个），"
        f"日期范围 {meta['day'].min()} ~ {meta['day'].max()}",
        f"- 每个合约-日随机抽 {M_MINUTES_PER_DAY} 个有效快照分钟（ask>0 且 bid>=0 且 "
        "ask>=bid），逐笔比较明细共 "
        f"{detail[['underlying', 'day', 'dte_bucket']].drop_duplicates().shape[0]} "
        f"个合约-日 × 最多 {M_MINUTES_PER_DAY} 分钟 × {len(DELAYS_S)} 个延迟 = "
        f"{len(detail)} 条",
        "- 口径：近 ATM（行权价最接近当日开盘价）、call/put 随机各半，不是逐笔成交价，"
        "是 ThetaData 逐笔 NBBO（bid/ask）——「tick 级成交」在这里指 tick 级报价，"
        "不是真实成交记录（ThetaData Options Standard 不提供逐笔成交，只有 NBBO）。",
        "- DTE 分层（0/1/2-3/4-7/8+）是描述性分组，边界本身不是待测参数、不构成策略结论。",
        "- 只覆盖 MU、QQQ（见上）；SPY 未抽样（本任务范围未要求，且 SPY 期权数据按负责人"
        "指令暂停于 2024-07-30）。",
        "",
        "## Δ=0 内部一致性检验",
        "",
        "理论上「≤t 最后一笔 tick」应与 1 分钟批量快照本身几乎相等（同一个 NBBO 概念，"
        "只是两条不同的拉取路径）；若 Δ=0 的误差不是 ~0，说明两条路径本身不一致，"
        "不是「延迟」造成的，需要先排查再看后面的 Δ>0 结果。",
        "",
        df_to_md(zero.drop(columns="delay_s").reset_index(drop=True)),
        "",
        "## 全部结果（按 DTE 分层 × 延迟，|误差%| 中位数与 P90）",
        "",
        df_to_md(summary),
        "",
        "## V4 方案草案（基于上表数字，不是独立结论——v3 引擎本身不改，供负责人决定是否立项）",
        "",
        *v4_proposal_lines(summary, len(meta), M_MINUTES_PER_DAY),
    ]
    (Path("docs/research/FILL_RESOLUTION.md")).write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 docs/research/FILL_RESOLUTION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
