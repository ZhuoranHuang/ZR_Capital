"""诊断（不做新研究）：负责人 D-030 MU 开仓买入委托提交前的秒级正股 / 期权微观特征
vs 同日同时段随机时刻 → docs/research/PRE_ENTRY_MICRO.md。

正股价 = 负责人所买合约同到期日、同行权价看涨 / 看跌的 ThetaData 逐笔 NBBO 中间价按平价
反推（S = C − P + K·e^(−rT)，r = 0.04；两条腿任一超过 10 秒未更新则该时刻不用；精度见
ENTRY_TIMING_DIAG.md）。时刻 t 的特征（方向化：看涨 +，看跌 −）：
- 前 10 / 30 / 60 / 120 秒正股方向化涨跌（bp）；同一窗口内最大单笔跳动（相邻两次反推价之差的
  绝对值，bp）与最大顺向 / 逆向单笔跳动；
- 距当日（09:30 起，长桥 1 分钟 K 高低点 + 当前分钟内的反推价）或近 5 / 15 / 30 分钟（反推价）
  高点 / 低点的距离（bp），以及是否「处在新高 / 新低」（距极值 ≤ 2 bp，约为反推价的噪声），
  分「顺向」（看涨看高点、看跌看低点）与「逆向」；
- 距最近 1 / 5 / 10 美元整数价位的距离（美元）；
- 所买期权中间价前 10 / 30 / 60 / 120 秒的变化（%）；报价更新次数（前 10 / 30 / 60 秒）及与
  前 60–120 秒每秒更新率之比。
对照：每张委托在同一交易日 [提交 − 30 分钟, 提交 + 30 分钟]（09:31–15:59，剔除提交前后
2 分钟）均匀抽 1,000 个时刻（固定种子）。报告负责人与随机的分布、中位数差，以及每张委托在其
自身 1,000 个随机时刻中的百分位（无差异时应在 0–1 上均匀，均值 0.5）。只列分布与差异。
运行：uv run python -m zr.research.pre_entry_micro
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.data.thetadata import lb_symbol
from zr.features.bars import ET, load_bars
from zr.ops.markdown import df_to_md
from zr.research import entry_timing_diag as etd
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.pre_entry_micro")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "PRE_ENTRY_MICRO.md"
WINS = (10, 30, 60, 120)
HL_MIN = (5, 15, 30)
N_RAND = 1000
NS = 1_000_000_000
MAX_AGE = 10 * NS
TOL_BP = 2.0  # 「处在新高 / 新低」= 距该窗口极值 ≤ 2 bp（平价反推价本身的噪声约 2 bp）


def implied_series(root: Path, o) -> tuple[np.ndarray, np.ndarray] | None:
    """两条腿报价合并时间轴上的反推价（两腿都在 10 秒内更新过才有值）。"""
    day = pd.Timestamp(int(o["submit_raw_ns"]), tz="UTC").tz_convert(ET).date()
    legs = []
    for right in ("call", "put"):
        tk = v4d.load_ticks(root, "MU", lb_symbol("MU", o["expiry"], right, o["strike"]), day)
        if tk is None or not len(tk.ts):
            return None
        legs.append((tk.ts, tk.mid))
    t = np.unique(np.r_[legs[0][0], legs[1][0]])
    vals = []
    for ts, mid in legs:
        i = np.searchsorted(ts, t, side="right") - 1
        ok = (i >= 0) & ((t - ts[np.clip(i, 0, None)]) <= MAX_AGE)
        vals.append(np.where(ok, mid[np.clip(i, 0, None)], np.nan))
    exp_close = pd.Timestamp(f"{o['expiry']} 16:00", tz=ET).value
    tt = np.maximum((exp_close - t) / (365 * 24 * 3600e9), 0.0)
    s = vals[0] - vals[1] + float(o["strike"]) * np.exp(-etd.R * tt)
    m = np.isfinite(s)
    return t[m], s[m]


def own_quotes(root: Path, o) -> tuple[np.ndarray, np.ndarray] | None:
    day = pd.Timestamp(int(o["submit_raw_ns"]), tz="UTC").tz_convert(ET).date()
    tk = v4d.load_ticks(root, "MU", o["symbol"], day)
    return None if tk is None or not len(tk.ts) else (tk.ts, tk.mid)


def _asof(ts, v, t):
    i = np.searchsorted(ts, t, side="right") - 1
    return np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)


def features(ts, s, q_ts, q_mid, day_hl, d: int, t: np.ndarray) -> pd.DataFrame:
    """t：一组时刻（ns）。day_hl：(1 分钟 K 收盘时刻, 截至该分钟的当日最高, 最低)。"""
    out = {}
    s_t = _asof(ts, s, t)
    for w in WINS:
        s0 = _asof(ts, s, t - w * NS)
        out[f"ret_{w}s_bp"] = d * (s_t / s0 - 1) * 1e4
        lo_i = np.searchsorted(ts, t - w * NS, side="left")
        hi_i = np.searchsorted(ts, t, side="right")
        jmp, jup, jdn = [], [], []
        diff = np.diff(s) / s[:-1] * 1e4 * d
        for a, b in zip(lo_i, hi_i, strict=True):
            seg = diff[max(a - 1, 0) : max(b - 1, 0)]
            jmp.append(np.abs(seg).max() if len(seg) else np.nan)
            jup.append(seg.max() if len(seg) else np.nan)
            jdn.append(-seg.min() if len(seg) else np.nan)
        out[f"jump_{w}s_bp"] = np.array(jmp)
        out[f"jump_with_{w}s_bp"] = np.array(jup)
        out[f"jump_against_{w}s_bp"] = np.array(jdn)
    for n in HL_MIN:
        lo_i = np.searchsorted(ts, t - n * 60 * NS, side="left")
        hi_i = np.searchsorted(ts, t, side="right")
        hi_v = np.array(
            [s[a:b].max() if b > a else np.nan for a, b in zip(lo_i, hi_i, strict=True)]
        )
        lo_v = np.array(
            [s[a:b].min() if b > a else np.nan for a, b in zip(lo_i, hi_i, strict=True)]
        )
        to_hi, to_lo = (hi_v / s_t - 1) * 1e4, (1 - lo_v / s_t) * 1e4  # 距高 / 低点 bp（≥ 0）
        with_d, against_d = (to_hi, to_lo) if d == 1 else (to_lo, to_hi)
        out[f"dist_ext_with_{n}m_bp"] = with_d
        out[f"dist_ext_against_{n}m_bp"] = against_d
        out[f"near_ext_with_{n}m"] = (with_d <= TOL_BP).astype(float)
        out[f"near_ext_against_{n}m"] = (against_d <= TOL_BP).astype(float)
    bc, bh, bl = day_hl
    j = np.searchsorted(bc, t, side="right") - 1
    dh = np.where(j >= 0, bh[np.clip(j, 0, None)], -np.inf)
    dl = np.where(j >= 0, bl[np.clip(j, 0, None)], np.inf)
    # 当前分钟内（最后一根完整 1 分钟 K 之后）的反推价极值
    m0 = np.where(j >= 0, bc[np.clip(j, 0, None)], 0)
    cur_hi = np.array(
        [
            s[np.searchsorted(ts, a) : np.searchsorted(ts, b, "right")].max()
            if np.searchsorted(ts, b, "right") > np.searchsorted(ts, a)
            else -np.inf
            for a, b in zip(m0, t, strict=True)
        ]
    )
    cur_lo = np.array(
        [
            s[np.searchsorted(ts, a) : np.searchsorted(ts, b, "right")].min()
            if np.searchsorted(ts, b, "right") > np.searchsorted(ts, a)
            else np.inf
            for a, b in zip(m0, t, strict=True)
        ]
    )
    day_hi, day_lo = np.maximum(dh, cur_hi), np.minimum(dl, cur_lo)
    to_hi, to_lo = (day_hi / s_t - 1) * 1e4, (1 - day_lo / s_t) * 1e4
    with_d, against_d = (to_hi, to_lo) if d == 1 else (to_lo, to_hi)
    out["dist_ext_with_day_bp"] = with_d
    out["dist_ext_against_day_bp"] = against_d
    out["near_ext_with_day"] = (with_d <= TOL_BP).astype(float)
    out["near_ext_against_day"] = (against_d <= TOL_BP).astype(float)
    for r in (1, 5, 10):
        out[f"dist_round{r}"] = np.abs(s_t - np.round(s_t / r) * r)
    for w in WINS:
        m1, m0q = _asof(q_ts, q_mid, t), _asof(q_ts, q_mid, t - w * NS)
        out[f"opt_mid_chg_{w}s_pct"] = (m1 / m0q - 1) * 100
    for w in (10, 30, 60):
        out[f"opt_updates_{w}s"] = (
            np.searchsorted(q_ts, t, "right") - np.searchsorted(q_ts, t - w * NS, "left")
        ).astype(float)
    base = (
        np.searchsorted(q_ts, t - 60 * NS, "right") - np.searchsorted(q_ts, t - 120 * NS, "left")
    ) / 60
    out["opt_update_rate_ratio_10s"] = np.where(
        base > 0, out["opt_updates_10s"] / 10 / base, np.nan
    )
    return pd.DataFrame(out)


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = etd.buy_orders(root)
    b = b[b["und"] == "MU"].reset_index(drop=True)
    w = etd.pair_windows(b)
    w["start"] = w["start"] - pd.Timedelta(minutes=30)
    w["end"] = w["end"] + pd.Timedelta(minutes=25)
    log.info(
        "合约对 tick（扩到 −60 / +30 分钟）：%s", v4d.pull_ticks(root, "MU", w.drop(columns="und"))
    )
    days = sorted({pd.Timestamp(n, tz="UTC").tz_convert(ET).date() for n in b["submit_raw_ns"]})
    one = load_bars(root, "MU", days[0], days[-1])  # RTH 1 分钟 K
    one_close = v4d.ns(one.index) + 60 * NS
    one_day = np.asarray(one.index.tz_convert(ET).date)
    rng = np.random.default_rng(0)
    own_rows, rnd_rows = [], []
    for k, o in b.iterrows():
        sub = int(o["submit_raw_ns"])
        day = pd.Timestamp(sub, tz="UTC").tz_convert(ET).date()
        ser, q = implied_series(root, o), own_quotes(root, o)
        if ser is None or q is None:
            continue
        m = one_day == day
        hl = (
            one_close[m],
            np.maximum.accumulate(one["high"].to_numpy()[m]),
            np.minimum.accumulate(one["low"].to_numpy()[m]),
        )
        d = int(o["dir"])
        f0 = features(ser[0], ser[1], q[0], q[1], hl, d, np.array([sub]))
        own_rows.append(
            f0.assign(
                order=k,
                submit_et=pd.Timestamp(sub, tz="UTC").tz_convert(ET).strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        lo = max(sub - 30 * 60 * NS, pd.Timestamp(f"{day} 09:31", tz=ET).value)
        hi = min(sub + 30 * 60 * NS, pd.Timestamp(f"{day} 15:59", tz=ET).value)
        cand = rng.integers(lo, hi, size=N_RAND * 3)
        cand = cand[np.abs(cand - sub) > 120 * NS][:N_RAND]
        fr = features(ser[0], ser[1], q[0], q[1], hl, d, np.sort(cand))
        rnd_rows.append(fr.assign(order=k))
    return pd.concat(own_rows, ignore_index=True), pd.concat(rnd_rows, ignore_index=True)


def compare(own: pd.DataFrame, rnd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in [x for x in own.columns if x not in ("order", "submit_et")]:
        a, r = own[c].astype(float), rnd[c].astype(float)
        pct = []
        for k, v in zip(own["order"], a, strict=True):
            g = rnd.loc[rnd["order"] == k, c].astype(float).dropna()
            if np.isfinite(v) and len(g):
                pct.append(((g < v).sum() + 0.5 * (g == v).sum()) / len(g))
        pct = np.array(pct)
        binary = set(np.unique(r.dropna())) <= {0.0, 1.0}
        row = {"特征": c}
        if binary:
            row.update(
                {
                    "负责人": f"{a.mean():.0%}（n={a.notna().sum()}）",
                    "随机": f"{r.mean():.0%}",
                    "差": f"{(a.mean() - r.mean()) * 100:+.0f} 个百分点",
                }
            )
        else:
            qa, qr = a.quantile([0.25, 0.5, 0.75]), r.quantile([0.25, 0.5, 0.75])
            row.update(
                {
                    "负责人": f"{qa[0.5]:.3g}（{qa[0.25]:.3g}～{qa[0.75]:.3g}）",
                    "随机": f"{qr[0.5]:.3g}（{qr[0.25]:.3g}～{qr[0.75]:.3g}）",
                    "差": f"{qa[0.5] - qr[0.5]:+.3g}",
                }
            )
        if len(pct):
            se = pct.std(ddof=1) / math.sqrt(len(pct)) if len(pct) > 1 else math.nan
            row.update(
                {
                    "自身随机中的百分位均值": round(float(pct.mean()), 3),
                    "≥90 分位": f"{(pct >= 0.9).mean():.0%}",
                    "≤10 分位": f"{(pct <= 0.1).mean():.0%}",
                    "t（均值 vs 0.5）": round(float((pct.mean() - 0.5) / se), 2)
                    if se and np.isfinite(se) and se > 0
                    else math.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    own, rnd = run(root)
    store = root / "research" / "pre_entry_micro"
    store.mkdir(parents=True, exist_ok=True)
    own.to_parquet(store / "owner.parquet", index=False)
    rnd.to_parquet(store / "random.parquet", index=False)
    cmp = compare(own, rnd)
    lines = [
        "# 负责人下单前的秒级微观特征 vs 随机时刻（诊断）",
        "",
        "`uv run python -m zr.research.pre_entry_micro`。诊断，不做新研究、不入任何检验族。"
        f"样本：D-030 入选段中 MU 的开仓买入委托，有完整合约对报价的 {own['order'].nunique()} 张；"
        f"随机对照每张 {N_RAND} 个时刻（同日、提交前后 30 分钟内、剔除提交前后 2 分钟）。"
        "方向化：看涨按正股上涨为正，看跌按下跌为正。百分位：负责人该时刻的值在其自身 1,000 个随"
        "机时刻中的位置。",
        "",
        df_to_md(cmp),
        "",
        "读法：百分位均值偏离 0.5、或 ≥90 / ≤10 分位的比例明显偏离 10%，表示负责人下单时刻在该特"
        "征上与同日"
        f"同时段随机时刻不同。共 {len(cmp)} 个特征同时比较，t 值未做多重比较校正，只作描述；"
        "「处在新高 / 新低」这类 0 / 1 特征几乎全为 0 时百分位方差极小，t 值不可靠。",
        "",
        "## 逐张委托",
        "",
        df_to_md(own.round(3)),
        "",
        "## 边界",
        "",
        "- 2026-06-09～2026-08-19 负责人 D-030 样本中的 MU 委托（日内单腿）；提交时刻 ≠ 决定时刻。",
        "- 正股价为平价反推（与长桥 1 分钟收盘的绝对误差中位数约 2 bp，偏差约 +1.3 bp）；反推价"
        "的更新频率取决于"
        "两条腿的报价，静默期内不更新。",
        "- 当日高低点 = 长桥 RTH 1 分钟 K（09:30 起）+ 当前分钟内的反推价；近 N 分钟高低点只用反"
        "推价。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(cmp.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
