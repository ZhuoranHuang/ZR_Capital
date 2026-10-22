"""feat/v4-decisive 第 0 节：指标口径与预热处理复核 + 三个状态校验点。

输出 docs/research/V4_CHECKPOINTS.md。

两件事：
1. 截图能否区分「RTH-only、按日重置」（D-027）与「含盘前盘后连续」：按 App 显示精度（3 位小数）
   逐值比较是否完全相等。
2. 每个交易日前 20 根 2 分钟 K（09:30–10:08 标签）BOLL 样本不足时的三种处理，各自给出负责人
   三个校验点（MU 2026-10-21：09:30 后急跌 IGNITION 下行、10:00–12:30 RANGE、12:18–13:36
   CHANNEL 上行）与 13:00 回踩点的结果：
   A RTH-only、按日重置、不足 20 根为空值（D-027 现实现）
   B RTH-only、按日重置、部分窗口（不足 20 根时按已有根数算）
   C 含盘前盘后连续、不按日重置（与长桥原生 2 分钟 K 同口径；09:30 的窗口含 08:52–09:28 盘前 K）
   RANGE 阈值对每种口径用同一规则重标：MU 2024 年 RTH bar 的 range_score 30% 分位。

运行：uv run python -m zr.research.v4_checkpoints
"""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.bars import ET, load_bars, load_extended_bars, resample_bars
from zr.features.regime import STATE_NAMES, indicators, load_config, regimes
from zr.ops.markdown import df_to_md
from zr.research import patterns_v3 as pt
from zr.research.indicator_parity import APP, boll, ema_first_seed, macd, nearest_close
from zr.research.method_v3_round1 import CHECK_DAY, CHECKS

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "V4_CHECKPOINTS.md"
CALIB_YEAR, CALIB_Q = 2024, 0.30
PULLBACK_LOW, PULLBACK_MID = 913.37, 912.69

CONVENTIONS = {
    "A": {"label": "RTH-only、按日重置、不足 20 根为空值（D-027 现实现）", "ext": False,
          "reset": True, "warmup": "nan"},
    "B": {"label": "RTH-only、按日重置、部分窗口", "ext": False, "reset": True,
          "warmup": "partial"},
    "C": {"label": "含盘前盘后连续、不按日重置（长桥原始 K，不清洗）", "ext": True, "reset": False,
          "warmup": "nan", "clean": False},
    "C*": {"label": "同 C，但用 D-026 的坏 bar 清洗（会误删 RTH 大振幅 bar，对照用）", "ext": True,
           "reset": False, "warmup": "nan", "clean": True},
}


def bars(root: Path, und: str, start: dt.date, end: dt.date, conv: dict) -> pd.DataFrame:
    if conv["ext"]:
        s = load_extended_bars(root, und, start, end, clean=conv.get("clean", True))
    else:
        s = load_bars(root, und, start, end)
    return resample_bars(s, "2min")


def rth_mask(index: pd.DatetimeIndex) -> np.ndarray:
    hm = np.asarray(index.tz_convert(ET).strftime("%H:%M"))
    return (hm >= "09:30") & (hm < "16:00")


def state_layer(b: pd.DataFrame, cfg: dict, conv: dict, thr: float | None = None):
    c = copy.deepcopy(cfg)
    if thr is not None:
        c["regime"]["range"]["score_threshold"] = thr
    x = indicators(b, c, daily_reset=conv["reset"], warmup=conv["warmup"])
    return x, regimes(b, x, c, 2), c


def calibrate(root: Path, cfg: dict, conv: dict) -> float:
    b = bars(root, "MU", dt.date(CALIB_YEAR, 1, 1), dt.date(CALIB_YEAR, 12, 31), conv)
    _, reg, _ = state_layer(b, cfg, conv)
    s = reg["range_score"][rth_mask(b.index)].dropna()
    return round(float(s.quantile(CALIB_Q)), 4)


def parity_table(root: Path) -> pd.DataFrame:
    """各口径在 3 个截图时刻的 15 个值，按 App 显示精度（3 位小数）是否完全相等。"""
    day = CHECK_DAY
    variants = {
        "RTH-only、按日重置（A/B 在截图时刻相同）": resample_bars(load_bars(root, "MU", day, day),
                                                    "2min"),
        "含盘前盘后连续（本地 1 分钟聚合，修正分箱、不清洗）": resample_bars(
            load_extended_bars(root, "MU", day - dt.timedelta(days=50), day, clean=False),
            "2min"),
        "RTH-only 跨日连续（不重置）": resample_bars(
            load_bars(root, "MU", day - dt.timedelta(days=50), day), "2min"),
    }
    direct = root / "diag" / "indicator_parity" / "min2_noadjust.parquet"
    if direct.exists():
        d2 = pd.read_parquet(direct).set_index("ts").sort_index()
        variants["含盘前盘后连续（长桥原生 2 分钟 K，10-19 起）"] = d2[d2.index.tz_convert(ET).date
                                                               <= day]
    rows = []
    for name, b in variants.items():
        c = b["close"]
        B, M = boll(c, 20, 2.0, 0), macd(c, 6, 13, 6, ema_first_seed)
        exact, worst = 0, 0.0
        for hm, w in APP.items():
            ts = nearest_close(c.index, hm, label_offset_min=2)
            for k, src in (("mid", B), ("upper", B), ("lower", B), ("dif", M), ("dea", M)):
                v = float(src.loc[ts, k])
                exact += round(v, 3) == w[k]
                worst = max(worst, abs(v - w[k]))
        rows.append({"口径": name, "3 位小数完全相等": f"{exact}/15",
                     "最大绝对差": round(worst, 5)})
    return pd.DataFrame(rows)


def checkpoint_rows(root: Path, cfg: dict, key: str, conv: dict, thr: float):
    b = bars(root, "MU", CHECK_DAY - dt.timedelta(days=20), CHECK_DAY, conv)
    x, reg, c = state_layer(b, cfg, conv, thr)
    et = b.index.tz_convert(ET)
    on = np.asarray(et.date) == CHECK_DAY
    hhmm = np.asarray(et.strftime("%H:%M"))
    rows = []
    for a, z, st, d, note in CHECKS:
        m = on & (hhmm >= a) & (hhmm <= z)
        names = [STATE_NAMES[v] for v in reg["state"].to_numpy()[m]]
        dirs = reg["dir"].to_numpy()[m]
        nan_bars = int(np.isnan(x["pctb"].to_numpy()[m]).sum())
        hit = np.mean([(n == st and (d == 0 or di == d)) for n, di in zip(names, dirs,
                                                                          strict=True)])
        dist = pd.Series([f"{n}{'' if di == 0 else ('↑' if di > 0 else '↓')}"
                          for n, di in zip(names, dirs, strict=True)]).value_counts().to_dict()
        rows.append({"口径": key, "核对点": note, "K 线数": int(m.sum()),
                     "BOLL 为空的 K": nan_bars, "符合比例": round(float(hit), 2),
                     "实际分布": ", ".join(f"{k} {v}" for k, v in dist.items())})
    # 13:00 回踩点
    lows = b["low"].to_numpy()
    k = np.flatnonzero(on & (np.abs(lows - PULLBACK_LOW) < 0.006))
    pb = {"口径": key, "回踩 K": None}
    if len(k):
        i = int(k[0])
        fired = [pt.entry_name(e) for e in pt.entry_configs(c) if e["entry"] == "E1"
                 and pt.entry_signal(b, x, reg, e, c)[i] == 1]
        pb = {"口径": key, "回踩 K（标签）": hhmm[i], "最低": round(float(lows[i]), 2),
              "中轨（我们）": round(float(x["mid"].iat[i]), 2), "中轨（App）": PULLBACK_MID,
              "状态": f"{STATE_NAMES[int(reg['state'].iat[i])]}{int(reg['dir'].iat[i]):+d}",
              "E1 识别定义数": f"{len(fired)}/36"}
    first_valid = et[on & ~np.isnan(x["mid"].to_numpy()) & rth_mask(b.index)]
    fv = first_valid[0].strftime("%H:%M") if len(first_valid) else "无"
    return rows, pb, fv


def main() -> int:
    root = get_settings().zr_data_root
    cfg = load_config()
    thr = {k: calibrate(root, cfg, v) for k, v in CONVENTIONS.items()}
    cp, pbs, fvs = [], [], {}
    for k, v in CONVENTIONS.items():
        r, pb, fv = checkpoint_rows(root, cfg, k, v, thr[k])
        cp += r
        pbs.append(pb)
        fvs[k] = fv
    conv_tab = pd.DataFrame([{"口径": k, "说明": v["label"], "RANGE 阈值（同规则重标）": thr[k],
                              "10-21 首个有效 BOLL 的 bar 标签": fvs[k]}
                             for k, v in CONVENTIONS.items()])
    L = [
        "# V4 第 0 节：指标口径复核与三个状态校验点",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        "复现 `uv run python -m zr.research.v4_checkpoints`",
        "",
        "## 1. 截图区分不了「RTH-only 按日重置」与「含盘前盘后连续」",
        "",
        "按 App 显示精度（3 位小数）比较负责人 3 个截图时刻的 15 个值（BOLL mid/upper/lower、"
        "DIF、DEA；时刻 +2 分钟换算为我们的 bar 标签）：",
        "",
        df_to_md(parity_table(root)),
        "",
        "- D-027 写的「含盘前盘后反而更不匹配（误差 44%）」**是本地分箱 bug，不是 App 行为**："
        "原始 1 分钟数据每天都有 04:00 那根，但 D-026 的坏 bar 清洗常把它删掉、当日首根变成 04:01"
        "（MU 767 天里 286 天、QQQ 276 天；不清洗 0 天），`resample_bars` 原来用 "
        "`origin=\"start\"`（锚在首根），于是全天 2 分钟分箱锚在奇数分钟（11:27/11:29 而不是 "
        "11:28）。本分支已改为 `origin=\"start_day\"`；RTH-only 数据每天首根都是 09:30"
        "（767/767），结果不变。D-026 时期含盘前盘后的状态层在这 37% 的天里分箱错位。",
        "- 截图唯一能排除的是「RTH-only 跨日连续」。「是否含盘前盘后」「是否按日重置」都没有被截图"
        "确定。D-027 的这两条应降级为「与截图一致的候选之一」。",
        "- 三个截图时刻都在 11:26 以后，**无法回答开盘前 20 根 K 的处理**（下一节）。",
        "",
        "## 2. 前 20 根 2 分钟 K 的处理与三个校验点",
        "",
        df_to_md(conv_tab),
        "",
        df_to_md(pd.DataFrame(cp)),
        "",
        "13:00 回踩点：",
        "",
        df_to_md(pd.DataFrame(pbs)),
        "",
        "C* 与 C 的差别只在 D-026 的坏 bar 清洗：清洗用跨 session 连续的 ATR，安静盘前之后的开盘"
        "急跌 bar（09:32）和 12:39 那根大振幅 bar 被当成坏 bar 删除，13:00 中轨因此偏 0.24。App "
        "的 912.69 与不清洗的数据一致，所以 C 不清洗。",
        "",
        "## 3. 结论（D-028）",
        "",
        "- **前 20 根 2 分钟 K 的处理**：与 App 一致的方式是口径 C——窗口向前延伸到盘前（09:30 的 "
        "BOLL 用 08:52–09:28 的盘前 K），开盘第一根就有完整 20 根窗口，**既不是部分窗口，也不是"
        "空值**。依据：负责人 3 个截图值 A、C 都完全相等（区分不了）；负责人在 App 上读到的"
        "「09:30 后急跌 IGNITION 下行」只有 C 能复现（A 空值时 8 根全为空，B 部分窗口 0 根命中）；"
        "13:00 回踩 C 与 App 一致；App 图表本身显示盘前 K。",
        "- **未直接核实**：开盘前 40 分钟 App 的 BOLL/MACD 数值没有截图。需要负责人任一交易日 "
        "09:32–10:08 ET 之间一根 2 分钟 K 的 BOLL mid/upper/lower 与 DIF/DEA 截图，可一次确定"
        "（A 空值、B 部分窗口、C 延伸到盘前三者在这段时间数值完全不同）。已列入 STATUS 待确认。",
        "- RANGE、CHANNEL 两个校验点三种口径下都远低于 1（RANGE 最高 0.34、CHANNEL 0.30），"
        "说明问题在状态定义（打分阈值、通道 N/M），不在指标数值；本轮按指令不改状态定义。",
        "- 边界：只有 MU 一天（2026-10-21）的 3 个截图时刻 + 负责人目测的 3 个时段；QQQ 未核对。",
        "",
    ]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
