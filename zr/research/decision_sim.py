"""决策模拟（METHOD_V2 规格 7，R-024）：模型样本外概率 ≥ 阈值的事件按回放规则入场，看真实期权盈亏。

- 阈值 {0.55, 0.60, 0.65}；同一 (日, 决策分钟, 方向) 多个检测器只入场一次（取最高概率）。
- 盈亏来自 zr/research/outcomes.py（持仓上限 10 与 30 各一版）。
- 对照（同一批样本外月份）：无模型 = 全部事件去重入场（等权基线）；E1T 回放 = 只用 E1T 事件。
- 回答「模型过滤有没有附加价值」：阈值组 vs 无模型组的每笔均值、合计、最大回撤、笔数/日。
"""

from __future__ import annotations

import pandas as pd

from zr.research import outcomes as oc
from zr.research.replay_rule import Lookup, stats

THRESHOLDS = (0.55, 0.60, 0.65)


def _trades(res: pd.DataFrame, lk: Lookup, cap: int) -> pd.DataFrame:
    e = res.copy()
    e["minute"] = e["bar_minute"] + 1
    e = e[(e["minute"] >= oc.FIRST_ENTRY) & (e["minute"] <= oc.LAST_ENTRY)]
    e = e.sort_values("p", ascending=False).drop_duplicates(["day", "minute", "direction"])
    parts = []
    for d in (1, -1):
        x = e[e["direction"] == d]
        parts.append(x.assign(net=lk.get(x["day"], x["minute"], d, cap)))
    return pd.concat(parts, ignore_index=True).dropna(subset=["net"]).sort_values(["day", "minute"])


def simulate(res: pd.DataFrame, lk: Lookup) -> tuple[pd.DataFrame, dict]:
    oos = res[res["p"].notna()]
    rows, curves = [], {}
    for cap in oc.CAPS:
        base = _trades(oos.assign(p=0.0), lk, cap)
        rows.append(
            {"group": "无模型（全部事件去重）", "cap": cap, **stats(base["net"], base["day"])}
        )
        e1t = _trades(oos[oos["detector"] == "E1T"].assign(p=0.0), lk, cap)
        rows.append({"group": "E1T 回放（同月份）", "cap": cap, **stats(e1t["net"], e1t["day"])})
        for thr in THRESHOLDS:
            t = _trades(oos[oos["p"] >= thr], lk, cap)
            r = {"group": f"模型 p ≥ {thr:.2f}", "cap": cap, **stats(t["net"], t["day"])}
            if len(t) and len(base):
                r["delta_mean_vs_no_model"] = round(float(t["net"].mean() - base["net"].mean()), 2)
            rows.append(r)
            curves[(cap, thr)] = t
        curves[(cap, "none")] = base
    return pd.DataFrame(rows), curves


def plot_curves(curves: dict, path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, cap in zip(axes, oc.CAPS, strict=True):
        for key, t in curves.items():
            if key[0] != cap or t.empty:
                continue
            lab = "no model" if key[1] == "none" else f"p>={key[1]:.2f}"
            daily = t.groupby("day")["net"].sum().cumsum()
            ax.plot(
                pd.to_datetime(daily.index), daily.to_numpy(), label=f"{lab} (n={len(t)})", lw=1
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"cap {cap} min")
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def months_of(res: pd.DataFrame) -> set:
    r = res[res["p"].notna()]
    return set(pd.to_datetime(r["day"]).dt.to_period("M").astype(str).unique())
