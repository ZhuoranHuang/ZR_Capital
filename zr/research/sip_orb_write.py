# ruff: noqa: E501  （中文报告行较长）
"""R-057 报告写出：docs/research/R057_SIP_ORB.md（第一句给结论）与案例图 docs/research/r057_cases/。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from zr.research import sip_orb as so
from zr.research.sip_orb_report import (
    PAPER_TAB,
    PERIODS,
    REPO,
    bars1_loader,
    r_multiples,
    sip_root,
)
from zr.research.timing.stats import fmt_ci, pooled

MAIN_KEY = "主口径（SIP 前 20，1 分钟，保守，D-013，滑点 0.01，10 万美元，4 倍）"


def _pct(x, nd=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


def criteria(X: dict, P: dict) -> dict:
    m = X[MAIN_KEY]
    D = m["p"]["daily"].set_index("date")
    yr = {}
    for y in (2024, 2025, 2026):
        r = D[[d.year == y for d in D.index]]["ret"]
        yr[y] = float(np.prod(1 + r) - 1) if len(r) else np.nan
    s02 = X["敏感性：滑点 0.02"]["m"]["mean"]
    mm = m["m"]
    return {
        "日收益均值 > 0 且按日 t ≥ 2.6": bool(mm["mean"] > 0 and mm["t"] >= 2.6),
        "夏普 > 1": bool(mm["sharpe"] > 1),
        "2024、2025、2026 三段都为正": bool(all(np.isfinite(v) and v > 0 for v in yr.values())),
        "每边滑点 0.02 美元仍为正": bool(s02 > 0),
        "超过安慰剂 (i) 95% 分位": bool(P["i"][0] > np.percentile(P["i"][1], 95)),
        "超过安慰剂 (ii) 95% 分位": bool(P["ii"][0] > np.percentile(P["ii"][1], 95)),
        "_years": yr,
    }


def case_charts(X: dict, out: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    out.mkdir(parents=True, exist_ok=True)
    tr = X[MAIN_KEY]["p"]["trades"]
    tr = tr[tr["shares"] > 0].sort_values(["date", "entry_min"])
    sel = pd.concat(
        [tr[tr["pnl"] > 0].tail(10).assign(g="盈利"), tr[tr["pnl"] <= 0].tail(10).assign(g="亏损")]
    )
    load1 = bars1_loader(sip_root())
    cards = []
    for k, r in enumerate(sel.itertuples()):
        b = load1(r.date)
        x = b[b["symbol"] == r.symbol].sort_values("ts")
        t = pd.DatetimeIndex(x["ts"]).tz_convert(so.ET)
        m = np.asarray(t.hour * 60 + t.minute)
        fig, ax = plt.subplots(figsize=(12, 5), dpi=90)
        xs = np.arange(len(x))
        for i, (o_, h_, l_, c_) in enumerate(x[["open", "high", "low", "close"]].to_numpy()):
            col = "#16a34a" if c_ >= o_ else "#dc2626"
            ax.vlines(i, l_, h_, color=col, lw=0.8)
            ax.vlines(i, min(o_, c_), max(o_, c_), color=col, lw=2.5)
        ax.axhline(r.or_high, color="#6b7280", ls=":", lw=1, label="开盘区间高 / 低")
        ax.axhline(r.or_low, color="#6b7280", ls=":", lw=1)
        stop = r.entry - r.dir * so.STOP_ATR * r.atr_prev
        ax.axhline(stop, color="#b91c1c", ls="--", lw=1, label=f"止损 {stop:.2f}")
        ie, ix = int(np.searchsorted(m, r.entry_min)), int(np.searchsorted(m, r.exit_min))
        ax.scatter([ie], [r.entry], color="#1d4ed8", zorder=5, label=f"入场 {r.entry:.2f}")
        ax.scatter(
            [min(ix, len(x) - 1)], [r.exit], color="#b45309", zorder=5, label=f"出场 {r.exit:.2f}"
        )
        ax.axvspan(-0.5, 4.5, color="#e5e7eb", alpha=0.5)
        step = max(1, len(x) // 12)
        ax.set_xticks(xs[::step])
        ax.set_xticklabels([t[i].strftime("%H:%M") for i in xs[::step]], fontsize=8)
        ax.legend(fontsize=8, loc="best")
        ax.set_title(
            f"{r.symbol} {r.date} {'做多' if r.dir > 0 else '做空'}  RelVol {r.relvol:.1f}  第 {int(r.rank)} 名  盈亏 {r.pnl:+,.0f} 美元（{r.shares} 股）",
            fontsize=10,
            loc="left",
        )
        fn = f"{r.g}_{k:02d}_{r.symbol}_{r.date}.png"
        fig.savefig(out / fn, bbox_inches="tight")
        plt.close(fig)
        cards.append((r.g, fn, f"{r.date} {r.symbol} {'多' if r.dir > 0 else '空'} {r.pnl:+,.0f}"))
    html = [
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>R-057 案例</title>",
        "<style>:root{--bg:#fff;--fg:#111827;--bd:#e5e7eb}@media (prefers-color-scheme:dark){:root{--bg:#111827;--fg:#f9fafb;--bd:#374151}}body{background:var(--bg);color:var(--fg);font-family:-apple-system,'PingFang SC',sans-serif;margin:16px}"
        ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}a{color:var(--fg);text-decoration:none;border:1px solid var(--bd);border-radius:6px;padding:6px}img{width:100%}</style></head><body>",
        "<h1>R-057 主口径：最近 10 笔盈利、10 笔亏损（1 分钟 K）</h1>",
    ]
    for g in ("盈利", "亏损"):
        html.append(
            f"<h2>{g}</h2><div class='g'>"
            + "".join(
                f"<a href='{fn}'><img src='{fn}' alt='{fn}'><div>{cap}</div></a>"
                for gg, fn, cap in cards
                if gg == g
            )
            + "</div>"
        )
    html.append("</body></html>")
    (out / "index.html").write_text("".join(html), encoding="utf-8")
    return str(out / "index.html")


def write_report(R: dict, X: dict, P: dict, B: pd.DataFrame) -> str:
    from zr.ops.markdown import df_to_md

    r = sip_root()
    prelim = R["prelim"]
    crit = criteria(X, P)
    passed = all(v for k, v in crit.items() if not k.startswith("_"))
    mm = X[MAIN_KEY]["m"]
    tag = "**初步，标的池不全**（成交额前 500 只）——" if prelim else ""
    first = (
        f"{tag}{'复现成立' if passed else '复现不成立'}：主口径（保守算法、D-013、每边 0.01 美元、10 万美元）日收益均值 {mm['mean'] * 100:+.3f}%（t {mm['t']:+.2f}），"
        f"夏普 {mm['sharpe']:.2f}，{sum(1 for k, v in crit.items() if not k.startswith('_') and v)} / 6 项通过标准满足。"
    )
    meta = json.loads((r / "universe" / "meta.json").read_text())
    L = [
        f"# R-057：Zarattini、Barbon、Aziz（2024）SIP 开盘区间突破的发表后样本外复现{'（初步）' if prelim else ''}",
        "",
        first,
        "",
        f"`python -m zr.research.sip_orb_report{' --prelim' if prelim else ''}`。预登记 RESEARCH_LOG R-057（运行前提交）。规则与参数全部取自原文，不调参。",
        "",
        "## 数据",
        "",
        f"- 标的名单：纳斯达克官方 Symbol Directory，文件日期 {meta['listed_date']}；过滤后普通股 {meta['included']} 只（剔除原因计数 {meta['reasons']}）。**只含当前上市股票，不含 2023-10 以来退市的（幸存者偏差）**；原文用 CRSP，含退市股。",
        f"- 本次回测的标的池：{len(R['pool'])} 只（{'成交额前 500 只' if prelim else '全部至少有一天合格且 5 分钟 K 已拉取的标的'}）；样本 {R['all_days'][0]}～{R['last']}，{len(R['all_days'])} 个交易日；原文样本 2016–2023，与本样本基本不重叠。",
        f"- 每日 RelVol 前 50 的 1 分钟 K：{R['top50'].shape[0]:,} 个标的日；基础版与安慰剂 (i) 用 5 分钟 K 执行（1 分钟 K 只拉了前 50）。",
        "",
    ]
    test = r / ".." / "sip_test" / "test_report.json"
    L += [
        "## 通过标准（主口径、10 万美元，全部满足才算「复现成立」）",
        "",
        "| 标准 | 结果 |",
        "|---|---|",
    ]
    L += [
        f"| {k} | {'满足' if v else '**不满足**'} |"
        for k, v in crit.items()
        if not k.startswith("_")
    ]
    L += ["", f"自然年收益：{ {y: _pct(v) for y, v in crit['_years'].items()} }。", ""]
    # 复现对照
    rows = []
    for name, v in PAPER_TAB.items():
        rows.append(
            {
                "组合": name,
                "样本": "2016–2023",
                "总收益": _pct(v["total"], 0),
                "年化": _pct(v["irr"]),
                "波动": _pct(v["vol"]),
                "夏普": f"{v['sharpe']:.2f}",
                "日胜率": _pct(v["hit"]),
                "每笔胜率": "—",
                "最大回撤": _pct(-v["mdd"]),
                "最差单日": _pct(v["worst"], 2),
                "alpha（年化）": _pct(v["alpha"]),
                "beta": f"{v['beta']:.2f}",
            }
        )
    for name, x in X.items():
        m = x["m"]
        rows.append(
            {
                "组合": name,
                "样本": f"{R['all_days'][0]}～{R['last']}",
                "总收益": _pct(m["total"], 0),
                "年化": _pct(m["irr"]),
                "波动": _pct(m["vol"]),
                "夏普": f"{m['sharpe']:.2f}",
                "日胜率": _pct(m["hit_day"]),
                "每笔胜率": _pct(m["hit_trade"]),
                "最大回撤": _pct(m["mdd"]),
                "最差单日": _pct(m["worst"], 2),
                "alpha（年化）": _pct(m["alpha"]),
                "beta": f"{m['beta']:.2f}",
            }
        )
    L += [
        "## 复现对照与全部账户 / 敏感性（敏感性只报告，不用于挑选）",
        "",
        "原文：表 1、表 2（SSRN 4729284，HSG 仓库 PDF，2026-11-14 读取；初始 25,000 美元、每股 0.0035 美元；原文「Hit Ratio」按日）。日胜率 = 有交易日中收益 > 0 的占比。alpha / beta = 日收益对 SPY 日收益回归（alpha × 252）。",
        "",
        df_to_md(pd.DataFrame(rows)),
        "",
    ]
    # 统计
    D = X[MAIN_KEY]["p"]["daily"].set_index("date").reindex(R["all_days"])
    rr = D["ret"].fillna(0)
    tr = r_multiples(X[MAIN_KEY]["p"]["trades"])
    st_R = pooled(tr["r_mult"].to_numpy(), tr["date"].to_numpy())
    st_pct = pooled(tr["pct"].to_numpy(), tr["date"].to_numpy())
    L += [
        "## 主口径统计",
        "",
        f"- 日收益均值 {mm['mean'] * 100:+.4f}%（按日整块重抽样 95% 区间 {mm['ci'][0] * 100:+.4f}%～{mm['ci'][1] * 100:+.4f}%），t {mm['t']:+.2f}，{mm['days']} 个交易日（有交易 {mm['traded_days']} 天），{mm['trades']:,} 笔。",
        f"- 每笔平均 R（扣费后，R = 0.10 × ATR）{fmt_ci({**st_R, 'mean': st_R['mean']}, 3).replace('%', '')} R；每笔占股价 {fmt_ci(st_pct, 3)}。",
        "",
    ]
    rows = []
    for name, a, b in PERIODS:
        x = tr[(tr["date"] >= a) & (tr["date"] <= b)]
        dd = rr[[a <= d <= b for d in rr.index]]
        if not len(dd):
            continue
        s = pooled(x["r_mult"].to_numpy(), x["date"].to_numpy(), n_boot=1000)
        rows.append(
            {
                "时段": name,
                "交易日": len(dd),
                "累计收益": _pct(float(np.prod(1 + dd) - 1)),
                "日均": f"{dd.mean() * 100:+.4f}%",
                "笔数": s["n"],
                "每笔 R（聚类 95%）": fmt_ci(s, 3).replace("%", ""),
            }
        )
    for nm, sgn in (("做多", 1), ("做空", -1)):
        x = tr[tr["dir"] == sgn]
        s = pooled(x["r_mult"].to_numpy(), x["date"].to_numpy(), n_boot=1000)
        rows.append(
            {
                "时段": f"全样本 {nm}",
                "交易日": "",
                "累计收益": "",
                "日均": "",
                "笔数": s["n"],
                "每笔 R（聚类 95%）": fmt_ci(s, 3).replace("%", ""),
            }
        )
    L += [df_to_md(pd.DataFrame(rows)), ""]
    # RelVol 分桶（基础版 5 分钟，主口径设定）
    bt = r_multiples(X["基础版（全部合格，5 分钟执行，主口径设定，10 万美元）"]["p"]["trades"])
    bt["rv_b"] = pd.cut(
        bt["relvol"],
        [0, 1, 2, 5, 10, 30, 1e12],
        right=False,
        labels=["< 1", "1–2", "2–5", "5–10", "10–30", "≥ 30"],
    )
    rows = []
    for lv, g in bt.groupby("rv_b", observed=True):
        s = pooled(g["r_mult"].to_numpy(), g["date"].to_numpy(), n_boot=500)
        rows.append(
            {
                "RelVol": lv,
                "笔数": s["n"],
                "每笔平均 R（聚类 95%）": fmt_ci(s, 3).replace("%", ""),
                "每笔占股价": f"{g['pct'].mean():+.3f}%",
            }
        )
    for lab, mask in (
        ("< 1（合计）", bt["relvol"] < 1),
        ("≥ 1（合计）", bt["relvol"] >= 1),
        ("≥ 30（合计）", bt["relvol"] >= 30),
    ):
        g = bt[mask]
        s = pooled(g["r_mult"].to_numpy(), g["date"].to_numpy(), n_boot=500)
        rows.append(
            {
                "RelVol": lab,
                "笔数": s["n"],
                "每笔平均 R（聚类 95%）": fmt_ci(s, 3).replace("%", ""),
                "每笔占股价": f"{g['pct'].mean():+.3f}%" if len(g) else "—",
            }
        )
    L += [
        "## 按 RelVol 分桶的每笔平均 R（基础版，5 分钟执行，主口径设定；对照原文图 4：< 100% −0.02R、> 100% +0.08R、> 30 倍 +0.38R，原文为扣佣金后）",
        "",
        df_to_md(pd.DataFrame(rows)),
        "",
    ]
    # 盈亏平衡滑点
    Bt = B.copy()
    for c in (
        "零滑点每笔盈亏（美元）",
        "盈亏平衡滑点（美元 / 边）",
        "占 R 中位数",
        "R 中位数（美元 / 股）",
    ):
        Bt[c] = Bt[c].map(
            lambda v: "—" if not np.isfinite(v) else (f"{v:.4f}" if abs(v) < 10 else f"{v:,.0f}")
        )
    L += [
        "## 盈亏平衡滑点（本报告最重要的数）",
        "",
        "保守算法、D-013、股数固定为主口径实际股数；每边滑点 0～0.10 美元（步长 0.005）重算入场、止损与平仓成交价，线性插值求每笔平均盈亏归零的滑点。0 = 零滑点已不赚钱；inf = 0.10 美元仍为正。",
        "",
        df_to_md(Bt),
        "",
    ]
    # 安慰剂
    rows = []
    for k, nm in (
        ("i", "(i) 合格 ∩ RelVol ≥ 1 中随机取 20 只（5 分钟执行）"),
        ("ii", "(ii) 选股不变、随机方向（1 分钟）"),
    ):
        act, sims = P[k]
        rows.append(
            {
                "安慰剂": nm,
                "实际日收益均值": f"{act * 100:+.4f}%",
                "安慰剂均值": f"{sims.mean() * 100:+.4f}%",
                "95% 分位": f"{np.percentile(sims, 95) * 100:+.4f}%",
                "实际所在分位": f"{(sims < act).mean():.1%}",
                "超过 95% 分位": "是" if act > np.percentile(sims, 95) else "否",
            }
        )
    L += [
        "## 安慰剂（各 1,000 次，固定权益 10 万美元的日收益，不复利）",
        "",
        df_to_md(pd.DataFrame(rows)),
        "",
    ]
    try:
        cases = case_charts(X, REPO / "docs" / "research" / "r057_cases")
        L += [f"案例图：`{Path(cases).relative_to(REPO)}`（主口径最近 10 笔盈利、10 笔亏损）。", ""]
    except Exception as e:  # noqa: BLE001
        L += [f"案例图生成失败：{type(e).__name__}: {e}", ""]
    if test.exists():
        t = json.loads(test.read_text())
        L += [
            "## 拉取实测（30 只股票）",
            "",
            f"- 日线 {t['daily']['requests']} 次请求 {t['daily']['sec']:.0f} 秒；5 分钟 K {t['bars5']['requests']} 次 {t['bars5']['sec']:.0f} 秒；1 分钟 K {t['bars1']['requests']} 次 {t['bars1']['sec']:.0f} 秒；错误 {t['errors']}。",
            "",
        ]
    L += [
        "## 边界",
        "",
        "- 无退市股（幸存者偏差，方向通常是高估收益）；名单是 2026-11-13 的当前上市普通股。",
        "- 1 分钟 K 的同根歧义：入场那根内先入场后止损无法分辨，保守 / 乐观两种算法都报；判定以保守为准。基础版与安慰剂 (i) 只有 5 分钟 K，歧义更大。",
        "- 做空未计借券费，也未核实能否借到；开盘时段的真实买卖价差未知（只用固定每边滑点）；长桥账户能否做 4 倍日内杠杆未核实。",
        "- 原文样本 2016–2023、本样本 2023-10 以后，基本不重叠；数据源不同（原文 IQFeed，本文长桥）。",
        "- 本检验不计入 MU / QQQ / SPY 留出期额度；之后若在这套数据上调参，2026-06 以后的数据须重新封存。",
        "",
    ]
    out = REPO / "docs" / "research" / ("R057_SIP_ORB_PRELIM.md" if prelim else "R057_SIP_ORB.md")
    out.write_text("\n".join(L), encoding="utf-8")
    return str(out)
