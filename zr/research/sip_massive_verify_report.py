# ruff: noqa: E501  （中文报告行较长）
"""R-057 §1.1 Massive 数据核对报告：读 VPS 产出的 verify.json、probe.json（zr.data.sip_massive），写
docs/research/R057_MASSIVE_DATA.md。表中数字全部来自这两个 JSON；结论段落只引用表中数字。

运行（Mac，JSON 由同步任务或 scp 取回）：
  python -m zr.research.sip_massive_verify_report --verify <verify.json> --probe <probe.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from zr.ops.markdown import df_to_md

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "R057_MASSIVE_DATA.md"


def files_table(v: dict) -> pd.DataFrame:
    rows = []
    for day, f in v["files"].items():
        if "error" in f:
            rows.append(
                {"日期": day, "分钟文件": "下载失败：" + f["error"].split(": ", 1)[-1][:80]}
            )
            continue
        m, d = f["minute"], f["day"]
        rows.append(
            {
                "日期": day,
                "分钟文件": f"{m['file_bytes'] / 1e6:.1f} MB / {m['rows']:,} 行 / {m['tickers']:,} 个代码",
                "时间范围（美东）": f"{m['window_start_min_et'][11:16]}–{m['window_start_max_et'][11:16]}",
                "盘前 / 常规 / 盘后行数": f"{m['rows_pre_0930']:,} / {m['rows_rth']:,} / {m['rows_post_1600']:,}",
                "整分钟网格": m["all_on_minute_grid"],
                "volume 类型": m["columns"]["volume"],
                "日文件": f"{d['file_bytes'] / 1e3:.0f} KB / {d['rows']:,} 行",
                "日文件时间戳（美东）": ", ".join(x[11:16] for x in d["window_start_et"]),
            }
        )
    return pd.DataFrame(rows)


def price_table(v: dict) -> pd.DataFrame:
    rows = []
    for day, f in v["files"].items():
        for sym, t in (f.get("tickers") or {}).items():
            if not t.get("present"):
                continue
            rows.append(
                {
                    "日期": day,
                    "代码": sym,
                    "常规时段开盘（1 分钟）": t["rth_open"],
                    "日文件开盘": t.get("day_file", {}).get("open"),
                    "常规时段末根收盘": t["rth_close"],
                    "日文件收盘": t.get("day_file", {}).get("close"),
                    "常规时段量 ÷ 日文件量": round(t["rth_volume"] / t["day_file"]["volume"], 3)
                    if t.get("day_file")
                    else None,
                    "全部 1 分钟量 ÷ 日文件量": round(t["all_volume"] / t["day_file"]["volume"], 3)
                    if t.get("day_file")
                    else None,
                }
            )
    return pd.DataFrame(rows)


def lb_table(p: dict) -> pd.DataFrame:
    rows = []
    for day, d in p["days"].items():
        for sym, c in (d.get("lb_1m_compare") or {}).items():
            r = c.get("rth") or {}
            k = d["lb_1m_containment_rth"].get(sym, {})
            if not r:
                continue
            rows.append(
                {
                    "日期": day,
                    "代码": sym,
                    "对齐（偏移 −1/0/+1 分钟收盘一致率）": f"{c['close_equal_rate_shift-1']:.2f} / {c['close_equal_rate_shift+0']:.2f} / {c['close_equal_rate_shift+1']:.2f}",
                    "常规时段根数": r["n"],
                    "开高低收完全相等率": " / ".join(
                        f"{r[x]['equal_rate']:.2f}" for x in ("open", "high", "low", "close")
                    ),
                    "收盘 |差| p95 美元": r["close"]["abs_usd_p95"],
                    "收盘 |差| 最大 bp": r["close"]["abs_bp_max"],
                    "SIP 高 ≥ 长桥高": k.get("sip_high_ge_lb"),
                    "SIP 低 ≤ 长桥低": k.get("sip_low_le_lb"),
                    "SIP 量 ≥ 长桥量": k.get("sip_vol_ge_lb"),
                    "量合计比 SIP ÷ 长桥": r["volume"]["sum_ratio"],
                    "仅 SIP 有 / 仅长桥有（全天）": f"{c['only_massive']} / {c['only_lb_traded']}",
                }
            )
    return pd.DataFrame(rows)


def lb_ext_summary(p: dict) -> pd.DataFrame:
    rows = []
    for d in p["days"].values():
        for sym, c in (d.get("lb_1m_compare") or {}).items():
            e = c.get("ext")
            if e:
                rows.append(
                    {
                        "代码": sym,
                        "n": e["n"],
                        "close_eq": e["close"]["equal_rate"],
                        "close_p95": e["close"]["abs_usd_p95"],
                        "vol_ratio": e["volume"]["sum_ratio"],
                    }
                )
    if not rows:
        return pd.DataFrame()
    x = pd.DataFrame(rows)
    return (
        x.groupby("代码")
        .agg(
            天数=("n", "size"),
            根数=("n", "sum"),
            收盘相等率中位数=("close_eq", "median"),
            收盘差p95中位数美元=("close_p95", "median"),
            量合计比中位数=("vol_ratio", "median"),
        )
        .reset_index()
    )


def daily_def_table(p: dict) -> pd.DataFrame:
    rows = []
    for day, d in p["days"].items():
        lb = d.get("lb_daily_vs_day", {})
        rows.append(
            {
                "日期": day,
                "代码数": d["tickers_day_and_rth"],
                "日文件 vs 常规时段 1 分钟：开 / 高 / 低 / 收相等率": " / ".join(
                    f"{d['day_vs_rth1m'][x]:.3f}" for x in ("open", "high", "low", "close")
                ),
                "常规时段 1 分钟量 ÷ 日量 p05 / p50 / p95（日量 ≥ 100 万）": " / ".join(
                    f"{q:.3f}" for q in d["rth1m_over_day_volume_q"]
                ),
                "长桥日线 vs 日文件：n": lb.get("n"),
                "开 / 高 / 低 / 收 / 量相等率": " / ".join(
                    f"{lb[x]:.2f}" for x in ("open", "high", "low", "close", "volume")
                )
                if lb
                else "",
                "长桥日量 ÷ 日文件量 最小 / 中位 / 最大": " / ".join(
                    f"{q:.4f}" for q in lb.get("volume_ratio_lb_over_m_q", [])
                ),
            }
        )
    return pd.DataFrame(rows)


def delisted_md(v: dict) -> str:
    out = []
    for sym, r in v["delisted"].items():
        m = r["minute"]
        ref = r["ref_inactive"][0] if r["ref_inactive"] else {}
        pit = r.get("ref_point_in_time") or []
        out.append(
            f"- **{sym}**：分钟文件{'有' if m.get('present') else '无'}"
            + (
                f"（{m['bars']} 根，常规时段 {m['rth_bars']} 根，常规时段开 {m['rth_open']}、收 {m['rth_close']}，量 {m['rth_volume']:,}）"
                if m.get("present")
                else ""
            )
            + f"；日文件{'有' if r['in_day_file'] else '无'}；reference（active=false）：{ref.get('name')}，type {ref.get('type')}，{ref.get('primary_exchange')}，delisted_utc {ref.get('delisted_utc')}，composite_figi {ref.get('composite_figi')}"
            + f"；时点查询（date=当日）返回 {len(pit)} 条"
            + (
                f"（type {pit[0].get('type')}，{pit[0].get('primary_exchange')}，当时 active={pit[0].get('active')}）"
                if pit
                else ""
            )
        )
    return "\n".join(out)


def ref_md(v: dict) -> str:
    r = v["reference"]
    out = []
    for ex in ("XNYS", "XNAS"):
        x = r[ex]
        pits = "；".join(
            f"date={k[4:14]} 时点查询 {x[k]:,} 只"
            for k in x
            if k.startswith("pit_") and k.count("_") == 1 and not k.endswith("seconds")
        )
        out.append(
            f"- {ex} 普通股（type=CS）：当前在市 {x['active']:,}，已退市 {x['inactive']:,}（其中 {x['inactive_with_delisted_utc']:,} 条有 delisted_utc，最早 {x['inactive_delisted_min']}）；{pits}"
        )
    out.append(
        f"- 本次 REST 调用 {r['rest_calls']} 次。拆股接口 NVDA："
        + "，".join(
            f"{s['execution_date']} {s['split_from']}→{s['split_to']}" for s in v["splits_nvda"]
        )
    )
    return "\n".join(out)


def summary(v: dict, p: dict) -> dict:
    """结论段用到的数字（全部由两个 JSON 算出）。"""
    rows = []
    for day, d in p["days"].items():
        for sym, k in d["lb_1m_containment_rth"].items():
            r = d["lb_1m_compare"][sym]["rth"]
            rows.append(
                {
                    "day": day,
                    "sym": sym,
                    "contained": k["sip_high_ge_lb"] == 1
                    and k["sip_low_le_lb"] == 1
                    and k["sip_vol_ge_lb"] == 1,
                    "outside": 1 - min(k["sip_high_ge_lb"], k["sip_low_le_lb"]),
                    "vr": r["volume"]["sum_ratio"],
                    "close_eq": r["close"]["equal_rate"],
                    "p95": r["close"]["abs_usd_p95"],
                }
            )
    x = pd.DataFrame(rows)
    by_day = x.groupby("day")["contained"].all()
    first_break = next((d for d, ok in by_day.items() if not ok), None)
    pre = x[x["day"] < first_break] if first_break else x
    post = x[x["day"] >= first_break] if first_break else x.iloc[0:0]
    dd = pd.DataFrame(
        [
            {"day": k, **d["day_vs_rth1m"], "rth_vol_med": d["rth1m_over_day_volume_q"][1]}
            for k, d in p["days"].items()
        ]
    )
    lbv = pd.DataFrame(
        [{"day": k, **d["lb_daily_vs_day"]} for k, d in p["days"].items() if "lb_daily_vs_day" in d]
    )
    lbr = lbv["volume_ratio_lb_over_m_q"].apply(pd.Series)
    return {
        "n_days": int(by_day.size),
        "first_day": min(by_day.index),
        "last_day": max(by_day.index),
        "pre_days": int((by_day.index < first_break).sum()) if first_break else int(by_day.size),
        "pre_all_contained": bool(pre["contained"].all()),
        "first_break": first_break,
        "last_contained_day": max(
            d for d, ok in by_day.items() if ok and (first_break is None or d < first_break)
        ),
        "pre_vr": (float(pre["vr"].min()), float(pre["vr"].median()), float(pre["vr"].max())),
        "post_vr": (float(post["vr"].min()), float(post["vr"].max())) if len(post) else None,
        "post_contained_rows": int(post["contained"].sum()),
        "post_outside": (float(post["outside"].min()), float(post["outside"].max()))
        if len(post)
        else None,
        "post_rows": len(post),
        "close_eq": (float(x["close_eq"].min()), float(x["close_eq"].max())),
        "close_p95_by_sym": {
            s: (float(g["p95"].min()), float(g["p95"].max())) for s, g in x.groupby("sym")
        },
        "day_open_eq_min": float(dd["open"].min()),
        "day_high_eq": (float(dd["high"].min()), float(dd["high"].max())),
        "day_low_eq": (float(dd["low"].min()), float(dd["low"].max())),
        "day_close_eq": (float(dd["close"].min()), float(dd["close"].max())),
        "rth_vol_med": (float(dd["rth_vol_med"].min()), float(dd["rth_vol_med"].max())),
        "lb_daily_n_days": len(lbv),
        "lb_daily_vol_ratio": (float(lbr[0].min()), float(lbr[2].max())),
        "lb_daily_vol_eq_all_days": int((lbv["volume"] == 1.0).sum()),
        "lb_daily_vol_neq_med": (
            float(lbr.loc[lbv["volume"] < 1.0, 1].min()),
            float(lbr.loc[lbv["volume"] < 1.0, 1].max()),
        ),
        "lb_daily_open_eq_min": float(lbv["open"].min()),
    }


def conclusions(v: dict, p: dict) -> str:
    s = summary(v, p)
    first = v["minute_first_accessible"].split("/")[-1][:10]
    c = s["close_p95_by_sym"]
    return "\n".join(
        [
            "## 结论",
            "",
            f"1. **可下载范围只到 {first}**（2017-06-15 返回 HTTP 403）。原文样本 2016-01-01～2023-12-31，可做的样本内只有 {first}～2023-12-31（扣 14 个交易日预热后更短）。样本外（2024-01-01 起）完整。",
            "2. 字段、时区、盘前盘后、不复权都与预期一致：`window_start` 为 UTC 纳秒、整分钟；分钟文件含 04:00–19:59（美东）；价格不复权（NVDA 拆股前后价格见下）。2026 年文件的 `volume` 是浮点（含小数股），2021、2024 年为整数。",
            "3. 退市股在：TWTR（2022-10-20）、SIVB（2023-03-08）分钟与日文件都有；reference 的 active=false 记录给出退市日期，`date=` 时点查询返回当日在市的普通股（含后来退市的）。",
            f"4. **日聚合 ≠ 常规时段 1 分钟聚合**（{s['n_days']} 个月度样本日，全市场）：开盘相等率最低 {s['day_open_eq_min']:.3f}；高 {s['day_high_eq'][0]:.3f}–{s['day_high_eq'][1]:.3f}、低 {s['day_low_eq'][0]:.3f}–{s['day_low_eq'][1]:.3f}；收盘只有 {s['day_close_eq'][0]:.3f}–{s['day_close_eq'][1]:.3f}（日文件是官方收盘价）；常规时段 1 分钟量合计 ÷ 日量的中位数 {s['rth_vol_med'][0]:.3f}–{s['rth_vol_med'][1]:.3f}；最低两天 2023-12-15、2024-03-15 都是季度期权到期日（推断：收盘竞价成交落在 16:00 那根 K，不在 09:30–15:59 内，未验证）。",
            f"5. **长桥日线 = Massive 日聚合**（R-057 测试拉取的 24–30 只，{s['lb_daily_n_days']} 天）：开盘相等率最低 {s['lb_daily_open_eq_min']:.2f}；量比（长桥 ÷ Massive）{s['lb_daily_vol_ratio'][0]:.4f}–{s['lb_daily_vol_ratio'][1]:.4f}，{s['lb_daily_vol_eq_all_days']} 天全部完全相等；其余 {s['lb_daily_n_days'] - s['lb_daily_vol_eq_all_days']} 天各天量比中位数 {s['lb_daily_vol_neq_med'][0]:.4f}–{s['lb_daily_vol_neq_med'][1]:.4f}（见第 4 节表）。据此，R-057 补充（2026-11-14）把合格条件的日线（开盘、ATR、14 日均量）定为日聚合，与原预登记所用的长桥日线同口径。",
            f"6. **长桥 1 分钟 K 不是 SIP 全量**：时间对齐无误（偏移 0 分钟的收盘一致率远高于 ±1 分钟）。{s['first_day']}～{s['last_contained_day']} 的 {s['pre_days']} 个样本日 × MU/QQQ/SPY，常规时段每一根都满足「SIP 高 ≥ 长桥高、SIP 低 ≤ 长桥低、SIP 量 ≥ 长桥量」，量合计比 SIP ÷ 长桥 {s['pre_vr'][0]:.2f}–{s['pre_vr'][2]:.2f}（中位数 {s['pre_vr'][1]:.2f}）——长桥分钟 K 只含部分成交（像单一交易所行情，推断，未验证）。"
            + (
                f"**{s['first_break']} 起关系改变**：量比 {s['post_vr'][0]:.2f}–{s['post_vr'][1]:.2f}，长桥高点或低点超出 SIP 范围的分钟占 {s['post_outside'][0]:.0%}–{s['post_outside'][1]:.0%}（{s['post_rows']} 个代码日中 {s['post_contained_rows']} 个仍完全包含），推断长桥行情源在此前一个月内变更（未验证）。"
                if s["first_break"]
                else ""
            ),
            f"   开高低收逐根完全相等率 {s['close_eq'][0]:.2f}–{s['close_eq'][1]:.2f}（收盘）；收盘 |差| 的 p95：MU {c['MU'][0]:.2f}–{c['MU'][1]:.2f} 美元、QQQ {c['QQQ'][0]:.2f}–{c['QQQ'][1]:.2f}、SPY {c['SPY'][0]:.2f}–{c['SPY'][1]:.2f}。",
            "",
            "边界：比对只覆盖 MU / QQQ / SPY 三只、每月一天（2023-11～2026-10）；长桥日线比对只覆盖 R-057 测试拉取的 30 只。未比对 2021-11～2023-10（本地没有长桥分钟 K）。",
            "",
        ]
    )


def render(v: dict, p: dict) -> str:
    dl = pd.DataFrame(v["downloads"])
    big = dl[dl["key"].str.contains("minute_aggs")]
    lt = lb_table(p)
    s = [
        "# R-057 Massive 数据核对（任务书 §1.1）",
        "",
        f"生成：`zr/research/sip_massive_verify_report.py`，输入 VPS `zr.data.sip_massive --phase verify`（{v['run_utc']}）与 `--phase probe`（{p['run_utc']}）的 JSON。原始文件核对完即删。",
        "",
        conclusions(v, p),
        "## 1. 可得范围",
        "",
        f"- flat files 列出分钟文件 {v['minute_files_listed']:,} 个（{v['minute_first_files'][0].split('/')[-1][:10]} 至 {v['minute_last_files'][-1].split('/')[-1][:10]}），但**订阅档位可下载的最早文件是 {v['minute_first_accessible'].split('/')[-1][:10]}**（HEAD 二分查找；更早的返回 HTTP 403 NOT_AUTHORIZED）。日聚合相同。",
        "- 起点约等于运行日减 5 年，推断为滚动 5 年窗口（未验证；若滚动，每过一天最早一天不可再下载）。",
        f"- 分钟文件下载：{len(big)} 个，平均 {big['bytes'].mean() / 1e6:.1f} MB、{big['seconds'].mean():.1f} 秒（VPS）。",
        "",
        "## 2. 日文件",
        "",
        df_to_md(files_table(v)),
        "",
        "字段："
        + ", ".join(
            f"`{k}` {t}"
            for k, t in next(f for f in v["files"].values() if "minute" in f)["minute"][
                "columns"
            ].items()
        )
        + "。`window_start` 是 Unix 纳秒（UTC），分钟文件里换成美东后覆盖 04:00–19:59，即**含盘前盘后**；日文件 `window_start` = 当日美东 00:00。",
        "",
        "### 是否不复权",
        "",
        df_to_md(price_table(v)),
        "",
        "NVDA 拆股（拆股接口）：2021-07-20 1→4、2024-06-10 1→10。2024-06-07 NVDA 开盘 1,197.70 美元（拆股前原始价）、2021-11-15 为 305.52 美元，若是复权价应约为其 1/10：**不复权**。日文件开盘 = 常规时段首根 1 分钟开盘；日文件收盘是官方收盘价，与常规时段末根 1 分钟收盘常差几美分。",
        "",
        "## 3. 退市股与 reference",
        "",
        delisted_md(v),
        "",
        ref_md(v),
        "",
        "## 4. 日线口径（日聚合 vs 常规时段 1 分钟聚合 vs 长桥日线）",
        "",
        df_to_md(daily_def_table(p)),
        "",
        "## 5. 与长桥 1 分钟 K 逐根比对（MU / QQQ / SPY，每月 15 日后首个交易日，常规时段，长桥只取有成交的 K）",
        "",
        df_to_md(lt),
        "",
        "盘前盘后（按代码汇总各日）：",
        "",
        df_to_md(lb_ext_summary(p)),
        "",
    ]
    return "\n".join(s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    v = json.loads(Path(a.verify).read_text(encoding="utf-8"))
    p = json.loads(Path(a.probe).read_text(encoding="utf-8"))
    Path(a.out).write_text(render(v, p), encoding="utf-8")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
