"""回放网格与规则卡（METHOD_V3 第 6 节）→ docs/research/RULE_CARDS_<UND>.md。

网格：周期 {1, 2, 5} × 入场形态及参数（zr/research/patterns_v3.py）× 出场组合（结构规则或 none ×
X6 止盈 / 止损 × X7 持仓上限）× 方向；另在 2 分钟上用 MACD(12, 26, 9) 跑一遍作参数扫描档。
引擎 zr/research/sim_v3.py（成交口径见其 docstring）。

规则卡：扣费后均值 > 0、n ≥ 100、且 2026 年（按年份拆分的样本外）均值 > 0 的行；附同时段随机对照
（每笔换成同日 ±15 分钟内的随机入场分钟，同方向同出场组合，1,000 次）与按日聚类的 t、Bonferroni
校正 p。
试验计数 = 去重后（等价定义只算一次）的网格行数。
运行：uv run python -m zr.research.grid_v3 --symbol MU.US
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from zr.config import get_settings
from zr.features.bars import ET, load_extended_bars
from zr.features.option_features import option_days
from zr.features.regime import load_config
from zr.ops.markdown import df_to_md
from zr.research import exits as ex
from zr.research import patterns_v3 as pt
from zr.research import sim_v3 as sv

log = logging.getLogger("zr.research.grid_v3")
STATE_OF = {"E10": "IGNITION", "E1": "CHANNEL", "E11": "RANGE"}


def dedup_names(cfg: dict) -> dict[str, str]:
    """入场名 → 代表名（E1 的 %b 区间与距中轨条件交集相同、N/M 相同即等价）。"""
    rep, out = {}, {}
    for c in pt.entry_configs(cfg):
        key = (c["entry"], c.get("eff"), c.get("N"), c.get("M"), c.get("k"), c.get("v"))
        if c["entry"] != "E1":
            key = (c["entry"], c.get("k"), c.get("M"), c.get("v"))
        name = pt.entry_name(c)
        out[name] = rep.setdefault(key, name)
    return out


def trial_counts(cfg: dict) -> dict:
    x6, x7 = cfg["exits"]["X6"], cfg["exits"]["X7"]
    k = len(x6["tp"]) * len(x6["sl"]) * len(x7["cap_min"])
    names = dedup_names(cfg)
    uniq = sorted(set(names.values()))
    per_period = sum(len(ex.exit_rules(cfg, n.split("|")[0])) * k * 2 for n in uniq)
    raw = sum(len(ex.exit_rules(cfg, n.split("|")[0])) * k * 2 for n in names)
    n_p = len(cfg["periods"])
    return {
        "entry_defs_raw": len(names),
        "entry_defs_unique": len(uniq),
        "rows_per_period": per_period,
        "rows_main": per_period * n_p,
        "rows_scan": per_period,
        "rows_total": per_period * (n_p + 1),
        "rows_total_raw": raw * (n_p + 1),
    }


def long_table(res: dict, cfg: dict, macd: str, only: list | None = None) -> pd.DataFrame:
    x6, x7 = cfg["exits"]["X6"], cfg["exits"]["X7"]
    reps = dedup_names(cfg)
    parts = []
    for p, o in res.items():
        rules = o["rules"]
        for name, a in o["accum"].items():
            if reps.get(name, name) != name or (only is not None and name not in only):
                continue
            fam = name.split("|")[0]
            ok_rules = set(ex.exit_rules(cfg, fam))
            idx = np.indices(a.n.shape).reshape(5, -1).T
            n = a.n.reshape(-1)
            keep = n > 0
            if not keep.any():
                continue
            idx, n = idx[keep], n[keep]
            g = lambda arr, k=keep: arr.reshape(-1)[k]  # noqa: E731
            s, win, wsum, lsum = g(a.s), g(a.win), g(a.wsum), g(a.lsum)
            nl = n - win
            d1, d2, dn = g(a.d1), g(a.d2), g(a.dn)
            with np.errstate(invalid="ignore", divide="ignore"):
                dm = d1 / dn
                dsd = np.sqrt(np.maximum(d2 / dn - dm**2, 0) * dn / np.maximum(dn - 1, 1))
                t_day = dm / dsd * np.sqrt(dn)
            b = a.bucket.reshape(-1, len(sv.BUCKET_NAMES))[keep]
            df = pd.DataFrame(
                {
                    "macd": macd,
                    "period": p,
                    "entry": name,
                    "family": fam,
                    "state": STATE_OF.get(fam, "OWN"),
                    "side": np.where(idx[:, 0] == 0, "long", "short"),
                    "exit_rule": [rules[i] for i in idx[:, 1]],
                    "tp": np.array(x6["tp"])[idx[:, 2]],
                    "sl": np.array(x6["sl"])[idx[:, 3]],
                    "cap": np.array(x7["cap_min"])[idx[:, 4]],
                    "n": n.astype(int),
                    "mean_net": s / n,
                    "win_rate": win / n,
                    "payoff": np.where(
                        (win > 0) & (nl > 0) & (lsum < 0),
                        (wsum / np.maximum(win, 1)) / (-lsum / np.maximum(nl, 1)),
                        np.nan,
                    ),
                    "sum_net": s,
                    "n_oos": g(a.n_oos).astype(int),
                    "mean_oos": np.where(
                        g(a.n_oos) > 0, g(a.s_oos) / np.maximum(g(a.n_oos), 1), np.nan
                    ),
                    "days": dn.astype(int),
                    "t_day": t_day,
                    "hold_bucket": np.array(sv.BUCKET_NAMES)[b.argmax(1)],
                    "hold_bucket_share": b.max(1) / n,
                }
            )
            parts.append(df[df["exit_rule"].isin(ok_rules)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def long_table_any(res: dict, cfg: dict, macd: str, names: list) -> pd.DataFrame:
    return long_table(res, cfg, macd, only=names)


def random_control_group(
    root: Path,
    und: str,
    stock: pd.DataFrame,
    cfg: dict,
    pdat: sv.PeriodData,
    cards: pd.DataFrame,
    trades: list,
    n_draw: int,
    window: int,
    seed: int = 0,
) -> list[dict]:
    """同时段随机对照（同一 周期 × MACD × 入场 × 方向 的多张卡共用一次期权评估）。

    每张卡的每笔实际成交换成同日 ±window 分钟内的随机入场分钟（同方向、该卡的出场组合），n_draw 次。
    """
    rules = pdat.rules
    x6, x7 = cfg["exits"]["X6"], cfg["exits"]["X7"]
    combos = [
        (
            rules.index(c["exit_rule"]),
            x6["tp"].index(c["tp"]),
            x6["sl"].index(c["sl"]),
            x7["cap_min"].index(c["cap"]),
        )
        for _, c in cards.iterrows()
    ]
    d = 1 if cards["side"].iat[0] == "long" else -1
    last_entry = sv.hm(cfg["execution"]["last_entry"])
    sday = np.asarray(stock.index.tz_convert(ET).date)
    rng = np.random.default_rng(seed)
    real = np.zeros(len(cards))
    rand = np.zeros((len(cards), n_draw))
    used = np.zeros(len(cards), dtype=int)
    for tr in trades:
        if tr["dir"] != d:
            continue
        taken = [tr["e"][tr["take"][(slice(None), *cb)]] for cb in combos]
        for k, cb in enumerate(combos):
            real[k] += float(tr["net"][(slice(None), *cb)][tr["take"][(slice(None), *cb)]].sum())
        allm = set(np.concatenate(taken).tolist()) if taken else set()
        if not allm:
            continue
        day = tr["day"]
        od = sv.load_day(root, und, day, cfg["execution"]["stale_snapshots"])
        trig = sv.day_triggers(pdat, pdat.day == day)
        c1 = stock[sday == day]["close"]
        spot_at = dict(zip(sv.minute_of(c1.index) + 1, c1.to_numpy(), strict=True))
        cand = sorted(
            {
                r
                for e in allm
                for r in range(max(1, e - window), min(last_entry, e + window) + 1)
                if r in spot_at
            }
        )
        if not cand or od is None:
            continue
        ev = pd.DataFrame({"e": cand, "direction": d, "spot": [spot_at[r] for r in cand]})
        res = sv.evaluate(od, ev, trig, cfg)
        if res is None:
            continue
        pos = {int(e): i for i, e in enumerate(res["meta"]["e"])}
        for k, cb in enumerate(combos):
            vals = res["net"][(slice(None), *cb)]
            for e in taken[k]:
                idx = [pos[r] for r in range(e - window, e + window + 1) if r in pos]
                if idx:
                    pool = vals[idx]
                    rand[k] += pool[rng.integers(0, len(pool), n_draw)]
                    used[k] += 1
    qty = cfg["execution"]["qty"]
    return [
        {
            "rand_p50_per_contract": round(float(np.median(rand[k])) / max(used[k], 1) / qty, 2),
            "rand_pctile": round(float((rand[k] <= real[k]).mean() * 100), 1),
            "rand_n_matched": int(used[k]),
        }
        for k in range(len(cards))
    ]


def run(und: str, n_random: int | None = None, out_dir: Path | None = None) -> dict:
    t0 = time.monotonic()
    cfg = load_config()
    root = get_settings().zr_data_root
    stock = load_extended_bars(root, und)
    days = option_days(root, und)
    res = sv.run_grid(root, und, stock, cfg, cfg["periods"], days)
    lt = long_table(res, cfg, "6,13,6")
    scan = sv.run_grid(root, und, stock, cfg, [2], days, macd_key="macd_scan")
    lt = pd.concat([lt, long_table(scan, cfg, "12,26,9")], ignore_index=True)
    tc = trial_counts(cfg)
    m = tc["rows_total"]
    lt["p_bonf"] = np.minimum(1.0, 2 * (1 - norm.cdf(np.abs(lt["t_day"].fillna(0)))) * m)
    lt["mean_per_contract"] = lt["mean_net"] / cfg["execution"]["qty"]
    store = root / "research" / "v3"
    store.mkdir(parents=True, exist_ok=True)
    lt.to_parquet(store / f"grid_{und}.parquet", index=False)
    log.info("%s 网格 %d 行 %.0fs", und, len(lt), time.monotonic() - t0)
    cc = cfg["cards"]
    cand = lt[(lt["mean_net"] > 0) & (lt["n"] >= cc["min_n"]) & (lt["mean_oos"] > 0)].copy()
    cand = cand.sort_values("mean_net", ascending=False)
    cards = []
    if len(cand):
        collect = {(int(r.period), r.entry) for r in cand.itertuples()}
        for macd, key in (("6,13,6", "macd"), ("12,26,9", "macd_scan")):
            sub = cand[cand["macd"] == macd]
            if sub.empty:
                continue
            per = sorted(sub["period"].unique())
            col = {(p, e) for p, e in collect if p in per and e in set(sub["entry"])}
            rr = sv.run_grid(root, und, stock, cfg, per, days, macd_key=key, collect=col)
            for p in per:
                pdat = sv.prepare_period(stock, p, cfg, key)
                for (entry, _side), grp in sub[sub["period"] == p].groupby(["entry", "side"]):
                    tr = rr[p]["trades"].get(entry, [])
                    rcs = random_control_group(
                        root,
                        und,
                        stock,
                        cfg,
                        pdat,
                        grp,
                        tr,
                        n_random or cc["n_random"],
                        cc["random_window_min"],
                    )
                    for (_, card), rc in zip(grp.iterrows(), rcs, strict=True):
                        cards.append({**card.to_dict(), **rc})
    cards = pd.DataFrame(cards)
    return {
        "grid": lt,
        "cards": cards,
        "trials": tc,
        "secs": time.monotonic() - t0,
        "cfg": cfg,
        "events": {
            p: pd.concat(o["events"], ignore_index=True) if o["events"] else pd.DataFrame()
            for p, o in res.items()
        },
    }


CARD_COLS = [
    "period",
    "macd",
    "state",
    "entry",
    "side",
    "exit_rule",
    "tp",
    "sl",
    "cap",
    "hold_bucket",
    "hold_bucket_share",
    "n",
    "win_rate",
    "payoff",
    "mean_per_contract",
    "n_oos",
    "mean_oos",
    "rand_pctile",
    "rand_p50_per_contract",
    "t_day",
    "p_bonf",
]


def negatives_summary(lt: pd.DataFrame) -> pd.DataFrame:
    g = lt.groupby(["macd", "period", "family", "side"])
    return pd.DataFrame(
        {
            "rows": g.size(),
            "rows_n_ge_100": g.apply(lambda x: int((x["n"] >= 100).sum())),
            "rows_mean_gt_0": g.apply(lambda x: int((x["mean_net"] > 0).sum())),
            "share_mean_le_0": g.apply(lambda x: round(float((x["mean_net"] <= 0).mean()), 3)),
            "median_per_contract": g["mean_per_contract"].median().round(2),
            "best_per_contract_n_ge_100": g.apply(
                lambda x: (
                    round(float(x.loc[x["n"] >= 100, "mean_per_contract"].max()), 2)
                    if (x["n"] >= 100).any()
                    else np.nan
                )
            ),
        }
    ).reset_index()


def report(und: str, r: dict) -> str:
    lt, cards, tc = r["grid"], r["cards"], r["trials"]
    tb = float(norm.ppf(1 - 0.05 / (2 * tc["rows_total"])))
    days = sorted(lt["days"].max() for _ in [0]) if len(lt) else [0]
    L = [
        f"# 规则卡 · {und}（METHOD_V3 第 6 节）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 耗时 "
        f"{r['secs']:.0f}s · "
        f"复现 `uv run python -m zr.research.grid_v3 --symbol {und}.US`",
        "",
        "口径：真实期权 1 分钟 NBBO；入场 = 信号 K 收盘 + 2 秒后的第一个快照买 ask，结构离场 = "
        "触发后第一个快照"
        "卖 bid，X6 用 mid 判定、下一快照 bid 成交；2 张；D-013 费用；D-001；15:50 清仓；"
        "同标的同时一笔。"
        "「每张均值」= 每笔净额 ÷ 2。t_day = 按日聚类（每日合计）的 t；p_bonf = Bonferroni "
        "校正后的双侧 p。",
        "",
        f"边界：标的 {und}；期权日 2023-10-02～2026-10-21；样本外检查 = 2026 "
        "年子样本（同一网格选出，"
        "不是独立的前瞻验证）；未覆盖：SPY（暂停）、DTE > 5 的合约、盘前盘后。",
        "",
        "## 试验计数（预登记 R-030）",
        "",
        df_to_md(pd.DataFrame([tc])),
        "",
        f"Bonferroni 双侧 5% 门槛：|t| > {tb:.2f}（m = {tc['rows_total']}）。",
        "",
    ]
    L += [
        f"## 规则卡（扣费后均值 > 0、n ≥ {r['cfg']['cards']['min_n']}、2026 年仍为正）："
        f"{len(cards)} 张",
        "",
    ]
    if len(cards):
        L += [
            df_to_md(cards[CARD_COLS].round(3)),
            "",
            f"通过 Bonferroni（p_bonf < 0.05）的卡：{int((cards['p_bonf'] < 0.05).sum())} 张；"
            f"同时段随机对照分位 ≥ 95 的卡：{int((cards['rand_pctile'] >= 95).sum())} 张。",
            "",
        ]
    else:
        L += ["无。", ""]
    L += [
        "## 全部网格汇总（含为负的，避免只展示好的）",
        "",
        df_to_md(negatives_summary(lt)),
        "",
        f"总行数 {len(lt)}（n>0 的行）；扣费后均值 > 0 的行 {int((lt['mean_net'] > 0).sum())}；"
        f"其中 n ≥ 100 的 {int(((lt['mean_net'] > 0) & (lt['n'] >= 100)).sum())}；"
        f"规则卡 {len(cards)} 张；Bonferroni 门槛 |t| > {tb:.2f}。",
        "",
    ]
    del days
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MU.US")
    ap.add_argument("--n-random", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    und = a.symbol.split(".")[0]
    r = run(und, a.n_random)
    out = Path(__file__).resolve().parents[2] / "docs" / "research" / f"RULE_CARDS_{und}.md"
    out.write_text(report(und, r), encoding="utf-8")
    root = get_settings().zr_data_root
    r["cards"].to_parquet(root / "research" / "v3" / f"cards_{und}.parquet", index=False)
    for p, ev in r["events"].items():
        if len(ev):
            ev.to_parquet(root / "research" / "v3" / f"events_{und}_p{p}.parquet", index=False)
    log.info("写入 %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
