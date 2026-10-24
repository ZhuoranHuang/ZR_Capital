"""R-052：出场定型（预登记见 RESEARCH_LOG R-052）。

第 1 步（样本内）：负责人 48 笔 MU 真实入场上替换出场——期权中间价 +6.5% 全卖；
止损 X ∈ {20%, 30%, 40%, 不设}；持仓上限 Y ∈ {30 分钟, 60 分钟, 到 15:50}；止盈 /
止损 / 上限都按负责人挂单策略 R-041 卖出，15:50 清仓吃买一。在 9 组有止损的组合中
选扣费后合计最接近实际 +9,734.90 的一组。
第 2 步：选定出场套用到 R-049 的 owner_rule_1..5 与调研三条（期权层，D-001，负责人
挂单 / 吃价差），只用已缓存的 330 个交易日（owner_rules_oos.CachedOptionData，不拉
新数据），BH q = 0.10（16 格）。
运行：uv run python -m zr.research.exit_finalize [--stage grid|apply|all]
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from zr.backtest import intrabar as ib
from zr.config import get_settings
from zr.features.bars import ET
from zr.research import exit_mining as xm
from zr.research import owner_rules_oos as o
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.exit_finalize")
NS = 1_000_000_000
TP = 6.5
SLS = (20.0, 30.0, 40.0, None)
CAPS = (30.0, 60.0, None)
ACTUAL = 9734.90
RULES = (
    "owner_rule_1",
    "owner_rule_2",
    "owner_rule_3",
    "owner_rule_4",
    "owner_rule_5",
    "survey_rsi2_pullback",
    "survey_ema_pullback",
    "survey_orb_breakout",
)


def store(root: Path) -> Path:
    p = root / "research" / "exit_finalize"
    (p / "option_days").mkdir(parents=True, exist_ok=True)
    return p


def exit_index(f: np.ndarray, start: int, stop: int, sl, cap) -> tuple[int, str]:
    """f = 逐秒浮盈 %；返回 (出场行, 原因)。"""
    seg = f[start:stop]
    hit_tp = np.flatnonzero(seg >= TP)
    hit_sl = np.flatnonzero(seg <= -sl) if sl is not None else np.array([], int)
    c = {"tp": hit_tp[0] if len(hit_tp) else None, "sl": hit_sl[0] if len(hit_sl) else None}
    if cap is not None and int(cap * 60) < stop - start:
        c["cap"] = int(cap * 60)
    c = {k: v for k, v in c.items() if v is not None}
    if not c:
        return stop, "flat"
    k = min(c, key=c.get)
    return start + int(c[k]), k


def grid(root: Path, seed: int = 0) -> pd.DataFrame:
    eps = xm.episodes(root)
    od = ib.OptionData(root, "MU")
    rng = np.random.default_rng(seed)
    rows = []
    for e in eps:
        day = e["day"]
        flat_ns = pd.Timestamp(f"{day} 15:50", tz=ET).value
        t = np.arange(pd.Timestamp(f"{day} 09:30", tz=ET).value, flat_ns + NS, NS)
        bk = od.book(
            e["symbol"],
            day,
            pd.Timestamp(e["entry_ns"], tz="UTC") - pd.Timedelta(minutes=1),
            pd.Timestamp(flat_ns, tz="UTC") + pd.Timedelta(minutes=3),
        )
        if bk is None:
            continue
        i0 = int(np.searchsorted(t, e["entry_ns"], side="right"))
        f = xm.state_arrays(bk, t, i0, e["entry_px"])["st_float_pct"]
        stop = len(t) - 1
        for sl, cap in itertools.product(SLS, CAPS):
            j, why = exit_index(f, i0, stop, sl, cap)
            tdec = int(t[j])
            order = {"mode": "taker", "lag_s": 2.0} if why == "flat" else {"mode": "owner"}
            n, px, _, _ = ib.place(bk, -1, e["qty"], tdec, order, e["symbol"], 0.9, None, rng)
            if n < e["qty"]:
                c = ib.cross_now(bk, -1, tdec, 2.0)
                px = ((px * n if n else 0.0) + c[1] * (e["qty"] - n)) / e["qty"] if c else px
            cash = (
                -e["entry_px"] * 100 * e["qty"]
                - e["buy_fees"]
                + px * 100 * e["qty"]
                - ib.fees(e["qty"], px, "Sell")
            )
            rows.append(
                {
                    "episode": e["episode"],
                    "sl": sl,
                    "cap": cap,
                    "reason": why,
                    "hold_min": (tdec - e["entry_ns"]) / NS / 60,
                    "sim": cash,
                    "actual": e["pnl"],
                }
            )
    return pd.DataFrame(rows)


def grid_table(g: pd.DataFrame) -> pd.DataFrame:
    t = (
        g.assign(sl=g["sl"].fillna(-1), cap=g["cap"].fillna(-1))
        .groupby(["sl", "cap"])
        .agg(
            total=("sim", "sum"),
            win=("sim", lambda x: (x > 0).mean()),
            hold_mean=("hold_min", "mean"),
            hold_med=("hold_min", "median"),
            n=("sim", "size"),
        )
        .reset_index()
    )
    t["diff_vs_actual"] = t["total"] - ACTUAL
    return t


def choose(t: pd.DataFrame) -> dict:
    ok = t[t["sl"] > 0]
    r = ok.loc[(ok["diff_vs_actual"]).abs().idxmin()]
    return {"sl": float(r["sl"]), "cap": None if r["cap"] < 0 else float(r["cap"])}


def rule_with_exit(rule: dict, ex: str, chosen: dict) -> dict:
    r = ib.with_execution(rule, ex, "d001")
    r["exit"] = {
        "tp": TP / 100,
        "sl": chosen["sl"] / 100,
        "max_hold_min": chosen["cap"] if chosen["cap"] is not None else 10_000,
        "order": r["exit"]["order"],
        "sl_order": "same",
    }
    return r


def _shard(root: Path, rules: dict, days: list, chosen: dict, seed: int) -> list:
    bs = ib.BarState(root, "MU", max(days))
    od = o.CachedOptionData(root, "MU")
    ck = store(root) / "option_days"
    out = []
    for day in days:
        p = ck / f"{day}.parquet"
        if p.exists():
            out += pd.read_parquet(p).to_dict("records")
            continue
        g = ib.second_grid(root, "MU", day)
        if g is None:
            continue
        rows = []
        try:
            for name, rule in rules.items():
                co = o.load_conds(root, day, name, len(g))
                if co is None or not any(v.any() for v in co.values()):
                    continue
                for ex in o.EXECUTIONS:
                    r = rule_with_exit(rule, ex, chosen)
                    for tr in ib.run_day(root, r, day, bs, od, seed, None, co):
                        rows.append({"rule": name, "exec": ex, **tr})
        except o.MissingData as e:
            log.info("%s 跳过：%s", day, e)
            continue
        pd.DataFrame(rows, columns=None if rows else ["rule", "exec", "day"]).to_parquet(
            p, index=False
        )
        out += rows
    return out


def apply(root: Path, chosen: dict, jobs: int, seed: int = 0) -> pd.DataFrame:
    rules = {k: v for k, v in o.load_rules().items() if k in RULES}
    used = sorted(
        pd.Timestamp(f.stem).date() for f in (o.store(root) / "option_days").glob("*.parquet")
    )
    ib.check_out_of_sample(used)
    shards = [used[k::jobs] for k in range(jobs)]
    parts = Parallel(n_jobs=jobs)(delayed(_shard)(root, rules, sh, chosen, seed) for sh in shards)
    df = pd.DataFrame([x for p in parts for x in p])
    df.assign(day=df["day"].astype(str)).to_parquet(store(root) / "option.parquet", index=False)
    return df


def apply_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rule, ex), sub in df.groupby(["rule", "exec"]):
        rows.append({"rule": rule, "exec": ex, **o.option_cell(sub)})
    t = pd.DataFrame(rows)
    t["bh"] = v4d.bh(t["p"].fillna(1).to_numpy(), 0.10)
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    st = store(root)
    if args.stage in ("grid", "all"):
        g = grid(root)
        g.to_parquet(st / "grid.parquet", index=False)
        t = grid_table(g)
        t.to_parquet(st / "grid_table.parquet", index=False)
        pd.to_pickle(choose(t), st / "chosen.pkl")
        print(t.round(2).to_string())
        print("选定", pd.read_pickle(st / "chosen.pkl"))
    if args.stage in ("apply", "all"):
        chosen = pd.read_pickle(st / "chosen.pkl")
        df = apply(root, chosen, args.jobs)
        t = apply_table(df)
        t.to_parquet(st / "apply_table.parquet", index=False)
        print(t.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
