"""R-042：负责人盈利是能力还是运气（预登记见 RESEARCH_LOG R-042）。

每笔真实入场换成同日 ±30 分钟内的随机时刻（同方向 / 方向随机两版）、同到期日、同虚实档位、同张数，
持仓上限从负责人真实持仓时长中抽样，出场按 R-052 选定组（+6.5% 止盈、−30% 止损），入场与出场用
负责人挂单策略 R-041，15:50 清仓吃买一，费用 D-013。期权盘口 = 本地 1 分钟 NBBO（options_1m），
不拉新数据。主比较 = 对账单合计在随机合计分布中的分位；同引擎对照 = 负责人真实入场在同一引擎下的
模拟合计的分位。
运行：uv run python -m zr.research.skill_vs_luck
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from zr.backtest import intrabar as ib
from zr.backtest.fill_limit import Book
from zr.config import get_settings
from zr.data.thetadata import lb_symbol
from zr.features.bars import ET
from zr.research import exit_finalize as xf
from zr.research import exit_mining as xm
from zr.research import owner_limit as ol

log = logging.getLogger("zr.research.skill_vs_luck")
NS = 1_000_000_000
N_DRAW = 1000
WIN = 30 * 60
SL = 30.0
SEED = 0
OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "SKILL_VS_LUCK.md"


def store(root: Path) -> Path:
    p = root / "research" / "skill_vs_luck"
    p.mkdir(parents=True, exist_ok=True)
    return p


class Day:
    """单日：1 秒网格秒级价、1 分钟报价盘口（按合约缓存）、每个到期日的行权价。"""

    def __init__(self, root: Path, day: dt.date):
        self.day = day
        g = ib.second_grid(root, "MU", day)
        self.t, self.S = g["t"].to_numpy(), pd.Series(g["S"].to_numpy()).ffill().to_numpy()
        q = pd.read_parquet(
            root / "options_1m" / "underlying=MU" / f"date={day}" / "quotes.parquet"
        )
        q["expiry"] = pd.to_datetime(q["expiry"]).dt.date
        self.q = {s: x for s, x in q.groupby("symbol")}
        self.strikes = {e: np.sort(x["strike"].unique()) for e, x in q.groupby("expiry")}
        self.flat = int(np.searchsorted(self.t, pd.Timestamp(f"{day} 15:50", tz=ET).value))
        self._books: dict = {}

    def book(self, sym: str):
        if sym not in self._books:
            x = self.q.get(sym)
            self._books[sym] = None if x is None or x.empty else Book.from_frames(x)
        return self._books[sym]


def steps(ks: np.ndarray, spot: float, k: float, right: str) -> int:
    """虚值档数：看涨 = 行权价在现价之上的第几档（1 = 第一档虚值），实值为负；看跌镜像。"""
    if right == "call":
        return (
            int((ks > spot).sum() - (ks > k).sum())
            if k > spot
            else -int(((ks > k) & (ks <= spot)).sum())
        )
    return (
        int((ks < spot).sum() - (ks < k).sum())
        if k < spot
        else -int(((ks < k) & (ks >= spot)).sum())
    )


def strike_at(ks: np.ndarray, spot: float, n: int, right: str) -> float | None:
    if right == "call":
        above, below = ks[ks > spot], ks[ks <= spot][::-1]
    else:
        above, below = ks[ks < spot][::-1], ks[ks >= spot]
    if n >= 1:
        return float(above[n - 1]) if len(above) >= n else None
    return (
        float(below[-n]) if len(below) > -n and n < 0 else (float(below[0]) if len(below) else None)
    )


def trade(D: Day, sym: str, k0: int, qty: float, cap_min: float, rng) -> float | None:
    """k0 = 决策时刻在秒网格中的行；返回扣费后盈亏（USD），未成交返回 0，无盘口返回 None。"""
    bk = D.book(sym)
    if bk is None:
        return None, False
    n, px0, te, _ = ib.place(
        bk, 1, qty, int(D.t[k0]), {"mode": "owner"}, sym, 0.9, int(D.t[D.flat]), rng
    )
    if n <= 0 or te is None:
        return 0.0, False
    i0 = int(np.searchsorted(D.t, te, side="right"))
    if i0 >= D.flat:
        i0 = D.flat - 1
    f = xm.state_arrays(bk, D.t, i0, px0)["st_float_pct"]
    j, why = xf.exit_index(f, i0, D.flat, SL, cap_min)
    tdec = int(D.t[min(j, len(D.t) - 1)])
    order = {"mode": "taker", "lag_s": 2.0} if why == "flat" else {"mode": "owner"}
    xn, px1, _, _ = ib.place(bk, -1, n, tdec, order, sym, 0.9, None, rng)
    if xn < n:
        c = ib.cross_now(bk, -1, tdec, 2.0)
        px1 = (
            ((px1 * xn if xn else 0.0) + c[1] * (n - xn)) / n
            if c
            else (px1 if np.isfinite(px1) else 0.0)
        )
    return (
        -px0 * 100 * n - ib.fees(n, px0, "Buy") + px1 * 100 * n - ib.fees(n, px1, "Sell"),
        True,
    )


def setup(root: Path) -> tuple[list[dict], np.ndarray]:
    sel, orders = ol.episode_orders(root)
    eps = xm.episodes(root)
    sub = orders[(orders["side"] == "Buy")].groupby("episode")["submit_ns"].min()
    out = []
    for e in eps:
        c = ib.parse_contract(e["symbol"])
        q = root / "options_1m" / "underlying=MU" / f"date={e['day']}" / "quotes.parquet"
        exps = pd.to_datetime(pd.read_parquet(q, columns=["expiry"])["expiry"]).dt.date.unique()
        e = dict(e)
        e["expiry"], e["strike"], e["right"] = c["expiry"], float(c["strike"]), c["right"]
        e["submit_ns"] = int(sub.loc[e["episode"]])
        et = pd.Timestamp(e["entry_ns"], tz="UTC").tz_convert(ET)
        e["group"] = "A" if (c["expiry"] == e["day"] and et.time() > dt.time(10, 15)) else "B"
        e["covered"] = c["expiry"] in set(exps)
        out.append(e)
    holds = sel[sel["und"] == "MU"]["hold_min"].to_numpy(float)
    return out, holds


def prepare(root: Path, eps: list[dict]) -> dict:
    days = {e["day"]: Day(root, e["day"]) for e in eps if e["covered"]}
    for e in eps:
        if not e["covered"]:
            continue
        D = days[e["day"]]
        k = int(np.searchsorted(D.t, e["submit_ns"], side="right")) - 1
        e["k"] = max(k, 0)
        e["n_steps"] = steps(D.strikes[e["expiry"]], float(D.S[e["k"]]), e["strike"], e["right"])
        lo = int(np.searchsorted(D.t, pd.Timestamp(f"{e['day']} 09:32", tz=ET).value))
        hi = int(np.searchsorted(D.t, pd.Timestamp(f"{e['day']} 15:40", tz=ET).value))
        e["lo"], e["hi"] = max(lo, e["k"] - WIN), min(hi, e["k"] + WIN)
    return days


def one_draw(root: Path, eps: list[dict], holds: np.ndarray, draw: int, random_dir: bool) -> dict:
    rng = np.random.default_rng([SEED, draw, int(random_dir)])
    days: dict = {}
    tot = {"A": 0.0, "B": 0.0, "fill_A": 0, "fill_B": 0}
    for e in eps:
        if not e["covered"]:
            continue
        if e["day"] not in days:
            days[e["day"]] = Day(root, e["day"])
        D = days[e["day"]]
        k = int(rng.integers(e["lo"], e["hi"] + 1))
        d = int(rng.choice([1, -1])) if random_dir else e["dir"]
        right = "call" if d == 1 else "put"
        K = strike_at(D.strikes[e["expiry"]], float(D.S[k]), e["n_steps"], right)
        cap = float(rng.choice(holds))
        pnl = (
            None
            if K is None
            else trade(D, lb_symbol("MU", e["expiry"], right, K), k, e["qty"], cap, rng)
        )
        tot[e["group"]] += 0.0 if pnl is None else pnl
    return tot


def owner_same_engine(root: Path, eps: list[dict], days: dict) -> dict:
    rng = np.random.default_rng(SEED)
    tot = {"A": 0.0, "B": 0.0, "fill_A": 0, "fill_B": 0}
    for e in eps:
        if not e["covered"]:
            continue
        D = days[e["day"]]
        hold = (max(e["sells"]) - e["entry_ns"]) / NS / 60
        pnl, filled = trade(D, e["symbol"], e["k"], e["qty"], hold, rng)
        tot[e["group"]] += 0.0 if pnl is None else pnl
        tot[f"fill_{e['group']}"] += int(filled)
    return tot


def run(root: Path, jobs: int = 10) -> dict:
    eps, holds = setup(root)
    days = prepare(root, eps)
    res = {}
    for rd in (False, True):
        draws = Parallel(n_jobs=jobs)(
            delayed(one_draw)(root, eps, holds, k, rd) for k in range(N_DRAW)
        )
        res["random_dir" if rd else "same_dir"] = pd.DataFrame(draws)
    res["owner_sim"] = owner_same_engine(root, eps, days)
    res["actual"] = {
        g: sum(e["pnl"] for e in eps if e["covered"] and e["group"] == g) for g in ("A", "B")
    }
    res["eps"] = pd.DataFrame(
        [
            {k: e[k] for k in ("episode", "day", "group", "covered", "dir", "qty", "pnl")}
            for e in eps
        ]
    )
    return res


def verdict(p: float) -> str:
    return "能力" if p >= 0.95 else ("不确定" if p >= 0.5 else "运气")


def table(res: dict) -> pd.DataFrame:
    rows = []
    for ver, lab in (("same_dir", "同方向"), ("random_dir", "方向随机")):
        d = res[ver].assign(
            合并=lambda x: x["A"] + x["B"], fill_合并=lambda x: x["fill_A"] + x["fill_B"]
        )
        os_ = dict(res["owner_sim"])
        os_["fill_合并"] = os_["fill_A"] + os_["fill_B"]
        for g, name in (("A", "A：10:15 后 0DTE"), ("B", "B：其余"), ("合并", "合并")):
            act = res["actual"]["A"] + res["actual"]["B"] if g == "合并" else res["actual"][g]
            sim = (
                res["owner_sim"]["A"] + res["owner_sim"]["B"]
                if g == "合并"
                else res["owner_sim"][g]
            )
            x = d[g].to_numpy()
            pa, ps = float((x < act).mean()), float((x < sim).mean())
            rows.append(
                {
                    "版本": lab,
                    "组": name,
                    "笔数": int(
                        (res["eps"]["covered"] & ((res["eps"]["group"] == g) | (g == "合并"))).sum()
                    ),
                    "对账单合计": act,
                    "随机中位数": float(np.median(x)),
                    "随机 P5": float(np.quantile(x, 0.05)),
                    "随机 P95": float(np.quantile(x, 0.95)),
                    "对账单分位": pa,
                    "判断（对账单）": verdict(pa),
                    "同引擎负责人模拟合计": sim,
                    "同引擎分位": ps,
                    "判断（同引擎）": verdict(ps),
                    "随机平均成交笔数": float(d[f"fill_{g}"].mean()),
                    "同引擎负责人成交笔数": int(os_[f"fill_{g}"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    res = run(root)
    st = store(root)
    for k in ("same_dir", "random_dir"):
        res[k].to_parquet(st / f"{k}.parquet", index=False)
    t = table(res)
    t.to_parquet(st / "table.parquet", index=False)
    pd.to_pickle({k: res[k] for k in ("owner_sim", "actual")}, st / "totals.pkl")
    res["eps"].to_parquet(st / "eps.parquet", index=False)
    print(t.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
