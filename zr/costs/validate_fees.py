"""用交易史验证费用公式：对每个有成交的期权订单，逐项比较实际费用与公式。

两个假设：
  H1 按订单：最低收费对整单生效一次，n = 订单成交张数
  H2 按成交笔数：每笔成交单独计费并各自取最低，再求和
输出 docs/FEE_VALIDATION.md。运行：uv run python -m zr.costs.validate_fees
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

import pandas as pd

from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.ops.markdown import df_to_md

FEE_CODES = [
    "OptionsCommission",
    "OptionsPlatformFee",
    "OptionsExchangeFee",
    "OptionsRegulatoryFee",
    "CATFee",
    "OptionsClearingFee",
    "OptionsSFCFee",
    "OptionsTradingActivityFee",
]


def load_journal(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    j = root / "journal"
    return (
        pd.read_parquet(j / "orders.parquet"),
        pd.read_parquet(j / "executions.parquet"),
        pd.read_parquet(j / "charges.parquet"),
    )


def actual_by_order(charges: pd.DataFrame) -> pd.DataFrame:
    piv = charges.pivot_table(
        index="order_id", columns="fee_code", values="amount", aggfunc="sum", fill_value=0.0
    )
    for c in FEE_CODES:
        if c not in piv:
            piv[c] = 0.0
    piv["total_actual"] = charges.drop_duplicates("order_id").set_index("order_id")["total_amount"]
    return piv[FEE_CODES + ["total_actual"]]


def formula_by_order(
    orders: pd.DataFrame, execs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (H1 按订单, H2 按成交) 的逐项公式费用。"""
    side = orders.set_index("order_id")["side"]
    h1_rows, h2_rows = {}, {}
    for oid, g in execs.groupby("order_id"):
        s = side.get(oid, "Buy")
        n = int(g["quantity"].sum())
        vwap = float((g["price"] * g["quantity"]).sum() / g["quantity"].sum())
        f1 = option_order_fees(n, s, vwap)
        h1_rows[oid] = {k: float(f1.get(k, Decimal(0))) for k in FEE_CODES} | {
            "total_formula": float(f1["total"]),
            "n_execs": len(g),
            "n_contracts": n,
            "side": s,
        }
        acc = dict.fromkeys(FEE_CODES, Decimal(0))
        tot = Decimal(0)
        for _, e in g.iterrows():
            f = option_order_fees(int(e["quantity"]), s, float(e["price"]))
            for k in FEE_CODES:
                acc[k] += f.get(k, Decimal(0))
            tot += f["total"]
        h2_rows[oid] = {k: float(acc[k]) for k in FEE_CODES} | {"total_formula": float(tot)}
    return pd.DataFrame.from_dict(h1_rows, orient="index"), pd.DataFrame.from_dict(
        h2_rows, orient="index"
    )


def compare(actual: pd.DataFrame, formula: pd.DataFrame) -> pd.DataFrame:
    idx = actual.index.intersection(formula.index)
    diff = pd.DataFrame(index=idx)
    for c in FEE_CODES:
        diff[c] = (actual.loc[idx, c] - formula.loc[idx, c]).round(2)
    diff["total"] = (actual.loc[idx, "total_actual"] - formula.loc[idx, "total_formula"]).round(2)
    return diff


def report(root: Path) -> str:
    orders, execs, charges = load_journal(root)
    actual = actual_by_order(charges)
    h1, h2 = formula_by_order(orders, execs)
    d1, d2 = compare(actual, h1), compare(actual, h2)
    multi = h1.index[h1["n_execs"] > 1]

    def summary(d: pd.DataFrame) -> list[str]:
        lines = [
            "| 项目 | 残差=0 的订单 | 残差均值 | 残差绝对值和 | 最大残差 |",
            "|---|---|---|---|---|",
        ]
        for c in FEE_CODES + ["total"]:
            col = d[c]
            lines.append(
                f"| {c} | {(col == 0).sum()}/{len(col)} | {col.mean():.4f} | "
                f"{col.abs().sum():.2f} | {col.abs().max():.2f} |"
            )
        return lines

    out = [
        "# 费用公式验证",
        "",
        f"订单 {len(actual)} 笔（有成交），其中多笔成交订单 {len(multi)} 笔。",
        "残差 = 实际 − 公式。脚本：`uv run python -m zr.costs.validate_fees`。",
        "",
        "## H1 按订单计最低收费（n = 整单成交张数）",
        "",
        *summary(d1),
        "",
        "## H2 按成交笔数各自计最低收费后求和",
        "",
        *summary(d2),
        "",
        "## 多笔成交订单上的对比",
        "",
        f"- H1 total 残差绝对值和：{d1.loc[multi, 'total'].abs().sum():.2f}",
        f"- H2 total 残差绝对值和：{d2.loc[multi, 'total'].abs().sum():.2f}",
        "",
    ]
    worse = d1.loc[d1["total"] != 0].copy()
    if len(worse):
        worse = worse.join(h1[["n_execs", "n_contracts", "side"]])
        out += [
            "## H1 total 残差 ≠ 0 的订单（订单号脱敏）",
            "",
            df_to_md(worse.reset_index(drop=True)),
            "",
        ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/FEE_VALIDATION.md")
    args = parser.parse_args(argv)
    text = report(get_settings().zr_data_root)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
