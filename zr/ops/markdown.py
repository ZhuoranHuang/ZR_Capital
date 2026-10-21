"""DataFrame → markdown 表格，避免引入 tabulate。"""

from __future__ import annotations

import math

import pandas as pd


def _cell(v: object) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-2 and v != 0 else f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v).replace("|", "\\|").replace("\n", " ")


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "（空）"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)
