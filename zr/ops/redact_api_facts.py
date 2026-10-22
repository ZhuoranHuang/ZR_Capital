"""脱敏 docs/API_FACTS.md：账户余额、持仓数量、订单号替换为 [REDACTED]。

用法：uv run python -m zr.ops.redact_api_facts docs/API_FACTS.md
原地改写；敏感小节内所有数字与 *_id 字符串都替换。另在 stderr 打印账户规模数量级（10^n），
不打印具体金额，供判断模拟 / 真实账户。
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

SENSITIVE_SECTIONS = (
    "account_balance",
    "stock_positions",
    "today_orders",
    "history_executions",
    "estimate_max_purchase_quantity",
)
NUM_RE = re.compile(r"(?<=[:\[,\s])-?\d+(?:\.\d+)?(?=[,\]\s}])")
ID_RE = re.compile(r'("(?:order_id|trade_id|account_channel)":\s*)"[^"]*"')


def _magnitude(value: float) -> str:
    if value == 0:
        return "0"
    return f"10^{int(math.floor(math.log10(abs(value))))}"


def _balance_magnitudes(body: str) -> dict[str, str]:
    """从 account_balance 小节 JSON 中取 net_assets / total_cash 的数量级。字段名未验证。"""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {}
    items = data if isinstance(data, list) else [data]
    out: dict[str, str] = {}
    for i, acc in enumerate(items):
        if not isinstance(acc, dict):
            continue
        for key in ("net_assets", "total_cash", "max_finance_amount", "buy_power"):
            v = acc.get(key)
            if isinstance(v, int | float):
                out[f"[{i}].{key}"] = _magnitude(v)
        cur = acc.get("currency")
        if cur:
            out[f"[{i}].currency"] = str(cur)
    return out


def redact(text: str) -> tuple[str, dict[str, str]]:
    sections = re.split(r"(?m)^(?=### )", text)
    magnitudes: dict[str, str] = {}
    out = []
    for sec in sections:
        title = sec.split("\n", 1)[0]
        if any(k in title for k in SENSITIVE_SECTIONS):
            m = re.search(r"```json\n(.*?)\n```", sec, re.S)
            if m and "account_balance" in title:
                magnitudes.update(_balance_magnitudes(m.group(1)))
            if m:
                body = ID_RE.sub(r'\1"[REDACTED]"', m.group(1))
                body = NUM_RE.sub("[REDACTED]", body)
                sec = sec[: m.start(1)] + body + sec[m.end(1) :]
        out.append(sec)
    return "".join(out), magnitudes


def main(argv: list[str] | None = None) -> int:
    path = Path((argv or sys.argv[1:])[0])
    text = path.read_text(encoding="utf-8")
    new, mags = redact(text)
    path.write_text(new, encoding="utf-8")
    print(f"redacted -> {path}", file=sys.stderr)
    for k, v in mags.items():
        print(f"magnitude {k} = {v}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
