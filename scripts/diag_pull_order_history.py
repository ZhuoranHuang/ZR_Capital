"""一次性诊断拉取（feat/fill-model 第 1 节）：期权订单的修改历史（order_detail.history），
只读不下单。

orders.parquet 只存订单的最终价格；改价前的挂单价只在 order_detail 的 history 里
（2026-10-22 VPS 实测：OrderDetail 有 history 属性）。写到
$ZR_DATA_ROOT/diag/order_history.parquet。
只在 VPS 跑（长桥 API，本机 DNS 污染）。每次调用间隔 ≥ 1.05 s。

uv run python scripts/diag_pull_order_history.py
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from zr.broker.longport_client import make_trade_ctx
from zr.broker.serialize import to_py
from zr.config import get_settings

log = logging.getLogger("diag_order_history")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    o = pd.read_parquet(root / "journal" / "orders.parquet")
    o = o[o["symbol"].str.contains(r"\d{6}[CP]\d+\.US$", regex=True)]
    ctx = make_trade_ctx()
    rows, first = [], True
    for oid in o["order_id"].astype(str):
        d = ctx.order_detail(oid)
        if first:
            log.info("order_detail 属性：%s", [a for a in dir(d) if not a.startswith("_")])
            first = False
        for h in getattr(d, "history", None) or []:
            r = to_py(h)
            r = (
                r
                if isinstance(r, dict)
                else {
                    a: to_py(getattr(h, a))
                    for a in dir(h)
                    if not a.startswith("_") and not callable(getattr(h, a))
                }
            )
            rows.append({"order_id": oid, **r})
        time.sleep(1.05)
    out = pd.DataFrame(rows)
    p = root / "diag" / "order_history.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    log.info("订单 %d，历史行 %d，列 %s -> %s", len(o), len(out), list(out.columns), p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
