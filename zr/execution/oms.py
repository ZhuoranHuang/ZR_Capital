"""OMS：唯一允许向券商提交订单的入口（CLAUDE.md 规则 6）。

当前阶段（纸面助手 feat/guardrail-assistant）**不实现任何下单**：
- 未同时满足环境变量 ZR_LIVE=1 与命令行 --live → 拒绝（LiveGateError）；
- 即使两者都满足，也拒绝（NotImplementedError）：是否把离场单提交到长桥模拟盘 / 实盘待负责人确认
  （STATUS 待确认），本模块不调用 TradeContext.submit_order。
"""

from __future__ import annotations

import os


class LiveGateError(PermissionError):
    pass


def live_allowed(cli_live: bool) -> bool:
    return cli_live and os.environ.get("ZR_LIVE", "") == "1"


def submit_order(*_args, cli_live: bool = False, **_kwargs):
    if not live_allowed(cli_live):
        raise LiveGateError("下单需要同时满足 ZR_LIVE=1 与 --live（CLAUDE.md 规则 6）")
    raise NotImplementedError("纸面阶段不向券商提交订单；是否提交到模拟盘待负责人确认")
