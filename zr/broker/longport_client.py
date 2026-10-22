"""长桥 OpenAPI 连接工厂（longport>=4.3）。

已核实（longport 4.3.7）：
- Config.from_apikey(app_key, app_secret, access_token)；不存在 Config.from_env()
- OptionQuote 没有 bid/ask 字段。期权盘口候选来源是 QuoteContext.depth(symbol)
  或 SubType.Depth 推送，每档字段 price / volume / order_num / position
  （未验证是否接受期权代码）
- history_candlesticks_by_offset(symbol, Period, AdjustType, forward, count, time, TradeSessions)
- option_chain_expiry_date_list(symbol)
- option_chain_info_by_date(symbol, date) 返回 [price, call_symbol, put_symbol]

实盘 / 模拟盘 token 分离（2026-10-22 负责人指令）：
- .env 分两个字段：LONGPORT_LIVE_ACCESS_TOKEN（实盘）、LONGPORT_PAPER_ACCESS_TOKEN（模拟盘）；
  App Key / App Secret 两种模式共用。旧字段 LONGPORT_ACCESS_TOKEN 作为 LIVE 的兼容别名
  （仍可用，但会在连接时打印改名提示；不读取、不打印 token 的值本身）。
- 连接时解码 access_token 的 JWT payload，读 `ac` 字段校验账户类型：paper 模式要求
  ac == "lb_papertrading"，live 模式要求 ac != "lb_papertrading"，不符则拒绝连接
  （**未验证**：token 为 JWT 且含 `ac` 字段这一事实来自负责人说法，本代理未独立核实长桥官方
  文档或反编译 SDK 核实；只做 payload 段的 base64url 解码读字段，不校验签名——没有验签密钥，
  且这只是客户端自检，不是安全边界）。
- 默认 mode="live"：行情与只读脚本（api_facts、recorder、trade_history、backfill_bars、universe）
  不传 mode 参数即可工作。只有未来的模拟盘下单脚本（paper_runner，尚未实现）需要显式传
  mode="paper"。

TradeContext 只允许被 zr/execution 的 OMS 使用；本模块不提供任何下单函数。
"""

from __future__ import annotations

import base64
import json
import warnings
from typing import Literal

from longport.openapi import Config, QuoteContext, TradeContext

from zr.config import Settings, get_settings

Mode = Literal["live", "paper"]
PAPER_AC = "lb_papertrading"


class MissingLongportKeysError(RuntimeError):
    pass


class LongportModeMismatchError(RuntimeError):
    """access_token 的账户类型（JWT ac 字段）与请求的 mode 不符。"""


def _jwt_claim(token: str, claim: str) -> str | None:
    """读 JWT payload 段的某个字段，不校验签名（无验签密钥，仅客户端自检）。

    不是合法的 3 段式 JWT、或该字段不存在时返回 None（调用方据此按「无法确认」处理）。
    """
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded))
    except Exception:  # noqa: BLE001 - 解码失败即视为无法确认，不是程序错误
        return None
    v = data.get(claim)
    return v if isinstance(v, str) else None


def make_config(mode: Mode = "live", settings: Settings | None = None) -> Config:
    s = settings or get_settings()
    if mode not in ("live", "paper"):
        raise ValueError(f"mode 必须是 'live' 或 'paper'：{mode!r}")
    if not s.has_longport_keys(mode):
        need = (
            "LONGPORT_PAPER_ACCESS_TOKEN"
            if mode == "paper"
            else ("LONGPORT_LIVE_ACCESS_TOKEN（或兼容别名 LONGPORT_ACCESS_TOKEN）")
        )
        raise MissingLongportKeysError(
            f"缺少长桥密钥（mode={mode}）：请在 .env 中设置 LONGPORT_APP_KEY、"
            f"LONGPORT_APP_SECRET、{need}（参考 .env.example）"
        )
    if mode == "live" and s.using_legacy_access_token_name:
        warnings.warn(
            "LONGPORT_ACCESS_TOKEN 已更名为 LONGPORT_LIVE_ACCESS_TOKEN，请更新 .env；"
            "当前仍按兼容别名读取，不影响运行。",
            stacklevel=2,
        )
    assert s.longport_app_key and s.longport_app_secret
    token = s.access_token_for(mode)
    assert token
    token_value = token.get_secret_value()
    ac = _jwt_claim(token_value, "ac")
    is_paper_token = ac == PAPER_AC
    if mode == "paper" and not is_paper_token:
        raise LongportModeMismatchError(
            f"paper 模式要求 token 的 JWT ac 字段为 {PAPER_AC!r}，实际读到 {ac!r}"
            "（None 表示无法解码/字段不存在）；拒绝连接，避免模拟盘代码连到实盘账户。"
        )
    if mode == "live" and is_paper_token:
        raise LongportModeMismatchError(
            f"live 模式检测到模拟盘 token（JWT ac={PAPER_AC!r}）；拒绝连接，"
            "避免实盘代码连到模拟盘账户。"
        )
    return Config.from_apikey(
        s.longport_app_key.get_secret_value(),
        s.longport_app_secret.get_secret_value(),
        token_value,
    )


def make_quote_ctx(mode: Mode = "live", settings: Settings | None = None) -> QuoteContext:
    # QuoteContext(config) 构造方式：未验证（不在已核实清单内）
    return QuoteContext(make_config(mode, settings))


def make_trade_ctx(mode: Mode = "live", settings: Settings | None = None) -> TradeContext:
    # TradeContext(config) 构造方式：未验证（不在已核实清单内）
    return TradeContext(make_config(mode, settings))
