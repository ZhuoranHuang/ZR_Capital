"""长桥 OpenAPI 连接工厂（longport>=4.3）。

已核实（longport 4.3.7）：
- Config.from_apikey(app_key, app_secret, access_token)；不存在 Config.from_env()
- OptionQuote 没有 bid/ask 字段。期权盘口候选来源是 QuoteContext.depth(symbol)
  或 SubType.Depth 推送，每档字段 price / volume / order_num / position
  （未验证是否接受期权代码）
- history_candlesticks_by_offset(symbol, Period, AdjustType, forward, count, time, TradeSessions)
- option_chain_expiry_date_list(symbol)
- option_chain_info_by_date(symbol, date) 返回 [price, call_symbol, put_symbol]

TradeContext 只允许被 zr/execution 的 OMS 使用；本模块不提供任何下单函数。
"""

from longport.openapi import Config, QuoteContext, TradeContext

from zr.config import Settings, get_settings


class MissingLongportKeysError(RuntimeError):
    pass


def make_config(settings: Settings | None = None) -> Config:
    s = settings or get_settings()
    if not s.has_longport_keys:
        raise MissingLongportKeysError(
            "缺少长桥密钥：请在 .env 中设置 LONGPORT_APP_KEY、LONGPORT_APP_SECRET、"
            "LONGPORT_ACCESS_TOKEN（参考 .env.example）"
        )
    assert s.longport_app_key and s.longport_app_secret and s.longport_access_token
    return Config.from_apikey(
        s.longport_app_key.get_secret_value(),
        s.longport_app_secret.get_secret_value(),
        s.longport_access_token.get_secret_value(),
    )


def make_quote_ctx(settings: Settings | None = None) -> QuoteContext:
    # QuoteContext(config) 构造方式：未验证（不在已核实清单内）
    return QuoteContext(make_config(settings))


def make_trade_ctx(settings: Settings | None = None) -> TradeContext:
    # TradeContext(config) 构造方式：未验证（不在已核实清单内）
    return TradeContext(make_config(settings))
