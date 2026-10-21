# 长桥 API 核验结果

生成时间（UTC）：2026-10-21T16:15:42+00:00  
生成脚本：`zr/ops/api_facts.py`，不手改。
每项为实际调用返回或实际异常；未出现在本文件的 API 行为视为「未验证」。
### make_quote_ctx



```json
"<builtins.QuoteContext object at 0x107916a30>"
```


## 行情包与交易日

### quote_level

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### quote_package_details

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### trading_session

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### trading_days(US, 近 20 天)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```


## 正股 quote / depth

### quote(正股)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### depth(SPY.US)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### depth(QQQ.US)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### depth(MU.US)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### depth(AMD.US)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```


## 1 分钟 bar 可回溯起点

### earliest_minute_bar(SPY.US)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### earliest_minute_bar(QQQ.US)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### earliest_minute_bar(MU.US)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### earliest_minute_bar(AMD.US)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```


## 期权链到期日

### option_chain_expiry_date_list(SPY.US) 最近 6 个

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### option_chain_expiry_date_list(QQQ.US) 最近 6 个

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### option_chain_expiry_date_list(MU.US) 最近 6 个

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### option_chain_expiry_date_list(AMD.US) 最近 6 个

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```


## SPY 最近到期 ATM call

### nearest_atm_call(SPY.US)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)
```

### make_trade_ctx



```json
"<builtins.TradeContext object at 0x1070ea700>"
```


## 交易账户（只读）

### account_balance

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/asset/account?): client error (Connect)
```

### stock_positions

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/asset/stock?): client error (Connect)
```

### today_orders

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/trade/order/today?): client error (Connect)
```

### history_executions(近 30 天)

**失败** `OpenApiException` code=None

```
OpenApiException: error sending request for url (https://openapi.longport.cn/v1/trade/execution/history?start_at=1790007342&end_at=1792599342): client error (Connect)
```
