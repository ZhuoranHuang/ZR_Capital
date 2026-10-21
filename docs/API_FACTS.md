# 长桥 API 核验结果

生成时间（UTC）：2026-10-21T18:43:45+00:00  
生成脚本：`zr/ops/api_facts.py`，不手改。
每项为实际调用返回或实际异常；未出现在本文件的 API 行为视为「未验证」。
### make_quote_ctx



```json
"<builtins.QuoteContext object at 0x7af25c89ff90>"
```


## 行情包与交易日

### quote_level



```json
"HKBK:OpenAPI|HKBK|Global|LV0;HKAB:OpenAPI|HKAB|Global|LV2;USAB:OpenAPI|USAB|Global|Delay;SHBK:OpenAPI|SHBK|Global|Delay;USAA:OpenAPI|USAA|Global|NBBO;USZD:OpenAPI|USZD|Global|LV0;SZAC:OpenAPI|SZAC|Global|Delay;HKZA:OpenAPI|HKZA|Global|LV0;SHBA:OpenAPI|SHBA|Global|Delay;SHAD:OpenAPI|SHAD|Global|Delay;USBK:OpenAPI|USBK|Global|LV0;SHAC:OpenAPI|SHAC|Global|Delay;SZAB:OpenAPI|SZAB|Global|LV1;USOQ:OpenAPI|USOQ|Global|LV1;USZC:OpenAPI|USZC|Global|LV0;SGBK:OpenAPI|SGBK|Global|Delay;HKAC:OpenAPI|HKAC|Global|LV1;SHZA:OpenAPI|SHZA|Global|Delay;USZB:OpenAPI|USZB|Global|LV0;USZA:OpenAPI|USZA|Global|LV0;SZAA:OpenAPI|SZAA|Global|LV1;SHAB:OpenAPI|SHAB|Global|LV1;SZAD:OpenAPI|SZAD|Global|Delay;HKZB:OpenAPI|HKZB|Global|LV0;HKAA:OpenAPI|HKAA|Global|LV1;USON:OpenAPI|USON|Global|LV1;USOP:OpenAPI|USOP|Global|LV1;SHAA:OpenAPI|SHAA|Global|LV1;SZBA:OpenAPI|SZBA|Global|Delay"
```

### quote_package_details



```json
[{"description": "Real-time quotes with 1 level bid & ask for A-shares from China Connect", "end_at": "2026-11-19T18:43:49", "key": "CN_Connect", "name": "ChinaConnect  LV1 Real-time Quotes", "start_at": "2026-10-20T18:43:49"}, {"description": "Hang Seng Index real-time quotes", "end_at": "2026-11-19T18:43:49", "key": "HK_HangSengIndex_AllTerminals", "name": "Real-time Quotes", "start_at": "2026-10-20T18:43:49"}, {"description": "OPRA US options real-time quotes with best bid and offer, for OpenAPI only", "end_at": "2026-11-22T07:27:42", "key": "US_OPRA_LV1_OpenAPI", "name": "LV1 Real-time Quotes", "start_at": "2026-10-21T07:27:42"}, {"description": "HKEX stocks real-time quotes with 1 level bid & ask, available across all platforms", "end_at": "2026-11-19T18:43:49", "key": "HK_L1_NonMainland_all_platforms", "name": "LV1 Real-time Quotes", "start_at": "2026-10-20T18:43:49"}, {"description": "Nasdaq real-time quotes with best bid and offer (including overnight trading), for OpenAPI only", "end_at": "2026-11-19T18:43:49", "key": "US_QBBO_OpenAPI", "name": "LV1 Real-time Quotes", "start_at": "2026-10-20T18:43:49"}]
```

### trading_session



```json
[{"market": "US", "trade_sessions": [{"begin_time": "04:00:00", "end_time": "09:30:00", "trade_session": "Pre"}, {"begin_time": "09:30:00", "end_time": "16:00:00", "trade_session": "Intraday"}, {"begin_time": "16:00:00", "end_time": "20:00:00", "trade_session": "Post"}]}, {"market": "HK", "trade_sessions": [{"begin_time": "09:30:00", "end_time": "12:00:00", "trade_session": "Intraday"}, {"begin_time": "13:00:00", "end_time": "16:00:00", "trade_session": "Intraday"}]}, {"market": "CN", "trade_sessions": [{"begin_time": "09:30:00", "end_time": "11:30:00", "trade_session": "Intraday"}, {"begin_time": "13:00:00", "end_time": "14:57:00", "trade_session": "Intraday"}]}, {"market": "SG", "trade_sessions": [{"begin_time": "09:00:00", "end_time": "12:00:00", "trade_session": "Intraday"}, {"begin_time": "13:00:00", "end_time": "17:00:00", "trade_session": "Intraday"}]}]
```

### trading_days(US, 近 20 天)



```json
{"half_trading_days": [], "trading_days": ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21"]}
```


## 正股 quote / depth

### quote(正股)



```json
[{"high": 774.43, "last_done": 774.19, "low": 768.4, "open": 771.2, "overnight_quote": null, "post_market_quote": {"high": 772.35, "last_done": 770.73, "low": 768.662, "prev_close": 769.05, "timestamp": "2026-10-20T23:59:54", "turnover": 2450966239.454, "volume": 3190010}, "pre_market_quote": {"high": 771.7, "last_done": 771.2, "low": 769.029, "prev_close": 769.05, "timestamp": "2026-10-21T13:30:00", "turnover": 255492719.915, "volume": 331478}, "prev_close": 769.05, "symbol": "SPY.US", "timestamp": "2026-10-21T18:43:49", "trade_status": "Normal", "turnover": 13130013757.103, "volume": 17015016}, {"high": 727.95, "last_done": 726.91, "low": 719.72, "open": 722.64, "overnight_quote": null, "post_market_quote": {"high": 724.47, "last_done": 722.849, "low": 717.55, "prev_close": 717.63, "timestamp": "2026-10-20T23:59:58", "turnover": 1364735684.003, "volume": 1898264}, "pre_market_quote": {"high": 723.99, "last_done": 722.61, "low": 718.61, "prev_close": 717.63, "timestamp": "2026-10-21T13:30:00", "turnover": 485036365.022, "volume": 672363}, "prev_close": 717.63, "symbol": "QQQ.US", "timestamp": "2026-10-21T18:43:49", "trade_status": "Normal", "turnover": 16163431942.328, "volume": 22347997}, {"high": 928.45, "last_done": 917.52, "low": 889.45, "open": 925.42, "overnight_quote": null, "post_market_quote": {"high": 922.5, "last_done": 915.17, "low": 894.18, "prev_close": 896.07, "timestamp": "2026-10-20T23:59:56", "turnover": 1596300534.525, "volume": 1770396}, "pre_market_quote": {"high": 928.262, "last_done": 924.5, "low": 910.25, "prev_close": 896.07, "timestamp": "2026-10-21T13:30:00", "turnover": 498670214.65, "volume": 542388}, "prev_close": 896.07, "symbol": "MU.US", "timestamp": "2026-10-21T18:43:49", "trade_status": "Normal", "turnover": 13711187194.702, "volume": 15064825}, {"high": 556.483, "last_done": 555.405, "low": 537.3, "open": 545.61, "overnight_quote": null, "post_market_quote": {"high": 551.0, "last_done": 550.65, "low": 533.39, "prev_close": 534.74, "timestamp": "2026-10-20T23:59:58", "turnover": 316028257.666, "volume": 586171}, "pre_market_quote": {"high": 552.5, "last_done": 546.68, "low": 540.13, "prev_close": 534.74, "timestamp": "2026-10-21T13:30:00", "turnover": 219908989.545, "volume": 405955}, "prev_close": 534.74, "symbol": "AMD.US", "timestamp": "2026-10-21T18:43:49", "trade_status": "Normal", "turnover": 6442549297.827, "volume": 11752452}]
```

### depth(SPY.US)



```json
{"asks": [{"order_num": 0, "position": 1, "price": 774.2, "volume": 182}], "bids": [{"order_num": 0, "position": 1, "price": 774.19, "volume": 80}]}
```

### depth(QQQ.US)



```json
{"asks": [{"order_num": 0, "position": 1, "price": 726.93, "volume": 840}], "bids": [{"order_num": 0, "position": 1, "price": 726.9, "volume": 220}]}
```

### depth(MU.US)



```json
{"asks": [{"order_num": 0, "position": 1, "price": 917.66, "volume": 2}], "bids": [{"order_num": 0, "position": 1, "price": 917.37, "volume": 2}]}
```

### depth(AMD.US)



```json
{"asks": [{"order_num": 0, "position": 1, "price": 555.46, "volume": 3}], "bids": [{"order_num": 0, "position": 1, "price": 555.29, "volume": 180}]}
```


## 1 分钟 bar 可回溯起点

（--skip-bars，跳过翻页；结果见上一次运行）


## 期权链到期日

### option_chain_expiry_date_list(SPY.US) 最近 6 个

核对 MU/AMD 是否为周一 / 周三到期。

```json
[{"date": "2026-10-15", "weekday": "Thursday"}, {"date": "2026-10-16", "weekday": "Friday"}, {"date": "2026-10-19", "weekday": "Monday"}, {"date": "2026-10-20", "weekday": "Tuesday"}, {"date": "2026-10-21", "weekday": "Wednesday"}, {"date": "2026-10-22", "weekday": "Thursday"}]
```

### option_chain_expiry_date_list(QQQ.US) 最近 6 个

核对 MU/AMD 是否为周一 / 周三到期。

```json
[{"date": "2026-10-15", "weekday": "Thursday"}, {"date": "2026-10-16", "weekday": "Friday"}, {"date": "2026-10-19", "weekday": "Monday"}, {"date": "2026-10-20", "weekday": "Tuesday"}, {"date": "2026-10-21", "weekday": "Wednesday"}, {"date": "2026-10-22", "weekday": "Thursday"}]
```

### option_chain_expiry_date_list(MU.US) 最近 6 个

核对 MU/AMD 是否为周一 / 周三到期。

```json
[{"date": "2026-10-16", "weekday": "Friday"}, {"date": "2026-10-19", "weekday": "Monday"}, {"date": "2026-10-21", "weekday": "Wednesday"}, {"date": "2026-10-23", "weekday": "Friday"}, {"date": "2026-10-26", "weekday": "Monday"}, {"date": "2026-10-28", "weekday": "Wednesday"}]
```

### option_chain_expiry_date_list(AMD.US) 最近 6 个

核对 MU/AMD 是否为周一 / 周三到期。

```json
[{"date": "2026-10-16", "weekday": "Friday"}, {"date": "2026-10-19", "weekday": "Monday"}, {"date": "2026-10-21", "weekday": "Wednesday"}, {"date": "2026-10-23", "weekday": "Friday"}, {"date": "2026-10-26", "weekday": "Monday"}, {"date": "2026-10-28", "weekday": "Wednesday"}]
```


## SPY 最近到期 ATM call

### nearest_atm_call(SPY.US)



```json
{"call_symbol": "SPY261021C774000.US", "expiry": "2026-10-21", "put_symbol": "SPY261021P774000.US", "spot": 774.21, "strike": 774.0}
```

### option_quote(SPY261021C774000.US)

OptionQuote 无 bid/ask（已核实）。

```json
[{"contract_multiplier": 100.0, "contract_size": 100.0, "contract_type": "American", "direction": "Call", "expiry_date": "2026-10-21", "high": 1.03, "historical_volatility": 0.0904, "implied_volatility": 0.068, "last_done": 0.81, "low": 0.14, "open": 0.5, "open_interest": 6240, "prev_close": 0.56, "strike_price": 774.0, "symbol": "SPY261021C774000.US", "timestamp": "2026-10-21T18:43:49", "trade_status": "Normal", "turnover": 25200868.0, "underlying_symbol": "SPY.US", "volume": 465846}]
```

### depth(SPY261021C774000.US)

关键测试：期权代码是否返回盘口；每档 price/volume/order_num/position。

```json
{"asks": [{"order_num": 0, "position": 1, "price": 0.82, "volume": 235}], "bids": [{"order_num": 0, "position": 1, "price": 0.81, "volume": 232}]}
```

### candlesticks(SPY261021C774000.US, Min_1, 100)



```json
[{"close": 0.62, "high": 0.64, "low": 0.6, "open": 0.63, "timestamp": "2026-10-21T17:04:00", "trade_session": "Intraday", "turnover": 73811.0, "volume": 1207}, {"close": 0.62, "high": 0.65, "low": 0.59, "open": 0.62, "timestamp": "2026-10-21T17:05:00", "trade_session": "Intraday", "turnover": 73706.0, "volume": 1187}, {"close": 0.69, "high": 0.69, "low": 0.62, "open": 0.63, "timestamp": "2026-10-21T17:06:00", "trade_session": "Intraday", "turnover": 126817.0, "volume": 1910}, {"close": 0.75, "high": 0.75, "low": 0.68, "open": 0.68, "timestamp": "2026-10-21T17:07:00", "trade_session": "Intraday", "turnover": 239296.0, "volume": 3333}, {"close": 0.76, "high": 0.78, "low": 0.74, "open": 0.74, "timestamp": "2026-10-21T17:08:00", "trade_session": "Intraday", "turnover": 284340.0, "volume": 3738}, {"close": 0.81, "high": 0.86, "low": 0.76, "open": 0.77, "timestamp": "2026-10-21T17:09:00", "trade_session": "Intraday", "turnover": 362807.0, "volume": 4427}, {"close": 0.85, "high": 0.86, "low": 0.8, "open": 0.8, "timestamp": "2026-10-21T17:10:00", "trade_session": "Intraday", "turnover": 269063.0, "volume": 3232}, {"close": 0.81, "high": 0.88, "low": 0.79, "open": 0.86, "timestamp": "2026-10-21T17:11:00", "trade_session": "Intraday", "turnover": 330834.0, "volume": 3980}, {"close": 0.76, "high": 0.84, "low": 0.7, "open": 0.81, "timestamp": "2026-10-21T17:12:00", "trade_session": "Intraday", "turnover": 246841.0, "volume": 3302}, {"close": 0.79, "high": 0.8, "low": 0.74, "open": 0.77, "timestamp": "2026-10-21T17:13:00", "trade_session": "Intraday", "turnover": 165765.0, "volume": 2156}, {"close": 0.78, "high": 0.85, "low": 0.77, "open": 0.78, "timestamp": "2026-10-21T17:14:00", "trade_session": "Intraday", "turnover": 190950.0, "volume": 2360}, {"close": 0.83, "high": 0.84, "low": 0.77, "open": 0.78, "timestamp": "2026-10-21T17:15:00", "trade_session": "Intraday", "turnover": 186083.0, "volume": 2256}, {"close": 0.92, "high": 0.96, "low": 0.83, "open": 0.84, "timestamp": "2026-10-21T17:16:00", "trade_session": "Intraday", "turnover": 472282.0, "volume": 5255}, {"close": 0.9, "high": 0.95, "low": 0.89, "open": 0.93, "timestamp": "2026-10-21T17:17:00", "trade_session": "Intraday", "turnover": 321100.0, "volume": 3496}, {"close": 0.91, "high": 0.95, "low": 0.88, "open": 0.91, "timestamp": "2026-10-21T17:18:00", "trade_session": "Intraday", "turnover": 151842.0, "volume": 1658}, {"close": 0.89, "high": 0.92, "low": 0.87, "open": 0.92, "timestamp": "2026-10-21T17:19:00", "trade_session": "Intraday", "turnover": 97639.0, "volume": 1095}, {"close": 0.98, "high": 0.99, "low": 0.87, "open": 0.9, "timestamp": "2026-10-21T17:20:00", "trade_session": "Intraday", "turnover": 223509.0, "volume": 2371}, {"close": 0.94, "high": 0.99, "low": 0.9, "open": 0.98, "timestamp": "2026-10-21T17:21:00", "trade_session": "Intraday", "turnover": 158463.0, "volume": 1664}, {"close": 0.9, "high": 0.99, "low": 0.87, "open": 0.95, "timestamp": "2026-10-21T17:22:00", "trade_session": "Intraday", "turnover": 150170.0, "volume": 1646}, {"close": 0.86, "high": 0.97, "low": 0.82, "open": 0.9, "timestamp": "2026-10-21T17:23:00", "trade_session": "Intraday", "turnover": 221036.0, "volume": 2519}, {"close": 0.84, "high": 0.87, "low": 0.83, "open": 0.86, "timestamp": "2026-10-21T17:24:00", "trade_session": "Intraday", "turnover": 433960.0, "volume": 5103}, {"close": 0.8, "high": 0.87, "low": 0.77, "open": 0.84, "timestamp": "2026-10-21T17:25:00", "trade_session": "Intraday", "turnover": 162237.0, "volume": 1973}, {"close": 0.81, "high": 0.84, "low": 0.78, "open": 0.8, "timestamp": "2026-10-21T17:26:00", "trade_session": "Intraday", "turnover": 112844.0, "volume": 1403}, {"close": 0.79, "high": 0.83, "low": 0.78, "open": 0.81, "timestamp": "2026-10-21T17:27:00", "trade_session": "Intraday", "turnover": 63011.0, "volume": 782}, {"close": 0.79, "high": 0.82, "low": 0.78, "open": 0.78, "timestamp": "2026-10-21T17:28:00", "trade_session": "Intraday", "turnover": 61469.0, "volume": …(截断)
```

### history_candlesticks_by_offset(SPY261021C774000.US, Min_1, 1000)



```json
[{"close": 1.34, "high": 1.34, "low": 1.34, "open": 1.34, "timestamp": "2026-10-19T15:34:00", "trade_session": "Intraday", "turnover": 4690.0, "volume": 35}, {"close": 1.38, "high": 1.38, "low": 1.31, "open": 1.31, "timestamp": "2026-10-19T15:35:00", "trade_session": "Intraday", "turnover": 1991.0, "volume": 15}, {"close": 1.47, "high": 1.47, "low": 1.42, "open": 1.42, "timestamp": "2026-10-19T15:36:00", "trade_session": "Intraday", "turnover": 431.0, "volume": 3}, {"close": 1.45, "high": 1.51, "low": 1.45, "open": 1.5, "timestamp": "2026-10-19T15:37:00", "trade_session": "Intraday", "turnover": 1644.0, "volume": 11}, {"close": 1.44, "high": 1.44, "low": 1.43, "open": 1.43, "timestamp": "2026-10-19T15:38:00", "trade_session": "Intraday", "turnover": 1726.0, "volume": 12}, {"close": 1.41, "high": 1.41, "low": 1.41, "open": 1.41, "timestamp": "2026-10-19T15:39:00", "trade_session": "Intraday", "turnover": 5640.0, "volume": 40}, {"close": 1.39, "high": 1.39, "low": 1.35, "open": 1.35, "timestamp": "2026-10-19T15:40:00", "trade_session": "Intraday", "turnover": 683.0, "volume": 5}, {"close": 1.39, "high": 1.39, "low": 1.39, "open": 1.39, "timestamp": "2026-10-19T15:41:00", "trade_session": "Intraday", "turnover": 139.0, "volume": 1}, {"close": 1.4, "high": 1.43, "low": 1.4, "open": 1.42, "timestamp": "2026-10-19T15:42:00", "trade_session": "Intraday", "turnover": 14582.0, "volume": 102}, {"close": 1.36, "high": 1.38, "low": 1.36, "open": 1.38, "timestamp": "2026-10-19T15:43:00", "trade_session": "Intraday", "turnover": 14480.0, "volume": 105}, {"close": 1.36, "high": 1.36, "low": 1.36, "open": 1.36, "timestamp": "2026-10-19T15:44:00", "trade_session": "Intraday", "turnover": 0.0, "volume": 0}, {"close": 1.3, "high": 1.3, "low": 1.26, "open": 1.28, "timestamp": "2026-10-19T15:45:00", "trade_session": "Intraday", "turnover": 29453.0, "volume": 232}, {"close": 1.32, "high": 1.33, "low": 1.32, "open": 1.33, "timestamp": "2026-10-19T15:46:00", "trade_session": "Intraday", "turnover": 4093.0, "volume": 31}, {"close": 1.32, "high": 1.32, "low": 1.32, "open": 1.32, "timestamp": "2026-10-19T15:47:00", "trade_session": "Intraday", "turnover": 0.0, "volume": 0}, {"close": 1.24, "high": 1.29, "low": 1.24, "open": 1.29, "timestamp": "2026-10-19T15:48:00", "trade_session": "Intraday", "turnover": 253.0, "volume": 2}, {"close": 1.24, "high": 1.24, "low": 1.24, "open": 1.24, "timestamp": "2026-10-19T15:49:00", "trade_session": "Intraday", "turnover": 0.0, "volume": 0}, {"close": 1.24, "high": 1.24, "low": 1.24, "open": 1.24, "timestamp": "2026-10-19T15:50:00", "trade_session": "Intraday", "turnover": 0.0, "volume": 0}, {"close": 1.28, "high": 1.31, "low": 1.27, "open": 1.31, "timestamp": "2026-10-19T15:51:00", "trade_session": "Intraday", "turnover": 644.0, "volume": 5}, {"close": 1.31, "high": 1.31, "low": 1.31, "open": 1.31, "timestamp": "2026-10-19T15:52:00", "trade_session": "Intraday", "turnover": 131.0, "volume": 1}, {"close": 1.31, "high": 1.31, "low": 1.31, "open": 1.31, "timestamp": "2026-10-19T15:53:00", "trade_session": "Intraday", "turnover": 0.0, "volume": 0}, {"close": 1.25, "high": 1.29, "low": 1.25, "open": 1.29, "timestamp": "2026-10-19T15:54:00", "trade_session": "Intraday", "turnover": 379.0, "volume": 3}, {"close": 1.22, "high": 1.25, "low": 1.22, "open": 1.25, "timestamp": "2026-10-19T15:55:00", "trade_session": "Intraday", "turnover": 1467.0, "volume": 12}, {"close": 1.24, "high": 1.24, "low": 1.21, "open": 1.21, "timestamp": "2026-10-19T15:56:00", "trade_session": "Intraday", "turnover": 245.0, "volume": 2}, {"close": 1.24, "high": 1.24, "low": 1.24, "open": 1.24, "timestamp": "2026-10-19T15:57:00", "trade_session": "Intraday", "turnover": 0.0, "volume": 0}, {"close": 1.21, "high": 1.22, "low": 1.21, "open": 1.21, "timestamp": "2026-10-19T15:58:00", "trade_session": "Intraday", "turnover": 1336.0, "volume": 11}, {"close": 1.21, "high": 1.21, "low": 1.21, "open": 1.21, "timestamp": "2026-10-19T15:59:00", "trade_session": "In …(截断)
```


## 订阅推送计数

### subscribe Quote+Depth+Trade 20s

按 symbol 计数；保存首条 Depth 样例。

```json
{"counts": {"depth": 99, "quote": 117, "trade": 115}, "depth_sample": {"SPY.US": {"asks": [{"order_num": 0, "position": 1, "price": 774.2, "volume": 1070}], "bids": [{"order_num": 0, "position": 1, "price": 774.18, "volume": 40}]}, "SPY261021C774000.US": {"asks": [{"order_num": 0, "position": 1, "price": 0.82, "volume": 846}], "bids": [{"order_num": 0, "position": 1, "price": 0.81, "volume": 59}]}}, "errors": [], "seconds": 20, "symbols": ["SPY.US", "SPY261021C774000.US"]}
```

### make_trade_ctx



```json
"<builtins.TradeContext object at 0x7af25d737cf0>"
```


## 交易账户（只读）

### account_balance



```json
[{"buy_power": [REDACTED], "cash_infos": [{"available_cash": [REDACTED], "currency": "USD", "frozen_cash": [REDACTED], "settling_cash": [REDACTED], "withdraw_cash": [REDACTED]}, {"available_cash": [REDACTED], "currency": "HKD", "frozen_cash": [REDACTED], "settling_cash": [REDACTED], "withdraw_cash": [REDACTED]}], "currency": "HKD", "frozen_transaction_fees": [], "init_margin": [REDACTED], "maintenance_margin": [REDACTED], "margin_call": [REDACTED], "max_finance_amount": [REDACTED], "net_assets": [REDACTED], "remaining_finance_amount": [REDACTED], "risk_level": [REDACTED], "total_cash": [REDACTED]}]
```

### stock_positions



```json
{"channels": [{"account_channel": "[REDACTED]", "positions": []}]}
```

### today_orders



```json
[]
```

### history_executions(近 30 天)



```json
[]
```

### estimate_max_purchase_quantity(SPY261021C774000.US, LO, Buy, 0.01)



```json
{"cash_max_qty": [REDACTED], "margin_max_qty": [REDACTED]}
```
