# 数据规范

根目录 `ZR_DATA_ROOT`（默认 `./data_store`，不入库）。原始数据只增不改；修正写新分区或新版本，不覆盖。

## 1. 正股 1 分钟 bar

- 标的：SPY、QQQ、MU、AMD（`ZR_SYMBOLS`）
- 范围：2023-01-01 至今
- 来源：长桥 `history_candlesticks_by_offset(symbol, Period.Min_1, AdjustType.NoAdjust, forward, count, time, TradeSessions)`；实际可回溯深度由 `zr/ops/api_facts.py` 核验后写入 `docs/API_FACTS.md`
- 路径：`bars_1m/underlying=XXX/date=YYYY-MM-DD/*.parquet`
- 列：`ts`(UTC, ns) · `open` `high` `low` `close`(float64) · `volume`(int64) · `turnover`(float64) · `session`(str: pre/regular/post)

## 2. 期权 NBBO（ThetaData Options Standard）

- DTE ≤ 5
- 行权价范围：SPY/QQQ 现价 ±3%，MU/AMD 现价 ±8%
- 范围：2024-01-01 至今
- 路径：`options_nbbo/underlying=XXX/date=YYYY-MM-DD/*.parquet`
- 列：`ts` · `option_symbol` · `expiry`(date) · `strike`(float64) · `right`(C/P) · `bid` `ask` `bid_size` `ask_size` · `last` · `volume` · `oi`
- Loader：`zr/data/thetadata.py`（Task 1.5，未写）

## 3. 自建录制器输出（`zr/data/recorder.py`）

- 路径：`recorder/date=YYYY-MM-DD/underlying=XXX/<kind>_<ts>.parquet`，`date` 取美东日期
- `kind` ∈ {quote, depth, trade}
- 列：`recv_ts_utc_ns`(int64 本地接收) · `push_ts`(int64 推送时间戳，秒或 ns 以原始为准) · `symbol` · `kind` · `payload`(str, JSON 原始字段)
- 去重键：(symbol, kind, push_ts, payload)
- 录制器是校验源，不替代 ThetaData；两者差异写进 W1 成本报告

## 4. 交易史（`zr/journal/trade_history.py`）

- `journal/orders.parquet`、`journal/executions.parquet`、`journal/charges.parquet`
- Decimal → float64；枚举 → 名称字符串；时间保留原时区

## 5. 命名

- 时间列统一 `ts`，UTC；展示层再转美东
- 期权代码沿用长桥格式 `^[A-Z.\-]+\d{6}[CP]\d+\.US$`
- 分区键只用 `date=` 与 `underlying=`
