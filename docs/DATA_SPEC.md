# 数据规范

根目录 `ZR_DATA_ROOT`（默认 `./data_store`，不入库）。原始数据只增不改；修正写新分区或新版本，不覆盖。

## 1. 正股 1 分钟 bar

- 标的：`config/universe.yaml`（见 `docs/UNIVERSE_SPEC.md`），`ZR_SYMBOLS` 只能选其子集；当前 SPY/QQQ/MU，AMD 暂停（D-016）
- 范围：2023-10-02 至今（长桥 1 分钟 K 实测最早 2023-10-02，见 API_FACTS 补充；2023-01～09 需另找数据源）
- 来源：长桥 `history_candlesticks_by_offset(symbol, Period.Min_1, AdjustType.NoAdjust, forward, count, time, TradeSessions)`；实际可回溯深度由 `zr/ops/api_facts.py` 核验后写入 `docs/API_FACTS.md`
- 路径：`bars_1m/underlying=XXX/year=YYYY/bars.parquet`（`zr/data/backfill_bars.py`，按 (ts, symbol) 去重）
- 列：`ts`(UTC, tz-aware) · `symbol` · `open` `high` `low` `close`(float64) · `volume`(int64) · `turnover`(float64) · `session`(Pre/Intraday/Post/Overnight)
- 2 / 5 分钟 K 从 1 分钟聚合，不单独拉取

## 2. 期权 NBBO（ThetaData Options Standard）

- DTE ≤ `max_dte`、行权价范围现价 ± `strike_band_pct`，值取自 `config/universe.yaml`（当前 SPY/QQQ ±3%，MU/AMD ±8%，DTE ≤ 5）
- 范围：2024-01-01 至今
- 路径：`options_1m/underlying=XXX/date=YYYY-MM-DD/{quotes,ohlc,oi}.parquet`（`zr/data/thetadata.py`，`manifest.jsonl` 记每日耗时与行数）
- 列（quotes）：`ts`(UTC) · `trade_date` · `symbol`(长桥格式) · `underlying` · `expiry` · `strike` · `right`(call/put) · `bid` `bid_size` `bid_exchange` `bid_condition` · `ask` `ask_size` `ask_exchange` `ask_condition`；ohlc：`open high low close volume count vwap`；oi：`open_interest`
- 逐笔：`options_ticks/underlying=XXX/date=YYYY-MM-DD/<contract>_<HHMM>-<HHMM>.parquet`（`ticks_window`，事件前后 ±15 分钟）
- 时间：Theta 返回美东无时区时间戳，入库统一转 UTC；1m 的 quote 为该分钟末次 NBBO，ohlc 时间戳为 bar 开盘时刻

## 3. 自建录制器输出（`zr/data/recorder.py`）

- 路径：`recorder/date=YYYY-MM-DD/underlying=XXX/<kind>_<ts>.parquet`，`date` 取美东日期
- `kind` ∈ {quote, depth, trade}
- 列：`recv_ts_utc_ns`(int64 本地接收) · `push_ts`(int64 推送时间戳，秒或 ns 以原始为准) · `symbol` · `kind` · `payload`(str, JSON 原始字段)
- 去重键：(symbol, kind, push_ts, payload)
- 录制器是校验源，不替代 ThetaData；两者差异写进 W1 成本报告

## 4. 交易史（`zr/journal/trade_history.py`）

- `journal/orders.parquet`、`journal/executions.parquet`、`journal/charges.parquet`
- Decimal → float64；枚举 → 名称字符串；时间保留原时区

## 5. 长桥 SDK 时间与期权链（2026-10-21 实测）

- SDK 返回的 `timestamp` 为 naive datetime，值是 UTC；落盘前统一加 `tzinfo=UTC`。
- 1 分钟 K 含 Pre / Post 时段，`trade_session` 字段区分；回溯深度 ≥ 30000 根（上限未探）。
- `option_chain_expiry_date_list` 含已过期日期，使用方负责过滤。
- 期权 `depth()` 只有 1 档，`order_num` 恒为 0。

## 6. 命名

- 时间列统一 `ts`，UTC；展示层再转美东
- 期权代码沿用长桥格式 `^[A-Z.\-]+\d{6}[CP]\d+\.US$`
- 分区键只用 `date=` 与 `underlying=`
