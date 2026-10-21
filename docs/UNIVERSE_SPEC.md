# 标的池规范

单一事实来源：[`config/universe.yaml`](../config/universe.yaml)，由 `zr/data/universe.py` 加载并校验。录制器、ThetaData loader、研究脚本都从这里读标的与期权链范围，不各自硬编码。

## 字段

| 字段 | 含义 | 用途 |
|---|---|---|
| `n_expiries` | 取 ≥ 今日的最近 n 个到期日 | 录制器订阅集 |
| `strikes_each_side` | ATM 两侧各取的档数 | 录制器订阅集 |
| `strike_band_pct` | 现价 ± 百分比 | ThetaData 历史拉取范围（DATA_SPEC 第 2 节） |
| `max_dte` | 历史数据最大 DTE | ThetaData 历史拉取范围 |
| `subscription_cap` | 全局订阅上限（SDK 限 500，留余量） | 录制器裁剪 |

## 当前值

| 标的 | n_expiries | strikes_each_side | strike_band_pct | max_dte |
|---|---|---|---|---|
| SPY.US | 2 | 10 | 3% | 5 |
| QQQ.US | 2 | 10 | 3% | 5 |
| MU.US | 2 | 8 | 8% | 5 |
| AMD.US（暂停，D-016） | 2 | 8 | 8% | 5 |

`subscription_cap` = 450。

## 规则

- `.env` 的 `ZR_SYMBOLS` 是运行时过滤器，只能选 universe 的子集；出现 universe 没有的代码时加载报错。当前默认 `SPY.US,QQQ.US,MU.US`。
- 到期日、行权价间距一律来自期权链 API（D-005），universe 只定义"取多少"，不定义"是哪些"。
- 改 universe 值要在 `docs/DECISIONS.md` 追加一条。MU/AMD 是否留在 v0 看 D-009。
