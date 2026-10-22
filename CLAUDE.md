# CLAUDE.md

ZR Capital：美股分钟级期权交易系统。代码在 `zr/`，测试在 `tests/`。
旧日频代码存档在 `legacy/`（只读参考）；长期机构化路线见 `docs/ROADMAP_LONGTERM.md`（仅供参考，不是操作指令）。

## 规则

1. 不奉承。直接说结论和问题。
2. 数字必须由代码算出，不凭记忆或估算填写。
3. 外部事实附来源；不确定的标「未验证」。
4. 每个任务一个分支，完成后合并进 main 并删除该分支；禁止 force push、禁止删除 main。禁止 `git reset --hard`。
5. 密钥只放 `.env`（已在 .gitignore），不得写进代码、配置文件、日志或提交记录。
6. 实盘下单只能经 `zr/execution` 的 OMS，且必须同时满足环境变量 `ZR_LIVE=1` 和命令行参数 `--live`。
7. 不硬编码到期日、交易日历、行权价间距；一律从数据源查询。
8. 需要负责人决策时，写入 `docs/STATUS.md` 的「待确认」，然后停止。
9. 每个任务结束时运行 `uv run ruff check .` 和 `uv run pytest -q`，并更新 `docs/STATUS.md`。
10. 负责人提出反驳时，先独立评估并给出自己的判断和依据，再决定是否调整计划；不自动把负责人的说法当作新基准。
11. 实验中的持仓时长、到期日选择、事件阈值一律是待测参数，不写死；任何数字结论必须来自仓库内可复现的脚本。
12. **边界声明**：每条研究结论必须写明数据范围（标的、日期、样本量）、口径（费用、对齐、标签定义）与未覆盖的情形；超出边界不外推。
13. **先直接检验**：任何规则或策略先用真实价格按原规则回放，再做特征/模型；模型的价值只以"相对无模型回放的附加值"来衡量。
14. **尺度遍历**：持仓与响应窗口从秒级到 30 分钟逐档测（10 秒、30 秒、1、2、3、5、10、15、30 分钟），不预设单一尺度；超过 30 分钟只作参考。

## 文档

- `docs/DECISIONS.md` 决策记录（只追加）；`docs/STRATEGY_V1.md` 策略定稿；`docs/DATA_SPEC.md` 数据规范
- `docs/PLAN_W1.md` 当周任务；`docs/STATUS.md` 当前状态与待确认；`docs/RESEARCH_LOG.md` 假设登记
- `docs/API_FACTS.md` 由 `zr/ops/api_facts.py` 生成，不手改（末尾「补充」小节除外，须注明来源脚本）；`docs/DATA_INVENTORY.md` 由 `zr/data/backfill_bars.py --inventory` 生成

## 常用命令

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

## 长桥 SDK 已核实事实（longport 4.3.7）

- 用 `Config.from_apikey(app_key, app_secret, access_token)`；不存在 `Config.from_env()`。
- `OptionQuote` 没有 bid/ask 字段。期权盘口候选来源是 `QuoteContext.depth(symbol)` 或 `SubType.Depth` 推送，每档字段为 price / volume / order_num / position（未验证是否接受期权代码）。
- `history_candlesticks_by_offset(symbol, Period, AdjustType, forward, count, time, TradeSessions)`。
- `option_chain_expiry_date_list(symbol)`；`option_chain_info_by_date(symbol, date)` 返回 [price, call_symbol, put_symbol]。
- 未列在此处的 API 用法，在代码注释里标「未验证」。

2026-10-21 在 VPS 实测补充（`docs/API_FACTS.md`）：
- `depth(期权代码)` 返回 1 档 bid/ask；`SubType.Depth` 推送对期权同样有效 → 期权盘口来源 = depth。
- `option_chain_expiry_date_list` 包含已过期日期，使用前必须过滤 `>= 今日`。
- SDK 返回的 datetime 为 naive，值是 UTC。
- `trading_days(Market.US, begin, end)` 返回 `MarketTradingDays{trading_days, half_trading_days}`。
- 期权 1 分钟 K：`candlesticks` 与 `history_candlesticks_by_offset` 都可用。
- 一次 `subscribe` 308 个代码 × Quote/Depth/Trade 正常；一分钟内反复订阅大量期权会触发 `code=301607 Too many option securities request within one minute`。
- systemd：`EnvironmentFile` 覆盖 `Environment=`，服务里的 `ZR_DATA_ROOT` 必须写在 `ExecStart=/usr/bin/env ...` 上。

## ThetaData 已核实事实（Theta Terminal v3，2026-10-22 实测）

- REST `http://127.0.0.1:25503/v3`；响应为 `{"response":[{"contract":{...},"data":[...]}]}` 嵌套，`zr/data/thetadata.py` 已展平。
- 当日数据必须指定 `expiration`；无数据可能返回 200 + 文本；`interval=1m` 时间戳 t 的报价 = t 时刻 NBBO
  （2026-10-22 复核：官方文档原文「the quote for each interval represents the last quote at the
  interval's timestamp」——即 t 时刻之前最近一次的真实报价，不是 [t, t+1) 或 [t−1, t) 区间聚合；
  文档未明说是区间起点还是终点，用 2026-10-21 录制器逐笔数据交叉验证：MU261023P940000 在 5 个比对
  分钟中 3 个与 recorder 在 t 之后几秒内收到的首个推送精确相等、2 个更接近 t 之前最后一次推送，
  与「t 时刻之前最近报价」一致，核实为正确，沿用不改）。
  来源：<https://thetadata.net/docs/operations/option_history_quote.html>（未验证：该页面未显式定义
  interval 边界是左闭右开还是其他）。
- 单到期日 `strike=*` 1 分钟 quote ≈ 22 MB / 5 s；SPY 一天（DTE≤5）≈ 31 s、1.9 MB parquet。
- 本机 LightGBM 缺 `libomp`（无 Homebrew），`zr/research/meta_model.py` 自动回退 sklearn HistGradientBoosting。
- 终端连上游用 gRPC，只认 HTTP CONNECT 代理；SOCKS 会导致 `MDDS upstream unavailable`。默认直连，`run_theta_terminal.sh --proxy` 才启用（需 `THETA_HTTP_PROXY`）。

## 指标标准口径（与长桥 App 对齐，2026-10-22 实测，D-027）

用 MU 2026-10-21 三个独立时刻（11:26/12:34/14:52 ET）的 App 截图值核对，系统遍历数据源、2 分钟对齐、历史范围、BOLL ddof、MACD EMA 初始化、复权、时间戳偏移后确定：

- **BOLL(20,2) 标准差用 ddof=0**（`zr/features/indicators.py` 的 `bollinger` 已经是这个口径，不用改）；MACD 用标准 EMA（`span=n, adjust=False`，即首值做种、α=2/(n+1)，不是 Wilder α=1/n）。
- **只用 RTH bar（09:30–16:00/提前收盘日对应收盘），不含盘前盘后**；BOLL/MACD **每个交易日从头算，不跨日累计**（`zr/features/regime.py::indicators(daily_reset=True)`，默认开启）。这推翻了 2026-10-22 早些时候 D-026 的假设（「含盘前盘后连续 K 线更像 App」）——补充：当时只看了日内四个人工校验点的比例，没有拿到 App 的精确数值截图，属于凭观感定的口径，这次补了带精确数值的截图才测出来是错的。
- **复权（NoAdjust vs ForwardAdjust）对近期数据几乎无影响**（MU 2026-10-19～22 期间 3,486 根 1 分钟 bar 里只有 1 根差 0.03，可忽略）；数据源（本地用长桥 1 分钟聚合成 2 分钟 vs 直接拉长桥 2 分钟 K）同样不影响。
- **App 显示的 K 线时刻比我们的 bar 标签（开盘时刻）少一个周期**：截图标「11:26」对应我们标签为「11:28」的 2 分钟 bar。换算时用 `App 时刻 + 周期 = 我们的 bar 标签`。机制未知（未拆解长桥前端），记为经验规律。
- 达到的精度：BOLL mid/upper/lower 共 9 个值误差 0%（四舍五入到 App 显示的 3 位小数后完全相等）；MACD DIF/DEA 残余误差（0.02%–0.15%）全部来自 App 只显示 3 位小数造成的量化（换算成绝对误差 <0.0005），不是口径差异。
- **副作用**：按日重置意味着每个交易日前 `BOLL 周期 × bar 分钟数`（例如 2 分钟 K、BOLL(20) ≈ 40 分钟，到 10:10 ET 左右）没有有效的 BOLL/MACD 值，这段时间 %b、DIF/DEA 为 NaN——如果 App 也是这个口径，这是**预期行为**，不是数据缺口。
- 过程与全部数值见 `docs/research/INDICATOR_PARITY.md`（`uv run python -m zr.research.indicator_parity`，可重复运行）。

## 运行环境

长桥 API 相关任务（核验、录制、交易史）只在 VPS `zr` 上跑；本机网络对 longportapp.com 有 DNS 污染。部署见 `docs/DEPLOY.md`。
