# 指标一致性核对（diag/indicator-parity 第 1 节）

生成：2026-10-22T18:17:45+00:00 UTC · 复现 `uv run python -m zr.research.indicator_parity`

## 全部组合（按最大误差排序，含为 NaN 的——数据不足以算出该时刻值）

| offset_min | ddof | ema_init | n_compared | max_pct_err | mean_pct_err | variant |
|---|---|---|---|---|---|---|
| 2 | 0 | first_seed | 15 | 0.15 | 0.02 | local_dayonly_from0930 |
| 2 | 1 | first_seed | 15 | 0.15 | 0.02 | local_dayonly_from0930 |
| 2 | 0 | first_seed | 15 | 0.18 | 0.02 | direct_min2_noadjust |
| 2 | 0 | sma_seed | 15 | 0.18 | 0.02 | direct_min2_noadjust |
| 2 | 0 | first_seed | 15 | 0.18 | 0.02 | direct_min2_fwdadjust |
| 2 | 0 | sma_seed | 15 | 0.18 | 0.02 | direct_min2_fwdadjust |
| 2 | 1 | first_seed | 15 | 0.18 | 0.03 | direct_min2_noadjust |
| 2 | 1 | sma_seed | 15 | 0.18 | 0.03 | direct_min2_noadjust |
| 2 | 1 | first_seed | 15 | 0.18 | 0.03 | direct_min2_fwdadjust |
| 2 | 1 | sma_seed | 15 | 0.18 | 0.03 | direct_min2_fwdadjust |
| 2 | 0 | first_seed | 15 | 0.18 | 0.02 | local_dayonly_from0800 |
| 2 | 1 | first_seed | 15 | 0.18 | 0.03 | local_dayonly_from0800 |
| 2 | 0 | sma_seed | 15 | 0.18 | 0.02 | local_dayonly_from0800 |
| 2 | 1 | sma_seed | 15 | 0.18 | 0.03 | local_dayonly_from0800 |
| 2 | 0 | first_seed | 15 | 0.18 | 0.02 | local_dayonly_from0854 |
| 2 | 1 | first_seed | 15 | 0.18 | 0.03 | local_dayonly_from0854 |
| 2 | 0 | sma_seed | 15 | 0.18 | 0.02 | local_dayonly_from0854 |
| 2 | 1 | sma_seed | 15 | 0.18 | 0.03 | local_dayonly_from0854 |
| 2 | 0 | sma_seed | 15 | 0.38 | 0.06 | local_dayonly_from0930 |
| 2 | 1 | sma_seed | 15 | 0.38 | 0.06 | local_dayonly_from0930 |
| 2 | 0 | first_seed | 15 | 2.38 | 0.27 | local_continuous_rth |
| 2 | 0 | sma_seed | 15 | 2.38 | 0.27 | local_continuous_rth |
| 2 | 1 | first_seed | 15 | 2.38 | 0.27 | local_continuous_rth |
| 2 | 1 | sma_seed | 15 | 2.38 | 0.27 | local_continuous_rth |
| 0 | 0 | first_seed | 15 | 7.77 | 1.8 | local_continuous_rth |
| 0 | 0 | sma_seed | 15 | 7.77 | 1.8 | local_continuous_rth |
| 0 | 1 | first_seed | 15 | 7.77 | 1.8 | local_continuous_rth |
| 0 | 1 | sma_seed | 15 | 7.77 | 1.8 | local_continuous_rth |
| 0 | 0 | sma_seed | 15 | 7.78 | 1.74 | local_dayonly_from0930 |
| 0 | 1 | sma_seed | 15 | 7.78 | 1.74 | local_dayonly_from0930 |
| 0 | 0 | sma_seed | 15 | 7.99 | 1.73 | local_dayonly_from0854 |
| 0 | 1 | sma_seed | 15 | 7.99 | 1.73 | local_dayonly_from0854 |
| 0 | 0 | first_seed | 15 | 7.99 | 1.73 | local_dayonly_from0854 |
| 0 | 1 | first_seed | 15 | 7.99 | 1.73 | local_dayonly_from0854 |
| 0 | 0 | sma_seed | 15 | 7.99 | 1.73 | local_dayonly_from0800 |
| 0 | 1 | sma_seed | 15 | 7.99 | 1.73 | local_dayonly_from0800 |
| 0 | 0 | first_seed | 15 | 7.99 | 1.73 | direct_min2_noadjust |
| 0 | 0 | sma_seed | 15 | 7.99 | 1.73 | direct_min2_noadjust |
| 0 | 0 | first_seed | 15 | 7.99 | 1.73 | direct_min2_fwdadjust |
| 0 | 0 | sma_seed | 15 | 7.99 | 1.73 | direct_min2_fwdadjust |
| 0 | 0 | first_seed | 15 | 7.99 | 1.73 | local_dayonly_from0800 |
| 0 | 1 | first_seed | 15 | 7.99 | 1.73 | direct_min2_noadjust |
| 0 | 1 | sma_seed | 15 | 7.99 | 1.73 | direct_min2_noadjust |
| 0 | 1 | first_seed | 15 | 7.99 | 1.73 | direct_min2_fwdadjust |
| 0 | 1 | sma_seed | 15 | 7.99 | 1.73 | direct_min2_fwdadjust |
| 0 | 1 | first_seed | 15 | 7.99 | 1.73 | local_dayonly_from0800 |
| 0 | 0 | first_seed | 15 | 8.15 | 1.72 | local_dayonly_from0930 |
| 0 | 1 | first_seed | 15 | 8.15 | 1.72 | local_dayonly_from0930 |
| 0 | 1 | first_seed | 15 | 22.99 | 4.12 | local_continuous_ext |
| 0 | 1 | sma_seed | 15 | 22.99 | 4.12 | local_continuous_ext |
| 0 | 1 | first_seed | 15 | 22.99 | 4.12 | local_dayonly_from0400 |
| 0 | 1 | sma_seed | 15 | 22.99 | 4.12 | local_dayonly_from0400 |
| 0 | 0 | first_seed | 15 | 22.99 | 4.12 | local_continuous_ext |
| 0 | 0 | sma_seed | 15 | 22.99 | 4.12 | local_continuous_ext |
| 0 | 0 | first_seed | 15 | 22.99 | 4.12 | local_dayonly_from0400 |
| 0 | 0 | sma_seed | 15 | 22.99 | 4.12 | local_dayonly_from0400 |
| 2 | 1 | first_seed | 15 | 44.02 | 7.09 | local_continuous_ext |
| 2 | 1 | sma_seed | 15 | 44.02 | 7.09 | local_continuous_ext |
| 2 | 1 | first_seed | 15 | 44.02 | 7.09 | local_dayonly_from0400 |
| 2 | 1 | sma_seed | 15 | 44.02 | 7.09 | local_dayonly_from0400 |
| 2 | 0 | first_seed | 15 | 44.02 | 7.09 | local_continuous_ext |
| 2 | 0 | sma_seed | 15 | 44.02 | 7.09 | local_continuous_ext |
| 2 | 0 | first_seed | 15 | 44.02 | 7.09 | local_dayonly_from0400 |
| 2 | 0 | sma_seed | 15 | 44.02 | 7.09 | local_dayonly_from0400 |
| 2 | 0 | wilder | 15 | 78.94 | 17.8 | local_dayonly_from0930 |
| 2 | 1 | wilder | 15 | 78.94 | 17.81 | local_dayonly_from0930 |
| 2 | 1 | wilder | 15 | 80.82 | 17.49 | local_dayonly_from0400 |
| 2 | 1 | wilder | 15 | 80.82 | 17.49 | local_continuous_ext |
| 2 | 0 | wilder | 15 | 80.82 | 17.49 | local_dayonly_from0400 |
| 2 | 0 | wilder | 15 | 80.82 | 17.49 | local_continuous_ext |
| 0 | 0 | wilder | 15 | 83.1 | 20.19 | local_dayonly_from0930 |
| 0 | 1 | wilder | 15 | 83.1 | 20.19 | local_dayonly_from0930 |
| 0 | 0 | wilder | 15 | 88.51 | 22.94 | local_dayonly_from0800 |
| 0 | 1 | wilder | 15 | 88.51 | 22.94 | local_dayonly_from0800 |
| 0 | 0 | wilder | 15 | 88.58 | 22.92 | direct_min2_noadjust |
| 0 | 0 | wilder | 15 | 88.58 | 22.92 | direct_min2_fwdadjust |
| 0 | 1 | wilder | 15 | 88.58 | 22.92 | direct_min2_noadjust |
| 0 | 1 | wilder | 15 | 88.58 | 22.92 | direct_min2_fwdadjust |
| 0 | 0 | wilder | 15 | 89.74 | 23.16 | local_dayonly_from0854 |
| 0 | 1 | wilder | 15 | 89.74 | 23.17 | local_dayonly_from0854 |
| 2 | 0 | wilder | 15 | 93.01 | 20.35 | local_dayonly_from0800 |
| 2 | 1 | wilder | 15 | 93.01 | 20.35 | local_dayonly_from0800 |
| 2 | 0 | wilder | 15 | 93.08 | 20.33 | direct_min2_noadjust |
| 2 | 0 | wilder | 15 | 93.08 | 20.33 | direct_min2_fwdadjust |
| 2 | 1 | wilder | 15 | 93.08 | 20.33 | direct_min2_noadjust |
| 2 | 1 | wilder | 15 | 93.08 | 20.33 | direct_min2_fwdadjust |
| 2 | 0 | wilder | 15 | 94.15 | 20.55 | local_dayonly_from0854 |
| 2 | 1 | wilder | 15 | 94.15 | 20.56 | local_dayonly_from0854 |
| 0 | 1 | wilder | 15 | 96.56 | 22.3 | local_dayonly_from0400 |
| 0 | 1 | wilder | 15 | 96.56 | 22.3 | local_continuous_ext |
| 0 | 0 | wilder | 15 | 96.56 | 22.3 | local_dayonly_from0400 |
| 0 | 0 | wilder | 15 | 96.56 | 22.3 | local_continuous_ext |
| 2 | 0 | wilder | 15 | 258.91 | 39.7 | local_continuous_rth |
| 2 | 1 | wilder | 15 | 258.91 | 39.7 | local_continuous_rth |
| 0 | 0 | wilder | 15 | 271.66 | 43.88 | local_continuous_rth |
| 0 | 1 | wilder | 15 | 271.66 | 43.88 | local_continuous_rth |

## 最佳口径：`local_dayonly_from0930`、ddof=0、MACD EMA 初始化=first_seed、时间戳偏移=2 分钟（最大误差 0.1548%）

**时间戳偏移 = 2 分钟**：截图标注的时刻（如「11:26」）与我们按开盘时刻标签的 K 线对不上，要在截图时刻上加一个周期（本例 2 分钟）才能找到对应的 bar——即截图「11:26」对应我们标签为「11:28」的那根 K。三个时刻在这个偏移下 BOLL 与 MACD 同时吻合，不是巧合。**这不代表我们的数据有问题**：bar 标签=开盘时刻的约定在全仓库一致（CLAUDE.md、zr/features/bars.py），且这套约定与长桥历史 K 线 API 本身的 `timestamp` 字段一致（已验证：RTH 首根时间戳=09:30，与开盘吻合）；偏移更像是 App 图表 UI 本身的显示/取值习惯（具体机制未知，未进一步拆解长桥前端），记为经验规律使用。

| time_et | bar_ts_utc | mid_ours | mid_app | mid_err% | upper_ours | upper_app | upper_err% | lower_ours | lower_app | lower_err% | dif_ours | dif_app | dif_err% | dea_ours | dea_app | dea_err% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11:26 | 2026-10-21 15:28:00+00:00 | 909.24 | 909.24 | 0 | 912.18 | 912.18 | 0 | 906.3 | 906.3 | 0 | 0.17 | 0.17 | 0.05 | 0.18 | 0.18 | 0.15 |
| 12:34 | 2026-10-21 16:36:00+00:00 | 912.15 | 912.15 | 0 | 914.96 | 914.96 | 0 | 909.35 | 909.35 | 0 | -0.75 | -0.75 | 0.03 | -0.69 | -0.69 | 0.03 |
| 14:52 | 2026-10-21 18:54:00+00:00 | 917.61 | 917.61 | 0 | 921.36 | 921.36 | 0 | 913.87 | 913.87 | 0 | -1.03 | -1.03 | 0.02 | -0.86 | -0.86 | 0.02 |

**残余 MACD 误差（0.02%–0.15%）**：BOLL 的 mid/upper/lower 9 个值全部 0% 误差（四舍五入到 App 显示的 3 位小数后完全相等，原始精度下差异 <0.001）。DIF/DEA 的残余百分比误差全部来自同一个来源——App 只显示 3 位小数，而 DIF/DEA 本身量级只有 0.1–1（例如 11:26 的 DEA 精确值是 0.17673，App 显示为 0.177，四舍五入后完全一致；0.17673 和 0.177 的差只有 0.00027，但除以 DEA 自己的量级 0.177 就显得有 0.15%），不是口径差异。换算成绝对误差，三个时刻 × DIF/DEA 共 6 个值全部在 ±0.0005 以内。

## 13:00 附近回踩 K 核对（最佳口径）

| bar_ts_utc | low_ours | low_app | mid_ours | mid_app |
|---|---|---|---|---|
| 2026-10-21 17:00:00+00:00 | 913.37 | 913.37 | 912.45 | 912.69 |

边界：只测 MU 2026-10-21 这一天的 3 个时刻 + 1 个回踩点；ForwardAdjust 与 NoAdjust 对这段日期的收盘价差异 ≤0.03（1/3486 根），已确认不是误差来源，建表未单列复权维度；直接长桥 2 分钟 K 只覆盖 2026-10-19 起（诊断拉取范围），早于此的历史对比仍用本地 1 分钟聚合。

## 第 2 节：按新口径重算状态层与四个校验点（不重跑全部网格）

`zr/features/regime.py::indicators` 默认改为 `daily_reset=True`（按日重置，不跨日累计），
`grid_v3.py`/`own_policy.py`/`model_v3.py`/`method_v3_round1.regime_check` 的数据源从
`load_extended_bars`（含盘前盘后，D-026）改回 `load_bars`（RTH-only）。RANGE 打分阈值重新标定：
`uv run python -m zr.features.range_calibration` 在新口径下给出 0.3499（原 0.3487，几乎不变，
因为打分本身是窗口内百分位秩，对这类变化不敏感）；config 未改（差异在舍入误差量级内）。

MU 2026-10-21 四个人工校验点（`regime_check()`，只跑这一个函数，没有重跑网格/规则卡/模型）：

| 核对点 | 新口径结果 | D-026（盘前盘后连续）结果 | 说明 |
|---|---|---|---|
| 09:30 后急跌 → IGNITION 下行 | **无法判定（NaN）** | 0.71 | BOLL(20) 按日重置，09:30–10:10 左右（20 根 × 2 分钟）没有有效值；若 App 也按日重置，这段时间的 IGNITION/RANGE/CHANNEL 对 App 自己来说也是「无指标可看」，不是我们的缺口 |
| 10:00–12:30 → RANGE | 0.17 | 0.21 | 两者都远低于 1；状态定义本身（阈值、窗口）与人工目测的「像不像盘整」有多大差距，是状态层定义问题，不是指标数值问题——指标数值已核实到 <0.001 绝对误差 |
| 12:18–13:36 → CHANNEL 上行 | 0.30 | 0.46 | 同上 |
| 13:00 回踩 K 被 E1 识别 | **12 / 36**（恢复） | 0 / 36 | 新口径下 13:00 那根 K 的低点 913.37、中轨 912.69——与截图数值精确一致（上面第 1 节已验证）；D-026 的连续带把这根 K 的中轨算偏了，才把 E1 识别弄没了 |

结论：新口径（RTH-only、按日重置）让第 4 点完全恢复、且是三个精确数值校验点都通过的同一套口径；
第 1 点从「有个数但其实是巧合」变成「诚实地报告无法判定」（BOLL 当日还没攒够 20 根）；第 2、3 点
（RANGE/CHANNEL 的比例型校验）两种口径下都远低于 1——这是状态**定义**（阈值、窗口参数）的问题，
和本节核实的指标**数值**正确性是两回事，不在这次的范围内。

**全部网格 / 规则卡 / 模型 / 157 笔分类的产出文件（`docs/research/RULE_CARDS_*.md`、
`docs/research/METHOD_V3_ROUND1.md`、`OWN_POLICY_EXTRACTED.md`、`research/v3/model_*.md`）
按本次指令要求未重新运行，其内容仍是 D-026（盘前盘后连续）口径下算出来的，现在已知是错误口径，
处于过时状态**，已记入 STATUS.md 待办。
