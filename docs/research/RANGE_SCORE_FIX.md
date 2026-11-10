# range_score 前视修复：新旧对比（D-037）

`python -m zr.research.range_score_fix` 生成。只报告，不重跑旧研究。旧实现窗口 = window_days × 整段输入每日根数中位数（含 t 之后的行）；新实现 = 当日之前已完成交易日根数中位数（按日递推），首日用名义值（RTH 390 ÷ P、含盘前盘后 960 ÷ P）。2 分钟 K，v4 状态配置；不一致 = range_score |差| > 1e-12 或一边为 NaN；状态 = regimes 的 state。

## MU · 口径 C（含盘前盘后、不按日重置）

数据：2023-10-02～2026-10-21，767 个交易日，367,759 根 2 分钟 K。旧窗口 480 根；新窗口取值 [480]。不一致：range_score 0 行，状态 0 行；涉及日期 0 天。

全部月份一致。

## MU · 口径 D-027（RTH-only、按日重置）

数据：2023-10-02～2026-10-21，767 个交易日，148,924 根 2 分钟 K。旧窗口 195 根；新窗口取值 [195]。不一致：range_score 0 行，状态 0 行；涉及日期 0 天。

全部月份一致。

## QQQ · 口径 C（含盘前盘后、不按日重置）

数据：2023-10-02～2026-10-21，767 个交易日，367,755 根 2 分钟 K。旧窗口 480 根；新窗口取值 [480]。不一致：range_score 0 行，状态 0 行；涉及日期 0 天。

全部月份一致。

## QQQ · 口径 D-027（RTH-only、按日重置）

数据：2023-10-02～2026-10-21，767 个交易日，148,920 根 2 分钟 K。旧窗口 195 根；新窗口取值 [195]。不一致：range_score 0 行，状态 0 行；涉及日期 0 天。

全部月份一致。

## 调用位置与输入长度（只列清单）

| 调用位置 | 输入 | 输入只有 1–2 天时是否受过影响 |
|---|---|---|
| zr/live/core.py（实盘 / 纸面助手） | 不调用 range_score；增量引擎 IncrementalEngine 的窗口固定 195（口径 A）/ 480（口径 C） | 否（固定名义值，与修复后首日名义值一致） |
| zr/strategy/incremental.py（增量引擎） | 窗口由调用方给定，独立实现，不调用 range_score | 否 |
| zr/research/shadow_replay.py:206（影子回放增量侧） | w = bars_per_day(seed)，seed = SEED_FROM 至回放窗口开始前的历史 K（多日、只含过去） | 否（只用过去数据） |
| zr/research/shadow_replay.py:325（影子回放批量对照侧） | regimes(b, …)，b = 回放日及之前的 2 分钟 K（多日） | 仅当 b 只有 1–2 天时；按代码 b 含种子期多日，中位数与修复后一致的可能性高（未逐日核实） |
| tests/test_incremental.py、tests/test_live_assistant.py | 单日夹具（MU 2026-10-21，口径 C 349 根至 15:36 / 口径 A RTH），引擎窗口曾取 bars_per_day(b) = 当日全天根数 | 是：旧测试用整日根数（349）作窗口，含未来信息；已改为首日名义窗口，与 zr/live/core.py 一致 |
| tests/test_synthetic_scenarios.py | 单日合成场景（约百根），按「窗口 = 场景长度」设计 | 是：已改为显式 bars_per_day=len(df) |
| zr/research/v4_decisive.py:247、v4_h5.py:177（含 state_features）、entry_timing_diag.py:144、intraday_single_leg.py:312、guardrail_evidence.py:48、owner_entry_mining.py:132、sim_v3.py:226、own_policy.py:145/178（含 state_features）、method_v3_round1.py:46、lookahead_audit.py:98（含 state_features）、v4_checkpoints.py:89 | 批量研究：全量或整年 / 多日 2 分钟 K（build_bars 读到 end 的全部历史，或研究窗口内全部日期） | 仅样本开头 1–2 天（见上表首日差异）；按日切片调用的（intraday_single_leg 等若逐日传入单日 K）会受影响——未逐个核实输入切片方式 |
| zr/research/model_v3.py:136（state_features） | reg 来自上游批量 regimes | 同上游 |
| zr/features/range_calibration.py:37（RANGE 阈值标定） | MU 2024 全年 2 分钟 K（RTH） | 仅 2024 年首日附近；阈值 0.3487 / 0.507 / 0.3499 未重标 |

## 边界

- 只比较 range_score 与状态标签；下游信号、交易、报告未重跑。
- 全量历史上首日名义值（480 / 195）恰好等于全样本中位数、之后各日的递推中位数也不变，所以上表四组全部一致；修复只改变输入很短（1–2 天、或只有部分交易日）时的结果，见下方调用清单。
- 调用清单按源码静态阅读整理，未逐个运行核实各研究脚本的实际输入切片。
