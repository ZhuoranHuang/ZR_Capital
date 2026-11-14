# 用了长桥分钟成交量或分钟高低点的已登记结果（D-038 清单）

2026-11-14 按负责人 0q 决定整理：只列清单、不重跑；下表每个报告末尾已追加「边界补充（2026-11-14，D-038）」，原结论不改。
依据：长桥 1 分钟 K 不是全市场成交，且 2026-05-15 前后口径不同（`docs/research/R057_MASSIVE_DATA.md` 结论 6）。
清单由子代理逐条读代码核对（证据为文件:行号），主会话抽查 R-E1、R-033、R-037、R-045、R-054、R-055 的证据属实。

| 登记 | 报告 | 用到的分钟数据 | 证据 | 样本期 | 与 2026-05-15 |
|---|---|---|---|---|---|
| R-E1 | `docs/research/EVENT_LABEL_TABLE.md` | 高低点 | 低点 / 高点触及布林中轨（zr/research/events.py:54,59） | 2023-10-02～2026-10-21 | 跨 |
| R-E2 | `docs/research/EVENT_LABEL_TABLE.md` | 高低点 | 30 分钟开盘区间高低（events.py:70-71） | 同上 | 跨 |
| R-E3 | `docs/research/EVENT_LABEL_TABLE.md` | 成交量与高低点 | 日内 VWAP（zr/features/indicators.py:79-89，典型价含高低） | 同上 | 跨 |
| R-E8 | `docs/research/EVENT_LABEL_TABLE.md` | 成交量与高低点 | VWAP 距离 ÷ 由高低算的 ATR（zr/features/features.py:65,78） | 同上 | 跨 |
| R-004、R-005 | `docs/research/EVENT_LABEL_TABLE.md` | 成交量与高低点 | 事件池含 E1/E2/E3/E8 | 同上 | 跨 |
| R-006 | `docs/research/SCAN_TABLE.md` | 成交量与高低点 | 趋势 B 用 VWAP；触及用低 / 高（zr/research/scan.py:44-47,96-97） | 同上 | 跨 |
| R-007 | `docs/research/SCAN_TABLE.md` | 高低点 | 15 分钟开盘区间高低 | 同上 | 跨 |
| R-009 | `docs/research/SCAN_TABLE.md` | 高低点 | 距前 N 根高 / 低的回撤（scan.py:129-130） | 同上 | 跨 |
| R-010、R-011 | `docs/research/SCREENSHOT_STRATEGIES.md` | 高低点 | KDJ、摆动高低点、按 K 线高低判止损（zr/research/screenshot_strategies.py:78,89-101,127,186-188） | 同上 | 跨 |
| R-012 | `docs/research/CONDITIONAL_TABLE.md` | 成交量与高低点 | E3/E8（VWAP）与 E1/E2 格子 | 同上 | 跨 |
| R-013 | `docs/research/META_MODEL.md` | 成交量与高低点 | vol_ratio、vwap_dist_atr、range_atr、or_pos（zr/features/features.py:65,77-84,111-115） | 同上 | 跨 |
| R-014 | `docs/research/BASELINE_IC.md` | 成交量与高低点 | 同 R-013 的特征 | 同上 | 跨 |
| R-017 | `docs/research/REPLAY_RULE_MU.md`、`docs/research/REPLAY_RULE_QQQ.md` | 高低点 | E1T 触及用低 / 高（zr/research/replay_rule.py:59-60） | MU 2023-10-02～2026-10-21 | 跨 |
| R-017～R-026 | `docs/research/METHOD_V2_ROUND1.md`、`docs/research/METHOD_V2_ROUND1_QQQ.md` | 成交量与高低点 | events_v2 含 VWAP 检测器；ever_up / MFE 标签用最高 / 最低（zr/research/labels_v2.py:48-49）；v2 特征含 vol_ratio、vwap_dist_atr | 2023-10-02～2026-10-21 | 跨 |
| R-028 | `docs/research/E4_MU_STOCK.md` | 高低点 | KDJ、ATR 止损、按 K 线高低判止损（zr/research/e4_mu_stock.py:82-104,149） | 2023-10-02～2026-10-21 | 跨 |
| R-029 | `docs/research/E1_PULLBACK_SWEEP_MU.md`、`docs/research/E1_PULLBACK_SWEEP_QQQ.md` | 高低点 | 回踩 L = 最低、H = 最高（zr/features/pullback.py:63-69） | 同上 | 跨 |
| R-030、R-032 | `docs/research/METHOD_V3_ROUND1.md`、`docs/research/RULE_CARDS_MU.md`、`docs/research/RULE_CARDS_QQQ.md` | 成交量与高低点 | E11 成交量 ÷ 前均值（zr/research/patterns_v3.py:105）；E1 / E11 高低、X5 摆动离场（zr/research/exits.py:42） | 2023-10～2026-10 | 跨 |
| R-031 | `docs/research/OWN_POLICY_EXTRACTED.md` | 成交量与高低点 | 候选特征含 sig_E11（量）、sig_E1（高低）；回放离场含 X5 | 负责人 2026-06～08 + 回放 2023-10～2026-10 | 跨 |
| R-033、R-035、R-037 | `docs/research/V4_DECISIVE.md` | 成交量与高低点 | H3 量比 ≥ 1.5、H2 回踩高低、前 20 根高低（zr/research/v4_decisive.py:99-108）；R-035 只在离场 (c) X5 用高低；R-037 量比项 vol_ratio（zr/research/intraday_single_leg.py:153） | R-033 2023-10-02～2026-10-21；R-035 / R-037 2024-10-22～2026-10-21 | 跨 |
| R-034 | `docs/research/V4_H5.md` | 成交量与高低点 | v2 特征 + H3 事件 | 同 R-033 | 跨 |
| R-036 | `docs/research/E4_REVIEW.md` | 高低点 | KDJ、ATR、摆动、按 K 线高低判止损（zr/research/e4_review.py:190） | 2023-10-02～2026-10-21 | 跨 |
| R-037 | `docs/research/INTRADAY_SINGLE_LEG.md` | 成交量与高低点 | vol_ratio > 0.6955（2 分钟量 ÷ 前 20 根均值）、pb_mid_hw 用最低 / 最高（intraday_single_leg.py:149,153） | 2024-10-22～2026-10-21（不含 06-09～08-19） | 跨 |
| R-040 | `docs/research/DISCIPLINE_RULES.md` | 成交量与高低点 | B3 量比 ≥ 1.5（zr/research/discipline.py:157-168）、B2 回踩、X5 | V4 样本 + 负责人 2026-06～08 | 跨 |
| R-041 | `docs/research/FILL_MODEL_RESULTS.md`、`docs/research/FILL_MODEL_CALIBRATION.md` | 成交量与高低点 | V4 H3 / H5 特征 | 同 R-040 | 跨 |
| R-043 | `docs/research/EXIT_HABITS.md`、`docs/research/EXIT_REVISION.md` | 成交量与高低点 | 入场用 V4 H2 / H3；「跌破前一更高低点」统计用低点 | 同 R-040 | 跨 |
| R-045 | `docs/research/SPY_INTRADAY_MOMENTUM.md` | 成交量 | VWAP = 长桥 turnover ÷ volume（zr/research/spy_intraday_momentum.py:87-95） | 2023-10-23～2026-10-20 | 跨 |
| R-049 | `docs/research/OWNER_RULES_OOS.md` | 成交量与高低点 | f_vol1m_cv*、v_rel1_20d、Aroon、ADX（zr/features/entry_features.py:321-329,486-500） | 正股层 2025-01-02～2026-05-31 + 2026-09-01～10-21；期权层 2025-01-02～2026-04-28 | 正股层跨；期权层在之前 |
| R-052 | `docs/research/EXIT_FINALIZE.md` | 成交量与高低点 | 第 2 步复用 owner_rule_1/3/5（量与 Aroon / ADX） | 第 2 步 2025-01-02～2026-04-28 | 全部在之前 |
| R-054 | `docs/research/R054_INDEX_STOCK_LAYER.md` | 高低点 | 开盘区间 = 09:30–09:34 最高 / 最低（zr/research/r054_index_stock_layer.py:122）、ADX | 2023-10-02～最新（不含 2026-06～08）；SPY 检验族 2023-10-02～2024-07-30 | QQQ 跨；SPY 检验族在之前 |
| R-055 | `docs/research/night_20261111/MORNING_REPORT.md`、`docs/research/night_20261111/FACTOR_TABLE.md`、`docs/research/night_20261111/ENTRY_TIMING_VAL.md` | 成交量与高低点 | OBV、vol_spike、vwap_z、ATR / KDJ / ADX；离场按 K 线高低判止损止盈（zr/research/timing/features.py:229,436,494；timing/labels.py:154-155） | 2023-10-02～2026-05-31 | 跨（只有验证段末尾 05-15～05-31） |
| R-056 | `docs/research/R056_MU_TREND_HOLDOUT.md` | 成交量与高低点 | VWAP 过滤（zr/research/timing/trend.py:36-37，典型价含高低）；入场离场用开盘价、无高低障碍 | 留出期 2026-06-01～10-20（开发 / 验证段在之前） | 留出期全部在之后 |

## 不在清单内

- 只用分钟收盘 / 开盘（不受成交量与高低点影响，未加说明）：R-E4、R-E5、R-E6、R-E7、R-008、R-015（2026-09 起，在之后）、R-016（2026-06～08，在之后）。注意它们与清单内条目共用的报告（EVENT_LABEL_TABLE、SCAN_TABLE）末尾的说明只针对清单内条目。
- 未运行或作废：R-001、R-002（待跑）、R-050（暂停）、R-039（作废）、R-057（本轮用 Massive 数据）。
- 不用正股分钟 K：R-003（交易日志）、R-027（录制器推送，2026-10-21）、R-042、R-046、R-047、R-048（期权平价秒级价与 ThetaData 报价；行权价选择只用长桥 1 分钟收盘）。
- L-001：代码不在仓库（/tmp 脚本），无法核对，未加说明。
- 未登记但用了长桥分钟高低点或成交量的报告（未加说明）：`docs/research/POPULAR_CONFLUENCE_STOCK.md`、`docs/research/OWNER_ENTRY_MINING.md`、`docs/research/PRE_ENTRY_MICRO.md`。
- 用了含盘前盘后的长桥分钟 K 的条目（R-033、R-034、R-035、R-037、R-040、R-041、R-043、R-049、R-054、R-055）：2026-05 的口径变化是否也影响盘前盘后，未核对。
- 部分报告由脚本生成，重新生成会丢掉末尾的说明；重跑前以本清单为准。
