# 前视审计修复前后对比（fix/lookahead-audit 第 5 节）

生成：2026-10-22T17:35:37+00:00 UTC · 「修复前」取自 commit `3dbd6c6`（RTH-only、RANGE 三条件同时满足）· 复现 `uv run python -m zr.research.compare_before_after`

## 规则卡（第 6 节网格）

| symbol | metric | before | after |
|---|---|---|---|
| MU | cards | 16 | 49 |
| MU | bonferroni | 0 | 0 |
| MU | rand_ge_95 | 0 | 4 |
| MU | rows_positive | 766 | 49 |
| QQQ | cards | 1640 | 1901 |
| QQQ | bonferroni | 0 | 0 |
| QQQ | rand_ge_95 | 79 | 3 |
| QQQ | rows_positive | 3877 | 2071 |

## 模型样本外 AUC（第 8 节）

| symbol | auc_fixed_before | auc_fixed_after | auc_optuna_before | auc_optuna_after |
|---|---|---|---|---|
| MU | 0.56 | 0.56 | 0.54 | 0.56 |
| QQQ | 0.54 | 0.53 | 0.53 | 0.54 |

## 157 笔按状态分类（第 7 节）

| state | n_before | win_rate_before | sum_pnl_before | mean_pnl_before | n_after | win_rate_after | sum_pnl_after | mean_pnl_after |
|---|---|---|---|---|---|---|---|---|
| CHANNEL 反向 | 27 | 0.78 | 5613.77 | 207.92 | 28 | 0.75 | 125.83 | 4.49 |
| CHANNEL 同向 | 27 | 0.48 | 4332.29 | 160.46 | 27 | 0.48 | 510.42 | 18.9 |
| IGNITION 反向 | 11 | 0.36 | -10008.89 | -909.9 | 12 | 0.33 | -5128.24 | -427.35 |
| IGNITION 同向 | 9 | 0.89 | 2809.51 | 312.17 | 7 | 0.86 | 1243.73 | 177.68 |
| OTHER | 46 | 0.61 | 3388.79 | 73.67 | 44 | 0.57 | 5204.74 | 118.29 |
| RANGE | 3 | 0 | -460.07 | -153.36 | 10 | 0.5 | -1121.96 | -112.2 |
| 未覆盖（入场在首根 K 收盘前） | 5 | 0 | -4840.88 | -968.18 |  |  |  |  |
| 未覆盖（无正股 1 分钟 K） | 29 | 0.52 | -2454.11 | -84.62 | 29 | 0.52 | -2454.11 | -84.62 |

## 四个人工校验点（MU 2026-10-21）

| checkpoint | 符合比例_before | 符合比例_after |
|---|---|---|
| 09:30 后急跌→IGNITION 下行 | 0 | 0.71 |
| 10:00–12:30→RANGE | 0.04 | 0.21 |
| 12:18–13:36→CHANNEL 上行 | 0.3 | 0.46 |
| 13:00 回踩 K 被 E1 识别的定义数（共 36） | 12/36 | 0/36 |

边界：卡片与模型对比只覆盖 MU、QQQ；SPY 未覆盖（期权拉取暂停）；「修复前」的分类/模型数字来自已合并进 main 的报告文本，用正则提取，提取失败显示 None。