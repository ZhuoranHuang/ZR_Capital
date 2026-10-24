"""OWNER_ENTRY_MINING.md 的生成（数字全部读 owner_entry_mining 的落盘结果或现算）。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.features import entry_features as ef
from zr.ops.markdown import df_to_md
from zr.research import owner_entry_mining as m


def _fmt(df: pd.DataFrame, nd: int = 3) -> pd.DataFrame:
    return df.round(nd)


def _top(u: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return u.reindex(u["z_perm"].abs().sort_values(ascending=False).index).head(n)


def subgroup_pct(s: pd.DataFrame, cols: list[str], mask: pd.Series) -> pd.Series:
    sub = s[mask & s["win"]]
    if sub.empty:
        return pd.Series(np.nan, index=cols)
    P, eidx, _ = m.group_pct(sub, cols)
    return pd.Series(np.nanmean(P[np.arange(len(eidx)), eidx], axis=0), index=cols)


def rule_cover(s: pd.DataFrame, rule: dict) -> dict:
    conds = [(c["name"], c["op"], float(c["value"])) for c in rule["entry"]["all"]]

    def hit(df):
        if df.empty:
            return np.array([], bool)
        return np.logical_and.reduce(
            [m._cond(df[f].astype(float).to_numpy(), op, v) for f, op, v in conds]
        )

    out = {}
    for lab, mk in (
        ("全部", pd.Series(True, index=s.index)),
        ("多", s["dir"] == 1),
        ("空", s["dir"] == -1),
        ("7 月", s["month"] == 7),
        ("8 月", s["month"] == 8),
        ("DTE 0", s["dte"] == 0),
        ("DTE ≥1", s["dte"] >= 1),
        ("10:15 前", s["before_1015"]),
        ("10:15 后", ~s["before_1015"]),
    ):
        w = s[mk & s["win"] & (s["kind"] == "entry")]
        r = s[mk & s["win"] & (s["kind"] == "random")]
        lo = s[mk & ~s["win"] & (s["kind"] == "entry")]
        out[lab] = (
            f"{int(hit(w).sum())}/{len(w)} vs 随机 {hit(r).mean():.0%}"
            f"（亏损 {int(hit(lo).sum())}/{len(lo)}）"
            if len(w)
            else "—"
        )
    return out


def write(root: Path, s: pd.DataFrame, cols: list[str]) -> None:
    st = m.store(root)
    uni = {
        k: pd.read_parquet(st / f"uni_{k}.parquet") for k in ("A", "B", "C", "A_long", "A_short")
    }
    tr = pd.read_parquet(st / "truncation.parquet")
    ent = s[s["kind"] == "entry"]
    n_feat_all = len([c for c in s.columns if c not in m.META])
    fam_counts = pd.Series([ef.family_of(c) for c in s.columns if c not in m.META]).value_counts()
    L = []
    a = L.append
    a("# 负责人入场规律反推（第一条）")
    a("")
    a(
        "`uv run python -m zr.research.owner_entry_mining`（特征 `zr/features/entry_features.py"
        "`）。"
        "研究性挖掘：样本小（35 个盈利入场），所有候选规则只是假设，结论以样本外检验"
        "（R-049，`OWNER_RULES_OOS.md`）为准。"
    )
    a("")
    a("## 1. 样本与口径")
    a("")
    a(
        f"- 事件：D-030 入选段中 MU 的开仓买入委托 {len(ent)} 张（{ent['day'].nunique()} 个交易日，"
        f"{min(ent['day'])}～{max(ent['day'])}），时刻 = 委托提交时刻（整秒向下取整，特征只用 ≤ "
        "该秒的"
        f"数据）。盈利入场 {int(ent['win'].sum())}（所在段净盈亏 > 0）、"
        f"亏损 {int((~ent['win']).sum())}；"
        f"看涨 {int((ent['dir'] == 1).sum())}（盈利 {int((ent['win'] & (ent['dir'] == 1)).sum())}"
        "）、"
        f"看跌 {int((ent['dir'] == -1).sum())}（盈利 {int((ent['win'] & (ent['dir'] == -1)).sum())}"
        "）。"
        f"月份：{ent['month'].value_counts().sort_index().to_dict()}（6 月没有 MU 委托）。"
    )
    a(
        f"- 随机对照：每张委托 {m.N_RAND} 个时刻（同日、提交前后 30 分钟、09:31–15:59、剔除前后 "
        "2 分钟，"
        f"固定种子），方向 = 所配对委托的方向，共 {int((s['kind'] == 'random').sum())} 个。"
    )
    a(
        "- 正股价：看跌看涨平价反推的秒级价（`zr/data/intrabar_pull.py`，与样本外数据同一构建；"
        "这 15 天按需补拉，RTH 覆盖率均为 1.0）；K 线口径 C（D-028）。"
    )
    a(
        f"- 特征：共 {n_feat_all} 个（{', '.join(f'{k} {v}' for k, v in fam_counts.items())}；"
        "前缀含义见 `entry_features.py` 文档）；去掉缺失 > 30% 或常数的列后用于统计的 "
        f"{len(cols)} 个。暴力因子（brute）{int(fam_counts.get('brute', 0))} 个 = 6 个窗口 × "
        "（对数价 4 个算子 + 5 条非方向序列 × 5 个算子）。"
    )
    a(
        f"- 截断检验：{tr.groupby('day')['n_times'].first().sum()} 个时刻（全部 48 个入场 + 每张"
        "委托 1 个"
        f"随机时刻）× {tr['feature'].nunique()} 个特征 × 2 个方向，删掉该时刻之后的全部数据重算，"
        f"不一致 {int(tr['mismatch'].sum())} 次。"
    )
    a(
        "- 方向化：看涨按上涨为正、看跌按下跌为正；「顺向极值」= 看涨看高点、看跌看低点；%b / R"
        "SI / "
        "KDJ 看跌取 1 − x 或 100 − x。状态层标签只作报告列（第 8 节），不作特征。"
    )
    a("")
    a("## 2. 单特征（置换检验 500 次，BH q = 0.10）")
    a("")
    for k, title in (
        ("A", "任务 A：盈利入场 vs 其随机对照"),
        ("C", "任务 C：全部入场 vs 随机对照"),
        ("B", "任务 B：盈利 vs 亏损入场"),
    ):
        u = uni[k]
        exp5 = 0.05 * len(u)
        a(f"### {title}")
        a("")
        a(
            f"{len(u)} 个特征；经验 p ≤ 0.05 的 {int((u['p_perm'] <= 0.05).sum())} 个（全为零假设"
            "时期望约"
            f" {exp5:.0f} 个），p ≤ 0.01 的 {int((u['p_perm'] <= 0.01).sum())} 个（期望约 "
            f"{0.01 * len(u):.0f}）；BH（经验 p）通过 {int(u['bh_perm'].sum())} 个；BH（按置换分"
            "布正态近似"
            f"的 p，补充）通过 {int(u['bh_z'].sum())} 个。注意：500 次置换的最小经验 p 为 1/501 "
            "≈ 0.002，"
            f"{len(u)} 个特征下 BH 第一名门槛为 {0.1 / len(u):.5f}，经验 p 本身达不到，只有至少 "
            f"{math.ceil(0.002 / (0.1 / len(u)))} 个特征同时取到最小 p 时才可能有通过者。"
        )
        a("")
        cols_show = (
            ["feature", "family", "pct_mean", "auc", "med_entry", "med_random", "p_perm", "z_perm"]
            if k != "B"
            else ["feature", "family", "auc", "med_win", "med_lose", "p_perm", "z_perm"]
        )
        a(df_to_md(_fmt(_top(u)[cols_show])))
        a("")
        a(
            "pct_mean：入场时刻在自身 50 个随机时刻中的平均百分位（无差异 = 0.5）；auc：入场 vs"
            " 随机"
            "（B 为盈利 vs 亏损）合并 AUC；med_*：中位数。"
            if k != "B"
            else "auc < 0.5 表示该特征在盈利入场上偏低。"
        )
        a("")
    a("## 3. 对称性（多 / 空分别做任务 A）")
    a("")
    la, sa = uni["A_long"].set_index("feature"), uni["A_short"].set_index("feature")
    both = la.join(sa, lsuffix="_long", rsuffix="_short")
    rho = both[["pct_mean_long", "pct_mean_short"]].corr(method="spearman").iloc[0, 1]
    topA = _top(uni["A"], 15)["feature"]
    sym = both.loc[topA, ["pct_mean_long", "p_perm_long", "pct_mean_short", "p_perm_short"]]
    a(
        f"看涨盈利入场只有 {int(uni['A_long']['n_entry'].iloc[0])} 个，看跌 "
        f"{int(uni['A_short']['n_entry'].iloc[0])} 个。全部特征上两侧组内百分位均值的 Spearman 相"
        "关 "
        f"{rho:.2f}。任务 A 前 15 个特征在两侧的表现："
    )
    a("")
    a(df_to_md(_fmt(sym.reset_index())))
    a("")
    a("## 4. 稳健性（任务 A 前 15 个特征的组内百分位均值，分组）")
    a("")
    groups = {
        "7 月": s["month"] == 7,
        "8 月": s["month"] == 8,
        "DTE 0": s["dte"] == 0,
        "DTE ≥1": s["dte"] >= 1,
        "10:15 前": s["before_1015"],
        "10:15 后": ~s["before_1015"],
    }
    rob = pd.DataFrame({g: subgroup_pct(s, list(topA), mk) for g, mk in groups.items()})
    nwin = {g: int((mk & s["win"] & (s["kind"] == "entry")).sum()) for g, mk in groups.items()}
    rob.columns = [f"{g}（n={nwin[g]}）" for g in rob.columns]
    a(df_to_md(_fmt(rob.reset_index().rename(columns={"index": "feature"}))))
    a("")
    a("## 5. 模型（按交易日留一交叉验证；置换 500 次）")
    a("")
    mp = st / "models.parquet"
    if mp.exists():
        md = pd.read_parquet(mp)
        a(
            "梯度提升树 = sklearn HistGradientBoosting（本机 LightGBM 缺 libomp、未装 XGBoost；"
            "叶 ≤ 8、"
            "深度 ≤ 3、叶最少 10 个样本、L2 = 10、≤ 200 轮、学习率 0.05、类别加权）；L1 逻辑回归"
            "（中位数填补、标准化、C = 0.05、类别加权）。oos_auc：留一日的样本外预测合并 AUC；"
            "oos_within_pct：入场时刻的样本外分数在自身随机对照中的平均百分位；p：标签按任务方式"
            "打乱"
            "（A / C 每组随机指定一个成员为入场，B 打乱盈亏）后整套交叉验证重做 500 次的单侧经验"
            " p。"
        )
        a("")
        a(df_to_md(_fmt(md)))
        a("")
    ip = st / "importance.parquet"
    if ip.exists():
        imp = pd.read_parquet(ip)
        a(
            f"### 任务 A 样本外置换重要性（HGB，前 20；留一日基准 AUC "
            f"{imp['base_auc'].iloc[0]:.3f}；测试折内逐列打乱 3 次的 AUC 平均下降）"
        )
        a("")
        a(df_to_md(_fmt(imp.drop(columns="base_auc").head(20), 4)))
        a("")
    for k in ("A", "B", "C"):
        tp = st / f"tree_{k}.parquet"
        if tp.exists():
            t = pd.read_parquet(tp)
            a(
                f"### 深度 ≤ 3 决策树（任务 {k}，全样本拟合；留一日 AUC {t['lodo_auc'].iloc[0]:.3f}"
                "）"
            )
            a("")
            a(df_to_md(_fmt(t.drop(columns="lodo_auc"))))
            a("")
    a("## 6. 候选规则（`config/intrabar/owner_rule_*.yaml`）")
    a("")
    rs = pd.read_pickle(st / "rule_search.pkl")
    cv = pd.read_parquet(st / "rule_cv.parquet")
    a(
        f"模板「趋势方向判定（双向对称）+ 触发条件 + 不做条件」：趋势从 {len(m.TREND_DEF)} 个"
        "「正值 = 处于方向 d 的趋势中」的 K 线特征里按任务 A 的 z 取前 8 个（阈值 > 0 或盈利入场"
        " 20% / "
        "30% 分位），触发从秒级 / 当前 K / 暴力因子、不做条件从期权 / 日历 / 量 / 跨标的里各取 "
        "z 绝对值"
        "前 8 个（阈值 = 盈利入场 20% / 30% 或 70% / 80% 分位）；共搜索 "
        f"{len(rs)} 条组合，保留覆盖 ≥ {m.MIN_WIN_COVER} 个盈利入场且盈利占比 > 随机占比的，按倍数"
        "（盈利占比 / 随机占比，加 0.5 平滑）取前 5 条（趋势、触发特征互不重复）。"
    )
    a("")
    rows = []
    for f in sorted(m.RULE_DIR.glob("owner_rule_*.yaml")):
        r = yaml.safe_load(f.read_text(encoding="utf-8"))
        cs = {c["role"]: f"{c['name']} {c['op']} {c['value']:.4g}" for c in r["entry"]["all"]}
        ins = r["in_sample"]
        rows.append(
            {
                "规则": f.stem,
                "趋势": cs.get("trend", ""),
                "触发": cs.get("trigger", ""),
                "满足才做": cs.get("no_trade", "—"),
                "盈利入场覆盖": ins["win_cover"],
                "盈利占比": ins["win_share"],
                "随机占比": ins["random_share"],
                "倍数": ins["lift"],
                "亏损入场覆盖": ins["lose_cover"],
            }
        )
    a(df_to_md(pd.DataFrame(rows)))
    a("")
    a("分组覆盖（盈利入场被覆盖数 / 该组盈利入场数，随机占比，亏损入场覆盖）：")
    a("")
    cov = []
    for f in sorted(m.RULE_DIR.glob("owner_rule_*.yaml")):
        r = yaml.safe_load(f.read_text(encoding="utf-8"))
        cov.append({"规则": f.stem, **rule_cover(s, r)})
    a(df_to_md(pd.DataFrame(cov)))
    a("")
    tw, tc = int(cv["test_wins"].sum()), int(cv["test_win_cover"].sum())
    rn, rc = int(cv["test_rand"].sum()), int(cv["test_rand_cover"].sum())
    a(
        "**规则搜索流程的留一日检验**：每折在其余交易日上重做单特征统计与模板搜索、取倍数最高的"
        "一条，"
        f"在留出日上看：盈利入场被触发 {tc}/{tw}（{tc / tw:.1%}），随机时刻被触发 {rc}/{rn}"
        f"（{rc / rn:.1%}）。"
        + (
            "留出日上盈利入场的触发比例不高于随机时刻——同一流程选出的规则在没见过的日子上没有"
            "区分盈利入场的能力，样本内倍数主要是选择偏差。"
            if tc / tw <= rc / rn
            else "留出日上盈利入场的触发比例高于随机时刻。"
        )
    )
    a("")
    a(df_to_md(_fmt(cv)))
    a("")
    a("## 7. 与以前诊断的对照")
    a("")
    a(_compare_prior(root, uni["C"]))
    a("")
    a("## 8. 状态层（报告列，不作特征）")
    a("")
    w = s[s["win"]]
    ct = (
        pd.crosstab(w["state"], w["kind"], normalize="columns")
        .round(3)
        .reset_index()
        .rename(columns={"entry": "盈利入场", "random": "其随机对照"})
    )
    a(
        "最后一根已完成 2 分钟 K（口径 C）的状态层标签（+1 / −1 为状态方向，未按所买方向调整）"
        "在盈利入场与其随机对照中的占比："
    )
    a("")
    a(df_to_md(ct))
    a("")
    a("## 9. 外部策略对照")
    a("")
    block = external_comparison(uni)
    a(block)
    a("")
    _write_survey_section(block)
    a("## 10. 边界")
    a("")
    a(
        "- 数据：MU，2026-07-07～2026-08-06 的 15 个交易日，48 张负责人 D-030 开仓买入委托（35 "
        "盈利、"
        "13 亏损；看涨只有 9 张、其中盈利 5 张）+ 2,400 个随机时刻。6 月无 MU 委托、8 月只有 1 天，"
        "月份稳健性实际只能比较 7 月与 8 月 6 日一天。"
    )
    a(
        "- 盈亏标签 = 所在段的净盈亏（含负责人真实的加减仓与离场），不是入场时刻本身的好坏；同一"
        "天的多"
        "张委托不独立。提交时刻 ≠ 决定时刻。"
    )
    a(
        "- 正股价为平价反推（误差中位数约 1–3 bp）；期权特征来自反推所用的最近到期、平值附近合约"
        "对，不是"
        "负责人所买的合约；期权成交量特征来自本地期权 1 分钟 bar（行权价带宽有限）。"
    )
    a(
        "- 置换 p 的分辨率 0.002；多重比较按 BH 处理后无单特征通过（第 2 节）；模型与规则的样本"
        "内结果"
        "不能外推，规则只作为 R-049 的预登记候选。"
    )
    a("- 2026-06～08 是成交参数校准窗口（D-033），本文结果不得作为任何规则的检验。")
    a("")
    m.OUT.write_text("\n".join(L), encoding="utf-8")


def _compare_prior(root: Path, uc: pd.DataFrame) -> str:
    from zr.research import pre_entry_micro as pem

    base = root / "research" / "pre_entry_micro"
    if not (base / "owner.parquet").exists():
        return "（PRE_ENTRY_MICRO 的落盘结果不存在，未对照）"
    cmp = pem.compare(
        pd.read_parquet(base / "owner.parquet"), pd.read_parquet(base / "random.parquet")
    )
    mp = {
        "ret_30s_bp": "s_ret30_bp",
        "ret_60s_bp": "s_ret60_bp",
        "ret_120s_bp": "s_ret120_bp",
        "dist_ext_with_5m_bp": "s_ext_with5m_bp",
        "dist_ext_against_5m_bp": "s_ext_against5m_bp",
        "dist_ext_with_15m_bp": "s_ext_with15m_bp",
        "dist_ext_against_15m_bp": "s_ext_against15m_bp",
        "dist_ext_against_30m_bp": "s_ext_against30m_bp",
        "dist_ext_against_day_bp": "s_ext_against_day_bp",
        "opt_updates_10s": "o_updates10s",
        "opt_updates_30s": "o_updates30s",
    }
    u = uc.set_index("feature")
    rows = []
    for a_, b_ in mp.items():
        r = cmp[cmp["特征"] == a_]
        rows.append(
            {
                "PRE_ENTRY_MICRO 特征": a_,
                "其百分位均值（1,000 随机）": float(r["自身随机中的百分位均值"].iloc[0])
                if len(r)
                else math.nan,
                "本文特征": b_,
                "本文任务 C 百分位均值（50 随机）": float(u.loc[b_, "pct_mean"])
                if b_ in u.index
                else math.nan,
                "本文经验 p": float(u.loc[b_, "p_perm"]) if b_ in u.index else math.nan,
            }
        )
    t = pd.DataFrame(rows).round(3)
    return (
        "PRE_ENTRY_MICRO.md（同一批 48 张委托，正股价用负责人所买合约同行权价的合约对反推，随机对照"
        "每张 1,000 个）与本文任务 C（平价价来自最近到期平值合约对，随机 50 个）在同义特征上的对照"
        "（两边的数据构建不同，数值应接近而不必相等）：\n\n"
        + df_to_md(t)
        + "\n\nENTRY_TIMING_DIAG.md：提交时刻在 2 分钟 K 中的位置分布、正在形成的 K 的柱方向与"
        "所买方向一致的比例（44%–50%）等，在本文对应 live2_* 与 b2_m6_* 特征（第 2 节）。"
    )


SURVEY_MAP = (
    ("价格在长均线趋势侧", "b2_dist_ma60_atr"),
    ("价格在长均线趋势侧", "b5_dist_ma60_atr"),
    ("均线排列", "b2_ma_align"),
    ("均线排列", "b5_ma_align"),
    ("通道 / 当日极值突破（距顺向当日极值，越小越近）", "s_ext_with_day_bp"),
    ("通道突破（处在 60 分钟新极值）", "s_at_ext_with60m"),
    ("开盘区间方向（相对开盘价）", "s_vs_open_bp"),
    ("ADX 强度", "b2_adx14"),
    ("ADX 强度", "b5_adx14"),
    ("摆动结构（高低点）", "b2_struct10"),
    ("高周期确认（15 分钟中轨同侧）", "b15_side_count10"),
    ("RSI(2) 回调（方向化，越低越是回调）", "b2_rsi2"),
    ("均值回归（%b 方向化，越低越靠逆向轨）", "live2_pctb"),
    ("负责人特征：距逆向 15 分钟极值", "s_ext_against15m_bp"),
    ("负责人特征：近 10 分钟方向化涨跌", "s_ret600_bp"),
)


def external_comparison(uni: dict) -> str:
    """调研中常见的「趋势怎么判断」→ 本文最接近的方向化特征，在负责人盈利入场上的表现。"""
    A, B = uni["A"].set_index("feature"), uni["B"].set_index("feature")
    rows = []
    for lab, f in SURVEY_MAP:
        if f not in A.index:
            continue
        rows.append(
            {
                "调研中的判定方式": lab,
                "对应特征": f,
                "盈利入场中位数": A.loc[f, "med_entry"],
                "随机中位数": A.loc[f, "med_random"],
                "任务 A 百分位均值": A.loc[f, "pct_mean"],
                "任务 A 经验 p": A.loc[f, "p_perm"],
                "任务 B AUC（盈 vs 亏）": B.loc[f, "auc"],
            }
        )
    t = pd.DataFrame(rows).round(3)
    fused = sorted(m.RULE_DIR.glob("fused_rule_*.yaml"))
    return (
        "调研（STRATEGY_SURVEY.md）中最常见的趋势判定 = 方向（价格 / 均线相对长均线、均线排列、通道"
        "或开盘区间突破）+ 强度（ADX 20–25）。这些判定在负责人盈利入场时刻与同日随机时刻上的差别"
        "（数据与口径同第 1 节；任务 A 百分位均值 0.5 = 无差别）：\n\n"
        + df_to_md(t)
        + "\n\n融合规则（`config/intrabar/fused_rule_*.yaml`，"
        f"{len(fused)} 条）：每条 owner_rule 加上调研最常见的「EMA50 在 EMA200 趋势侧 + ADX(14) "
        "> 20」，"
        "参数取调研默认值、未在任何数据上调整；与 owner_rule、调研三条必测规则一起进入 R-049 样本外"
        "检验（OWNER_RULES_OOS.md）。"
    )


def _write_survey_section(block: str) -> None:
    p = m.REPO / "docs" / "research" / "STRATEGY_SURVEY.md"
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8")
    i = txt.find("## 7. 与负责人入场规律的对照")
    j = txt.find("\n## 8.", i)
    if i < 0 or j < 0:
        return
    new = (
        "## 7. 与负责人入场规律的对照\n\n由 `zr/research/owner_entry_report.py` 生成"
        f"（第一条 OWNER_ENTRY_MINING.md 第 9 节）。\n\n{block}\n"
    )
    p.write_text(txt[:i] + new + txt[j:], encoding="utf-8")
