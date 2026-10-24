"""POPULAR_CONFLUENCE_STOCK.md 的生成（数字全部来自 popular_confluence 的回测结果）。"""

from __future__ import annotations

import math

import pandas as pd

from zr.ops.markdown import df_to_md
from zr.research import popular_confluence as pc

RULES = {
    "squeeze_ema200": (
        "LazyBear Squeeze Momentum：BB(20, 2) 与 KC(SMA20 ± 1.5 × SMA(TR, 20))，动能 = "
        "linreg(收盘 − (20 根高低中点与 SMA20 的均值), 20)；挤压释放（上一根 BB 在 KC 内、"
        "本根 BB 放出 KC）且动能 > 0 并收盘在 EMA200 上方做多（镜像做空）；动能连续两根走弱"
        "或变号出场。",
        "https://www.tradingview.com/script/nqQ1DT5a/（默认参数经页面源码核实）；EMA200 过滤"
        "为负责人指定",
    ),
    "supertrend_adx_ema200": (
        "SuperTrend(10, 3)（ATR = RMA(TR, 10)，源 hl2）由下转上、ADX(14) > 25、收盘在 EMA200 "
        "上方做多（镜像做空）；反向翻转出场。",
        "https://www.tradingview.com/script/P5Gu6F8k/（KivancOzbilgic SuperTrend，参数据调研"
        "记录，页面源码未验证）；ADX / EMA200 过滤为负责人指定",
    ),
    "bb_rsi_chartart": (
        "负责人给定参数：BB(200, 2)、RSI(14)；RSI 上穿 30 且收盘上穿下轨做多，RSI 下穿 70 且"
        "收盘下穿上轨做空；反向信号出场（反手）。",
        "https://www.tradingview.com/script/uCV8I4xA/（ChartArt「Bollinger + RSI, Double "
        "Strategy」）",
    ),
    "bb_rsi_chartart_src": (
        "同一脚本的源码默认：RSI(6) 以 50 为界（上穿 50 且收盘上穿下轨做多，下穿 50 且收盘下穿"
        "上轨做空），BB(200, 2)；反向信号出场。负责人给定的 RSI(14) 30/70 与源码不同，两版都报。",
        "https://www.tradingview.com/script/uCV8I4xA/（RSIlength = 6、RSIoverSold = "
        "RSIoverBought = 50，页面源码核实）",
    ),
    "wavetrend_mfi": (
        "LazyBear WaveTrend（hlc3，n1 = 10、n2 = 21，wt2 = SMA(wt1, 4)，±60）：wt1 在 −60 下方"
        "上穿 wt2 且 MFI(14) 上升做多；wt1 在 +60 上方下穿 wt2 且 MFI 下降做空；反向交叉出场。",
        "https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/"
        "（参数经页面源码核实）；MFI 条件为负责人指定",
    ),
    "four_confluence": (
        "四指标共振：MACD(12, 26, 9) 柱正负、RSI(14) 相对 50、KDJ(9, 3, 3) K 相对 D、收盘相对 "
        "BB(20, 2) 中轨；四项同向入场，任意两项反向出场。",
        "负责人描述（常见「多指标共振」写法，无单一原作者）",
    ),
    "macd_nsga2": (
        "MACD 复合（模仿截图 NSGA-II 策略）：MACD 金叉做多 / 死叉做空，可选过滤（EMA 趋势、ADX "
        "阈值、RSI 区间、成交量 > 倍数 × SMA20）；NSGA-II（pymoo，种群 40 × 25 代；目标：样本内"
        "扣费后收益最大、最大回撤最小，10 万美元口径）只在样本内优化，帕累托前沿上样本内夏普最高"
        "的一组冻结后测样本外；反向交叉出场。",
        "方法参考 https://arxiv.org/abs/2111.13364；截图原策略的过滤条件未找到"
        "（STRATEGY_SURVEY.md）",
    ),
}


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def _tbl(d: pd.DataFrame, macd: bool = False) -> pd.DataFrame:
    cols = {
        "标的": d["und"],
        "周期": d["tf"],
        "策略": d["strategy"],
    }
    if macd:
        cols["段"] = d["macd_part"]
    cols.update(
        {
            "方向": d["side"],
            "资金": d["capital"],
            "笔数": d["trades"],
            "胜率": d["win_rate"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}"),
            "盈亏比": d["payoff"].round(2),
            "扣费前": d["ret_gross"].map(_pct),
            "扣费后": d["ret"].map(_pct),
            "年化": d["ann"].map(_pct),
            "最大回撤": d["mdd"].map(lambda x: f"{x * 100:.1f}%"),
            "夏普": d["sharpe"].round(2),
            "买入持有": d["bh"].map(_pct),
            "相对买入持有": d["vs_bh"].map(_pct),
            "随机方向分位": d["rand_pct"].map(lambda x: "—" if pd.isna(x) else f"{x:.0%}"),
            "爆仓": d["wiped"].map(lambda x: "是" if x else ""),
        }
    )
    return pd.DataFrame(cols)


def write(df: pd.DataFrame) -> None:
    order = {"1d": 0, "1h": 1, "15m": 2}
    df = df.assign(_o=df["tf"].map(order)).sort_values(["und", "_o", "strategy", "side", "capital"])
    main = df[df["strategy"] != "macd_nsga2"]
    mac = df[df["strategy"] == "macd_nsga2"]
    win = main.groupby("tf")["window"].first().to_dict()
    rates = "、".join(f"{k} {v}" for k, v in pc.USDCNY.items())
    L = [
        "# 网上流行的单标的共振策略：正股回测（MU、QQQ）",
        "",
        "`uv run python -m zr.research.popular_confluence`。只做正股。收益、回撤、价格变动均为"
        "百分比；「随机方向分位」= 策略扣费后收益在 1,000 次随机对照（同样笔数与持仓根数、随机"
        "时点、随机方向；只做多版本只做多）中的分位。",
        "",
        "## 设定",
        "",
        f"- 数据：日线 = 长桥前复权日 K（VPS 只读拉取一次，2018-11 起用于指标预热），测试 "
        f"{win.get('1d')}；1 小时 / 15 分钟 = 本地长桥 RTH 1 分钟 K 按 09:30 起的网格合成（1 小时"
        f"最后一根为 15:30–16:00），测试 {win.get('15m')}，指标跨日连续计算（不按日重置）。",
        "- 成交：信号 K 收盘后、下一根 K 开盘，1 tick（0.01 美元）不利滑点；费用 D-013 正股逐单"
        "公式；满仓复利、整数股、同一时间一个仓位；持仓中出现反向入场信号先平后反手。",
        "- 日内版 15:50 ET（提前收盘日为收盘前 10 分钟）按该分钟开盘价清仓，不过夜；日线版可过夜。",
        "- 做空：不计借券费与保证金要求（未覆盖）；满仓做空遇大涨使权益 ≤ 0 时按收盘强平并停止"
        "（表中「爆仓」）。",
        "- 资金：1 万人民币按测试起始日 USD/CNY 折美元（FRED DEXCHUS，"
        f"https://fred.stlouisfed.org/series/DEXCHUS；{rates}）；10 万美元。",
        "- 买入持有：测试期首根开盘买入整数股、末根收盘卖出（含费用与滑点）。",
        "",
        "## 主表（MACD 复合见下一节）",
        "",
        df_to_md(_tbl(main)),
        "",
        "## MACD 复合（NSGA-II 样本内优化 → 冻结 → 样本外）",
        "",
        "样本内 / 样本外：日线 2021-10-01～2023-12-29 / 2024-01-02～2026-10-21；日内 "
        "2023-10-02～2024-06-28 / 2024-07-01～2026-10-21。每个标的 × 周期 × 方向（多空 / 只做多）"
        "各优化一次（10 万美元口径），两档资金共用同一组参数。",
        "",
        df_to_md(_tbl(mac, macd=True)),
        "",
        "冻结参数：",
        "",
        df_to_md(
            mac.drop_duplicates(["und", "tf", "side"])[["und", "tf", "side", "macd_params"]].rename(
                columns={"und": "标的", "tf": "周期", "side": "方向", "macd_params": "参数"}
            )
        ),
        "",
        "## 每条策略的规则与来源",
        "",
    ]
    for k, (rule, src) in RULES.items():
        L.append(f"- **{k}**：{rule} 来源：{src}")
    L += [
        "",
        "## 边界",
        "",
        "- 只有 MU、QQQ；日线约 5 年、日内约 3 年；参数为原作者默认或负责人指定（MACD 复合例外），"
        "结论不外推到其他参数、标的或时期。",
        "- 指标与 Pine 的差异：EMA 首值按本项目口径（span = n、adjust = False、首值做种），"
        "Pine 部分函数以 SMA 起算，预热后差异可忽略；RMA / ATR / RSI / ADX 为 Wilder。",
        "- 1 万人民币档在 MU 涨到数百美元后常买不起 1 股（信号被跳过）；D-013 每单最低 1 美元平台费"
        "对小资金占比高。",
        "- 日线版 MU 期间涨幅极大，满仓做空可能爆仓（见「爆仓」列）；真实账户会更早被追加保证金。",
        "",
    ]
    pc.OUT.write_text("\n".join(L), encoding="utf-8")
