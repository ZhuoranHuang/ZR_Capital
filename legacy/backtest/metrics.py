import numpy as np
import pandas as pd

def total_return(portfolio):
    """总收益率"""
    return (portfolio.iloc[-1] - portfolio.iloc[0]) / portfolio.iloc[0] * 100

def annualized_return(portfolio, trading_days=252):
    """年化收益率"""
    total = portfolio.iloc[-1] / portfolio.iloc[0]
    years = len(portfolio) / trading_days
    return (total ** (1 / years) - 1) * 100

def max_drawdown(portfolio):
    """最大回撤：从最高点跌到最低点的最大幅度"""
    peak = portfolio.cummax()
    drawdown = (portfolio - peak) / peak
    return drawdown.min() * 100

def sharpe_ratio(portfolio, risk_free_rate=0.04, trading_days=252):
    """
    夏普比率：每承受一单位风险能获得多少超额收益
    > 1 = 良好  > 2 = 优秀  > 3 = 卓越
    """
    daily_returns = portfolio.pct_change().dropna()
    excess_returns = daily_returns - risk_free_rate / trading_days
    if excess_returns.std() == 0:
        return 0
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days)

def calmar_ratio(portfolio, trading_days=252):
    """
    卡玛比率：年化收益率 / 最大回撤
    机构常用指标，越高越好
    """
    ann_ret = annualized_return(portfolio, trading_days)
    max_dd = abs(max_drawdown(portfolio))
    if max_dd == 0:
        return 0
    return ann_ret / max_dd

def win_rate(trades):
    """胜率：盈利交易次数 / 总交易次数"""
    sells = [t for t in trades if t['action'] == '卖出' and 'pnl' in t]
    if not sells:
        return 0
    wins = [t for t in sells if t['pnl'] > 0]
    return len(wins) / len(sells) * 100

def payoff_ratio(trades):
    """
    盈亏比 (payoff ratio): 平均盈利 / 平均亏损
    > 1.5 = 良好  > 2 = 优秀
    """
    sells = [t for t in trades if t['action'] == '卖出' and 'pnl' in t]
    profits = [t['pnl'] for t in sells if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in sells if t['pnl'] < 0]
    if not losses:
        return float('inf')
    if not profits:
        return 0
    return np.mean(profits) / np.mean(losses)


def profit_factor(trades):
    """盈亏因子 (profit factor): 总盈利 / 总亏损"""
    sells = [t for t in trades if t['action'] == '卖出' and 'pnl' in t]
    profits = [t['pnl'] for t in sells if t['pnl'] > 0]
    losses = [abs(t['pnl']) for t in sells if t['pnl'] < 0]
    if not losses:
        return float('inf')
    if not profits:
        return 0
    return sum(profits) / sum(losses)

def get_full_report(portfolio, trades, initial_cash, risk_free_rate=0.04):
    """生成完整机构级回测报告"""
    report = {
        '初始资金':            f"${initial_cash:,.2f}",
        '最终资产':            f"${portfolio.iloc[-1]:,.2f}",
        '总收益率':            f"{total_return(portfolio):.2f}%",
        '年化收益率':          f"{annualized_return(portfolio):.2f}%",
        '最大回撤':            f"{max_drawdown(portfolio):.2f}%",
        '夏普比率':            f"{sharpe_ratio(portfolio, risk_free_rate=risk_free_rate):.2f}",
        '卡玛比率':            f"{calmar_ratio(portfolio):.2f}",
        '交易次数':            len([t for t in trades if t['action'] == '买入']),
        '胜率':                f"{win_rate(trades):.2f}%",
        '盈亏比(payoff)':      f"{payoff_ratio(trades):.2f}",
        '盈亏因子(profit factor)': f"{profit_factor(trades):.2f}",
    }
    return report