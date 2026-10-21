import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from strategy.technical.ma_rsi_strategy import generate_signals, get_trade_signal
from backtest.metrics import get_full_report


def run_backtest(data, initial_cash=100000, ma_short=20, ma_long=60,
                 rsi_window=14, stop_loss=0.08):
    """
    运行回测（无前视偏差版本）
    决策时点：第 i-1 日收盘后
    执行时点：第 i 日开盘价
    最后一个 bar 上产生的信号无法被执行（没有下一日开盘价），自然丢弃
    返回：(data, trades, report)
    """
    data = generate_signals(data, ma_short, ma_long, rsi_window)

    cash = initial_cash
    shares = 0
    buy_price = 0
    position = 0  # 0=空仓 1=持仓

    portfolio = []
    trades = []

    dates = data.index

    for i in range(len(data)):
        row = data.iloc[i]
        close_today = float(row['Close'])

        if i == 0:
            portfolio.append(cash + shares * close_today)
            continue

        prev = data.iloc[i - 1]
        cross = prev['Cross']
        rsi_val = prev['RSI']

        if pd.isna(cross) or pd.isna(rsi_val):
            portfolio.append(cash + shares * close_today)
            continue

        cross = float(cross)
        rsi_val = float(rsi_val)
        prev_close = float(prev['Close'])
        open_today = float(row['Open'])

        signal = get_trade_signal(cross, rsi_val, position, buy_price, prev_close, stop_loss)

        if signal == 'buy':
            shares = cash / open_today
            cash = 0
            buy_price = open_today
            position = 1
            trades.append({
                'date': dates[i].date(),
                'action': '买入',
                'price': round(open_today, 2),
                'rsi': round(rsi_val, 1),
                'assets': round(shares * open_today, 2)
            })

        elif signal == 'sell':
            cash = shares * open_today
            pnl = (open_today - buy_price) / buy_price * 100
            trades.append({
                'date': dates[i].date(),
                'action': '卖出',
                'price': round(open_today, 2),
                'rsi': round(rsi_val, 1),
                'assets': round(cash, 2),
                'pnl': round(pnl, 2)
            })
            shares = 0
            position = 0
            buy_price = 0

        portfolio.append(cash + shares * close_today)

    data = data.copy()
    data['Portfolio'] = portfolio

    report = get_full_report(data['Portfolio'], trades, initial_cash)

    return data, trades, report
