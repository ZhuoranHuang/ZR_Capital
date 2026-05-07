import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.ma_rsi_strategy import generate_signals, get_trade_signal

def run_backtest(data, initial_cash=100000, ma_short=20, ma_long=60, 
                 rsi_window=14, stop_loss=0.08):
    """
    运行回测
    返回：交易记录、资产曲线、最终统计
    """
    data = generate_signals(data, ma_short, ma_long, rsi_window)
    
    cash = initial_cash
    shares = 0
    buy_price = 0
    position = 0  # 0=空仓 1=持仓
    
    portfolio = []
    trades = []

    for i, row in data.iterrows():
        try:
            price = float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close'])
            cross = float(row['Cross'].iloc[0]) if hasattr(row['Cross'], 'iloc') else float(row['Cross'])
            rsi_val = float(row['RSI'].iloc[0]) if hasattr(row['RSI'], 'iloc') else float(row['RSI'])
        except:
            portfolio.append(cash + shares * price)
            continue

        if str(cross) == 'nan' or str(rsi_val) == 'nan':
            portfolio.append(cash + shares * price)
            continue

        signal = get_trade_signal(cross, rsi_val, position, buy_price, price, stop_loss)

        if signal == 'buy':
            shares = cash / price
            cash = 0
            buy_price = price
            position = 1
            trades.append({
                'date': i.date(),
                'action': '买入',
                'price': round(price, 2),
                'rsi': round(rsi_val, 1),
                'assets': round(shares * price, 2)
            })

        elif signal == 'sell':
            cash = shares * price
            pnl = (price - buy_price) / buy_price * 100
            trades.append({
                'date': i.date(),
                'action': '卖出',
                'price': round(price, 2),
                'rsi': round(rsi_val, 1),
                'assets': round(cash, 2),
                'pnl': round(pnl, 2)
            })
            shares = 0
            position = 0
            buy_price = 0

        portfolio.append(cash + shares * price)

    data['Portfolio'] = portfolio
    final = data['Portfolio'].iloc[-1]
    
    stats = {
        'initial': initial_cash,
        'final': round(final, 2),
        'return_pct': round((final - initial_cash) / initial_cash * 100, 2),
        'n_trades': len([t for t in trades if t['action'] == '买入']),
    }

    return data, trades, stats