import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.indicators import moving_average, rsi, cross_signal

def generate_signals(data, ma_short=20, ma_long=60, rsi_window=14):
    """生成买卖信号"""
    data = data.copy()
    data['MA_short'] = moving_average(data, ma_short)
    data['MA_long'] = moving_average(data, ma_long)
    data['RSI'] = rsi(data, rsi_window)
    data['Cross'] = cross_signal(data['MA_short'], data['MA_long'])
    return data

def get_trade_signal(cross, rsi_value, position, buy_price, current_price, stop_loss=0.08):
    """
    判断当前是否买入或卖出
    规则：
    - 买入：金叉 + RSI < 70（不超买）
    - 卖出：死叉 + RSI > 30（不超卖）
    - 止损：亏损超过stop_loss比例强制卖出
    返回: 'buy' / 'sell' / 'hold'
    """
    # 止损检查（优先级最高）
    if position == 1 and buy_price > 0:
        loss = (current_price - buy_price) / buy_price
        if loss < -stop_loss:
            return 'sell'

    # 金叉买入
    if cross == 1 and position == 0 and rsi_value < 70:
        return 'buy'

    # 死叉卖出
    if cross == -1 and position == 1 and rsi_value > 30:
        return 'sell'

    return 'hold'