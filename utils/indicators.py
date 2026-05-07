import pandas as pd

def moving_average(data, window):
    """计算移动均线"""
    return data['Close'].rolling(window=window).mean()

def rsi(data, window=14):
    """计算RSI相对强弱指数"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def cross_signal(ma_short, ma_long):
    """计算金叉死叉信号"""
    signal = (ma_short > ma_long).astype(int)
    cross = signal.diff()
    return cross