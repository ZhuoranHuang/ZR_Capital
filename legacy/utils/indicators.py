import pandas as pd

def moving_average(data, window):
    """计算移动均线"""
    return data['Close'].rolling(window=window).mean()

def rsi(data, window=14):
    """计算RSI（Wilder平滑法，1978标准，与TradingView等主流平台一致）"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi

def cross_signal(ma_short, ma_long):
    """计算金叉死叉信号"""
    signal = (ma_short > ma_long).astype(int)
    cross = signal.diff()
    return cross