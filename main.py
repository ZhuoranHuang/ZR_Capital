import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backtest.engine import run_backtest

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

# ===== 参数设置 =====
TICKER = "AAPL"       # 股票代码
PERIOD = "5y"         # 回测周期
INITIAL_CASH = 100000 # 初始资金
MA_SHORT = 20         # 短期均线
MA_LONG = 60          # 长期均线
RSI_WINDOW = 14       # RSI周期
STOP_LOSS = 0.08      # 止损比例8%

# ===== 下载数据 =====
print(f"正在下载 {TICKER} {PERIOD} 数据...")
data = yf.download(TICKER, period=PERIOD)
print(f"数据下载完成，共 {len(data)} 条记录")

# ===== 运行回测 =====
data, trades, report = run_backtest(
    data,
    initial_cash=INITIAL_CASH,
    ma_short=MA_SHORT,
    ma_long=MA_LONG,
    rsi_window=RSI_WINDOW,
    stop_loss=STOP_LOSS
)

# ===== 打印交易记录 =====
print("\n===== 交易记录 =====")
for t in trades:
    if t['action'] == '买入':
        print(f"买入 {t['date']} 价格:${t['price']} RSI:{t['rsi']}")
    else:
        print(f"卖出 {t['date']} 价格:${t['price']} RSI:{t['rsi']} 盈亏:{t['pnl']}% 资产:${t['assets']}")

# ===== 打印统计 =====
print("\n===== 回测统计 =====")
for k, v in report.items():
    print(f"{k}: {v}")

# ===== 画图 =====
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

ax1.plot(data['Close'], label='收盘价', color='blue', alpha=0.6)
ax1.plot(data['MA_short'], label=f'MA{MA_SHORT}', color='orange')
ax1.plot(data['MA_long'], label=f'MA{MA_LONG}', color='red')
ax1.set_title(f'ZR Capital - {TICKER} 回测系统')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(data['RSI'], label='RSI', color='purple')
ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
ax2.set_ylabel('RSI')
ax2.legend()
ax2.grid(alpha=0.3)

ax3.plot(data['Portfolio'], label='策略收益', color='green')
ax3.axhline(y=INITIAL_CASH, color='gray', linestyle='--', label='初始资金')
ax3.set_ylabel('资产（美元）')
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()