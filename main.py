import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backtest.engine import run_backtest
from config.loader import load_settings, load_strategy
from data.downloader import get_source

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

# ===== 参数加载 =====
settings = load_settings()
strategy = load_strategy('ma_rsi')

TICKER       = strategy['universe']['ticker']
PERIOD       = strategy['universe']['period']
INITIAL_CASH = strategy['capital']['initial_cash']
MA_SHORT     = strategy['params']['ma_short']
MA_LONG      = strategy['params']['ma_long']
RSI_WINDOW   = strategy['params']['rsi_window']
STOP_LOSS    = strategy['params']['stop_loss']

# ===== 加载数据 =====
print(f"正在加载 {TICKER} {PERIOD} 数据...")
source = get_source(settings['data']['default_source'])
data = source.get(TICKER, PERIOD, ttl_hours=settings['data']['cache_ttl_hours'])
print(f"数据加载完成，共 {len(data)} 条记录")

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