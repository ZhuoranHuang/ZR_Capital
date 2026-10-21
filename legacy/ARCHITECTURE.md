# ZR Capital Architecture Reference

This document is the quick reference for understanding the system. Read this whenever you feel lost in the codebase.

## Execution Flow: What happens when you run `python main.py`

1. main.py loads parameters from YAML via config/loader.py
2. main.py asks data/downloader.py for stock data (with 24h parquet cache)
3. main.py passes data and parameters to backtest/engine.py's run_backtest()
4. backtest/engine.py simulates each historical day:
   - For each day, calls strategy/technical/ma_rsi_strategy.py to get buy/sell signal
   - strategy/ uses utils/indicators.py to compute MA and RSI
   - Engine simulates the trade based on signal
5. backtest/metrics.py computes Sharpe, max drawdown, etc.
6. main.py plots results with matplotlib

## Folder Responsibilities

| Folder | Role | Input | Output | Called by |
|--------|------|-------|--------|-----------|
| config/ | Configuration center | YAML files | Parameter dicts | main.py |
| data/ | Data factory | Ticker, period | OHLCV DataFrame | main.py |
| utils/ | Toolbox | Price series | Technical indicators | strategy/ |
| strategy/ | Strategy brain | Data + params | Buy/sell signals | backtest/engine |
| backtest/ | Simulator | Data + strategy + params | Trade records, metrics | main.py |
| portfolio/ | Portfolio mgmt | Multiple signals | Position weights | (future) |
| risk/ | Risk control | Positions, signals | Risk alerts, circuit breaker | (future) |
| execution/ | Live trading | Signals | Real orders | (future) |
| monitor/ | Monitoring | System state | Logs, alerts | (future) |
| dashboard/ | Visualization | All data | Web UI | (future) |
| research/ | Research playground | Any | Factors, signals | (future) |

## Currently Active (Phase 1-2)

Only 5 folders are actually in use right now:
- config/ ✅
- data/ ✅
- utils/ ✅
- strategy/ ✅
- backtest/ ✅

Other folders are placeholders for future phases.

**Current status (2026-05-21):** Phase 1 code runs but a full audit found 5 critical issues (look-ahead bias, non-standard RSI, MultiIndex band-aid, etc.) — see the "Phase 1 Code Audit" section in CLAUDE.md. The Phase 1 backtest numbers are NOT yet trustworthy. A 5-step rework is pending before moving to new Phase 2/3 work.

## Key Design Principles

1. **Separation of concerns**: main.py only orchestrates; it does not contain business logic
2. **Vendor isolation**: External library imports are confined to single modules (e.g., yfinance only in data/downloader.py, yaml only in config/loader.py)
3. **Configuration externalization**: All strategy parameters live in YAML, not in code
4. **Strategy as pure function**: Strategies take data and params, return signals. They don't know about backtest vs live execution.

## When You're Lost

1. Read main.py top to bottom — it shows the full story
2. Check this ARCHITECTURE.md for folder responsibilities
3. Check CLAUDE.md for the 10-phase roadmap
