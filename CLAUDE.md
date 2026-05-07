# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ZR Capital is being built as a quantitative trading system. The long-term goal is to grow this codebase into a **professional, world-class, institutional quantitative hedge fund** — not a personal hobby project. Every architectural decision should be evaluated against that bar: would a multi-PM institutional shop accept this design in production?

The project is in early scaffolding stage: only `main.py`, `backtest/`, `strategy/technical/ma_rsi_strategy.py`, and `utils/indicators.py` contain code. Most other top-level dirs (`config/`, `dashboard/`, `data/`, `execution/`, `monitor/`, `portfolio/`, `research/`, `risk/`, `tests/`) are empty placeholders that define the intended architecture. Code comments, log output, and chart labels are in Chinese.

## Vision & roadmap

The buildout is staged from idea generation → portfolio construction → live execution → monitoring. The current state is **Phase 1**, baseline technical strategy with a working backtest loop and institutional metrics. Stages are roughly sequential but later phases can begin once their dependencies are stable.

| Phase | Theme | What gets built | Exit criteria |
|---|---|---|---|
| 1 | **Technical strategy baseline** *(current)* | MA+RSI rules, single-asset backtest engine, full metrics report (Sharpe, Calmar, drawdown, win rate, profit factor) | Reproducible backtest on a single ticker with institutional-grade metrics ✅ |
| 2 | **Strategy library expansion** | Multi-asset universe, more `strategy/technical/*` rules (breakout, momentum, mean-reversion bands), parameter sweeps | A registry of strategies all conforming to the `generate_signals` + `get_trade_signal` interface |
| 3 | **Statistical & arbitrage strategies** | `strategy/statistical/` — pairs trading, cointegration, stat-arb baskets, vol-targeted carry | Strategies that depend on cross-sectional data, not just per-ticker series |
| 4 | **Fundamental overlay** | `strategy/fundamental/` — quality/value/growth screens, earnings/macro filters, fundamental data ingestion under `data/` | Strategies that combine price-based signals with fundamental gates |
| 5 | **Research infrastructure** | `research/factors/` (factor library), `research/signals/` (alpha research), `research/notebooks/` (Jupyter exploration); IC/IR analysis; factor decay studies | A research-to-production path: notebook → vetted signal → strategy module |
| 6 | **ML strategies** | `strategy/ml/` — feature pipelines off the factor library, classical ML (XGBoost, ridge), then sequence models (LSTM/Transformer) for return forecasting; walk-forward CV, leakage controls, hyperparam tracking | ML models that beat the technical baselines net of costs on out-of-sample data |
| 7 | **Risk & portfolio construction** | `risk/` — VaR, CVaR, exposure caps, vol targeting, correlation/regime monitoring, stop-loss policies; `portfolio/` — position sizing, multi-strategy capital allocation, Kelly/risk-parity/mean-variance optimizers, rebalance scheduling | A combined book that respects risk limits and allocates capital across strategies |
| 8 | **Execution** | `execution/` — paper-trading layer first, then broker adapters (IBKR / Alpaca / crypto venues), order routing, slippage & cost modeling, TWAP/VWAP/POV algos, smart order types | Live-equivalent order flow with realistic fills feeding back into the backtest cost model |
| 9 | **Monitoring & dashboards** | `monitor/` — real-time PnL, drawdown alerts, signal-decay & strategy-health watchdogs, kill-switches; `dashboard/{components,pages}` — operator UI for live book, trade blotter, factor exposures, risk reports | An ops console that an institutional risk committee would accept |
| 10 | **Production hardening** | `data/` pipelines (vendor ingestion, point-in-time storage, corporate actions), `config/` per-environment settings, `tests/` regression + property + integration tests, CI, scheduled jobs, audit logs, post-trade attribution | Reproducible, observable, auditable system suitable for outside capital |

## Planned modules

Every existing top-level directory is part of the architecture — none are decorative. The intended contents:

- **`strategy/`** — strategy families, each module exposes `generate_signals(data, **params)` + `get_trade_signal(...)`.
  - `technical/` — price/volume rules (MA+RSI baseline lives here; add breakout, momentum, mean-reversion).
  - `statistical/` — pairs, cointegration, stat-arb, vol-targeted carry.
  - `fundamental/` — value/quality/growth screens, earnings filters.
  - `ml/` — supervised models on engineered features, walk-forward CV, model registry.
- **`research/`** — alpha research that feeds `strategy/`.
  - `factors/` — reusable factor calculations (value, momentum, quality, vol, etc.) with point-in-time correctness.
  - `signals/` — alpha signal candidates (IC/IR analysis, decay studies, combination logic).
  - `notebooks/` — exploratory Jupyter work; promoted ideas graduate into `factors/` or `signals/`.
- **`backtest/`** — simulation engine and reporting.
  - `engine.py` — current row-by-row loop; will evolve to vectorized + event-driven with multi-asset support.
  - `metrics.py` — institutional-grade metrics report (Sharpe, Calmar, drawdown, win rate, profit factor; expand with CAGR, Sortino, Omega, tail ratios, factor exposures).
- **`risk/`** — pre-trade and post-trade risk controls: VaR/CVaR, exposure & concentration limits, vol targeting, correlation/regime detection, drawdown circuit breakers.
- **`portfolio/`** — capital allocation across strategies and positions: Kelly, risk-parity, mean-variance, Black-Litterman; rebalance scheduling; multi-strategy book aggregation.
- **`execution/`** — order lifecycle: paper-trading first, then broker adapters (IBKR, Alpaca, crypto venues), order routing, slippage/cost models, execution algos (TWAP/VWAP/POV).
- **`monitor/`** — runtime observability: live PnL, drawdown & exposure alerts, signal-decay watchdogs, kill-switches, post-trade attribution.
- **`dashboard/`** — operator UI on top of `monitor/`.
  - `components/` — reusable charts/tables (price, equity curve, exposure, factor heatmaps).
  - `pages/` — composed views (live book, trade blotter, risk report, strategy health).
- **`data/`** — market & alternative data ingestion, point-in-time storage, corporate actions, vendor adapters; should be the only layer that talks to external data sources.
- **`config/`** — environment configs (dev/paper/live), strategy parameter files, risk limits, broker credentials (loaded from env, never committed).
- **`utils/`** — pure-function shared helpers (`indicators.py` lives here; keep it stateless and dependency-light).
- **`tests/`** — unit, property-based, and integration tests; regression suites that pin known backtest results so refactors don't silently change PnL.

## Running

There is no test suite, lint config, or build step yet. `requirements.txt` is empty — install dependencies ad hoc.

```powershell
# Run the end-to-end backtest demo (downloads AAPL from yfinance, runs MA+RSI strategy, plots)
python main.py

# Required packages used by current code
pip install yfinance matplotlib pandas numpy
```

The Matplotlib charts hard-code `Microsoft YaHei` for Chinese glyph rendering — change this if running on a system without that font.

## Architecture

The runtime data flow for a backtest is:

1. `main.py` downloads OHLCV with `yfinance` and calls `backtest.engine.run_backtest`.
2. `backtest/engine.py` calls `strategy.technical.ma_rsi_strategy.generate_signals` to enrich the dataframe with `MA_short`, `MA_long`, `RSI`, `Cross` columns (computed via `utils/indicators.py`), then iterates row-by-row asking `get_trade_signal` for `buy` / `sell` / `hold`.
3. The engine maintains cash/shares/position state, records each fill into `trades`, and returns `(data, trades, report)` — where `report` is the formatted dict from `backtest.metrics.get_full_report` (Sharpe, Calmar, max drawdown, win rate, profit factor, etc., values pre-formatted as strings with Chinese keys).

Key seam: `get_trade_signal(cross, rsi_value, position, buy_price, current_price, stop_loss)` in `strategy/technical/ma_rsi_strategy.py` is the pure decision function. Stop-loss is checked first (highest priority), then golden-cross buy (requires RSI < 70), then death-cross sell (requires RSI > 30). New strategies — across all phases below — should follow the same `generate_signals` + `get_trade_signal` shape so the engine loop stays generic.

## Things to be aware of before editing

- `backtest/engine.py` and `strategy/technical/ma_rsi_strategy.py` both manipulate `sys.path` to add the project root so sibling-package imports resolve when scripts are run directly. Packages have `__init__.py` files, so prefer plain `from strategy.technical... import ...` over re-adding path hacks.
- `engine.py` defends against `yfinance` returning multi-column frames by checking `hasattr(row['Close'], 'iloc')` on each row — preserve this pattern when modifying the loop, or normalize the dataframe upstream instead.
