> 2027 机构化长期路线参考，已被新 CLAUDE.md 取代，不再作为操作指令

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ZR Capital is being built as a quantitative trading system. The long-term goal is to grow this codebase into a **professional, world-class, institutional quantitative hedge fund** — not a personal hobby project. Every architectural decision should be evaluated against that bar: would a multi-PM institutional shop accept this design in production?

The project is in early scaffolding stage. Files with real code: `main.py`, `backtest/` (`engine.py`, `metrics.py`), `strategy/technical/ma_rsi_strategy.py`, `utils/indicators.py`, `data/downloader.py`, and `config/loader.py`. Other dirs are placeholders defining intended architecture. Code comments, log output, and chart labels are in Chinese.

## Vision & roadmap

The buildout is staged from idea generation → portfolio construction → live execution → monitoring. The current state is **Phase 1**, baseline technical strategy with a working backtest loop and institutional metrics. Stages are roughly sequential but later phases can begin once their dependencies are stable.

| Phase | Theme | What gets built | Exit criteria |
|---|---|---|---|
| 1 | **Basic strategy + backtest + metrics** *(NEEDS REWORK — see Phase 1 Audit section)* | MA+RSI rules, single-asset backtest engine, full metrics report (Sharpe, Calmar, drawdown, win rate, profit factor) | Backtest runs but a 2026-05-21 code audit found look-ahead bias and 4 other issues; results not yet trustworthy ⚠️ |
| 2 | **Data layer + statistical foundations** *(next)* | `data/` (downloader, cleaner, storage, features, universe, validators), `utils/statistics.py` (ADF/KPSS, correlation, IC/rank-IC, significance testing), unified schema and point-in-time storage | A clean, point-in-time data spine and statistical toolkit usable by every later phase |
| 3 | **Factor engineering** | Technical, fundamental, and alpha factors built on the data layer; IC / rank-IC analysis, factor decay studies, factor combination logic under `research/factors/` and `research/signals/` | A vetted factor library with documented IC/decay properties feeding strategies and ML |
| 4 | **Multi-factor models + tree-based ML** | `strategy/ml/tree.py` (XGBoost, LightGBM) as the primary ML track; walk-forward CV, leakage controls, hyperparam tracking; ensembles via `strategy/ml/ensemble.py`; neural models (`neural.py`) optional follow-on | Tree-based models that beat technical baselines net of costs on out-of-sample data |
| 5 | **Portfolio management** | `portfolio/optimizer.py` (Markowitz, extending to risk-parity / Kelly / Black-Litterman), `portfolio/weighting.py`, `portfolio/rebalance.py` | A combined book that allocates capital across strategies with scheduled / threshold rebalances |
| 6 | **Risk control** | `risk/position.py` (pyramid scaling), `risk/var.py`, `risk/drawdown.py`, `risk/exposure.py`, `risk/compliance.py`; pre-trade checks and post-trade attribution | Risk limits enforced before every order; kill-switches on limit breach |
| 7 | **Execution layer** | `execution/longbridge.py` (live broker), `execution/simulator.py` (shared with backtest), `execution/slippage.py`, `execution/cost.py` | Live-equivalent order flow with realistic fills feeding back into the backtest cost model |
| 8 | **Monitoring** | `monitor/logger.py`, `monitor/alert.py`, `monitor/performance.py` — structured logs, drawdown / signal-decay / data-quality alerts, real-time PnL and exposure | Runtime observability an institutional risk committee would accept |
| 9 | **Dashboard (Streamlit)** | `dashboard/app.py` plus `dashboard/pages/{backtest,live,research}.py` and shared `components/` | Operator console for backtest visualization, live book, and research exploration |
| 10 | **Systematization** | Scheduled runs, automation (CI, cron jobs, retraining pipelines), audit logs, per-environment `config/` (dev/paper/live), regression + property + integration tests, documentation | Reproducible, observable, auditable system suitable for outside capital |

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
- `engine.py` currently works around `yfinance` MultiIndex columns with a per-row `hasattr(row['Close'], 'iloc')` check. The 2026-05-21 audit flagged this as a band-aid (issue #5) — the fix is to flatten columns to single level in `downloader.py`, NOT to preserve this pattern. Remove the hack during the engine rework.

## Roadmap refinements (agreed)

These refinements override the surface reading of the phase table above. When the table and this section disagree, this section wins.

1. **Data layer + factor engineering are core foundations, not late-stage work.** Pull them forward into Phase 2–3. A factor library and clean data spine must exist before strategies multiply, not after — otherwise every later phase inherits broken plumbing.
2. **Statistical tooling is a Phase 2 deliverable.** Stationarity tests (ADF/KPSS), correlation matrices, IC / rank-IC analysis, and significance testing (t-stats, bootstrap, multiple-testing correction) ship alongside the strategy registry — they are how strategies get vetted, not optional research polish.
3. **Tree-based models (XGBoost, LightGBM) are the primary ML track.** Phase 4 starts here. Sequence models (LSTM/Transformer/neural nets) are optional follow-ons, justified only when tabular models hit a ceiling on a specific problem.
4. **Mid-to-low frequency is the operating regime.** Holding periods range from days to weeks. This is set by individual-grade infrastructure (no co-location, no L2 feeds). HFT/microstructure strategies are explicitly out of scope until infra changes.

## Core quant principles

5. **Quant is data engineering, not strategy engineering.** Most edge in this codebase will come from clean, consistent, point-in-time data — not from cleverer signals. Phase 2 must include:
   - **Unified schema** across all assets and time frames (consistent column names, dtypes, timezones, index conventions).
   - **Database storage** (Parquet + DuckDB or Postgres) — no more ad-hoc `yfinance.download` calls scattered across strategies.
   - **Multi-period data management** — daily / weekly / intraday bars stored with explicit period tags; resampling is a single audited code path.
   - **Field naming standards** — `close_adj`, `ret_1d`, `vol_20d`, etc., defined once in a data dictionary.
   - **Data quality checks** — missing bars, splits/dividends, survivorship-bias filters, outlier detection.
   - **Backtest-vs-live data consistency** — the same field on the same date must return the same value in research and production. Snapshot vendor data at ingestion time.
6. **MA crossovers are lagging by construction.** Golden/death crosses confirm trends that already happened. Phase 3 must introduce **leading signals**:
   - Capital flow / order flow (net buying, large-trade imbalance).
   - Options chain anomalies (IV skew shifts, unusual volume, put/call ratios).
   - Fundamental changes (earnings revisions, guidance, insider activity).
   - Multi-factor combinations that anticipate price moves rather than confirm them.
   Treat the MA+RSI baseline as a benchmark to beat, not a template to replicate.
7. **Position management uses pyramid scaling, not all-in / all-out.** Phase 6 (`risk/position.py`, working with `portfolio/weighting.py`) implements:
   - **Scaling-in** — add to winners in tranches as the thesis confirms (e.g., 1/3, 1/3, 1/3 across confirmation levels).
   - **Scaling-out** — trim winners progressively at target levels; never sell the whole position on a single signal.
   - Position size is a continuous function of signal strength and risk, not a binary flag.

## Known risks (must be addressed in their phases)

Each item below is a known way this system will produce misleading results if ignored. The phase listed is where the fix lands.

8. **Look-ahead bias** → **Phase 2 data layer.** Any computation that uses information not yet available at the decision timestamp (e.g., today's close to decide today's open trade, restated fundamentals, future-dated index membership). Fix: strict point-in-time data in `data/storage.py`, decision-time slicing, leakage audit on every feature in `data/features.py`.
9. **Fill-ability issues** → **Phase 7 slippage model.** Assuming we always fill at the printed price ignores spread, depth, and adverse selection. Fix: explicit slippage model (`execution/slippage.py`) parameterized by volume share, volatility, and side; reject fills that exceed a notional fraction of bar volume.
10. **Backtest matching bias** → **Phase 7 execution simulation.** Backtest fills must mirror what the live broker would actually do (queue position, partial fills, halts, locked markets). Fix: `execution/simulator.py` shared between backtest and paper-trading, fed the same order book / bar assumptions.
11. **Risk exposure control gap** → **Phase 6 risk module.** Without VaR/CVaR, correlation monitoring, sector/factor exposure limits, and gross/net caps, a "winning" book can blow up on a single regime change. Fix: pre-trade exposure checks and post-trade attribution; kill-switch on limit breaches.
12. **Distribution shift** → **Phase 4 ML.** Market regimes change; a model fit on 2015–2020 may be actively harmful in 2026. Fix: rolling-window retraining, walk-forward CV, out-of-sample holdout that is never touched during research, regime-conditioned performance reports.
13. **LLM training data contamination.** Any LLM-based strategy (signal generation, news scoring, agentic research) must be backtested **only on data dated AFTER the model's training cutoff** — otherwise the model has already "seen" the answers and results are uninvestable. Traditional ML (XGBoost, LightGBM, ridge, etc.) is **not** affected by this; standard walk-forward CV is sufficient for those.

## Data source strategy (US equities)

14. `data/downloader.py` will expose a **multi-source abstraction layer** so strategies never name a vendor directly. Vendor swaps stay local to the downloader.
    - **Market data (OHLCV, quotes):** `yfinance` (free, research) + **Longbridge API** (live trading) + **Polygon.io** (premium tier, intraday + historical depth).
    - **Fundamentals:** `yfinance` + **Longbridge**.
    - **News / sentiment:** **Bloomberg API** / **Finnhub** (future; gated behind cost/access).
    - **Macro:** **FRED** (Federal Reserve Economic Data — official, free, definitive for US macro series).

   Contract: every strategy and research notebook reads through `data/` interfaces; no module outside `data/` may import a vendor SDK directly.

## Future exploration (not main roadmap)

Items below are tracked but not committed milestones. Revisit when their prerequisites are stable.

15. **MACD indicator (DIF / DEA / histogram)** — additive to `utils/indicators.py` once the leading-signal work in Phase 3 begins; useful as a momentum confirmation companion to MA+RSI.
16. **Anthropic Financial Skills** — evaluate as a Phase 3–4 research aid (fundamental parsing, filing summarization). Subject to the LLM-contamination rule in item 13.
17. **TradingAgents** (multi-agent research system, ~65K-star open source) — consider in Phase 3–4 as a research-assistant layer (not an execution layer). Sandbox first, then wire into `research/` if it produces vetted signals.

## TradingView Integration Plan

TradingView is an **auxiliary tool, not a replacement** for our core Python system. Core strategy logic, the backtest engine, and execution must remain in Python. TradingView is layered on top for visualization, alerting, and operator UX.

Integration points across phases:

**Phase 2–3 (Research stage):**
- Use free TradingView web for visual chart inspection and pattern recognition.
- Cross-reference our Python signals with TV charts for validation.
- Cost: $0.

**Phase 7–8 (Execution + Monitoring):**
- TradingView Pro subscription ($15–60/month) with Webhook alerts.
- TV price/indicator alerts → Webhook → push to Telegram / email / our `monitor/alert.py`.
- Reduces need for constant manual monitoring.

**Phase 9 (Dashboard):**
- Embed TradingView Widget (free) for live quotes and heatmaps in our Streamlit dashboard.
- Consider TradingView Charting Library (paid, enterprise) for professional-grade interactive K-line charts, comparable to professional broker terminals (verify which specific charting library any given broker uses before relying on this).

**Phase 10+ (After institutional setup):**
- Evaluate TradingView Charting Library commercial license for branded dashboard.

**DO NOT use TradingView for:**
- Writing core strategy logic (Pine Script locks you into their platform).
- Primary backtest engine (data accuracy and execution assumptions are suboptimal).
- Live trading execution (must go through Longbridge API).

## Cross-sectional analysis (核心方法论)

Cross-sectional analysis is a **core methodology, not optional**. It is a widely-used approach in quantitative investing, grounded in the Fama-French three-factor model (1992) and Jegadeesh-Titman momentum research (1993). We implement it explicitly in our system.

**Key distinction:**
- **Time-series analysis:** study one stock's history (MA, RSI, etc.) — what Phase 1 did.
- **Cross-sectional analysis:** at a given time, compare *all* stocks in the universe — rank by factors, select top N%, neutralize by sector/size.

**Phase 3 implementation:**
- All factors in `research/factors/` must support cross-sectional scoring.
- Add `research/cross_section/` module with:
  - Ranking utilities (percentile, z-score, decile).
  - Sector neutralization.
  - Market-cap neutralization.
  - Factor IC (Information Coefficient) computation across the universe.

**Phase 4 implementation:**
- Multi-factor models combine multiple cross-sectional scores.
- ML models (XGBoost, LightGBM) trained on cross-sectional features predict *relative ranking*, not absolute price.

**Phase 5 implementation:**
- Portfolio construction selects top-decile names from cross-sectional scores.
- Long-short strategies short the bottom decile, long the top decile (cross-sectional alpha).

**Rationale:** most retail quants only do time-series; cross-sectional methods are a major source of documented alpha in the academic literature.

## Phase 3 prerequisite — Engine-strategy decoupling refactor

Before adding any new strategies in Phase 3, **the backtest engine must first be decoupled from specific strategy logic.** Do this as **Phase 3 Step 0**, before any other Phase 3 work.

**Current problem:**
- `backtest/engine.py:run_backtest()` accepts strategy-specific parameters (`ma_short`, `ma_long`, `rsi_window`) directly.
- Adding new strategies (Bollinger, Momentum, ML-based) requires modifying the engine.
- This violates the open-closed principle and blocks Phase 3 expansion.

**Required refactor:**
1. Define an abstract `Strategy` base class in `strategy/base.py` with:
   - Abstract method: `generate_signal(data) -> 'buy' | 'sell' | 'hold'`.
   - Abstract method: `get_name() -> str`.
2. Refactor `strategy/technical/ma_rsi_strategy.py` to inherit from the `Strategy` base class.
3. Refactor `backtest/engine.py:run_backtest()` signature:
   - **Old:** `run_backtest(data, ma_short, ma_long, rsi_window, stop_loss, ...)`.
   - **New:** `run_backtest(data, strategy: Strategy, initial_cash, stop_loss, ...)`.
4. Engine calls `strategy.generate_signal(today_data)` instead of `generate_signals(..., ma_short, ...)`.
5. `main.py` instantiates the strategy object from YAML config and passes it to the engine.
6. **Verify:** existing MA+RSI backtest produces identical results to baseline commit `cd948ef` (6 trades, $105,353.76 terminal, −21.44% drawdown).

After this refactor, adding new strategies becomes trivial: subclass `Strategy`, no engine changes needed. This is the foundation for Phase 3 (multi-strategy factor research) and Phase 4 (ML strategies).

**Rationale:** separation of concerns; the engine should be strategy-agnostic. This is how mainstream open-source backtest frameworks are structured — zipline, backtrader, and qlib all use this strategy-base-class pattern (verifiable in their public repos).

## Phase 3 addition — Market regime detection

Add a new module: **`research/regime/`** for market state classification.

- Detect market regimes: trending, mean-reverting, high-vol, low-vol.
- Use indicators like ADX, realized volatility, cross-asset correlations.
- Tag each strategy with applicable regimes.
- System auto-switches or re-weights strategies based on the current regime.
- **Rationale:** the same strategy performs very differently across regimes; regime-aware allocation prevents drawdowns during unfavorable conditions.

## Phase 6 addition — Circuit breaker system

Add a new module: **`risk/circuit_breaker.py`**.

- Auto-halt trading when triggered by:
  - Single-day loss exceeds X% (configurable threshold).
  - Consecutive N-day losses.
  - Data source anomalies.
  - Strategy drawdown exceeds historical max + buffer.
- Requires manual override to resume.
- **Rationale:** protect capital during black-swan events (e.g., historical flash crashes and liquidity-vacuum events); systems without circuit breakers compound losses, systems with them survive.

## Phase 1 Code Audit (2026-05-21)

A line-by-line audit of all 6 code files. Conclusion: the Phase 1 backtest result (6 trades, Sharpe -0.41) is NOT trustworthy and must be reproduced after fixing the issues below.

### Critical issues (must fix before trusting any backtest)
1. **engine.py — look-ahead bias.** Signal computed from day i's data, fill executed at day i's CLOSE. Reality: you cannot trade at a close price that is only known at close. Fix: signal on day i -> fill at day i+1 OPEN.
2. **engine.py — bare `except:` swallows errors** and may reuse a stale `price` value, hiding bugs. Fix: remove bare except, handle NaN and boundary rows explicitly.
3. **indicators.py — RSI uses simple moving average, not Wilder's smoothing** (the 1978 standard). Result differs from every mainstream platform (TradingView etc.). Fix: use `ewm(alpha=1/window, adjust=False)`.
4. **indicators.py — RSI division-by-zero unhandled.** When avg_loss=0 (sustained rally) it produces NaN. Fix: RSI=100 when avg_loss=0.
5. **downloader.py + engine.py — MultiIndex band-aid.** yfinance 1.3.0 returns MultiIndex columns even for one ticker; engine hacks around it with `.iloc[0]`, which silently breaks for multiple tickers. Fix: flatten columns to single level in downloader.

### Minor issues (fix when convenient)
6. **metrics.py — profit_factor is mislabeled.** The function named `profit_factor` actually computes mean-profit / mean-loss (payoff ratio). True profit factor = sum(profits)/sum(losses). The earlier "3.01" was payoff ratio, not profit factor.
7. **downloader.py — make adjustment explicit.** yfinance 1.3.0 defaults auto_adjust=True (verified), but relying on the default is not reproducible. Write auto_adjust=True explicitly.
8. **metrics.py — risk-free rate hardcoded at 0.04**, should be read from config.

### Deferred (design simplifications, later phases)
9. All-in / all-out sizing -> Phase 5 position management.
10. No transaction cost / slippage -> execution layer.
11. Stop-loss uses close price, not intrabar -> Phase 6 risk.

### Verified correct (do not change)
- moving_average: rolling() correct, no look-ahead.
- cross_signal: golden/death cross via diff(), event definition correct.
- max_drawdown, total_return, annualized_return, sharpe_ratio, calmar_ratio, win_rate: algorithms correct.
- DataSource abstract base class design: correct, matches Qlib/Backtrader patterns.

### Rework order (this replaces the vague "Phase 2 next" framing)
Step 1: fix indicators.py (RSI Wilder + div-zero).
Step 2: fix downloader.py (explicit auto_adjust, flatten MultiIndex, empty-data check).
Step 3: refactor engine.py (look-ahead fix: signal day i -> fill day i+1 open; remove MultiIndex band-aid; fix exception handling; optionally do the Strategy-base-class decoupling here).
Step 4: fix metrics.py (profit_factor naming, risk-free from config).
Step 5: re-run backtest, compare before/after, document the new (trustworthy) numbers.
