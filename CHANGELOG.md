# Changelog

All notable changes to `alpha-lens` are recorded here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Initial release

The first public release.

### Added
- Core statistics: Sharpe, Sortino, Calmar, CAGR, win rate, drawdown,
  rolling Sharpe, turnover.
- Rank Information Coefficient (Spearman) with proper forward-period
  shifting to prevent look-ahead bias.
- Regime detection: rule-based (default) and optional HMM via `hmmlearn`.
- Per-regime performance summaries and a transition matrix.
- Drawdown event detection with regime tagging (weighted by loss
  magnitude, not just frequency).
- Factor attribution via OLS, with t-stats, R², annualized alpha, and
  rolling-window attribution.
- Alpha decay analysis: IC at multiple forward horizons, exponential
  half-life fit, rolling IC and stability score.
- Factor correlation matrix and Variance Inflation Factors.
- **Production Readiness Score** with 7 components:
  - Deflated Sharpe Ratio (Bailey & López de Prado 2014).
  - Probability of Backtest Overfitting (PBO) via CSCV.
  - Minimum Backtest Length.
  - Out-of-sample 70/30 split and 5-window walk-forward.
  - Bootstrap Sharpe confidence interval and subsample stability.
  - Regime robustness.
  - Transaction cost sensitivity with break-even bisection.
- Standalone HTML report with embedded Plotly.js for offline use.
- 85 tests covering every analysis module.
- Three runnable examples: `quickstart.py`, `momentum_autopsy.py`, and
  `compare_real_vs_lucky.py`.
- API and methodology documentation under `docs/`.

[0.1.0]: https://github.com/alpha-lens/alpha-lens/releases/tag/v0.1.0
