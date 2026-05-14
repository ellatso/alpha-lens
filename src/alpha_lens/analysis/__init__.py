"""Analysis modules — the intellectual core of alpha-lens."""

from alpha_lens.analysis.attribution import attribute_returns, rolling_attribution
from alpha_lens.analysis.correlation import analyze_correlation, compute_vif
from alpha_lens.analysis.cost_analysis import analyze_costs, apply_cost, find_breakeven_cost
from alpha_lens.analysis.decay import analyze_decay, estimate_half_life, rolling_ic
from alpha_lens.analysis.drawdown import analyze_drawdowns, find_drawdown_events
from alpha_lens.analysis.overfitting import (
    analyze_overfitting,
    deflated_sharpe_ratio,
    minimum_backtest_length,
    probability_of_backtest_overfitting,
)
from alpha_lens.analysis.regime import (
    analyze_regimes,
    detect_regimes_hmm,
    detect_regimes_rule_based,
)
from alpha_lens.analysis.robustness import (
    analyze_robustness,
    bootstrap_sharpe,
    subsample_sharpes,
)
from alpha_lens.analysis.scoring import compute_readiness_score
from alpha_lens.analysis.statistics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    cumulative_returns,
    drawdown_series,
    excess_returns,
    information_coefficient,
    max_drawdown,
    rank_information_coefficient,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    turnover_from_positions,
    win_rate,
)
from alpha_lens.analysis.validation import (
    analyze_validation,
    train_test_split,
    walk_forward_sharpes,
)

__all__ = [
    "annualized_volatility",
    "cagr",
    "calmar_ratio",
    "cumulative_returns",
    "drawdown_series",
    "excess_returns",
    "information_coefficient",
    "max_drawdown",
    "rank_information_coefficient",
    "rolling_sharpe",
    "sharpe_ratio",
    "sortino_ratio",
    "total_return",
    "turnover_from_positions",
    "win_rate",
    "analyze_regimes",
    "detect_regimes_hmm",
    "detect_regimes_rule_based",
    "analyze_drawdowns",
    "find_drawdown_events",
    "attribute_returns",
    "rolling_attribution",
    "analyze_decay",
    "estimate_half_life",
    "rolling_ic",
    "analyze_correlation",
    "compute_vif",
    "analyze_overfitting",
    "deflated_sharpe_ratio",
    "minimum_backtest_length",
    "probability_of_backtest_overfitting",
    "analyze_validation",
    "train_test_split",
    "walk_forward_sharpes",
    "analyze_robustness",
    "bootstrap_sharpe",
    "subsample_sharpes",
    "analyze_costs",
    "apply_cost",
    "find_breakeven_cost",
    "compute_readiness_score",
]
