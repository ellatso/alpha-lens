"""Configuration constants and dataclasses for alpha-lens.

Concentrating tunable knobs in one place makes it easy to reason about
how analysis decisions are made. Every default here is documented with
a rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Time-series constants
# ----------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
"""Number of trading days used for annualization. NOT 365.

This is the standard convention in equity quant finance. Using 365 would
underestimate Sharpe by a factor of √(365/252) ≈ 1.20.
"""

DEFAULT_RISK_FREE_RATE: float = 0.0
"""Default risk-free rate (annualized, decimal). 0 is a safe baseline.

For real analysis users should pass a sensible value (e.g. 0.04 for ~4%
T-bill yield). We don't default to anything market-specific.
"""


# ----------------------------------------------------------------------------
# Regime detection
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds for rule-based regime detection.

    The defaults work for daily US equity returns. For higher-frequency
    data or different markets, scale the windows accordingly.
    """

    short_sma_window: int = 20
    """Short-term moving average window (days)."""

    long_sma_window: int = 60
    """Long-term moving average window (days)."""

    vol_window: int = 20
    """Window for realized volatility estimation."""

    vol_high_percentile: float = 0.75
    """Percentile threshold above which volatility is considered 'high'."""

    bear_drawdown_threshold: float = 0.10
    """Drawdown from peak required to confirm a bear regime."""

    hmm_n_states: int | None = None
    """Number of HMM states. If None, auto-select via BIC (2-4)."""


# ----------------------------------------------------------------------------
# Overfitting detection
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class OverfittingConfig:
    """Parameters for overfitting diagnostics.

    These follow López de Prado (2018, "Advances in Financial Machine
    Learning") and Bailey & López de Prado (2014).
    """

    n_trials_assumed: int = 100
    """Number of strategy variants assumed to have been tested.

    The Deflated Sharpe Ratio penalizes a strategy by how many
    alternatives the researcher tried before picking this one. 100 is
    a moderate default — pessimistic enough to catch garden-variety
    overfitting, generous enough that a strategy with truly strong
    signal can still survive.
    """

    cscv_n_partitions: int = 16
    """Number of partitions for Combinatorially Symmetric Cross-Validation.

    Must be even. 16 partitions give 12,870 IS/OOS splits — enough
    statistical power without being slow.
    """

    walk_forward_n_windows: int = 5
    """Number of walk-forward windows for OOS validation."""

    train_test_split: float = 0.7
    """Fraction of data used for in-sample. The rest is held out."""


# ----------------------------------------------------------------------------
# Robustness testing
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessConfig:
    """Parameters for robustness and stability tests."""

    bootstrap_n_samples: int = 1000
    """Number of bootstrap resamples for Sharpe CI."""

    bootstrap_seed: int = 42
    """Random seed for reproducible bootstrap. Set to None for non-reproducible."""

    n_subsamples: int = 4
    """Number of contiguous subsamples to split the data into.

    With n=4, each subsample is ~25% of the data. A strategy should be
    positive in at least 3 of 4 subsamples to be considered stable.
    """


# ----------------------------------------------------------------------------
# Production Readiness scoring
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringConfig:
    """Weights for the Production Readiness Score.

    The weights reflect what actually predicts production failure:
    overfitting and OOS degradation matter most because they directly
    answer "will the backtest hold up". Raw performance matters less
    because anyone can produce a high-Sharpe backtest by trying enough
    variants.

    Weights must sum to 1.0.
    """

    overfitting_weight: float = 0.25
    """Deflated Sharpe + PBO. The single most important dimension."""

    validation_weight: float = 0.20
    """OOS Sharpe, degradation ratio, walk-forward consistency."""

    robustness_weight: float = 0.15
    """Bootstrap CI, subsample stability."""

    regime_weight: float = 0.10
    """Performance consistency across market regimes."""

    decay_weight: float = 0.10
    """IC stability and half-life."""

    cost_weight: float = 0.10
    """Breakeven cost and turnover."""

    performance_weight: float = 0.10
    """Raw OOS Sharpe. Last because it's the most overfittable single metric."""

    # Score buckets for the verdict.
    ready_threshold: float = 80.0
    conditional_threshold: float = 60.0
    not_ready_threshold: float = 40.0

    def __post_init__(self) -> None:
        weights = [
            self.overfitting_weight,
            self.validation_weight,
            self.robustness_weight,
            self.regime_weight,
            self.decay_weight,
            self.cost_weight,
            self.performance_weight,
        ]
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.4f}. "
                f"Adjust the weights in ScoringConfig."
            )


# ----------------------------------------------------------------------------
# Cost analysis
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CostConfig:
    """Parameters for transaction-cost sensitivity analysis."""

    cost_levels_bps: tuple[float, ...] = (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0)
    """Cost levels (in basis points per unit turnover) to sweep over."""


# ----------------------------------------------------------------------------
# Aggregate config
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AutopsyConfig:
    """All configuration in one place."""

    regime: RegimeConfig = field(default_factory=RegimeConfig)
    overfitting: OverfittingConfig = field(default_factory=OverfittingConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    costs: CostConfig = field(default_factory=CostConfig)

    rolling_window: int = 63
    """Default rolling-statistic window. ~3 months."""

    ic_horizons: tuple[int, ...] = (1, 5, 20, 60)
    """Forward-return horizons (days) for IC analysis."""
