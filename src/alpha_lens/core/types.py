"""Pydantic data models for alpha-lens.

These models define the contract between analysis modules and the report
generator. All public APIs accept and return these typed structures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ----------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------


class RegimeLabel(str, Enum):
    """Market regime classification.

    Used by both rule-based and HMM-based regime detection. The four-way
    split is empirically useful: it separates trend (bull/bear) from
    volatility (high_vol) and gives a residual bucket (sideways) for
    everything else.
    """

    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    SIDEWAYS = "sideways"


class RegimeMethod(str, Enum):
    """Method used to detect market regimes."""

    RULE_BASED = "rule_based"
    HMM = "hmm"


class ReadinessVerdict(str, Enum):
    """High-level recommendation from the Production Readiness Score.

    These map roughly to score buckets:
        READY:       score >= 80  — deploy with normal monitoring
        CONDITIONAL: 60-79        — paper trade or deploy with caveats
        NOT_READY:   40-59        — substantial issues, iterate before deploy
        REJECT:      < 40         — likely overfitted or fundamentally broken
    """

    READY = "ready"
    CONDITIONAL = "conditional"
    NOT_READY = "not_ready"
    REJECT = "reject"


# ----------------------------------------------------------------------------
# Base model with pandas support
# ----------------------------------------------------------------------------


class AlphaLensModel(BaseModel):
    """Base model that allows arbitrary types (pandas Series/DataFrames) as fields."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        # Don't validate on assignment — analysis pipelines mutate intermediate state.
        validate_assignment=False,
    )


# ----------------------------------------------------------------------------
# Input models
# ----------------------------------------------------------------------------


class AlphaData(AlphaLensModel):
    """Container for the inputs to an autopsy run.

    Attributes:
        returns: Strategy returns as a pandas Series indexed by DatetimeIndex.
            Returns should be in decimal form (e.g. 0.01 for 1%).
        benchmark_returns: Optional benchmark returns aligned to ``returns``.
        factor_values: Optional DataFrame of factor values (rows = dates,
            columns = factor names). Used for IC/decay analysis. Must be
            shifted forward such that ``factor_values.loc[t]`` represents
            information known AT time ``t`` — never future data.
        custom_factors: Optional DataFrame of factor RETURNS (not values)
            for use in attribution regression.
        risk_free_rate: Annualized risk-free rate (decimal). Default 0.0.
        positions: Optional DataFrame of portfolio positions (rows = dates,
            columns = assets, values = weights). Required for turnover and
            capacity analysis.
    """

    returns: pd.Series
    benchmark_returns: pd.Series | None = None
    factor_values: pd.DataFrame | None = None
    custom_factors: pd.DataFrame | None = None
    risk_free_rate: float = 0.0
    positions: pd.DataFrame | None = None

    @field_validator("returns")
    @classmethod
    def _returns_must_have_datetime_index(cls, v: pd.Series) -> pd.Series:
        if not isinstance(v.index, pd.DatetimeIndex):
            raise ValueError(
                f"`returns` must have a DatetimeIndex, got {type(v.index).__name__}. "
                f"Convert with: returns.index = pd.to_datetime(returns.index)"
            )
        if v.empty:
            raise ValueError("`returns` is empty — cannot run autopsy on no data.")
        return v


# ----------------------------------------------------------------------------
# Analysis result models
# ----------------------------------------------------------------------------


class CoreStatistics(AlphaLensModel):
    """Headline performance statistics.

    All ratios are annualized assuming 252 trading days per year.
    """

    total_return: float = Field(..., description="Cumulative total return (decimal).")
    cagr: float = Field(..., description="Compound annual growth rate (decimal).")
    annualized_volatility: float = Field(..., description="Annualized std of returns.")
    sharpe_ratio: float = Field(..., description="Annualized Sharpe (mean / std × √252).")
    sortino_ratio: float = Field(..., description="Sharpe using only downside std.")
    calmar_ratio: float = Field(..., description="CAGR / |max drawdown|.")
    max_drawdown: float = Field(..., description="Worst peak-to-trough drawdown (negative).")
    win_rate: float = Field(..., description="Fraction of positive return periods.")
    n_observations: int = Field(..., description="Number of return observations.")
    start_date: datetime
    end_date: datetime
    years: float = Field(..., description="Backtest length in years.")


class RegimeSummary(AlphaLensModel):
    """Performance breakdown for a single regime."""

    regime: RegimeLabel
    n_days: int
    fraction: float = Field(..., description="Fraction of total period in this regime.")
    mean_return_annualized: float
    volatility_annualized: float
    sharpe_ratio: float
    max_drawdown: float


class RegimeAnalysis(AlphaLensModel):
    """Output of regime detection and per-regime performance analysis."""

    method: RegimeMethod
    labels: pd.Series  # DatetimeIndex → RegimeLabel.value
    summaries: list[RegimeSummary]
    transition_matrix: pd.DataFrame
    robustness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0-1 score: how consistent performance is across regimes.",
    )


class FactorAttribution(AlphaLensModel):
    """Output of factor attribution regression."""

    factor_names: list[str]
    betas: dict[str, float] = Field(..., description="Full-sample factor loadings.")
    t_stats: dict[str, float] = Field(..., description="t-statistics for each loading.")
    alpha_annualized: float = Field(..., description="Annualized intercept (residual alpha).")
    alpha_t_stat: float
    r_squared: float
    rolling_betas: pd.DataFrame | None = None
    uniqueness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="1 - R² against known factors; fraction of variance NOT explained.",
    )


class DrawdownEvent(AlphaLensModel):
    """A single peak-to-trough-to-recovery drawdown episode."""

    peak_date: datetime
    trough_date: datetime
    recovery_date: datetime | None = Field(
        None, description="None if the strategy never recovered within the sample."
    )
    depth: float = Field(..., description="Drawdown depth (negative number).")
    duration_days: int = Field(..., description="Peak to trough in calendar days.")
    recovery_days: int | None = Field(None, description="Trough to recovery in calendar days.")
    dominant_regime: RegimeLabel | None = None
    worst_days: list[tuple[datetime, float]] = Field(
        default_factory=list,
        description="Top contributing single-day losses during the drawdown.",
    )


class DrawdownAnalysis(AlphaLensModel):
    """Aggregated drawdown statistics."""

    events: list[DrawdownEvent]
    n_drawdowns: int
    avg_depth: float
    avg_duration_days: float
    avg_recovery_days: float | None
    regime_concentration: dict[str, float] = Field(
        default_factory=dict,
        description="Fraction of drawdown severity attributable to each regime.",
    )


class DecayMetrics(AlphaLensModel):
    """Alpha decay diagnostics."""

    ic_by_horizon: dict[int, float] = Field(
        ..., description="Spearman IC at each forward horizon (days)."
    )
    ic_pvalue_by_horizon: dict[int, float]
    estimated_half_life_days: float | None = Field(
        None, description="Exponential decay half-life of IC. None if no decay detectable."
    )
    rolling_ic: pd.Series | None = None
    ic_stability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of rolling-IC windows with same-sign IC as full sample.",
    )


class CorrelationAnalysis(AlphaLensModel):
    """Factor correlation and multicollinearity analysis."""

    correlation_matrix: pd.DataFrame
    vif_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Variance inflation factor for each factor. >5 indicates concern.",
    )
    max_correlation: float = Field(..., description="Maximum absolute off-diagonal correlation.")


# ----------------------------------------------------------------------------
# Production Readiness models (the differentiating layer)
# ----------------------------------------------------------------------------


class OverfittingDiagnostics(AlphaLensModel):
    """Statistical tests for overfitting risk.

    These implement the framework from López de Prado (2018) and
    Bailey & López de Prado (2014). The key insight: a Sharpe ratio
    computed from a backtest is biased upward by the search process
    that produced the strategy. These diagnostics try to quantify and
    correct that bias.
    """

    deflated_sharpe_ratio: float = Field(
        ...,
        description="Sharpe adjusted for n_trials, skewness, kurtosis (Bailey-LdP 2014).",
    )
    deflated_sharpe_pvalue: float = Field(
        ...,
        description="P-value: probability the true Sharpe is <= 0.",
    )
    probability_of_backtest_overfitting: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="PBO from CSCV. >0.5 means backtest selection is no better than random.",
    )
    minimum_backtest_length_years: float = Field(
        ...,
        description="Minimum years of data needed for the Sharpe to be statistically meaningful.",
    )
    actual_backtest_length_years: float
    minimum_length_satisfied: bool


class ValidationResults(AlphaLensModel):
    """Out-of-sample and walk-forward validation results."""

    in_sample_sharpe: float
    out_of_sample_sharpe: float
    degradation_ratio: float = Field(
        ...,
        description="OOS Sharpe / IS Sharpe. <0.5 indicates serious overfitting.",
    )
    walk_forward_sharpes: list[float] = Field(
        default_factory=list,
        description="Sharpe ratio from each walk-forward window.",
    )
    walk_forward_consistency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of walk-forward windows with positive Sharpe.",
    )


class RobustnessResults(AlphaLensModel):
    """Robustness and stability diagnostics."""

    sharpe_confidence_interval: tuple[float, float] = Field(
        ...,
        description="Bootstrap 95% CI for the Sharpe ratio.",
    )
    sharpe_std_bootstrap: float
    subsample_sharpes: list[float] = Field(
        default_factory=list,
        description="Sharpe in each contiguous time subsample.",
    )
    subsample_positive_fraction: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of subsamples with positive Sharpe.",
    )


class CostAnalysis(AlphaLensModel):
    """Transaction cost sensitivity and capacity."""

    cost_sensitivity: dict[float, float] = Field(
        ...,
        description="Map of cost (bps per turnover) → resulting annualized Sharpe.",
    )
    breakeven_cost_bps: float | None = Field(
        None,
        description="Cost level at which Sharpe drops to zero. None if no cost reaches it.",
    )
    annualized_turnover: float | None = Field(
        None,
        description="Annualized two-way turnover (only computed if positions provided).",
    )


class ReadinessComponent(AlphaLensModel):
    """One line item in the production readiness scorecard."""

    name: str
    score: float = Field(..., ge=0.0, le=100.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    status: str = Field(..., description="'pass', 'warn', or 'fail'.")
    value: str = Field(..., description="Human-readable value for the report (e.g. '1.8' or '12 bps').")
    detail: str = Field(..., description="One-sentence explanation of what this measures.")


class ProductionReadinessScore(AlphaLensModel):
    """Top-level scorecard — the headline number a PM sees first.

    Combines weighted sub-scores from each diagnostic dimension. The
    component weights reflect priorities: overfitting and OOS validation
    matter most because they directly answer "will this work in
    production"; raw performance matters less because anyone can show
    you a backtest with high Sharpe.
    """

    overall_score: float = Field(..., ge=0.0, le=100.0)
    verdict: ReadinessVerdict
    components: list[ReadinessComponent]
    recommendation: str = Field(
        ...,
        description="Plain-language recommendation: deploy, paper-trade, iterate, or reject.",
    )
    top_risks: list[str] = Field(
        default_factory=list,
        description="The 1-3 most important risks for the reviewer to know about.",
    )


# ----------------------------------------------------------------------------
# Top-level report container
# ----------------------------------------------------------------------------


class AutopsyReport(AlphaLensModel):
    """Complete output of an autopsy run.

    This is what ``autopsy()`` returns. Every analysis module fills in
    one field. ``readiness`` aggregates everything into a single
    actionable score.
    """

    # Inputs (echoed back for the report)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Analysis results
    statistics: CoreStatistics
    regime: RegimeAnalysis | None = None
    attribution: FactorAttribution | None = None
    drawdowns: DrawdownAnalysis
    decay: DecayMetrics | None = None
    correlation: CorrelationAnalysis | None = None

    # Production readiness layer
    overfitting: OverfittingDiagnostics
    validation: ValidationResults
    robustness: RobustnessResults
    costs: CostAnalysis | None = None
    readiness: ProductionReadinessScore

    # Optional LLM interpretation
    llm_interpretation: str | None = None

    def save(self, path: str) -> str:
        """Save the report as a standalone HTML file.

        Args:
            path: File path to write to.

        Returns:
            The absolute path of the written file.
        """
        # Local import to avoid pulling in plotly/jinja2 at module import time.
        from alpha_lens.report.html_report import render_html_report

        return render_html_report(self, path)
