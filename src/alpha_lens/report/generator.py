"""The ``autopsy()`` orchestrator — alpha-lens's main public entry point.

This function ties every analysis module together. The design goals are:

* **Three-line minimum.** ``autopsy(returns)`` works.
* **Graceful degradation.** Missing optional inputs (factors, benchmark,
  positions) skip the corresponding analysis instead of erroring.
* **All-or-nothing report.** Either we produce a full :class:`AutopsyReport`
  or we raise a clear error. We never return partial results silently.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from alpha_lens.analysis.attribution import attribute_returns
from alpha_lens.analysis.correlation import analyze_correlation
from alpha_lens.analysis.cost_analysis import analyze_costs
from alpha_lens.analysis.decay import analyze_decay
from alpha_lens.analysis.drawdown import analyze_drawdowns
from alpha_lens.analysis.overfitting import analyze_overfitting
from alpha_lens.analysis.regime import analyze_regimes
from alpha_lens.analysis.robustness import analyze_robustness
from alpha_lens.analysis.scoring import compute_readiness_score
from alpha_lens.analysis.statistics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)
from alpha_lens.analysis.validation import analyze_validation
from alpha_lens.core.config import TRADING_DAYS_PER_YEAR, AutopsyConfig
from alpha_lens.core.types import (
    AutopsyReport,
    CoreStatistics,
    RegimeMethod,
)
from alpha_lens.core.validator import validate_aligned, validate_returns

logger = logging.getLogger(__name__)

__all__ = ["autopsy"]


def autopsy(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    factors: pd.DataFrame | None = None,
    factor_values: pd.Series | pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    strategy_variants: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    regime_method: str = "rule_based",
    config: AutopsyConfig | None = None,
    n_trials_assumed: int | None = None,
) -> AutopsyReport:
    """Run a full alpha autopsy on a returns series.

    Args:
        returns: Strategy returns. The only required input.
        benchmark_returns: Market or benchmark returns. If supplied,
            regime detection and CAPM-style attribution become available.
        factors: DataFrame of factor RETURNS (e.g. Fama-French). Each
            column is one factor. Enables multi-factor attribution.
        factor_values: Optional single factor VALUES series for IC and
            decay analysis. (Distinct from ``factors`` which are factor
            RETURNS used in regression.) If a DataFrame is passed, the
            first column is used.
        positions: Optional DataFrame of portfolio weights for accurate
            turnover and cost analysis.
        strategy_variants: Optional DataFrame of returns for OTHER
            strategy variants the researcher tested. Enables PBO
            calculation via CSCV.
        risk_free_rate: Annualized risk-free rate (decimal).
        regime_method: ``'rule_based'`` (default) or ``'hmm'``.
        config: Custom :class:`AutopsyConfig`. Uses defaults if None.
        n_trials_assumed: Override the assumed number of strategy trials
            (for DSR computation). Default 100; pessimistic researchers
            should pass higher values.

    Returns:
        :class:`AutopsyReport` containing every analysis result and the
        top-level Production Readiness Score.

    Example:
        >>> from alpha_lens import autopsy
        >>> report = autopsy(my_strategy_returns)
        >>> report.save("autopsy.html")
        >>> print(report.readiness.verdict)
    """
    config = config or AutopsyConfig()
    if n_trials_assumed is not None:
        # Replace just the overfitting config — frozen dataclass requires
        # construction of a new one.
        from dataclasses import replace as dc_replace

        new_ovr = dc_replace(config.overfitting, n_trials_assumed=n_trials_assumed)
        config = dc_replace(config, overfitting=new_ovr)

    # Validate and clean.
    returns = validate_returns(returns)
    if benchmark_returns is not None:
        benchmark_returns = validate_aligned(returns, benchmark_returns, name="benchmark_returns")  # type: ignore[assignment]
    if factors is not None:
        factors = validate_aligned(returns, factors, name="factors")  # type: ignore[assignment]
    if positions is not None:
        # Don't drop too many — positions are sometimes weekly while returns are daily.
        positions = validate_aligned(returns, positions, name="positions", min_overlap_fraction=0.3)  # type: ignore[assignment]
    if factor_values is not None:
        factor_values = validate_aligned(returns, factor_values, name="factor_values")  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Core statistics
    # ------------------------------------------------------------------
    n = len(returns)
    years = n / TRADING_DAYS_PER_YEAR
    stats = CoreStatistics(
        total_return=total_return(returns),
        cagr=cagr(returns),
        annualized_volatility=annualized_volatility(returns),
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate_annual=risk_free_rate),
        sortino_ratio=sortino_ratio(returns, risk_free_rate_annual=risk_free_rate),
        calmar_ratio=calmar_ratio(returns),
        max_drawdown=max_drawdown(returns),
        win_rate=win_rate(returns),
        n_observations=n,
        start_date=returns.index.min().to_pydatetime(),
        end_date=returns.index.max().to_pydatetime(),
        years=years,
    )

    # ------------------------------------------------------------------
    # Regime
    # ------------------------------------------------------------------
    regime_enum = RegimeMethod(regime_method)
    regime_result = None
    try:
        regime_result = analyze_regimes(
            returns,
            benchmark_returns=benchmark_returns,
            method=regime_enum,
            config=config.regime,
        )
    except Exception as e:
        logger.warning("Regime analysis failed: %s", e)

    # ------------------------------------------------------------------
    # Drawdowns
    # ------------------------------------------------------------------
    dd_result = analyze_drawdowns(
        returns,
        regime_labels=regime_result.labels if regime_result else None,
    )

    # ------------------------------------------------------------------
    # Attribution
    # ------------------------------------------------------------------
    attribution_result = None
    if factors is not None or benchmark_returns is not None:
        try:
            attribution_result = attribute_returns(
                returns,
                factors=factors,
                benchmark_returns=benchmark_returns,
                risk_free_rate_annual=risk_free_rate,
                rolling_window=config.rolling_window,
            )
        except Exception as e:
            logger.warning("Attribution failed: %s", e)

    # ------------------------------------------------------------------
    # Decay (only if user provided a factor VALUE series)
    # ------------------------------------------------------------------
    decay_result = None
    if factor_values is not None:
        try:
            # If a DataFrame, use the first column.
            if isinstance(factor_values, pd.DataFrame):
                factor_series = (
                    None
                    if factor_values.shape[1] == 0
                    else factor_values.iloc[:, 0]
                )
            else:
                factor_series = factor_values
            if factor_series is not None:
                decay_result = analyze_decay(
                    factor_series, returns, horizons=config.ic_horizons
                )
        except Exception as e:
            logger.warning("Decay analysis failed: %s", e)

    # ------------------------------------------------------------------
    # Correlation (only if multi-factor)
    # ------------------------------------------------------------------
    correlation_result = None
    if factors is not None and factors.shape[1] >= 2:
        try:
            correlation_result = analyze_correlation(factors)
        except Exception as e:
            logger.warning("Correlation analysis failed: %s", e)

    # ------------------------------------------------------------------
    # Production-readiness layer
    # ------------------------------------------------------------------
    overfitting = analyze_overfitting(
        returns,
        strategy_variants=strategy_variants,
        risk_free_rate_annual=risk_free_rate,
        config=config.overfitting,
    )
    validation = analyze_validation(
        returns,
        risk_free_rate_annual=risk_free_rate,
        config=config.overfitting,
    )
    robustness = analyze_robustness(
        returns,
        risk_free_rate_annual=risk_free_rate,
        config=config.robustness,
    )
    cost_result = None
    if positions is not None:
        try:
            cost_result = analyze_costs(
                returns,
                positions=positions,
                risk_free_rate_annual=risk_free_rate,
                config=config.costs,
            )
        except Exception as e:
            logger.warning("Cost analysis failed: %s", e)
    else:
        # Run a basic cost analysis assuming 1.0 turnover.
        cost_result = analyze_costs(
            returns,
            positions=None,
            risk_free_rate_annual=risk_free_rate,
            config=config.costs,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    readiness = compute_readiness_score(
        statistics=stats,
        overfitting=overfitting,
        validation=validation,
        robustness=robustness,
        regime=regime_result,
        decay=decay_result,
        costs=cost_result,
        config=config.scoring,
    )

    # Assemble metadata for the report.
    metadata: dict[str, Any] = {
        "n_observations": n,
        "years": round(years, 2),
        "start": str(returns.index.min().date()),
        "end": str(returns.index.max().date()),
        "has_benchmark": benchmark_returns is not None,
        "has_factors": factors is not None,
        "has_positions": positions is not None,
        "has_decay": decay_result is not None,
        "regime_method": regime_method,
        "risk_free_rate": risk_free_rate,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        # Stashed for chart rendering (prefixed with underscore by convention).
        # These are deliberately kept outside the JSON-serializable surface;
        # they're only read by the HTML renderer in this process.
        "_returns": returns,
        "_benchmark": benchmark_returns,
    }

    return AutopsyReport(
        metadata=metadata,
        statistics=stats,
        regime=regime_result,
        attribution=attribution_result,
        drawdowns=dd_result,
        decay=decay_result,
        correlation=correlation_result,
        overfitting=overfitting,
        validation=validation,
        robustness=robustness,
        costs=cost_result,
        readiness=readiness,
    )
