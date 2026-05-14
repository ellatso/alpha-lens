"""Robustness diagnostics: bootstrap CI and subsample stability.

These tests answer: "is the observed performance fragile?"

* **Bootstrap confidence interval for Sharpe.** Resample returns with
  replacement many times; compute Sharpe each time; report the 2.5%
  and 97.5% percentiles. A tight CI around a positive number is
  reassuring; a wide CI that crosses zero says the Sharpe could plausibly
  be due to luck.

* **Subsample stability.** Split the data into N contiguous chunks and
  check the Sharpe in each. A strategy that worked in 3 of 4 quarters
  is more robust than one that hit one big winner.

Both are simple but underused in practice. They turn a single
point-estimate into a sense of *how much* you should trust it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lens.analysis.statistics import sharpe_ratio
from alpha_lens.core.config import RobustnessConfig
from alpha_lens.core.types import RobustnessResults

__all__ = ["analyze_robustness", "bootstrap_sharpe", "subsample_sharpes"]


def analyze_robustness(
    returns: pd.Series,
    *,
    risk_free_rate_annual: float = 0.0,
    config: RobustnessConfig | None = None,
) -> RobustnessResults:
    """Run all robustness tests.

    Args:
        returns: Strategy returns.
        risk_free_rate_annual: Annualized risk-free rate.
        config: Robustness parameters.

    Returns:
        :class:`RobustnessResults`.
    """
    config = config or RobustnessConfig()
    ci, std = bootstrap_sharpe(
        returns,
        n_samples=config.bootstrap_n_samples,
        seed=config.bootstrap_seed,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    sub_sharpes = subsample_sharpes(
        returns,
        n_subsamples=config.n_subsamples,
        risk_free_rate_annual=risk_free_rate_annual,
    )
    positive_frac = (
        sum(1 for s in sub_sharpes if s > 0) / len(sub_sharpes) if sub_sharpes else 0.0
    )

    return RobustnessResults(
        sharpe_confidence_interval=ci,
        sharpe_std_bootstrap=std,
        subsample_sharpes=sub_sharpes,
        subsample_positive_fraction=positive_frac,
    )


def bootstrap_sharpe(
    returns: pd.Series,
    *,
    n_samples: int = 1000,
    seed: int | None = 42,
    risk_free_rate_annual: float = 0.0,
    confidence: float = 0.95,
) -> tuple[tuple[float, float], float]:
    """Bootstrap the Sharpe ratio.

    We use simple IID bootstrap (resample individual days with
    replacement). For time-series with autocorrelation, a stationary
    bootstrap would be more rigorous — but for daily returns of most
    strategies, autocorrelation is small enough that the simple
    bootstrap is a reasonable first cut. We note this as a caveat in
    the report.

    Args:
        returns: Strategy returns.
        n_samples: Number of bootstrap resamples.
        seed: Random seed for reproducibility. None for non-reproducible.
        risk_free_rate_annual: Annualized risk-free rate.
        confidence: Confidence level for the CI.

    Returns:
        Tuple of ((lower_bound, upper_bound), std_of_sharpe).
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < 30:
        sr = sharpe_ratio(returns, risk_free_rate_annual=risk_free_rate_annual)
        return (sr, sr), 0.0

    values = returns.values
    sharpes = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        sample = pd.Series(values[idx])
        sharpes[i] = sharpe_ratio(sample, risk_free_rate_annual=risk_free_rate_annual)

    sharpes = sharpes[~np.isnan(sharpes)]
    if len(sharpes) == 0:
        return (0.0, 0.0), 0.0

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(sharpes, alpha))
    upper = float(np.quantile(sharpes, 1.0 - alpha))
    std = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else 0.0
    return (lower, upper), std


def subsample_sharpes(
    returns: pd.Series,
    *,
    n_subsamples: int = 4,
    risk_free_rate_annual: float = 0.0,
) -> list[float]:
    """Compute Sharpe ratio in each contiguous subsample.

    Unlike walk-forward (which is chronologically ordered), subsamples
    are mostly used for stability checks: did the strategy work in
    every part of the sample, or just one lucky stretch?

    Args:
        returns: Strategy returns.
        n_subsamples: Number of equal-size subsamples.
        risk_free_rate_annual: Annualized risk-free rate.

    Returns:
        List of Sharpe values, one per subsample.
    """
    n = len(returns)
    if n < n_subsamples * 30:
        return []

    chunks = np.array_split(returns, n_subsamples)
    return [
        sharpe_ratio(pd.Series(chunk), risk_free_rate_annual=risk_free_rate_annual)
        for chunk in chunks
    ]
