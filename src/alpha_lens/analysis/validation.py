"""Out-of-sample validation: train/test split and walk-forward analysis.

The most basic test of a backtest: does it hold up on data the strategy
was not built on? This module produces three numbers:

* **In-sample Sharpe** — what you saw when building the strategy
* **Out-of-sample Sharpe** — what you'd see after deployment
* **Degradation ratio** — OOS/IS. Below 0.5 is a serious red flag.

Walk-forward analysis extends this: instead of one IS/OOS split, we
simulate the strategy being repeatedly retrained and re-evaluated on
chronologically forward data. This catches strategies that worked in
the past but no longer work — a common failure mode that a single
split can miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lens.analysis.statistics import sharpe_ratio
from alpha_lens.core.config import OverfittingConfig
from alpha_lens.core.types import ValidationResults

__all__ = ["analyze_validation", "train_test_split", "walk_forward_sharpes"]


def analyze_validation(
    returns: pd.Series,
    *,
    risk_free_rate_annual: float = 0.0,
    config: OverfittingConfig | None = None,
) -> ValidationResults:
    """Run the full validation suite.

    Args:
        returns: Strategy returns.
        risk_free_rate_annual: Annualized risk-free rate.
        config: Overfitting/validation parameters.

    Returns:
        :class:`ValidationResults` with IS/OOS Sharpes, degradation, walk-forward.
    """
    config = config or OverfittingConfig()
    is_sr, oos_sr = train_test_split(
        returns,
        train_fraction=config.train_test_split,
        risk_free_rate_annual=risk_free_rate_annual,
    )
    degradation = _safe_ratio(oos_sr, is_sr)

    walk_forward = walk_forward_sharpes(
        returns,
        n_windows=config.walk_forward_n_windows,
        risk_free_rate_annual=risk_free_rate_annual,
    )
    positive_fraction = (
        sum(1 for s in walk_forward if s > 0) / len(walk_forward) if walk_forward else 0.0
    )

    return ValidationResults(
        in_sample_sharpe=is_sr,
        out_of_sample_sharpe=oos_sr,
        degradation_ratio=degradation,
        walk_forward_sharpes=walk_forward,
        walk_forward_consistency=positive_fraction,
    )


def train_test_split(
    returns: pd.Series,
    *,
    train_fraction: float = 0.7,
    risk_free_rate_annual: float = 0.0,
) -> tuple[float, float]:
    """Simple chronological IS/OOS split.

    Args:
        returns: Strategy returns.
        train_fraction: Fraction of data used for IS.
        risk_free_rate_annual: Annualized risk-free rate.

    Returns:
        Tuple of (in_sample_sharpe, out_of_sample_sharpe).
    """
    n = len(returns)
    if n < 100:
        sr = sharpe_ratio(returns, risk_free_rate_annual=risk_free_rate_annual)
        return sr, sr

    split = int(n * train_fraction)
    is_part = returns.iloc[:split]
    oos_part = returns.iloc[split:]

    is_sr = sharpe_ratio(is_part, risk_free_rate_annual=risk_free_rate_annual)
    oos_sr = sharpe_ratio(oos_part, risk_free_rate_annual=risk_free_rate_annual)
    return is_sr, oos_sr


def walk_forward_sharpes(
    returns: pd.Series,
    *,
    n_windows: int = 5,
    risk_free_rate_annual: float = 0.0,
) -> list[float]:
    """Compute Sharpe in each of N equal-size contiguous windows.

    This is a poor man's walk-forward: a true WF would retrain the
    strategy at each step. Since we don't have access to the strategy's
    parameters, we instead measure performance in each forward chunk.

    Args:
        returns: Strategy returns.
        n_windows: Number of equal-size windows.
        risk_free_rate_annual: Annualized risk-free rate.

    Returns:
        List of length ``n_windows`` with the Sharpe in each window.
    """
    n = len(returns)
    if n < n_windows * 20:
        return []

    chunks = np.array_split(returns, n_windows)
    return [sharpe_ratio(pd.Series(c), risk_free_rate_annual=risk_free_rate_annual) for c in chunks]


def _safe_ratio(num: float, denom: float) -> float:
    """Compute num/denom robustly when either may be zero, negative, or NaN.

    Conventions for the degradation ratio:
        * Both positive: standard ratio.
        * IS positive, OOS non-positive: 0 (complete degradation).
        * IS non-positive: undefined → return 0 (strategy was never good).
    """
    if np.isnan(num) or np.isnan(denom):
        return 0.0
    if denom <= 0:
        return 0.0
    if num <= 0:
        return 0.0
    return float(num / denom)
