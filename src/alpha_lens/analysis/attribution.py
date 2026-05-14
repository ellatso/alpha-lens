"""Factor attribution: decompose strategy returns into known factors.

The fundamental question: "is my alpha actually alpha, or is it just
hidden exposure to known factors?" A high-Sharpe strategy that is 80%
explained by the size and value factors is not really alpha — it's a
levered factor portfolio.

We support two modes:

* **Custom factor regression** (default): user provides a DataFrame of
  factor RETURNS (one column per factor), we run OLS:

      strategy_excess_return ~ factor_1 + factor_2 + ... + intercept

  and report betas, t-stats, R², and the intercept (which is the
  residual "true" alpha).

* **Auto-attribution against benchmark**: if no factors are provided
  but a benchmark is, we run a simple CAPM regression to extract the
  market beta and alpha.

Fama-French data download is intentionally not bundled here — that
belongs in ``data/providers.py`` so that users can mock it for tests.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from alpha_lens.analysis.statistics import excess_returns
from alpha_lens.core.config import TRADING_DAYS_PER_YEAR
from alpha_lens.core.types import FactorAttribution

logger = logging.getLogger(__name__)

__all__ = ["attribute_returns", "rolling_attribution"]


def attribute_returns(
    returns: pd.Series,
    factors: pd.DataFrame | None = None,
    *,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate_annual: float = 0.0,
    rolling_window: int | None = None,
) -> FactorAttribution:
    """Run OLS attribution of strategy returns against factor returns.

    Either ``factors`` or ``benchmark_returns`` must be provided. If
    both are given, ``factors`` is used and ``benchmark_returns`` is
    ignored.

    Args:
        returns: Strategy returns.
        factors: DataFrame of factor returns (columns are factor names).
        benchmark_returns: Used as the single market factor if ``factors``
            is None.
        risk_free_rate_annual: Annualized risk-free rate. Returns are
            converted to excess returns before regression.
        rolling_window: If set, also compute rolling betas with this window.

    Returns:
        :class:`FactorAttribution` with betas, t-stats, alpha, R².

    Raises:
        ValueError: If neither factors nor benchmark is provided, or if
            there's insufficient overlapping data.
    """
    if factors is None and benchmark_returns is None:
        raise ValueError(
            "attribute_returns needs either `factors` or `benchmark_returns`. "
            "Provide one of them, or skip attribution entirely."
        )

    # Build the factor matrix.
    if factors is None:
        # Synthesize a single-factor "MKT" model from the benchmark.
        excess_bench = excess_returns(benchmark_returns, risk_free_rate_annual)  # type: ignore[arg-type]
        factor_matrix = pd.DataFrame({"MKT": excess_bench})
    else:
        factor_matrix = factors.copy()
        # If the user passed RAW (not excess) factor returns and we have a
        # risk-free rate, we leave that to the user — they know whether
        # their factors are already in excess form. The standard Fama-French
        # factors ARE excess (MKT-RF), so converting again would double-count.

    strategy_excess = excess_returns(returns, risk_free_rate_annual)

    # Align.
    aligned = pd.concat([strategy_excess.rename("y"), factor_matrix], axis=1).dropna()
    if len(aligned) < 30:
        raise ValueError(
            f"Need at least 30 overlapping observations for attribution, "
            f"got {len(aligned)} after aligning returns with factors."
        )

    y = aligned["y"].values
    X = aligned[factor_matrix.columns].values
    X = sm.add_constant(X, has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit()

    factor_names = list(factor_matrix.columns)
    # First param is the intercept (alpha), rest are factor betas.
    alpha_daily = float(model.params[0])
    alpha_t = float(model.tvalues[0])
    betas = {name: float(model.params[i + 1]) for i, name in enumerate(factor_names)}
    t_stats = {name: float(model.tvalues[i + 1]) for i, name in enumerate(factor_names)}

    r_squared = float(model.rsquared)
    alpha_annualized = alpha_daily * TRADING_DAYS_PER_YEAR

    # Uniqueness: 1 - R². Fraction of variance NOT explained by factors.
    uniqueness = float(np.clip(1.0 - r_squared, 0.0, 1.0))

    # Optional rolling betas.
    rolling_betas_df: pd.DataFrame | None = None
    if rolling_window is not None and len(aligned) >= rolling_window * 2:
        rolling_betas_df = _rolling_betas(aligned, factor_names, window=rolling_window)

    return FactorAttribution(
        factor_names=factor_names,
        betas=betas,
        t_stats=t_stats,
        alpha_annualized=alpha_annualized,
        alpha_t_stat=alpha_t,
        r_squared=r_squared,
        rolling_betas=rolling_betas_df,
        uniqueness_score=uniqueness,
    )


def rolling_attribution(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    window: int = 63,
    risk_free_rate_annual: float = 0.0,
) -> pd.DataFrame:
    """Compute rolling-window factor betas.

    Args:
        returns: Strategy returns.
        factors: Factor returns DataFrame.
        window: Rolling window size in periods.
        risk_free_rate_annual: Annualized risk-free rate.

    Returns:
        DataFrame indexed by date with one column per factor. The intercept
        (alpha) column is named ``alpha_daily``.
    """
    strategy_excess = excess_returns(returns, risk_free_rate_annual)
    aligned = pd.concat([strategy_excess.rename("y"), factors], axis=1).dropna()
    return _rolling_betas(aligned, list(factors.columns), window=window)


def _rolling_betas(
    aligned: pd.DataFrame,
    factor_names: list[str],
    *,
    window: int,
) -> pd.DataFrame:
    """Helper: compute rolling betas via repeated OLS fits.

    For performance, this uses a vectorized approach on chunks rather
    than calling statsmodels per window.
    """
    results = []
    y_all = aligned["y"].values
    X_all = aligned[factor_names].values
    X_all = np.column_stack([np.ones(len(X_all)), X_all])

    n = len(aligned)
    for end in range(window, n + 1):
        start = end - window
        y_win = y_all[start:end]
        X_win = X_all[start:end]
        try:
            # OLS closed form: beta = (X'X)^-1 X'y
            xtx = X_win.T @ X_win
            xty = X_win.T @ y_win
            beta = np.linalg.solve(xtx, xty)
            results.append((aligned.index[end - 1], beta))
        except np.linalg.LinAlgError:
            results.append((aligned.index[end - 1], np.full(len(factor_names) + 1, np.nan)))

    if not results:
        return pd.DataFrame(columns=["alpha_daily", *factor_names])

    idx = [r[0] for r in results]
    coefs = np.array([r[1] for r in results])
    return pd.DataFrame(
        coefs,
        index=pd.DatetimeIndex(idx),
        columns=["alpha_daily", *factor_names],
    )
