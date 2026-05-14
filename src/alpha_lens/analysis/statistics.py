"""Foundation statistics: Sharpe, Sortino, drawdown, IC, etc.

All functions in this module are pure (no side effects, deterministic
output) and vectorized. They form the substrate for every other
analysis module.

Conventions:
    * All ratios are annualized using TRADING_DAYS_PER_YEAR = 252.
    * Returns are in decimal form (0.01 = 1%), never percent.
    * Excess returns subtract the period-equivalent risk-free rate.
    * NaN-handling: each function specifies its behavior — most drop NaN,
      a few propagate to avoid silent corruption (e.g. cumulative product).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from alpha_lens.core.config import TRADING_DAYS_PER_YEAR

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
]


# ----------------------------------------------------------------------------
# Returns and cumulative metrics
# ----------------------------------------------------------------------------


def excess_returns(returns: pd.Series, risk_free_rate_annual: float = 0.0) -> pd.Series:
    """Subtract the period-equivalent risk-free rate from returns.

    Args:
        returns: Periodic returns (assumed daily for annualization).
        risk_free_rate_annual: Annualized risk-free rate (decimal).

    Returns:
        Excess returns of the same shape as input.
    """
    if risk_free_rate_annual == 0.0:
        return returns
    # Convert annual to per-period using simple division. This is the
    # convention in most academic papers; geometric conversion is more
    # accurate at high rates but the difference is negligible at typical
    # risk-free rates.
    daily_rf = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    return returns - daily_rf


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compound returns into a cumulative wealth index starting at 1.0.

    Args:
        returns: Periodic returns.

    Returns:
        Series of cumulative wealth, same length as input.
    """
    return (1.0 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    """Total compounded return over the full sample.

    Args:
        returns: Periodic returns.

    Returns:
        Total return (decimal). 0.50 means +50%.
    """
    if len(returns) == 0:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound annual growth rate.

    Args:
        returns: Periodic returns.
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        CAGR (decimal).
    """
    n = len(returns)
    if n == 0:
        return 0.0
    total = (1.0 + returns).prod()
    if total <= 0:
        # Strategy lost everything — CAGR is meaningless. Return the floor.
        return -1.0
    years = n / periods_per_year
    return float(total ** (1.0 / years) - 1.0)


# ----------------------------------------------------------------------------
# Volatility and risk-adjusted ratios
# ----------------------------------------------------------------------------


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualized standard deviation of returns.

    Args:
        returns: Periodic returns.
        periods_per_year: Number of periods per year.

    Returns:
        Annualized volatility (decimal). Returns 0.0 if returns is empty
        or constant.
    """
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio.

    Uses √N annualization. This assumes returns are roughly IID — a
    standard but imperfect assumption.

    Args:
        returns: Periodic returns.
        risk_free_rate_annual: Annualized risk-free rate (decimal).
        periods_per_year: Annualization factor.

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if volatility is zero.
    """
    if len(returns) < 2:
        return 0.0
    excess = excess_returns(returns, risk_free_rate_annual)
    std = excess.std(ddof=1)
    # Use absolute tolerance, not == 0, because floating-point on
    # near-constant series can yield std ~ 1e-19, blowing up the ratio.
    if std < 1e-12 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio (uses only downside deviation in the denominator).

    Args:
        returns: Periodic returns.
        risk_free_rate_annual: Annualized risk-free rate.
        periods_per_year: Annualization factor.

    Returns:
        Annualized Sortino ratio. Returns 0.0 if downside deviation is zero.
    """
    if len(returns) < 2:
        return 0.0
    excess = excess_returns(returns, risk_free_rate_annual)
    downside = excess[excess < 0]
    if len(downside) < 2:
        # No (or one) negative returns — Sortino is undefined / infinite.
        # Return a large finite value to avoid NaN propagation, but cap it.
        return 0.0 if excess.mean() <= 0 else float("inf")
    # By convention, divide by RMS of negatives, treating zero/positive as 0.
    # This is the López de Prado convention.
    downside_dev = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    if downside_dev < 1e-12:
        return 0.0
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """CAGR divided by absolute max drawdown.

    Args:
        returns: Periodic returns.
        periods_per_year: Annualization factor.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return cagr(returns, periods_per_year) / abs(mdd)


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with strictly positive returns.

    Args:
        returns: Periodic returns.

    Returns:
        Win rate in [0, 1].
    """
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


# ----------------------------------------------------------------------------
# Drawdown
# ----------------------------------------------------------------------------


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the drawdown at each point in time.

    Drawdown is the percentage decline from the running peak of the
    cumulative-return curve.

    Args:
        returns: Periodic returns.

    Returns:
        Series of drawdowns (all values <= 0).
    """
    cum = cumulative_returns(returns)
    peak = cum.cummax()
    return cum / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum (worst) drawdown over the full sample.

    Args:
        returns: Periodic returns.

    Returns:
        Max drawdown as a negative number (e.g. -0.25 for a 25% drawdown).
    """
    if len(returns) == 0:
        return 0.0
    return float(drawdown_series(returns).min())


# ----------------------------------------------------------------------------
# Rolling statistics
# ----------------------------------------------------------------------------


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Rolling Sharpe ratio.

    Args:
        returns: Periodic returns.
        window: Window length in periods.
        risk_free_rate_annual: Annualized risk-free rate.
        periods_per_year: Annualization factor.

    Returns:
        Rolling Sharpe series. Initial ``window-1`` values are NaN.
    """
    excess = excess_returns(returns, risk_free_rate_annual)
    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std(ddof=1)
    sharpe = rolling_mean / rolling_std * np.sqrt(periods_per_year)
    return sharpe.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------
# Information Coefficient
# ----------------------------------------------------------------------------


def information_coefficient(
    factor: pd.Series,
    returns: pd.Series,
    forward_periods: int = 1,
) -> tuple[float, float]:
    """Pearson IC: correlation between factor values and FUTURE returns.

    For most quant applications, prefer :func:`rank_information_coefficient`.

    Args:
        factor: Factor values, indexed by date. ``factor.loc[t]`` is the
            signal known at time ``t``.
        returns: Returns series.
        forward_periods: Number of periods forward to look. With
            ``forward_periods=1``, factor at ``t`` is correlated with
            return at ``t+1``.

    Returns:
        Tuple of (IC, p-value).
    """
    return _ic_impl(factor, returns, forward_periods, method="pearson")


def rank_information_coefficient(
    factor: pd.Series,
    returns: pd.Series,
    forward_periods: int = 1,
) -> tuple[float, float]:
    """Spearman rank IC: rank-correlation between factor and future returns.

    Rank IC is the standard choice in quant finance because it's robust
    to outliers and doesn't require normality.

    Args:
        factor: Factor values, indexed by date.
        returns: Returns series.
        forward_periods: Number of periods forward.

    Returns:
        Tuple of (rank IC, p-value).
    """
    return _ic_impl(factor, returns, forward_periods, method="spearman")


def _ic_impl(
    factor: pd.Series,
    returns: pd.Series,
    forward_periods: int,
    method: str,
) -> tuple[float, float]:
    """Shared implementation for Pearson/Spearman IC.

    Critical: we shift FACTOR forward (equivalently, shift RETURNS
    backward by ``forward_periods``). This means the factor value at
    time ``t`` is compared against returns from ``t+1`` to ``t+forward_periods``.
    This prevents look-ahead bias.
    """
    if forward_periods < 1:
        raise ValueError(f"forward_periods must be >= 1, got {forward_periods}")

    # Align inputs.
    aligned = pd.concat([factor.rename("factor"), returns.rename("returns")], axis=1).dropna()
    if len(aligned) < 30:
        return float("nan"), float("nan")

    # Sum returns over the forward window (compound for accuracy, but for
    # IC purposes the sum is close enough and is the textbook convention).
    fwd_returns = (
        aligned["returns"].rolling(forward_periods).sum().shift(-forward_periods)
    )

    # Combine and drop NaN.
    df = pd.DataFrame({"factor": aligned["factor"], "fwd": fwd_returns}).dropna()
    if len(df) < 30:
        return float("nan"), float("nan")

    if method == "spearman":
        result = stats.spearmanr(df["factor"], df["fwd"])
    else:
        result = stats.pearsonr(df["factor"], df["fwd"])

    # scipy returns objects with .statistic and .pvalue (or tuples in older versions).
    if hasattr(result, "statistic"):
        return float(result.statistic), float(result.pvalue)
    return float(result[0]), float(result[1])


# ----------------------------------------------------------------------------
# Turnover
# ----------------------------------------------------------------------------


def turnover_from_positions(positions: pd.DataFrame) -> pd.Series:
    """Compute two-way turnover from a positions DataFrame.

    Two-way turnover at time t is sum |w_t - w_{t-1}| across assets,
    which counts both opening and closing legs of a trade.

    Args:
        positions: DataFrame indexed by date with one column per asset.
            Values are portfolio weights (need not sum to 1 — leverage OK).

    Returns:
        Series of per-period two-way turnover.
    """
    if positions.empty:
        return pd.Series(dtype=float)
    return positions.diff().abs().sum(axis=1)
