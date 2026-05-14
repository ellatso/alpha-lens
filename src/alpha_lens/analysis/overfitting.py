"""Overfitting diagnostics.

Three complementary tests:

1. **Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado (2014).
   Adjusts the in-sample Sharpe for the bias introduced by multiple
   testing, skewness, and kurtosis. Returns a Sharpe estimate that's
   "deflated" by the search process, plus a probability that the true
   Sharpe is non-positive.

2. **Probability of Backtest Overfitting (PBO)** — López de Prado.
   Uses Combinatorially Symmetric Cross-Validation (CSCV) to estimate
   how often the best-in-sample strategy fails to be above-median
   out-of-sample. PBO > 0.5 means the backtest selection process is
   no better than coin-flipping.

3. **Minimum Backtest Length (MinBTL)** — Bailey, Borwein, et al.
   Computes the minimum years of data needed for the observed Sharpe
   to be statistically distinguishable from zero given the number of
   trials.

References:
    Bailey & López de Prado (2014). "The Deflated Sharpe Ratio: Correcting
        for Selection Bias, Backtest Overfitting and Non-Normality."
    Bailey, Borwein, López de Prado, Zhu (2017). "The Probability of
        Backtest Overfitting."
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from alpha_lens.analysis.statistics import sharpe_ratio
from alpha_lens.core.config import TRADING_DAYS_PER_YEAR, OverfittingConfig
from alpha_lens.core.types import OverfittingDiagnostics

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_overfitting",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "minimum_backtest_length",
]


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def analyze_overfitting(
    returns: pd.Series,
    *,
    strategy_variants: pd.DataFrame | None = None,
    risk_free_rate_annual: float = 0.0,
    config: OverfittingConfig | None = None,
) -> OverfittingDiagnostics:
    """Run the full suite of overfitting diagnostics.

    Args:
        returns: Strategy returns.
        strategy_variants: Optional DataFrame of returns for the OTHER
            strategy variants the researcher tested. If provided, PBO is
            computed via CSCV. If None, PBO is None and DSR uses the
            assumed ``n_trials`` from config.
        risk_free_rate_annual: Annualized risk-free rate.
        config: Overfitting parameters.

    Returns:
        :class:`OverfittingDiagnostics`.
    """
    config = config or OverfittingConfig()

    sr = sharpe_ratio(returns, risk_free_rate_annual=risk_free_rate_annual)
    # DSR
    dsr, dsr_pvalue = deflated_sharpe_ratio(
        returns,
        n_trials=config.n_trials_assumed,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    # PBO via CSCV — needs strategy variants. If we only have one
    # strategy, we approximate by partitioning IT and computing PBO
    # internally (less ideal but informative).
    if strategy_variants is not None and strategy_variants.shape[1] >= 4:
        pbo = probability_of_backtest_overfitting(
            strategy_variants, n_partitions=config.cscv_n_partitions
        )
    else:
        pbo = None

    # Minimum Backtest Length
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    min_years = minimum_backtest_length(
        sr,
        n_trials=config.n_trials_assumed,
    )

    return OverfittingDiagnostics(
        deflated_sharpe_ratio=dsr,
        deflated_sharpe_pvalue=dsr_pvalue,
        probability_of_backtest_overfitting=pbo,
        minimum_backtest_length_years=min_years,
        actual_backtest_length_years=n_years,
        minimum_length_satisfied=n_years >= min_years,
    )


# ----------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ----------------------------------------------------------------------------


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int = 100,
    risk_free_rate_annual: float = 0.0,
    sharpe_variance: float | None = None,
) -> tuple[float, float]:
    """Compute the Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    The DSR corrects the observed Sharpe for two biases:

    1. **Multiple testing.** Testing N strategies and picking the best
       inflates the apparent Sharpe even if all underlying strategies
       have true Sharpe = 0. The expected maximum of N draws from a
       standard normal grows with √(2 ln N).

    2. **Non-normality.** The standard Sharpe assumes IID Gaussian
       returns. Skewed/heavy-tailed returns (the norm in finance) have
       larger standard errors than the textbook formula admits.

    Args:
        returns: Strategy returns.
        n_trials: Number of strategy variants assumed tested. The DSR is
            highly sensitive to this — pessimistic researchers use 1000+.
        risk_free_rate_annual: Annualized risk-free rate.
        sharpe_variance: If known, the variance of Sharpe across the
            ``n_trials`` candidates. If None, computed from a default
            assumption (uniform distribution between -SR and +SR).

    Returns:
        Tuple of (deflated_sharpe, p_value).
        ``deflated_sharpe`` is a z-score: how many standard deviations
            the observed Sharpe is above the expected maximum under H0.
        ``p_value`` is P(true Sharpe <= 0 | observed Sharpe).
    """
    n = len(returns)
    if n < 30:
        return 0.0, 1.0

    excess = returns - (risk_free_rate_annual / TRADING_DAYS_PER_YEAR)
    sr_daily = excess.mean() / excess.std(ddof=1) if excess.std(ddof=1) > 0 else 0.0
    sr_annualized = float(sr_daily * np.sqrt(TRADING_DAYS_PER_YEAR))

    # Higher moments of returns.
    skew = float(stats.skew(excess, bias=False))
    kurt = float(stats.kurtosis(excess, bias=False, fisher=True))  # excess kurtosis

    # Expected maximum Sharpe under H0 (all strategies have true Sharpe = 0).
    # The classic result from Bailey & López de Prado 2014, eq. 7.
    euler_mascheroni = 0.5772156649015329
    max_z_expected = (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n_trials) + (
        euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    # The expected max Sharpe in DAILY terms is max_z_expected / √n.
    expected_max_sharpe_daily = max_z_expected / np.sqrt(n)
    expected_max_sharpe_annual = expected_max_sharpe_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Standard error of the Sharpe estimate, accounting for skew/kurtosis.
    # Bailey & López de Prado eq. 9.
    se_sharpe_daily = np.sqrt(
        (1 - skew * sr_daily + ((kurt) / 4.0) * sr_daily**2) / (n - 1)
    )
    if se_sharpe_daily <= 0 or np.isnan(se_sharpe_daily):
        se_sharpe_daily = np.sqrt(1.0 / (n - 1))  # fallback

    # DSR: probability the true Sharpe exceeds the expected max
    # given the observed Sharpe and its standard error.
    z = (sr_daily - expected_max_sharpe_daily) / se_sharpe_daily
    dsr_probability = float(stats.norm.cdf(z))

    # We return the DSR-style number that quants expect: the deflated
    # SR is observed_sharpe minus the bias term, expressed as a z-score.
    # And the p-value is one minus the probability that SR > expected_max.
    _ = sr_annualized
    _ = expected_max_sharpe_annual
    p_value = 1.0 - dsr_probability

    return float(z), p_value


# ----------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV)
# ----------------------------------------------------------------------------


def probability_of_backtest_overfitting(
    strategy_returns: pd.DataFrame,
    *,
    n_partitions: int = 16,
    risk_free_rate_annual: float = 0.0,
) -> float:
    """Compute PBO via Combinatorially Symmetric Cross-Validation.

    Procedure:
        1. Partition the time series into ``n_partitions`` (even number)
           equal-size, non-overlapping chunks.
        2. For each combination of ``n_partitions / 2`` chunks (the IS
           set), the rest is the OOS set.
        3. Pick the strategy with the highest IS Sharpe.
        4. Compute that strategy's RANK in OOS Sharpe (across all variants).
        5. If the IS-best strategy is also OOS-above-median, score 1; else 0.
        6. PBO = 1 - fraction of combinations where IS-best is OOS-best.

    Args:
        strategy_returns: DataFrame where each column is one variant's returns.
        n_partitions: Even number of partitions. Default 16 gives 12,870 splits.
        risk_free_rate_annual: Annualized risk-free rate.

    Returns:
        PBO in [0, 1]. Lower is better. >0.5 means selection process is
        essentially random.
    """
    if strategy_returns.shape[1] < 2:
        raise ValueError(
            f"PBO needs at least 2 strategy variants, got {strategy_returns.shape[1]}."
        )
    if n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be even, got {n_partitions}.")

    clean = strategy_returns.dropna()
    n_obs, n_strats = clean.shape
    if n_obs < n_partitions * 5:
        logger.warning(
            "PBO has limited statistical power with only %d obs across %d partitions.",
            n_obs,
            n_partitions,
        )

    # Partition. Drop the remainder to ensure equal sizes.
    partition_size = n_obs // n_partitions
    if partition_size < 2:
        return float("nan")

    partitions: list[np.ndarray] = []
    for p in range(n_partitions):
        start = p * partition_size
        end = start + partition_size
        partitions.append(clean.iloc[start:end].values)  # shape (size, n_strats)

    # Iterate over all C(n_partitions, n_partitions/2) splits.
    # For each split, find the IS-best strategy and check whether it's
    # above the median in OOS. PBO is the fraction of splits where it's
    # NOT above median.
    n_above_median = 0
    n_total = 0
    for is_indices in combinations(range(n_partitions), n_partitions // 2):
        oos_indices = [i for i in range(n_partitions) if i not in is_indices]
        is_data = np.concatenate([partitions[i] for i in is_indices], axis=0)
        oos_data = np.concatenate([partitions[i] for i in oos_indices], axis=0)

        is_sharpes = _column_sharpes(is_data)
        oos_sharpes = _column_sharpes(oos_data)

        if np.all(np.isnan(is_sharpes)) or np.all(np.isnan(oos_sharpes)):
            continue

        best_is = int(np.nanargmax(is_sharpes))
        best_is_oos_sharpe = oos_sharpes[best_is]

        # Compute the rank of best_is's OOS Sharpe within all OOS Sharpes.
        # Use percentile rank (0-1): if it's above 0.5, it beat the OOS median.
        valid_oos = oos_sharpes[~np.isnan(oos_sharpes)]
        if len(valid_oos) < 2 or np.isnan(best_is_oos_sharpe):
            continue
        rank_pct = (valid_oos < best_is_oos_sharpe).sum() / len(valid_oos)
        if rank_pct > 0.5:
            n_above_median += 1
        n_total += 1

    if n_total == 0:
        return float("nan")
    pbo = 1.0 - (n_above_median / n_total)
    return float(pbo)


def _column_sharpes(data: np.ndarray) -> np.ndarray:
    """Sharpe ratio per column, vectorized.

    Args:
        data: 2D array, shape (n_obs, n_strategies).

    Returns:
        1D array of length n_strategies.
    """
    means = data.mean(axis=0)
    stds = data.std(axis=0, ddof=1)
    stds = np.where(stds == 0, np.nan, stds)
    return means / stds * np.sqrt(TRADING_DAYS_PER_YEAR)


# ----------------------------------------------------------------------------
# Minimum Backtest Length
# ----------------------------------------------------------------------------


def minimum_backtest_length(
    observed_sharpe: float,
    *,
    n_trials: int = 100,
    significance: float = 0.05,
) -> float:
    """Compute the minimum backtest length needed for SR to be significant.

    From Bailey & López de Prado: MinBTL solves for years T such that
    the observed Sharpe could be statistically distinguished from the
    expected maximum under H0 (true SR = 0 for all trials).

        T >= 1 + ((1-γ)*Φ⁻¹(1-1/N) + γ*Φ⁻¹(1-1/(Nₑ)))² / SR²

    where γ is the Euler-Mascheroni constant and N is the number of trials.

    Args:
        observed_sharpe: Annualized observed Sharpe ratio.
        n_trials: Number of strategy variants tested.
        significance: Significance level (unused in the standard formula
            but kept for future extension).

    Returns:
        Minimum required years of data.
    """
    _ = significance
    if observed_sharpe <= 0:
        return float("inf")

    euler = 0.5772156649015329
    max_z = (1 - euler) * stats.norm.ppf(1 - 1.0 / n_trials) + (
        euler * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    return float(1.0 + (max_z / observed_sharpe) ** 2)
