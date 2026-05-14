"""Shared fixtures and synthetic data generators for tests.

Synthetic data with KNOWN ground truth is essential for testing
quantitative code. Real market data is too noisy to verify whether an
estimator is computing the right thing — but with synthetic data, we
can construct, e.g., a returns series with a known half-life and check
that the estimator recovers it.

These generators are intentionally simple — they're not meant to fool
real quants, they're meant to produce data with controllable
properties for testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ----------------------------------------------------------------------------
# Basic generators
# ----------------------------------------------------------------------------


def make_iid_returns(
    n_days: int = 504,
    *,
    annual_return: float = 0.10,
    annual_vol: float = 0.15,
    seed: int = 42,
    start: str = "2020-01-01",
) -> pd.Series:
    """Generate IID Gaussian daily returns with known annualized properties.

    Args:
        n_days: Number of business days.
        annual_return: Annualized mean return (decimal).
        annual_vol: Annualized standard deviation.
        seed: Random seed.
        start: Start date.

    Returns:
        Daily returns Series.
    """
    rng = np.random.default_rng(seed)
    daily_mean = annual_return / 252
    daily_std = annual_vol / np.sqrt(252)
    dates = pd.date_range(start, periods=n_days, freq="B")
    returns = rng.normal(daily_mean, daily_std, n_days)
    return pd.Series(returns, index=dates, name="strategy")


def make_trending_returns(
    n_days: int = 504,
    *,
    annual_return: float = 0.20,
    annual_vol: float = 0.10,
    seed: int = 42,
) -> pd.Series:
    """Generate a strongly trending series (a 'bull market')."""
    return make_iid_returns(
        n_days=n_days,
        annual_return=annual_return,
        annual_vol=annual_vol,
        seed=seed,
    )


def make_crash(
    n_days: int = 504,
    *,
    crash_start: int = 100,
    crash_depth: float = -0.50,
    crash_duration: int = 20,
    seed: int = 42,
) -> pd.Series:
    """Generate a returns series with a sharp crash embedded.

    Args:
        n_days: Total days.
        crash_start: Day index at which the crash begins.
        crash_depth: Cumulative drop during the crash (e.g. -0.50 for -50%).
        crash_duration: Number of days the crash lasts.
        seed: Random seed.

    Returns:
        Returns Series with the crash embedded.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.008, n_days)
    # During the crash period, daily return is set such that cumulative drops to crash_depth.
    daily_crash = (1 + crash_depth) ** (1 / crash_duration) - 1
    # Add some noise around the crash drift.
    rets[crash_start : crash_start + crash_duration] = (
        daily_crash + rng.normal(0, 0.025, crash_duration)
    )
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.Series(rets, index=dates, name="crash_strategy")


def make_returns_with_drawdown(
    n_days: int = 252,
    *,
    drawdown_start: int = 100,
    depth: float = -0.25,
    duration: int = 30,
    recovery: int = 40,
    seed: int = 42,
) -> pd.Series:
    """Generate returns with a controlled drawdown-and-recovery pattern.

    Args:
        n_days: Total length.
        drawdown_start: Day at which drawdown begins.
        depth: Drawdown depth.
        duration: Days from peak to trough.
        recovery: Days from trough to recovery.
        seed: Random seed.

    Returns:
        Returns Series.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.007, n_days)

    # Drawdown phase: daily losses summing to ``depth``.
    daily_down = (1 + depth) ** (1 / duration) - 1
    rets[drawdown_start : drawdown_start + duration] = (
        daily_down + rng.normal(0, 0.008, duration)
    )

    # Recovery phase: daily gains undoing the drawdown.
    recover_target = 1 / (1 + depth) - 1  # gain needed
    daily_up = (1 + recover_target) ** (1 / recovery) - 1
    end = drawdown_start + duration + recovery
    if end <= n_days:
        rets[drawdown_start + duration : end] = (
            daily_up + rng.normal(0, 0.008, recovery)
        )

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.Series(rets, index=dates, name="dd_strategy")


# ----------------------------------------------------------------------------
# Factor generators
# ----------------------------------------------------------------------------


def make_market_factor(n_days: int = 504, *, seed: int = 7) -> pd.Series:
    """A simple 'market' factor with realistic properties."""
    return make_iid_returns(
        n_days=n_days, annual_return=0.08, annual_vol=0.18, seed=seed, start="2020-01-01"
    )


def make_levered_strategy(
    market: pd.Series, leverage: float = 1.5, alpha_annual: float = 0.0, seed: int = 11
) -> pd.Series:
    """Construct a strategy = leverage * market + iid noise + alpha."""
    rng = np.random.default_rng(seed)
    daily_alpha = alpha_annual / 252
    noise = rng.normal(0, 0.001, len(market))
    return market * leverage + daily_alpha + pd.Series(noise, index=market.index)


def make_decaying_alpha_factor(
    n_days: int = 1000,
    *,
    base_ic: float = 0.10,
    half_life: int = 30,
    annual_vol: float = 0.12,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Generate a factor with KNOWN exponential decay.

    Returns at time t+h are correlated with the factor at time t with
    coefficient ic_at_h = base_ic * exp(-ln(2) * h / half_life).

    Args:
        n_days: Number of days.
        base_ic: IC at horizon h=1.
        half_life: Days for IC to drop by 50%.
        annual_vol: Volatility of the returns series.
        seed: Random seed.

    Returns:
        Tuple of (factor_values, returns).
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)

    # Generate factor values (standardized).
    factor_vals = rng.standard_normal(n_days)

    # Build returns: r_{t+h} has correlation IC_h with factor_t.
    # We construct returns as a sum of weighted shifted factor values plus noise.
    returns = np.zeros(n_days)
    lam = np.log(2) / half_life
    # Use horizons 1 through ~3 half-lives.
    max_h = min(n_days - 1, half_life * 5)
    for h in range(1, max_h + 1):
        ic_h = base_ic * np.exp(-lam * (h - 1))
        if abs(ic_h) < 0.005:
            break
        # The weight per shift to achieve correlation ic_h is ~ic_h * daily_vol
        # (under simplifying assumptions).
        weight = ic_h * daily_vol / max_h
        returns[h:] += factor_vals[:-h] * weight
    # Add idiosyncratic noise.
    returns += rng.normal(0, daily_vol * 0.95, n_days)

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return (
        pd.Series(factor_vals, index=dates, name="factor"),
        pd.Series(returns, index=dates, name="returns"),
    )


def make_regime_switching_returns(
    n_days: int = 1000,
    *,
    seed: int = 42,
) -> tuple[pd.Series, list[tuple[int, int, str]]]:
    """Generate a series with clearly delineated bull / bear / sideways regimes.

    Returns:
        Tuple of (returns_series, list of (start_idx, end_idx, regime_label)).
    """
    rng = np.random.default_rng(seed)
    segments = [
        (0, 250, "bull"),         # 250 days of strong bull
        (250, 350, "bear"),       # 100 days of bear
        (350, 600, "bull"),       # recovery
        (600, 800, "sideways"),   # flat
        (800, 1000, "high_vol"),  # volatile but flat-mean
    ]
    rets = np.zeros(n_days)
    for start, end, regime in segments:
        n = end - start
        if regime == "bull":
            rets[start:end] = rng.normal(0.0012, 0.008, n)
        elif regime == "bear":
            rets[start:end] = rng.normal(-0.0015, 0.018, n)
        elif regime == "sideways":
            rets[start:end] = rng.normal(0.0001, 0.006, n)
        elif regime == "high_vol":
            rets[start:end] = rng.normal(0.0000, 0.025, n)

    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    return pd.Series(rets, index=dates, name="regime_strategy"), segments


# ----------------------------------------------------------------------------
# Pytest fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def iid_returns() -> pd.Series:
    """Standard IID daily returns, ~Sharpe 0.67."""
    return make_iid_returns()


@pytest.fixture
def market_returns() -> pd.Series:
    return make_market_factor()


@pytest.fixture
def crash_returns() -> pd.Series:
    return make_crash()


@pytest.fixture
def regime_returns() -> tuple[pd.Series, list[tuple[int, int, str]]]:
    return make_regime_switching_returns()


@pytest.fixture
def decaying_factor() -> tuple[pd.Series, pd.Series]:
    return make_decaying_alpha_factor()
