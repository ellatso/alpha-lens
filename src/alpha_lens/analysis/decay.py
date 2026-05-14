"""Alpha decay analysis: how fast does the signal lose predictive power?

All alphas decay. The question is *how fast*. Decay analysis answers:

* What's the IC at different forward horizons (1d, 5d, 20d, 60d)?
* What's the half-life — how many days until IC drops by 50%?
* Is the IC stable over time, or did it work in 2018 but stop in 2021?
* Is the decay regime-dependent?

These metrics matter because they determine how often you need to
rebalance and how soon you'll need to re-research the factor.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import optimize

from alpha_lens.analysis.statistics import rank_information_coefficient
from alpha_lens.core.types import DecayMetrics

logger = logging.getLogger(__name__)

__all__ = ["analyze_decay", "rolling_ic", "estimate_half_life"]


def analyze_decay(
    factor: pd.Series,
    returns: pd.Series,
    *,
    horizons: tuple[int, ...] = (1, 5, 20, 60),
    rolling_window: int = 252,
) -> DecayMetrics:
    """Run a full alpha-decay analysis.

    Args:
        factor: Factor values, indexed by date. ``factor.loc[t]`` is the
            signal as it would be known at time ``t``.
        returns: Returns series.
        horizons: Forward-return horizons (days) at which to compute IC.
        rolling_window: Window for rolling IC computation.

    Returns:
        :class:`DecayMetrics` with horizon-IC map, half-life estimate, and
        rolling-IC stability.
    """
    ic_by_horizon: dict[int, float] = {}
    ic_pvalue_by_horizon: dict[int, float] = {}
    for h in horizons:
        ic, pval = rank_information_coefficient(factor, returns, forward_periods=h)
        ic_by_horizon[h] = float(ic)
        ic_pvalue_by_horizon[h] = float(pval)

    half_life = estimate_half_life(ic_by_horizon)

    rolling = rolling_ic(factor, returns, forward_periods=1, window=rolling_window)
    full_ic = ic_by_horizon.get(1, 0.0)
    ic_stability = _ic_stability(rolling, reference_ic=full_ic)

    return DecayMetrics(
        ic_by_horizon=ic_by_horizon,
        ic_pvalue_by_horizon=ic_pvalue_by_horizon,
        estimated_half_life_days=half_life,
        rolling_ic=rolling,
        ic_stability=ic_stability,
    )


def rolling_ic(
    factor: pd.Series,
    returns: pd.Series,
    *,
    forward_periods: int = 1,
    window: int = 252,
) -> pd.Series:
    """Compute rolling rank-IC.

    For each rolling window, rank both the factor values and the
    forward-shifted returns, then compute the Pearson correlation of
    those ranks (which equals Spearman correlation of the originals).

    Args:
        factor: Factor values.
        returns: Returns series.
        forward_periods: Forward-return horizon.
        window: Rolling window length.

    Returns:
        Series of rolling rank-IC values.
    """
    fwd = returns.rolling(forward_periods).sum().shift(-forward_periods)
    aligned = pd.concat([factor.rename("f"), fwd.rename("r")], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    return _rolling_spearman(aligned["f"], aligned["r"], window=window)


def _rolling_spearman(x: pd.Series, y: pd.Series, *, window: int) -> pd.Series:
    """True rolling Spearman: rank within each window, then correlate."""
    df = pd.concat([x, y], axis=1).dropna()
    df.columns = ["x", "y"]
    out_index = df.index[window - 1 :]
    out_values = np.empty(len(out_index))

    x_arr = df["x"].values
    y_arr = df["y"].values

    for i, end in enumerate(range(window, len(df) + 1)):
        start = end - window
        x_win = x_arr[start:end]
        y_win = y_arr[start:end]
        # ranks
        x_rank = pd.Series(x_win).rank().values
        y_rank = pd.Series(y_win).rank().values
        # Pearson on ranks = Spearman on values
        x_centered = x_rank - x_rank.mean()
        y_centered = y_rank - y_rank.mean()
        num = (x_centered * y_centered).sum()
        denom = np.sqrt((x_centered**2).sum() * (y_centered**2).sum())
        out_values[i] = num / denom if denom > 0 else np.nan

    return pd.Series(out_values, index=out_index, name="rolling_ic")


def estimate_half_life(ic_by_horizon: dict[int, float]) -> float | None:
    """Fit exponential decay to IC-vs-horizon and report half-life.

    Model: IC(h) = IC(1) * exp(-λ * (h - 1)), so half-life = ln(2) / λ.

    Args:
        ic_by_horizon: Map of horizon (days) → IC value.

    Returns:
        Half-life in days, or None if no decay is detectable (IC doesn't
        decline with horizon, or IC is too noisy to fit).
    """
    if len(ic_by_horizon) < 3:
        return None

    horizons = np.array(sorted(ic_by_horizon.keys()), dtype=float)
    ics = np.array([ic_by_horizon[int(h)] for h in horizons])

    # If IC at horizon 1 is near zero or negative, no decay to estimate.
    base_ic = ics[0]
    if abs(base_ic) < 0.005:
        return None

    # Normalize by base IC so we're fitting a pure exponential.
    normalized = ics / base_ic if base_ic > 0 else -ics / abs(base_ic)

    # The data should be monotonically decreasing (in normalized form).
    # If not, decay isn't a good model.
    if np.any(normalized[1:] > normalized[:-1] + 0.1):
        # IC is rising with horizon — that's not decay, that's mean reversion
        # or a multi-day signal. Half-life is undefined.
        return None

    # Fit log(IC) = log(IC_0) - λ*(h-1)
    # Equivalently: log(normalized) = -λ*(h-1)
    valid = normalized > 0
    if valid.sum() < 2:
        return None

    h_valid = horizons[valid] - 1.0  # start at h=1
    log_n_valid = np.log(normalized[valid])

    try:
        # Constrained least squares: enforce λ >= 0.
        def loss(lam: float) -> float:
            pred = -lam * h_valid
            return float(((pred - log_n_valid) ** 2).sum())

        result = optimize.minimize_scalar(loss, bounds=(0.0, 10.0), method="bounded")
        lam = float(result.x)
        if lam < 1e-6:
            return None
        return float(np.log(2) / lam)
    except Exception as e:
        logger.debug("Half-life fit failed: %s", e)
        return None


def _ic_stability(rolling: pd.Series, *, reference_ic: float) -> float:
    """Fraction of rolling-IC windows with same sign as the full-sample IC.

    A stable alpha has consistent IC sign over time. A regime-dependent
    one flips signs.

    Args:
        rolling: Rolling-IC series.
        reference_ic: Full-sample IC to compare against.

    Returns:
        Fraction in [0, 1]. 1.0 = perfectly consistent.
    """
    clean = rolling.dropna()
    if len(clean) == 0:
        return 0.0
    if abs(reference_ic) < 1e-6:
        return 0.5
    target_sign = np.sign(reference_ic)
    return float((np.sign(clean) == target_sign).mean())
