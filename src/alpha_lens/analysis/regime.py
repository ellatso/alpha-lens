"""Market regime detection.

A regime is a persistent state of the market — bull, bear, high-vol,
sideways — with characteristic statistical properties. Strategies often
"work" in some regimes and fail in others; identifying these patterns
is essential for explaining backtest results and assessing robustness.

Two methods are provided:

* **Rule-based** (default): transparent thresholds on moving averages,
  volatility, and drawdown. Easy to explain to a stakeholder.
* **HMM** (optional): Hidden Markov Model on returns. Discovers regimes
  data-driven without preset thresholds. Requires `hmmlearn`.

Both methods produce a Series of :class:`RegimeLabel` values aligned to
the input index, plus a transition matrix and per-regime performance
summary.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from alpha_lens.analysis.statistics import (
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)
from alpha_lens.core.config import TRADING_DAYS_PER_YEAR, RegimeConfig
from alpha_lens.core.types import (
    RegimeAnalysis,
    RegimeLabel,
    RegimeMethod,
    RegimeSummary,
)

logger = logging.getLogger(__name__)

__all__ = ["analyze_regimes", "detect_regimes_rule_based", "detect_regimes_hmm"]


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def analyze_regimes(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    method: RegimeMethod = RegimeMethod.RULE_BASED,
    config: RegimeConfig | None = None,
) -> RegimeAnalysis:
    """Detect regimes and analyze per-regime performance.

    The regime classification is computed on the BENCHMARK series if
    provided (more meaningful: regimes are properties of the market,
    not of the strategy), otherwise on the strategy returns themselves.

    Args:
        returns: Strategy returns.
        benchmark_returns: Market returns used to define regimes. If None,
            regimes are inferred from ``returns`` — less ideal because
            the strategy may smooth or amplify market behavior.
        method: ``RULE_BASED`` (transparent thresholds) or ``HMM``.
        config: Regime detection parameters. Defaults from
            :class:`RegimeConfig` are used if not provided.

    Returns:
        :class:`RegimeAnalysis` with per-regime breakdown.
    """
    config = config or RegimeConfig()
    series_for_regime = benchmark_returns if benchmark_returns is not None else returns
    # Align to the strategy returns' index.
    series_for_regime = series_for_regime.reindex(returns.index).ffill()

    if method == RegimeMethod.HMM:
        labels = detect_regimes_hmm(series_for_regime, config=config)
    else:
        labels = detect_regimes_rule_based(series_for_regime, config=config)

    # Per-regime summaries computed on STRATEGY returns.
    summaries = _summarize_regimes(returns, labels)
    transitions = _transition_matrix(labels)
    robustness = _robustness_score(summaries)

    return RegimeAnalysis(
        method=method,
        labels=labels,
        summaries=summaries,
        transition_matrix=transitions,
        robustness_score=robustness,
    )


# ----------------------------------------------------------------------------
# Rule-based regime detection
# ----------------------------------------------------------------------------


def detect_regimes_rule_based(
    returns: pd.Series,
    *,
    config: RegimeConfig | None = None,
) -> pd.Series:
    """Classify each date into one of four regimes via simple rules.

    Decision tree (checked in order):
        1. Realized vol > high threshold AND drawdown > bear threshold → bear
        2. Realized vol > high threshold → high_vol
        3. Short SMA > long SMA → bull
        4. Otherwise → sideways

    Args:
        returns: Series of returns (typically the benchmark/market).
        config: Thresholds. Uses defaults if None.

    Returns:
        Series of regime labels (as strings) aligned to ``returns.index``.
    """
    config = config or RegimeConfig()
    cum = (1.0 + returns).cumprod()

    # Compute regime ingredients.
    short_sma = cum.rolling(config.short_sma_window).mean()
    long_sma = cum.rolling(config.long_sma_window).mean()
    realized_vol = returns.rolling(config.vol_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Use a ROLLING peak (last 252 days) for the bear drawdown check.
    # If we used the all-time peak, then once any drawdown occurred, we'd
    # be flagged as in-drawdown forever — even during subsequent bull legs.
    rolling_peak_window = max(TRADING_DAYS_PER_YEAR, config.long_sma_window * 4)
    rolling_peak = cum.rolling(rolling_peak_window, min_periods=config.long_sma_window).max()
    rolling_dd = cum / rolling_peak - 1.0

    # Volatility threshold: the rolling-vol percentile across the full sample.
    # We compute this on the in-sample vol distribution. Using a fixed
    # value would be cleaner but markets vary too much across periods.
    vol_threshold = realized_vol.quantile(config.vol_high_percentile)

    # Build labels.
    labels = pd.Series(RegimeLabel.SIDEWAYS.value, index=returns.index, dtype=object)
    is_high_vol = realized_vol > vol_threshold
    is_in_bear_dd = rolling_dd < -config.bear_drawdown_threshold
    is_uptrend = short_sma > long_sma
    is_downtrend = short_sma < long_sma

    # Apply rules in order — later assignments win where conditions overlap.
    # 1. Uptrend without crisis vol → bull.
    labels[is_uptrend & ~is_high_vol] = RegimeLabel.BULL.value
    # 2. High volatility (regardless of trend) → high_vol.
    labels[is_high_vol] = RegimeLabel.HIGH_VOL.value
    # 3. Genuine bear: actively going down AND in a meaningful drawdown.
    #    Pure downtrend without drawdown = early correction (sideways).
    #    Pure drawdown without downtrend = recovery (not bear).
    #    Only when BOTH conditions hold do we call it bear.
    labels[is_in_bear_dd & is_downtrend & ~is_high_vol] = RegimeLabel.BEAR.value
    # 4. Bear with crisis vol: that's still bear (worst case).
    labels[is_in_bear_dd & is_downtrend & is_high_vol] = RegimeLabel.BEAR.value

    # Warmup period (before SMAs are valid) defaults to sideways.
    warmup_idx = labels.index[: config.long_sma_window]
    labels.loc[warmup_idx] = RegimeLabel.SIDEWAYS.value

    labels.name = "regime"
    return labels


# ----------------------------------------------------------------------------
# HMM regime detection
# ----------------------------------------------------------------------------


def detect_regimes_hmm(
    returns: pd.Series,
    *,
    config: RegimeConfig | None = None,
) -> pd.Series:
    """Fit a Gaussian HMM to returns and label hidden states.

    States are labeled by their statistical signature:
        * Highest mean return → BULL
        * Highest variance + negative mean → BEAR
        * Highest variance + non-negative mean → HIGH_VOL
        * Lowest variance → SIDEWAYS

    Args:
        returns: Series of returns.
        config: Regime config. Uses ``hmm_n_states`` if specified, else
            auto-selects via BIC.

    Returns:
        Series of regime labels.

    Raises:
        ImportError: If ``hmmlearn`` is not installed.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as exc:
        raise ImportError(
            "HMM regime detection requires `hmmlearn`. "
            "Install with: pip install 'alpha-lens[hmm]'"
        ) from exc

    config = config or RegimeConfig()
    clean = returns.dropna()
    if len(clean) < 100:
        warnings.warn(
            f"HMM needs >= 100 observations, got {len(clean)}. "
            f"Falling back to rule-based detection.",
            stacklevel=2,
        )
        return detect_regimes_rule_based(returns, config=config)

    X = clean.values.reshape(-1, 1)

    # Auto-select n_states via BIC if not specified.
    if config.hmm_n_states is None:
        best_bic, best_model = np.inf, None
        for n in range(2, 5):
            try:
                model = GaussianHMM(
                    n_components=n,
                    covariance_type="diag",
                    n_iter=200,
                    random_state=42,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X)
                log_likelihood = model.score(X)
                # BIC = -2*ll + k*log(n) where k = number of parameters
                k = n * n + 2 * n  # transitions + means + variances
                bic = -2 * log_likelihood + k * np.log(len(X))
                if bic < best_bic:
                    best_bic, best_model = bic, model
                    _ = n  # n_states inferred from best_model below
            except Exception as e:
                logger.debug("HMM with n=%d failed: %s", n, e)
        model = best_model
    else:
        model = GaussianHMM(
            n_components=config.hmm_n_states,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)

    if model is None:
        warnings.warn("HMM fitting failed; falling back to rule-based.", stacklevel=2)
        return detect_regimes_rule_based(returns, config=config)

    states = model.predict(X)
    labels = _label_hmm_states(states, model, len(clean))

    out = pd.Series(RegimeLabel.SIDEWAYS.value, index=returns.index, dtype=object)
    out.loc[clean.index] = labels
    out.name = "regime"
    return out


def _label_hmm_states(states: np.ndarray, model: object, _n: int) -> list[str]:
    """Map opaque HMM state indices to interpretable regime labels."""
    means = model.means_.flatten()  # type: ignore[attr-defined]
    variances = model.covars_.flatten()  # type: ignore[attr-defined]

    n_states = len(means)
    # Rank states.
    by_mean = np.argsort(-means)  # highest mean first
    by_var = np.argsort(-variances)  # highest variance first

    label_map: dict[int, str] = {}

    # Highest mean = BULL.
    label_map[int(by_mean[0])] = RegimeLabel.BULL.value

    # Lowest mean: BEAR if also high variance, else SIDEWAYS.
    lowest_mean = int(by_mean[-1])
    if lowest_mean not in label_map:
        if means[lowest_mean] < 0 and variances[lowest_mean] > variances.mean():
            label_map[lowest_mean] = RegimeLabel.BEAR.value
        else:
            label_map[lowest_mean] = RegimeLabel.SIDEWAYS.value

    # Highest variance among remaining: HIGH_VOL (unless already labeled).
    for s in by_var:
        s = int(s)
        if s not in label_map:
            label_map[s] = RegimeLabel.HIGH_VOL.value
            break

    # Anything else: SIDEWAYS.
    for s in range(n_states):
        if s not in label_map:
            label_map[s] = RegimeLabel.SIDEWAYS.value

    return [label_map[int(s)] for s in states]


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------


def _summarize_regimes(returns: pd.Series, labels: pd.Series) -> list[RegimeSummary]:
    """Compute per-regime performance statistics."""
    summaries: list[RegimeSummary] = []
    total_days = len(returns)
    for regime in RegimeLabel:
        mask = labels == regime.value
        n = int(mask.sum())
        if n < 5:
            continue
        regime_returns = returns[mask]
        summaries.append(
            RegimeSummary(
                regime=regime,
                n_days=n,
                fraction=n / total_days if total_days > 0 else 0.0,
                mean_return_annualized=float(regime_returns.mean() * TRADING_DAYS_PER_YEAR),
                volatility_annualized=annualized_volatility(regime_returns),
                sharpe_ratio=sharpe_ratio(regime_returns),
                max_drawdown=max_drawdown(regime_returns),
            )
        )
    return summaries


def _transition_matrix(labels: pd.Series) -> pd.DataFrame:
    """Compute the empirical transition matrix between regimes."""
    states = [r.value for r in RegimeLabel]
    matrix = pd.DataFrame(0.0, index=states, columns=states)
    prev = labels.shift(1)
    pairs = pd.DataFrame({"from": prev, "to": labels}).dropna()
    if pairs.empty:
        return matrix
    counts = pairs.groupby(["from", "to"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=states, columns=states, fill_value=0)
    # Normalize each row to sum to 1 (where possible).
    row_sums = counts.sum(axis=1)
    return counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)


def _robustness_score(summaries: list[RegimeSummary]) -> float:
    """Score 0-1: how consistent is the strategy across regimes?

    A strategy that has positive Sharpe in every regime gets 1.0. A
    strategy whose Sharpe is 3.0 in bull but -2.0 in bear gets a low
    score even though average performance might look great.
    """
    if not summaries:
        return 0.0

    sharpes = [s.sharpe_ratio for s in summaries if not np.isnan(s.sharpe_ratio)]
    if not sharpes:
        return 0.0

    # Fraction of regimes with positive Sharpe, weighted by time spent.
    weighted_positive = sum(
        s.fraction for s in summaries if s.sharpe_ratio > 0 and not np.isnan(s.sharpe_ratio)
    )
    total_weight = sum(s.fraction for s in summaries)
    if total_weight == 0:
        return 0.0

    # Combine: fraction of time in positive-Sharpe regimes, penalized by spread.
    base = weighted_positive / total_weight
    if len(sharpes) > 1:
        # Penalize cases where Sharpe varies wildly across regimes.
        sharpe_std = float(np.std(sharpes))
        sharpe_penalty = min(sharpe_std / 3.0, 0.5)  # cap penalty at 0.5
        base = base * (1.0 - sharpe_penalty)

    return float(np.clip(base, 0.0, 1.0))
