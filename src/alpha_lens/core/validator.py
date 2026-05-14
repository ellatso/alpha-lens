"""Input validation and data-quality checks.

Bad inputs produce confusing analysis. This module catches common
mistakes early and produces actionable error messages — never silent
type errors or "TypeError: unsupported operand type" messages.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationWarning:
    """A non-fatal issue with the input data."""

    code: str
    message: str
    severity: str  # "info", "warning", "error"


class InputValidationError(ValueError):
    """Raised when input data is unusable for analysis."""


def validate_returns(returns: pd.Series, *, min_observations: int = 60) -> pd.Series:
    """Validate and lightly clean a returns series.

    Args:
        returns: Strategy returns. Must be a pandas Series with DatetimeIndex.
        min_observations: Minimum number of non-NaN observations to proceed.

    Returns:
        The cleaned returns series (sorted, deduplicated, NaN-handled).

    Raises:
        InputValidationError: If the series is fundamentally unusable.
    """
    if not isinstance(returns, pd.Series):
        raise InputValidationError(
            f"`returns` must be a pandas Series, got {type(returns).__name__}. "
            f"If you have a DataFrame, select a column: df['strategy']"
        )

    if not isinstance(returns.index, pd.DatetimeIndex):
        try:
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        except Exception as exc:
            raise InputValidationError(
                f"`returns` index must be convertible to DatetimeIndex. "
                f"Original index type: {type(returns.index).__name__}"
            ) from exc

    # Sort and deduplicate.
    returns = returns.sort_index()
    if returns.index.duplicated().any():
        n_dups = int(returns.index.duplicated().sum())
        warnings.warn(
            f"Found {n_dups} duplicate dates in returns; keeping last occurrence.",
            stacklevel=2,
        )
        returns = returns[~returns.index.duplicated(keep="last")]

    # Drop NaNs but warn if we lost a lot.
    n_before = len(returns)
    returns = returns.dropna()
    n_dropped = n_before - len(returns)
    if n_dropped > 0:
        frac_dropped = n_dropped / n_before
        if frac_dropped > 0.1:
            warnings.warn(
                f"Dropped {n_dropped} NaN values ({frac_dropped:.1%} of input). "
                f"Check your returns calculation.",
                stacklevel=2,
            )

    if len(returns) < min_observations:
        raise InputValidationError(
            f"Need at least {min_observations} return observations after cleaning, "
            f"got {len(returns)}. A meaningful autopsy requires roughly 3+ months of data."
        )

    # Detect catastrophic returns that suggest unit-confusion.
    if (returns.abs() > 1.0).any():
        n_extreme = int((returns.abs() > 1.0).sum())
        warnings.warn(
            f"Found {n_extreme} return values with magnitude > 100%. "
            f"Returns should be in DECIMAL form (e.g. 0.01 for 1%), not percent. "
            f"If your data is in percent, divide by 100.",
            stacklevel=2,
        )

    # Detect look-ahead bias smoking gun: too-high Sharpe.
    if len(returns) > 60:
        rough_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        if rough_sharpe > 6:
            warnings.warn(
                f"Implied Sharpe ratio of {rough_sharpe:.1f} is implausibly high. "
                f"Common causes: look-ahead bias, survivorship bias, or "
                f"counting future returns as known. Double-check your backtest.",
                stacklevel=2,
            )

    return returns


def validate_aligned(
    primary: pd.Series,
    secondary: pd.Series | pd.DataFrame,
    name: str = "secondary",
    *,
    min_overlap_fraction: float = 0.5,
) -> pd.Series | pd.DataFrame:
    """Reindex ``secondary`` to ``primary``'s index, checking for sufficient overlap.

    Args:
        primary: The reference index (typically the returns series).
        secondary: Series or DataFrame to align.
        name: Name of the secondary input for error messages.
        min_overlap_fraction: Minimum fraction of primary dates that must
            be covered by secondary before raising an error.

    Returns:
        The aligned secondary input.
    """
    if not isinstance(secondary.index, pd.DatetimeIndex):
        secondary = secondary.copy()
        secondary.index = pd.to_datetime(secondary.index)

    overlap = primary.index.intersection(secondary.index)
    overlap_frac = len(overlap) / len(primary) if len(primary) > 0 else 0.0

    if overlap_frac < min_overlap_fraction:
        raise InputValidationError(
            f"`{name}` overlaps with `returns` on only {overlap_frac:.1%} of dates "
            f"(need at least {min_overlap_fraction:.0%}). "
            f"Returns spans {primary.index.min().date()} to {primary.index.max().date()}; "
            f"`{name}` spans {secondary.index.min().date()} to {secondary.index.max().date()}."
        )

    aligned = secondary.reindex(primary.index)

    # ffill briefly to handle minor calendar mismatches (e.g. monthly factors).
    if isinstance(aligned, pd.DataFrame):
        aligned = aligned.ffill(limit=5)
    else:
        aligned = aligned.ffill(limit=5)

    return aligned


def check_data_quality(returns: pd.Series) -> list[ValidationWarning]:
    """Run a battery of data-quality checks and return any issues found.

    These are heuristics — they catch the common ways backtest data goes
    wrong. None of them are guarantees; they're flags for the user to
    investigate.

    Args:
        returns: A validated returns series.

    Returns:
        List of ValidationWarning objects. Empty if data looks clean.
    """
    issues: list[ValidationWarning] = []

    # Zero-return runs: suggest the strategy was idle (which is fine) or
    # that returns weren't computed properly (which isn't).
    zero_runs = _max_run_length(returns == 0)
    if zero_runs > 20:
        issues.append(
            ValidationWarning(
                code="long_zero_run",
                severity="warning",
                message=(
                    f"Found a run of {zero_runs} consecutive zero returns. "
                    f"If the strategy was inactive, this is fine. If not, your "
                    f"returns may be padded or have join issues."
                ),
            )
        )

    # Identical consecutive returns: strong sign of forward-fill bugs.
    if len(returns) > 10:
        diffs = returns.diff().abs()
        n_identical = int((diffs == 0).sum())
        if n_identical / len(returns) > 0.3:
            issues.append(
                ValidationWarning(
                    code="forward_fill_suspected",
                    severity="warning",
                    message=(
                        f"{n_identical / len(returns):.0%} of consecutive return pairs are "
                        f"identical. This often indicates forward-filled prices, which "
                        f"inflates Sharpe and distorts drawdowns."
                    ),
                )
            )

    # Extreme skew/kurtosis: not necessarily wrong, but worth knowing.
    if len(returns) > 30:
        kurt = float(returns.kurt())
        if abs(kurt) > 20:
            issues.append(
                ValidationWarning(
                    code="extreme_kurtosis",
                    severity="info",
                    message=(
                        f"Excess kurtosis is {kurt:.1f} — heavy tails. "
                        f"Standard Sharpe ratio assumes normality and will overstate "
                        f"performance. The Deflated Sharpe Ratio corrects for this."
                    ),
                )
            )

    return issues


def _max_run_length(mask: pd.Series) -> int:
    """Return the length of the longest run of True in a boolean Series."""
    if mask.empty:
        return 0
    # Group runs by where the value changes.
    grp = (mask != mask.shift()).cumsum()
    return int(mask.groupby(grp).sum().max())
