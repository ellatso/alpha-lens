"""Drawdown autopsy: detect every drawdown event and characterize it.

A drawdown is a peak-to-trough-to-recovery episode in the cumulative
return curve. This module identifies each one and answers:

* How deep, how long, how long to recover?
* What regime was the market in?
* Which days contributed most?

The output drives the "Drawdown Autopsy" section of the report — the
single section reviewers spend the most time on because it tells them
what happens when the strategy stops working.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

from alpha_lens.analysis.statistics import cumulative_returns
from alpha_lens.core.types import DrawdownAnalysis, DrawdownEvent, RegimeLabel

__all__ = ["analyze_drawdowns", "find_drawdown_events"]


def analyze_drawdowns(
    returns: pd.Series,
    *,
    regime_labels: pd.Series | None = None,
    min_depth: float = 0.05,
    top_n: int = 10,
    worst_days_per_event: int = 5,
) -> DrawdownAnalysis:
    """Run a full drawdown autopsy on a returns series.

    Args:
        returns: Periodic returns.
        regime_labels: Optional Series of regime labels for attribution.
        min_depth: Minimum drawdown depth to report (e.g. 0.05 = 5%).
            Smaller drawdowns are filtered out.
        top_n: Return at most this many of the worst drawdowns.
        worst_days_per_event: Number of worst single-day contributors to
            keep per drawdown event.

    Returns:
        :class:`DrawdownAnalysis` containing ranked events and aggregates.
    """
    events = find_drawdown_events(
        returns,
        regime_labels=regime_labels,
        min_depth=min_depth,
        worst_days_per_event=worst_days_per_event,
    )

    # Rank by depth (most severe first).
    events.sort(key=lambda e: e.depth)
    top_events = events[:top_n]

    n = len(events)
    avg_depth = float(np.mean([e.depth for e in events])) if events else 0.0
    avg_duration = float(np.mean([e.duration_days for e in events])) if events else 0.0
    recovery_days_list = [e.recovery_days for e in events if e.recovery_days is not None]
    avg_recovery = float(np.mean(recovery_days_list)) if recovery_days_list else None

    # Compute regime concentration: of the TOTAL drawdown severity, what
    # fraction occurred in each regime?
    regime_conc: dict[str, float] = {}
    if regime_labels is not None and events:
        total_severity = sum(abs(e.depth) for e in events)
        if total_severity > 0:
            regime_severity: dict[str, float] = {}
            for event in events:
                if event.dominant_regime is not None:
                    key = event.dominant_regime.value
                    regime_severity[key] = regime_severity.get(key, 0.0) + abs(event.depth)
            regime_conc = {k: v / total_severity for k, v in regime_severity.items()}

    return DrawdownAnalysis(
        events=top_events,
        n_drawdowns=n,
        avg_depth=avg_depth,
        avg_duration_days=avg_duration,
        avg_recovery_days=avg_recovery,
        regime_concentration=regime_conc,
    )


def find_drawdown_events(
    returns: pd.Series,
    *,
    regime_labels: pd.Series | None = None,
    min_depth: float = 0.05,
    worst_days_per_event: int = 5,
) -> list[DrawdownEvent]:
    """Identify every peak-to-trough-to-recovery drawdown event.

    Algorithm:
        1. Walk through the cumulative-return series.
        2. A new "peak" is set whenever cumulative returns reach a new high.
        3. A drawdown starts when cumulative falls below the peak.
        4. Track the trough (lowest point) during the drawdown.
        5. The drawdown "recovers" when cumulative returns reach the peak again.
        6. If the sample ends before recovery, the event is unrecovered.

    Args:
        returns: Periodic returns.
        regime_labels: Optional regime labels for attribution per event.
        min_depth: Filter out drawdowns shallower than this.
        worst_days_per_event: Number of worst-day contributors to record.

    Returns:
        Unsorted list of drawdown events.
    """
    if len(returns) == 0:
        return []

    cum = cumulative_returns(returns)
    events: list[DrawdownEvent] = []

    # Track state as we walk forward.
    peak_value = cum.iloc[0]
    peak_idx = 0
    trough_value = peak_value
    trough_idx = peak_idx
    in_drawdown = False

    cum_values = cum.values

    for i in range(1, len(cum_values)):
        v = cum_values[i]

        if not in_drawdown:
            if v >= peak_value:
                # New peak.
                peak_value = v
                peak_idx = i
            elif v < peak_value:
                # Drawdown begins.
                in_drawdown = True
                trough_value = v
                trough_idx = i
        else:
            # Already in drawdown.
            if v < trough_value:
                trough_value = v
                trough_idx = i
            if v >= peak_value:
                # Recovered.
                event = _build_event(
                    returns,
                    peak_idx=peak_idx,
                    trough_idx=trough_idx,
                    recovery_idx=i,
                    peak_value=peak_value,
                    trough_value=trough_value,
                    regime_labels=regime_labels,
                    worst_days_per_event=worst_days_per_event,
                )
                if abs(event.depth) >= min_depth:
                    events.append(event)
                # Reset state.
                in_drawdown = False
                peak_value = v
                peak_idx = i
                trough_value = v
                trough_idx = i

    # Open drawdown at sample end (unrecovered).
    if in_drawdown:
        event = _build_event(
            returns,
            peak_idx=peak_idx,
            trough_idx=trough_idx,
            recovery_idx=None,
            peak_value=peak_value,
            trough_value=trough_value,
            regime_labels=regime_labels,
            worst_days_per_event=worst_days_per_event,
        )
        if abs(event.depth) >= min_depth:
            events.append(event)

    return events


def _build_event(
    returns: pd.Series,
    *,
    peak_idx: int,
    trough_idx: int,
    recovery_idx: int | None,
    peak_value: float,
    trough_value: float,
    regime_labels: pd.Series | None,
    worst_days_per_event: int,
) -> DrawdownEvent:
    """Construct a DrawdownEvent from its boundary indices."""
    depth = float(trough_value / peak_value - 1.0)
    peak_date = returns.index[peak_idx]
    trough_date = returns.index[trough_idx]
    recovery_date = returns.index[recovery_idx] if recovery_idx is not None else None

    duration_days = (trough_date - peak_date).days
    recovery_days = (recovery_date - trough_date).days if recovery_date is not None else None

    # Dominant regime during the drawdown (peak → trough).
    # Weight each day by its loss magnitude — the days that actually
    # caused the drawdown matter more than incidental days where the
    # strategy was flat. Otherwise a long flat period before the real
    # crash dominates a frequency count.
    dominant_regime: RegimeLabel | None = None
    if regime_labels is not None:
        window_labels = regime_labels.iloc[peak_idx : trough_idx + 1]
        window_returns = returns.iloc[peak_idx : trough_idx + 1]
        if len(window_labels) > 0:
            # Weight: magnitude of NEGATIVE returns (positive days get 0 weight).
            weights = (-window_returns).clip(lower=0)
            if weights.sum() > 0:
                weighted = (
                    pd.DataFrame({"r": window_labels, "w": weights})
                    .groupby("r")["w"]
                    .sum()
                )
                top_label = str(weighted.idxmax())
            else:
                # No negative days — just take the mode.
                mode = window_labels.mode()
                top_label = str(mode.iloc[0]) if len(mode) > 0 else None  # type: ignore[assignment]
            if top_label is not None:
                with contextlib.suppress(ValueError):
                    dominant_regime = RegimeLabel(top_label)

    # Worst single-day losses during the drawdown.
    window_returns = returns.iloc[peak_idx : trough_idx + 1]
    worst = window_returns.nsmallest(min(worst_days_per_event, len(window_returns)))
    worst_days = [(idx.to_pydatetime(), float(val)) for idx, val in worst.items()]

    return DrawdownEvent(
        peak_date=peak_date.to_pydatetime(),
        trough_date=trough_date.to_pydatetime(),
        recovery_date=recovery_date.to_pydatetime() if recovery_date is not None else None,
        depth=depth,
        duration_days=duration_days,
        recovery_days=recovery_days,
        dominant_regime=dominant_regime,
        worst_days=worst_days,
    )
