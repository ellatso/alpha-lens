"""Tests for drawdown analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lens.analysis.drawdown import analyze_drawdowns, find_drawdown_events


class TestDrawdown:
    def test_single_known_drawdown(self) -> None:
        # +10%, +10%, -20%, -20%, +30% (recovers).
        rets = pd.Series(
            [0.10, 0.10, -0.20, -0.20, 0.30],
            index=pd.date_range("2020-01-01", periods=5, freq="B"),
        )
        events = find_drawdown_events(rets, min_depth=0.0)
        assert len(events) == 1
        evt = events[0]
        # Peak after day 2 (cum=1.21). Trough after day 4 (cum=1.21*0.8*0.8=0.7744).
        # Depth = 0.7744 / 1.21 - 1 = -0.36.
        assert abs(evt.depth - (-0.36)) < 0.01

    def test_unrecovered_drawdown_has_no_recovery_date(self) -> None:
        # Just goes down and stays down.
        rets = pd.Series(
            [0.05, -0.10, -0.05, -0.02],
            index=pd.date_range("2020-01-01", periods=4, freq="B"),
        )
        events = find_drawdown_events(rets, min_depth=0.0)
        assert len(events) == 1
        assert events[0].recovery_date is None
        assert events[0].recovery_days is None

    def test_min_depth_filters_small_drawdowns(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0.001, 0.005, 200),
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )
        events = find_drawdown_events(rets, min_depth=0.001)
        events_large = find_drawdown_events(rets, min_depth=0.20)
        assert len(events) >= len(events_large)

    def test_full_analysis_with_regimes(self) -> None:
        """Drawdown events should be tagged with the dominant regime."""
        rng = np.random.default_rng(1)
        n = 252
        rets = rng.normal(0.001, 0.008, n)
        # Inject a drawdown.
        rets[100:130] = rng.normal(-0.01, 0.02, 30)
        returns = pd.Series(rets, index=pd.date_range("2020-01-01", periods=n, freq="B"))
        regime_labels = pd.Series(
            ["bull"] * 100 + ["bear"] * 30 + ["bull"] * 122,
            index=returns.index,
        )
        result = analyze_drawdowns(returns, regime_labels=regime_labels, min_depth=0.03)
        assert result.n_drawdowns >= 1
        # The biggest drawdown should be tagged "bear".
        worst = result.events[0]
        assert worst.dominant_regime is not None
        assert worst.dominant_regime.value == "bear"

    def test_events_ranked_by_severity(self) -> None:
        """Returned events should be sorted by depth (most severe first)."""
        rng = np.random.default_rng(2)
        rets = rng.normal(0.001, 0.012, 500)
        returns = pd.Series(rets, index=pd.date_range("2020-01-01", periods=500, freq="B"))
        result = analyze_drawdowns(returns, min_depth=0.02)
        if len(result.events) > 1:
            for i in range(len(result.events) - 1):
                assert result.events[i].depth <= result.events[i + 1].depth

    def test_empty_returns(self) -> None:
        events = find_drawdown_events(pd.Series(dtype=float))
        assert events == []
