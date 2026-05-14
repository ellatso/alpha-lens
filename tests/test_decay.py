"""Tests for decay analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lens.analysis.decay import (
    analyze_decay,
    estimate_half_life,
    rolling_ic,
)


class TestEstimateHalfLife:
    def test_known_exponential_decay(self) -> None:
        # IC at horizons 1, 5, 10, 20 with half-life of 10:
        # IC(h) = 0.1 * exp(-ln(2)*(h-1)/10)
        true_hl = 10.0
        lam = np.log(2) / true_hl
        ic_map = {h: 0.1 * np.exp(-lam * (h - 1)) for h in [1, 5, 10, 20, 40]}
        est_hl = estimate_half_life(ic_map)
        assert est_hl is not None
        assert abs(est_hl - true_hl) < 3.0  # within 3 days

    def test_no_decay_returns_none(self) -> None:
        # IC the same at every horizon → not exponentially decaying.
        ic_map = {1: 0.05, 5: 0.05, 10: 0.05, 20: 0.05}
        est = estimate_half_life(ic_map)
        # With perfectly flat IC, the optimizer returns ~0 lam → None.
        # Allow either None or a very long half-life.
        assert est is None or est > 100

    def test_zero_ic_returns_none(self) -> None:
        ic_map = {1: 0.001, 5: 0.0005, 10: 0.0001}
        assert estimate_half_life(ic_map) is None

    def test_rising_ic_returns_none(self) -> None:
        # IC rising with horizon — not decay.
        ic_map = {1: 0.02, 5: 0.05, 10: 0.08, 20: 0.10}
        assert estimate_half_life(ic_map) is None


class TestAnalyzeDecay:
    def test_recovers_strong_ic_at_horizon_1(self) -> None:
        """A factor that strongly predicts t+1 should have detectable IC at h=1."""
        rng = np.random.default_rng(0)
        n = 1000
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Construct factor that predicts next-day returns.
        rets_arr = rng.normal(0, 0.012, n)
        factor_arr = np.zeros(n)
        for i in range(n - 1):
            factor_arr[i] = 0.5 * rets_arr[i + 1] + rng.normal(0, 0.012)
        factor = pd.Series(factor_arr, index=dates)
        returns = pd.Series(rets_arr, index=dates)

        metrics = analyze_decay(factor, returns, horizons=(1, 5, 20))
        # h=1 should show meaningful IC.
        assert abs(metrics.ic_by_horizon[1]) > 0.1
        # Higher horizons should have weaker IC.
        assert abs(metrics.ic_by_horizon[20]) < abs(metrics.ic_by_horizon[1])

    def test_random_factor_has_low_ic(self) -> None:
        rng = np.random.default_rng(0)
        n = 1000
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        factor = pd.Series(rng.standard_normal(n), index=dates)
        returns = pd.Series(rng.normal(0, 0.012, n), index=dates)
        metrics = analyze_decay(factor, returns, horizons=(1, 5, 20))
        for _h, ic in metrics.ic_by_horizon.items():
            assert abs(ic) < 0.1


class TestRollingIC:
    def test_rolling_ic_shape(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        factor = pd.Series(rng.standard_normal(n), index=dates)
        returns = pd.Series(rng.normal(0, 0.012, n), index=dates)
        result = rolling_ic(factor, returns, window=100)
        # Should produce some output.
        assert len(result) > 0
