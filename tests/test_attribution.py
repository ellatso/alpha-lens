"""Tests for factor attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lens.analysis.attribution import attribute_returns


class TestAttribution:
    def test_pure_market_beta(self) -> None:
        """Strategy = 1.5x market should recover beta=1.5, alpha=0, R²=1."""
        rng = np.random.default_rng(0)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        market = pd.Series(rng.normal(0.0005, 0.012, n), index=dates)
        strategy = market * 1.5

        result = attribute_returns(strategy, benchmark_returns=market)
        assert abs(result.betas["MKT"] - 1.5) < 0.001
        assert abs(result.alpha_annualized) < 0.005
        assert result.r_squared > 0.999

    def test_pure_alpha_low_r_squared(self) -> None:
        """Independent strategy should have R² near 0 and uniqueness near 1."""
        rng = np.random.default_rng(1)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        market = pd.Series(rng.normal(0.0005, 0.012, n), index=dates)
        # Independent strategy with positive drift.
        strategy = pd.Series(
            rng.normal(0.0008, 0.008, n), index=dates
        )
        result = attribute_returns(strategy, benchmark_returns=market)
        assert result.r_squared < 0.05
        assert result.uniqueness_score > 0.95
        # Alpha should be approximately the strategy's mean return * 252.
        expected_alpha = strategy.mean() * 252
        assert abs(result.alpha_annualized - expected_alpha) < 0.02

    def test_mixed_strategy(self) -> None:
        """A strategy with both beta and alpha components."""
        rng = np.random.default_rng(2)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        market = pd.Series(rng.normal(0.0005, 0.012, n), index=dates)
        independent = pd.Series(rng.normal(0.0010, 0.008, n), index=dates)
        # Strategy = 0.8 * market + independent
        strategy = market * 0.8 + independent
        result = attribute_returns(strategy, benchmark_returns=market)
        # Beta should be close to 0.8.
        assert abs(result.betas["MKT"] - 0.8) < 0.05
        # R² should be moderate.
        assert 0.3 < result.r_squared < 0.95

    def test_multiple_factors(self) -> None:
        """Custom factor regression with multiple factors."""
        rng = np.random.default_rng(3)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        mkt = pd.Series(rng.normal(0.0005, 0.012, n), index=dates)
        smb = pd.Series(rng.normal(0.0002, 0.008, n), index=dates)
        hml = pd.Series(rng.normal(0.0003, 0.007, n), index=dates)
        factors = pd.DataFrame({"MKT": mkt, "SMB": smb, "HML": hml})

        # Strategy: 1.0 * MKT + 0.5 * SMB + (-0.3) * HML + alpha.
        strategy = mkt + 0.5 * smb - 0.3 * hml + 0.0005

        result = attribute_returns(strategy, factors=factors)
        assert abs(result.betas["MKT"] - 1.0) < 0.05
        assert abs(result.betas["SMB"] - 0.5) < 0.05
        assert abs(result.betas["HML"] - (-0.3)) < 0.05

    def test_raises_when_no_factors_or_benchmark(self) -> None:
        rets = pd.Series(
            [0.001] * 100, index=pd.date_range("2020-01-01", periods=100, freq="B")
        )
        with pytest.raises(ValueError, match="either"):
            attribute_returns(rets)

    def test_raises_on_insufficient_overlap(self) -> None:
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 0.01, 100), index=pd.date_range("2020-01-01", periods=100, freq="B"))
        b = pd.Series(rng.normal(0, 0.01, 100), index=pd.date_range("2025-01-01", periods=100, freq="B"))
        with pytest.raises(ValueError):
            attribute_returns(a, benchmark_returns=b)
