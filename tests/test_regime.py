"""Tests for regime detection."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from alpha_lens.analysis.regime import (
    analyze_regimes,
    detect_regimes_rule_based,
)
from alpha_lens.core.types import RegimeLabel


class TestRuleBasedRegime:
    def test_pure_bull_market(self) -> None:
        rng = np.random.default_rng(0)
        # Strong uptrend, low vol.
        rets = pd.Series(
            rng.normal(0.0015, 0.006, 504),
            index=pd.date_range("2020-01-01", periods=504, freq="B"),
        )
        labels = detect_regimes_rule_based(rets)
        # After warmup, bull should dominate.
        non_warmup = labels.iloc[60:]
        bull_fraction = (non_warmup == RegimeLabel.BULL.value).mean()
        assert bull_fraction > 0.7

    def test_crash_detected_as_bear_or_high_vol(self) -> None:
        rng = np.random.default_rng(1)
        n = 500
        rets = rng.normal(0.0005, 0.008, n)
        # Inject a crash starting at day 200.
        crash_start = 200
        crash_days = 20
        rets[crash_start : crash_start + crash_days] = rng.normal(-0.025, 0.030, crash_days)
        returns = pd.Series(rets, index=pd.date_range("2020-01-01", periods=n, freq="B"))
        labels = detect_regimes_rule_based(returns)
        # During and shortly after the crash, labels should be bear or high_vol.
        crash_period = labels.iloc[crash_start : crash_start + crash_days + 20]
        bad_regime_fraction = crash_period.isin(
            [RegimeLabel.BEAR.value, RegimeLabel.HIGH_VOL.value]
        ).mean()
        assert bad_regime_fraction > 0.6, (
            f"Expected crash to be flagged as bear/high_vol, got {Counter(crash_period)}"
        )

    def test_labels_aligned_to_index(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0.001, 0.01, 300),
            index=pd.date_range("2020-01-01", periods=300, freq="B"),
        )
        labels = detect_regimes_rule_based(rets)
        pd.testing.assert_index_equal(labels.index, rets.index)


class TestRegimeAnalysis:
    def test_per_regime_summaries_make_sense(self) -> None:
        """Bull regime should have positive mean, bear should have negative."""
        rng = np.random.default_rng(2)
        n = 1000
        rets = np.zeros(n)
        rets[:500] = rng.normal(0.0015, 0.008, 500)  # bull
        rets[500:] = rng.normal(-0.0015, 0.020, 500)  # bear
        returns = pd.Series(rets, index=pd.date_range("2018-01-01", periods=n, freq="B"))
        result = analyze_regimes(returns)

        # We should find at least 2 regimes.
        assert len(result.summaries) >= 2

        # Bull-labeled days should have positive average return; bear negative.
        for s in result.summaries:
            if s.regime == RegimeLabel.BULL and s.n_days > 50:
                assert s.mean_return_annualized > 0
            if s.regime == RegimeLabel.BEAR and s.n_days > 50:
                assert s.mean_return_annualized < 0

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        rng = np.random.default_rng(3)
        rets = pd.Series(
            rng.normal(0.0005, 0.012, 500),
            index=pd.date_range("2020-01-01", periods=500, freq="B"),
        )
        result = analyze_regimes(rets)
        # Each row of the transition matrix should sum to 1 (or 0 for unseen regimes).
        for _, row in result.transition_matrix.iterrows():
            row_sum = row.sum()
            assert abs(row_sum - 1.0) < 1e-6 or row_sum == 0.0

    def test_robustness_score_high_for_consistent_strategy(self) -> None:
        """A strategy positive in every regime should score high."""
        rng = np.random.default_rng(4)
        n = 1000
        rets = np.zeros(n)
        rets[:500] = rng.normal(0.0015, 0.008, 500)
        rets[500:] = rng.normal(-0.0015, 0.020, 500)
        market_returns = pd.Series(
            rets, index=pd.date_range("2018-01-01", periods=n, freq="B")
        )
        # Construct a strategy that does well regardless of regime.
        strategy = pd.Series(
            rng.normal(0.0008, 0.008, n), index=market_returns.index
        )
        result = analyze_regimes(strategy, benchmark_returns=market_returns)
        assert result.robustness_score > 0.3
