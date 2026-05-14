"""Tests for overfitting diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lens.analysis.overfitting import (
    deflated_sharpe_ratio,
    minimum_backtest_length,
    probability_of_backtest_overfitting,
)


class TestDeflatedSharpe:
    def test_low_dsr_for_high_n_trials(self) -> None:
        """Same returns, more trials → lower DSR."""
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0.0005, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B"),
        )
        dsr_10, _ = deflated_sharpe_ratio(rets, n_trials=10)
        dsr_1000, _ = deflated_sharpe_ratio(rets, n_trials=1000)
        assert dsr_10 > dsr_1000

    def test_p_value_in_range(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0.0005, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B"),
        )
        _, p = deflated_sharpe_ratio(rets, n_trials=100)
        assert 0 <= p <= 1

    def test_handles_short_series(self) -> None:
        rets = pd.Series(
            [0.001] * 20,
            index=pd.date_range("2020-01-01", periods=20, freq="B"),
        )
        dsr, p = deflated_sharpe_ratio(rets)
        assert dsr == 0.0
        assert p == 1.0


class TestMinimumBacktestLength:
    def test_high_sharpe_needs_short_backtest(self) -> None:
        # A strategy with Sharpe 3.0 needs less data than one with Sharpe 0.5.
        sr_high = minimum_backtest_length(3.0, n_trials=100)
        sr_low = minimum_backtest_length(0.5, n_trials=100)
        assert sr_high < sr_low

    def test_more_trials_need_more_data(self) -> None:
        sr = 1.0
        few = minimum_backtest_length(sr, n_trials=10)
        many = minimum_backtest_length(sr, n_trials=1000)
        assert many > few

    def test_zero_or_negative_sharpe_requires_infinite_data(self) -> None:
        assert minimum_backtest_length(0.0) == float("inf")
        assert minimum_backtest_length(-1.0) == float("inf")


class TestPBO:
    def test_pbo_for_real_signal_is_low(self) -> None:
        """A real signal embedded among noise should have LOW PBO."""
        rng = np.random.default_rng(7)
        n = 2000
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        # One strategy has real signal (high mean), others are pure noise.
        real = rng.normal(0.0015, 0.01, n)  # Sharpe ~2.4
        noise = rng.normal(0, 0.012, (n, 19))
        all_strats = np.column_stack([real.reshape(-1, 1), noise])
        df = pd.DataFrame(all_strats, index=dates, columns=[f"s{i}" for i in range(20)])
        pbo = probability_of_backtest_overfitting(df, n_partitions=8)
        assert pbo < 0.3

    def test_pbo_for_all_noise_is_high(self) -> None:
        """All-noise should have PBO near 0.5 on average."""
        # Run a few seeds to check that PBO straddles 0.5.
        pbos = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            n = 1500
            df = pd.DataFrame(
                rng.normal(0, 0.012, (n, 15)),
                index=pd.date_range("2020-01-01", periods=n, freq="B"),
                columns=[f"s{i}" for i in range(15)],
            )
            pbos.append(probability_of_backtest_overfitting(df, n_partitions=8))
        # On average should be around 0.5.
        assert 0.3 < np.mean(pbos) < 0.8

    def test_raises_on_single_strategy(self) -> None:
        df = pd.DataFrame(
            {"s0": [0.01] * 100},
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
        )
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(df)

    def test_raises_on_odd_partitions(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.normal(0, 0.01, (500, 5)),
            index=pd.date_range("2020-01-01", periods=500, freq="B"),
            columns=[f"s{i}" for i in range(5)],
        )
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(df, n_partitions=7)
