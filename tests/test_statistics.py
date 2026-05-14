"""Tests for core statistics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lens.analysis import statistics as st


class TestSharpeRatio:
    def test_zero_returns_gives_zero(self) -> None:
        rets = pd.Series([0.0] * 100, index=pd.date_range("2020-01-01", periods=100, freq="B"))
        assert st.sharpe_ratio(rets) == 0.0

    def test_constant_returns_handles_zero_std(self) -> None:
        rets = pd.Series([0.001] * 100, index=pd.date_range("2020-01-01", periods=100, freq="B"))
        # No variance → Sharpe is undefined. We return 0 to avoid NaN propagation.
        assert st.sharpe_ratio(rets) == 0.0

    def test_known_sharpe_recovered(self, iid_returns: pd.Series) -> None:
        # Generator: annual_return=0.10, annual_vol=0.15 → Sharpe ≈ 0.667.
        sr = st.sharpe_ratio(iid_returns)
        # With 504 days, finite-sample Sharpe will have some noise.
        assert 0.4 < sr < 1.0

    def test_annualization_uses_sqrt_252(self) -> None:
        # Sharpe should scale with √252, not √365.
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0.001, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B"),
        )
        sr_default = st.sharpe_ratio(rets)
        sr_calendar = st.sharpe_ratio(rets, periods_per_year=365)
        ratio = sr_calendar / sr_default
        expected = np.sqrt(365 / 252)
        assert abs(ratio - expected) < 0.01

    def test_excess_return_subtraction(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0.001, 0.01, 500),
            index=pd.date_range("2020-01-01", periods=500, freq="B"),
        )
        sr_rf_zero = st.sharpe_ratio(rets, risk_free_rate_annual=0.0)
        sr_rf_high = st.sharpe_ratio(rets, risk_free_rate_annual=0.10)
        # Higher RF → lower Sharpe.
        assert sr_rf_high < sr_rf_zero

    def test_empty_returns(self) -> None:
        assert st.sharpe_ratio(pd.Series(dtype=float)) == 0.0


class TestSortinoRatio:
    def test_sortino_higher_than_sharpe_when_upside_skewed(self) -> None:
        # Construct returns with mostly small losses and occasional big wins.
        rng = np.random.default_rng(0)
        rets_arr = rng.normal(-0.0001, 0.005, 500)
        rets_arr[::20] += 0.05  # big positive shocks
        rets = pd.Series(rets_arr, index=pd.date_range("2020-01-01", periods=500, freq="B"))
        sharpe = st.sharpe_ratio(rets)
        sortino = st.sortino_ratio(rets)
        # When skew is positive, Sortino should exceed Sharpe.
        assert sortino > sharpe


class TestMaxDrawdown:
    def test_no_drawdown_when_monotonic_up(self) -> None:
        rets = pd.Series(
            [0.001] * 100,
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
        )
        assert st.max_drawdown(rets) == 0.0

    def test_known_drawdown(self) -> None:
        # Build a series: +10%, +10%, -50% (single day).
        rets = pd.Series(
            [0.10, 0.10, -0.50],
            index=pd.date_range("2020-01-01", periods=3, freq="B"),
        )
        # Peak after day 2 is 1.21. After day 3, 0.605. Drawdown = -50%.
        mdd = st.max_drawdown(rets)
        assert abs(mdd - (-0.50)) < 1e-6

    def test_drawdown_is_always_non_positive(self, iid_returns: pd.Series) -> None:
        assert st.max_drawdown(iid_returns) <= 0


class TestInformationCoefficient:
    def test_perfect_predictor(self) -> None:
        # Factor = next day's return. IC should be exactly 1.
        rng = np.random.default_rng(0)
        n = 500
        rets = pd.Series(
            rng.normal(0, 0.01, n),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        # factor at time t = return at time t+1
        factor = rets.shift(-1)
        ic, p = st.rank_information_coefficient(factor.iloc[:-1], rets, forward_periods=1)
        assert ic > 0.99
        assert p < 1e-10

    def test_random_factor_ic_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        n = 1000
        rets = pd.Series(
            rng.normal(0, 0.01, n),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        # Independent factor.
        factor = pd.Series(rng.standard_normal(n), index=rets.index)
        ic, p = st.rank_information_coefficient(factor, rets, forward_periods=1)
        assert abs(ic) < 0.10

    def test_look_ahead_bias_prevented(self) -> None:
        """IC must use FUTURE returns, not contemporaneous.

        If we compute IC with forward_periods=1, the factor at time t
        should correlate with return at t+1, not return at t.
        """
        rng = np.random.default_rng(0)
        n = 500
        rets = pd.Series(
            rng.normal(0, 0.01, n),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        # Factor that exactly EQUALS today's return.
        factor = rets.copy()
        # With forward_periods=1, this factor should NOT have high IC —
        # because correlation between today's return and tomorrow's return
        # is near zero by construction.
        ic, _ = st.rank_information_coefficient(factor, rets, forward_periods=1)
        assert abs(ic) < 0.15

    def test_invalid_forward_periods_raises(self) -> None:
        rets = pd.Series([0.01] * 100, index=pd.date_range("2020-01-01", periods=100, freq="B"))
        factor = rets.copy()
        with pytest.raises(ValueError):
            st.rank_information_coefficient(factor, rets, forward_periods=0)


class TestCAGR:
    def test_known_cagr(self) -> None:
        # 21% over 1 trading year should give CAGR = 0.21.
        n = 252
        rng = np.random.default_rng(0)
        # Tiny perturbations to introduce variance but preserve total.
        target_total = 0.21
        daily = (1 + target_total) ** (1 / n) - 1
        rets = pd.Series(
            np.full(n, daily) + rng.normal(0, 1e-6, n),
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        cagr = st.cagr(rets)
        assert abs(cagr - target_total) < 0.001

    def test_catastrophic_loss(self) -> None:
        rets = pd.Series([-0.50, -0.50, -0.50], index=pd.date_range("2020-01-01", periods=3, freq="B"))
        # Total return: -87.5%. CAGR should be -1 (floor).
        cagr = st.cagr(rets)
        assert cagr == -1.0


class TestRollingSharpe:
    def test_warmup_period_is_nan(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0, 0.01, 252),
            index=pd.date_range("2020-01-01", periods=252, freq="B"),
        )
        rolling = st.rolling_sharpe(rets, window=63)
        # First 62 values should be NaN.
        assert rolling.iloc[:62].isna().all()
        assert not rolling.iloc[62:].isna().any()


class TestTurnover:
    def test_no_turnover_when_positions_constant(self) -> None:
        positions = pd.DataFrame(
            [[0.5, 0.5]] * 10,
            index=pd.date_range("2020-01-01", periods=10, freq="B"),
            columns=["A", "B"],
        )
        to = st.turnover_from_positions(positions)
        # First entry is NaN (no prior); rest are zero.
        assert (to.iloc[1:] == 0).all()

    def test_full_rotation(self) -> None:
        # Switch from 100% A to 100% B → two-way turnover = 2.
        positions = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=pd.date_range("2020-01-01", periods=2, freq="B"),
            columns=["A", "B"],
        )
        to = st.turnover_from_positions(positions)
        assert abs(to.iloc[1] - 2.0) < 1e-9
