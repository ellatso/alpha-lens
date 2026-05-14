"""Tests for validation, robustness, costs, and the scoring layer."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from alpha_lens.analysis.cost_analysis import analyze_costs, apply_cost
from alpha_lens.analysis.robustness import analyze_robustness, bootstrap_sharpe
from alpha_lens.analysis.scoring import compute_readiness_score
from alpha_lens.analysis.statistics import sharpe_ratio
from alpha_lens.analysis.validation import (
    analyze_validation,
    train_test_split,
    walk_forward_sharpes,
)
from alpha_lens.core.types import (
    CoreStatistics,
    OverfittingDiagnostics,
    ReadinessVerdict,
    RobustnessResults,
    ValidationResults,
)

# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


class TestValidation:
    def test_train_test_split_returns_two_sharpes(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0.0005, 0.01, 500),
            index=pd.date_range("2020-01-01", periods=500, freq="B"),
        )
        is_sr, oos_sr = train_test_split(rets)
        assert isinstance(is_sr, float)
        assert isinstance(oos_sr, float)

    def test_walk_forward_yields_n_values(self) -> None:
        rng = np.random.default_rng(1)
        rets = pd.Series(
            rng.normal(0.0005, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B"),
        )
        sharpes = walk_forward_sharpes(rets, n_windows=5)
        assert len(sharpes) == 5

    def test_degradation_is_nonneg_when_both_positive(self) -> None:
        rng = np.random.default_rng(2)
        rets = pd.Series(
            rng.normal(0.001, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B"),
        )
        result = analyze_validation(rets)
        if result.in_sample_sharpe > 0 and result.out_of_sample_sharpe > 0:
            assert result.degradation_ratio > 0


# ----------------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------------


class TestRobustness:
    def test_bootstrap_ci_contains_point_estimate_on_average(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0.001, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B"),
        )
        actual_sr = sharpe_ratio(rets)
        (lo, hi), std = bootstrap_sharpe(rets, n_samples=500)
        # CI should generally bracket the point estimate.
        assert lo - 0.5 <= actual_sr <= hi + 0.5
        assert std > 0

    def test_subsample_results_returned(self) -> None:
        rng = np.random.default_rng(1)
        rets = pd.Series(
            rng.normal(0.001, 0.01, 1000),
            index=pd.date_range("2020-01-01", periods=1000, freq="B"),
        )
        result = analyze_robustness(rets)
        assert len(result.subsample_sharpes) == 4
        assert 0 <= result.subsample_positive_fraction <= 1


# ----------------------------------------------------------------------------
# Cost
# ----------------------------------------------------------------------------


class TestCost:
    def test_higher_cost_lowers_sharpe(self) -> None:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        rets = pd.Series(rng.normal(0.0008, 0.01, 500), index=dates)
        turnover = pd.Series([1.0] * 500, index=dates)
        sr_0 = sharpe_ratio(apply_cost(rets, turnover, cost_bps_per_unit=0))
        sr_50 = sharpe_ratio(apply_cost(rets, turnover, cost_bps_per_unit=50))
        assert sr_50 < sr_0

    def test_breakeven_cost_is_positive(self) -> None:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        # Strategy with moderate Sharpe.
        rets = pd.Series(rng.normal(0.0010, 0.01, 500), index=dates)
        positions = pd.DataFrame(
            {"A": [0.5, 0.5] * 250, "B": [0.5, -0.5] * 250},
            index=dates,
        )
        result = analyze_costs(rets, positions=positions)
        # Breakeven should exist and be positive (strategy is profitable).
        if result.breakeven_cost_bps is not None:
            assert result.breakeven_cost_bps > 0


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def _make_stats(sharpe: float, max_dd: float = -0.10) -> CoreStatistics:
    return CoreStatistics(
        total_return=0.15,
        cagr=0.08,
        annualized_volatility=0.15,
        sharpe_ratio=sharpe,
        sortino_ratio=sharpe * 1.2,
        calmar_ratio=0.08 / abs(max_dd),
        max_drawdown=max_dd,
        win_rate=0.55,
        n_observations=1000,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        years=4.0,
    )


def _make_overfitting(p: float, pbo: float | None = None, min_len_ok: bool = True) -> OverfittingDiagnostics:
    return OverfittingDiagnostics(
        deflated_sharpe_ratio=1.5,
        deflated_sharpe_pvalue=p,
        probability_of_backtest_overfitting=pbo,
        minimum_backtest_length_years=3.0 if min_len_ok else 10.0,
        actual_backtest_length_years=4.0,
        minimum_length_satisfied=min_len_ok,
    )


def _make_validation(is_sr: float, oos_sr: float, wf_consistency: float = 0.8) -> ValidationResults:
    return ValidationResults(
        in_sample_sharpe=is_sr,
        out_of_sample_sharpe=oos_sr,
        degradation_ratio=oos_sr / is_sr if is_sr > 0 else 0.0,
        walk_forward_sharpes=[oos_sr] * 5,
        walk_forward_consistency=wf_consistency,
    )


def _make_robustness(ci_lo: float = 0.5, ci_hi: float = 1.5) -> RobustnessResults:
    return RobustnessResults(
        sharpe_confidence_interval=(ci_lo, ci_hi),
        sharpe_std_bootstrap=0.3,
        subsample_sharpes=[1.0, 1.2, 0.8, 1.1],
        subsample_positive_fraction=1.0,
    )


class TestScoring:
    def test_strong_strategy_gets_ready_verdict(self) -> None:
        score = compute_readiness_score(
            statistics=_make_stats(1.8),
            overfitting=_make_overfitting(p=0.01, pbo=0.1),
            validation=_make_validation(is_sr=1.8, oos_sr=1.6, wf_consistency=1.0),
            robustness=_make_robustness(ci_lo=1.0, ci_hi=2.0),
        )
        assert score.overall_score >= 70
        assert score.verdict in (ReadinessVerdict.READY, ReadinessVerdict.CONDITIONAL)

    def test_weak_strategy_gets_reject_verdict(self) -> None:
        score = compute_readiness_score(
            statistics=_make_stats(0.2, max_dd=-0.40),
            overfitting=_make_overfitting(p=0.80, pbo=0.7, min_len_ok=False),
            validation=_make_validation(is_sr=1.5, oos_sr=-0.5, wf_consistency=0.2),
            robustness=_make_robustness(ci_lo=-0.5, ci_hi=0.5),
        )
        assert score.overall_score < 50
        assert score.verdict in (ReadinessVerdict.NOT_READY, ReadinessVerdict.REJECT)

    def test_top_risks_populated_when_components_fail(self) -> None:
        score = compute_readiness_score(
            statistics=_make_stats(0.5),
            overfitting=_make_overfitting(p=0.6, pbo=0.6),
            validation=_make_validation(is_sr=2.0, oos_sr=-0.5, wf_consistency=0.4),
            robustness=_make_robustness(ci_lo=-0.5, ci_hi=1.5),
        )
        assert len(score.top_risks) >= 1
