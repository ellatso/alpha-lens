"""End-to-end integration tests for the autopsy pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lens import autopsy
from alpha_lens.core.types import ReadinessVerdict


@pytest.fixture
def basic_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    n = 1000
    return pd.Series(
        rng.normal(0.0008, 0.012, n),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


@pytest.fixture
def basic_benchmark() -> pd.Series:
    rng = np.random.default_rng(43)
    n = 1000
    return pd.Series(
        rng.normal(0.0003, 0.014, n),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


class TestAutopsyMinimal:
    def test_returns_only(self, basic_returns: pd.Series) -> None:
        """Verify autopsy works with just a returns series."""
        report = autopsy(basic_returns)
        assert report.statistics.n_observations == len(basic_returns)
        assert report.statistics.sharpe_ratio is not None
        # Without benchmark, regime detection still works (rule-based).
        assert report.regime is not None
        # Attribution requires factors or benchmark — should be None.
        assert report.attribution is None
        # Decay requires factor_values — should be None.
        assert report.decay is None
        # Readiness is always computed.
        assert report.readiness is not None
        assert 0 <= report.readiness.overall_score <= 100
        assert report.readiness.verdict in ReadinessVerdict

    def test_with_benchmark_enables_attribution(
        self, basic_returns: pd.Series, basic_benchmark: pd.Series
    ) -> None:
        report = autopsy(basic_returns, benchmark_returns=basic_benchmark)
        assert report.attribution is not None
        assert "MKT" in report.attribution.betas

    def test_with_factors_enables_multifactor(self, basic_returns: pd.Series) -> None:
        rng = np.random.default_rng(7)
        n = len(basic_returns)
        factors = pd.DataFrame(
            {
                "MKT": rng.normal(0.0003, 0.014, n),
                "SMB": rng.normal(0.0001, 0.008, n),
                "HML": rng.normal(0.0002, 0.007, n),
            },
            index=basic_returns.index,
        )
        report = autopsy(basic_returns, factors=factors)
        assert report.attribution is not None
        assert set(report.attribution.factor_names) == {"MKT", "SMB", "HML"}
        assert report.correlation is not None
        assert report.correlation.correlation_matrix.shape == (3, 3)

    def test_with_factor_values_enables_decay(self, basic_returns: pd.Series) -> None:
        rng = np.random.default_rng(11)
        factor_vals = pd.Series(
            rng.standard_normal(len(basic_returns)),
            index=basic_returns.index,
        )
        report = autopsy(basic_returns, factor_values=factor_vals)
        assert report.decay is not None
        assert len(report.decay.ic_by_horizon) > 0


class TestReadinessScore:
    def test_score_within_bounds(self, basic_returns: pd.Series) -> None:
        report = autopsy(basic_returns)
        assert 0.0 <= report.readiness.overall_score <= 100.0
        for c in report.readiness.components:
            assert 0.0 <= c.score <= 100.0
            assert c.status in ("pass", "warn", "fail")
            assert 0.0 <= c.weight <= 1.0

    def test_weights_sum_to_one(self, basic_returns: pd.Series) -> None:
        """Components are re-normalized after potentially being skipped."""
        report = autopsy(basic_returns)
        total_weight = sum(c.weight for c in report.readiness.components)
        assert abs(total_weight - 1.0) < 1e-6

    def test_strong_strategy_scores_well(self) -> None:
        """Build a deliberately strong (synthetic) strategy."""
        rng = np.random.default_rng(0)
        n = 2000
        # Sharpe ~2.0 with low autocorrelation.
        rets = pd.Series(
            rng.normal(0.001, 0.008, n),
            index=pd.date_range("2018-01-01", periods=n, freq="B"),
        )
        report = autopsy(rets)
        # We don't assert exact verdict because PBO has stochasticity,
        # but the score should be in the upper half.
        assert report.readiness.overall_score >= 50

    def test_garbage_strategy_scores_poorly(self) -> None:
        """A negative-Sharpe strategy should be rejected."""
        rng = np.random.default_rng(0)
        n = 1000
        rets = pd.Series(
            rng.normal(-0.0005, 0.015, n),  # negative drift
            index=pd.date_range("2020-01-01", periods=n, freq="B"),
        )
        report = autopsy(rets)
        assert report.readiness.verdict in (
            ReadinessVerdict.NOT_READY,
            ReadinessVerdict.REJECT,
        )


class TestReportRendering:
    def test_html_file_written(self, basic_returns: pd.Series) -> None:
        report = autopsy(basic_returns)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)
        try:
            written = report.save(str(path))
            written_path = Path(written)
            assert written_path.exists()
            size = written_path.stat().st_size
            # Should include Plotly.js — expect 3-6MB.
            assert size > 1_000_000
            assert size < 10_000_000
            # Should contain key visual elements.
            content = written_path.read_text(encoding="utf-8")
            assert "Production Readiness" in content
            assert "alpha-lens" in content or "alpha&#8209;" in content
            # All 6 default tabs should appear at minimum.
            assert "panel-overview" in content
            assert "panel-drawdowns" in content
            assert "panel-readiness" in content
            # Plotly should be embedded.
            assert "Plotly.newPlot" in content
            # No leftover template placeholders.
            for placeholder in [
                "$title", "$score_svg", "$verdict_text",
                "$components_html", "$panels_html",
            ]:
                assert placeholder not in content, f"Unsubstituted: {placeholder}"
        finally:
            if path.exists():
                path.unlink()

    def test_renders_with_no_benchmark(self, basic_returns: pd.Series) -> None:
        report = autopsy(basic_returns)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = Path(f.name)
        try:
            report.save(str(path))
            content = path.read_text(encoding="utf-8")
            # No attribution tab.
            assert "panel-attribution" not in content
        finally:
            if path.exists():
                path.unlink()


class TestConfigOverride:
    def test_n_trials_override_changes_dsr(self, basic_returns: pd.Series) -> None:
        report_low = autopsy(basic_returns, n_trials_assumed=10)
        report_high = autopsy(basic_returns, n_trials_assumed=10000)
        # Higher n_trials → stricter DSR p-value.
        assert (
            report_high.overfitting.deflated_sharpe_pvalue
            >= report_low.overfitting.deflated_sharpe_pvalue
        )
