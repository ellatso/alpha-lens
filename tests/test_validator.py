"""Tests for input validator and types."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from alpha_lens.core.types import AlphaData
from alpha_lens.core.validator import (
    InputValidationError,
    check_data_quality,
    validate_aligned,
    validate_returns,
)


class TestValidateReturns:
    def test_raises_on_non_series(self) -> None:
        with pytest.raises(InputValidationError, match="Series"):
            validate_returns([0.01, 0.02, 0.03])  # type: ignore[arg-type]

    def test_converts_string_index(self) -> None:
        # Generate unique date strings: 100 days starting from 2020-01-01.
        date_strings = [
            (pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(100)
        ]
        rets = pd.Series(
            np.random.normal(0, 0.01, 100),
            index=date_strings,
        )
        result = validate_returns(rets)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_warns_on_percent_input(self) -> None:
        # Returns in percent (e.g. 5 instead of 0.05).
        rets = pd.Series(
            np.random.normal(0, 2.0, 200),
            index=pd.date_range("2020-01-01", periods=200, freq="B"),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_returns(rets)
            assert any("DECIMAL" in str(warning.message) for warning in w)

    def test_raises_on_too_few_observations(self) -> None:
        rets = pd.Series(
            [0.001] * 30,
            index=pd.date_range("2020-01-01", periods=30, freq="B"),
        )
        with pytest.raises(InputValidationError, match="60"):
            validate_returns(rets, min_observations=60)

    def test_drops_duplicates(self) -> None:
        idx = list(pd.date_range("2020-01-01", periods=100, freq="B"))
        idx[50] = idx[49]
        rets = pd.Series(np.random.normal(0, 0.01, 100), index=idx)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = validate_returns(rets)
        assert not result.index.duplicated().any()


class TestValidateAligned:
    def test_warns_on_low_overlap(self) -> None:
        primary = pd.Series(
            np.random.normal(0, 0.01, 100),
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
        )
        secondary = pd.Series(
            np.random.normal(0, 0.01, 100),
            index=pd.date_range("2030-01-01", periods=100, freq="B"),
        )
        with pytest.raises(InputValidationError, match="overlap"):
            validate_aligned(primary, secondary)


class TestAlphaDataModel:
    def test_rejects_non_datetime_index(self) -> None:
        rets = pd.Series([0.01, 0.02, 0.03], index=[1, 2, 3])
        with pytest.raises(Exception):  # noqa: B017 — pydantic raises ValidationError
            AlphaData(returns=rets)

    def test_accepts_valid_input(self) -> None:
        rets = pd.Series(
            np.random.normal(0, 0.01, 100),
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
        )
        data = AlphaData(returns=rets)
        assert len(data.returns) == 100


class TestDataQuality:
    def test_no_issues_with_clean_data(self) -> None:
        rng = np.random.default_rng(0)
        rets = pd.Series(
            rng.normal(0.001, 0.01, 500),
            index=pd.date_range("2020-01-01", periods=500, freq="B"),
        )
        issues = check_data_quality(rets)
        # Clean random data shouldn't trigger major warnings.
        assert len([i for i in issues if i.severity == "error"]) == 0

    def test_flags_long_zero_runs(self) -> None:
        rets = pd.Series(
            [0.0] * 50 + list(np.random.normal(0, 0.01, 50)),
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
        )
        issues = check_data_quality(rets)
        codes = [i.code for i in issues]
        assert "long_zero_run" in codes
