"""Factor correlation and multicollinearity analysis.

When attribution uses multiple factors, correlated factors create
unstable estimates and inflate apparent explanatory power. This module
quantifies that risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lens.core.types import CorrelationAnalysis

__all__ = ["analyze_correlation", "compute_vif"]


def analyze_correlation(factors: pd.DataFrame) -> CorrelationAnalysis:
    """Compute pairwise correlations and VIFs for a set of factors.

    Args:
        factors: DataFrame where each column is a factor.

    Returns:
        :class:`CorrelationAnalysis`.
    """
    clean = factors.dropna()
    if len(clean) < 30 or clean.shape[1] < 2:
        return CorrelationAnalysis(
            correlation_matrix=clean.corr() if not clean.empty else pd.DataFrame(),
            vif_scores={},
            max_correlation=0.0,
        )

    corr = clean.corr()

    # Max off-diagonal absolute correlation.
    # Use .copy() because .values can return a read-only view in newer pandas.
    abs_corr = corr.abs().values.copy()
    np.fill_diagonal(abs_corr, 0)
    max_corr = float(abs_corr.max()) if abs_corr.size > 0 else 0.0

    vif_scores = compute_vif(clean)

    return CorrelationAnalysis(
        correlation_matrix=corr,
        vif_scores=vif_scores,
        max_correlation=max_corr,
    )


def compute_vif(factors: pd.DataFrame) -> dict[str, float]:
    """Compute the Variance Inflation Factor for each column.

    VIF_i = 1 / (1 - R²_i) where R²_i is the R² of regressing column i
    on all other columns. VIF > 5 indicates problematic multicollinearity;
    VIF > 10 indicates severe.

    Args:
        factors: DataFrame of factors.

    Returns:
        Dict mapping factor name to VIF value.
    """
    clean = factors.dropna()
    cols = list(clean.columns)
    if len(cols) < 2 or len(clean) < len(cols) + 1:
        return {}

    vifs: dict[str, float] = {}
    for col in cols:
        y = clean[col].values
        X = clean.drop(columns=[col]).values
        X_with_const = np.column_stack([np.ones(len(X)), X])

        try:
            # OLS closed form.
            xtx = X_with_const.T @ X_with_const
            xty = X_with_const.T @ y
            beta = np.linalg.solve(xtx, xty)
            y_pred = X_with_const @ beta
            ss_res = float(((y - y_pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r_squared = float(np.clip(r_squared, 0.0, 0.9999))  # avoid division by zero
            vifs[col] = 1.0 / (1.0 - r_squared)
        except np.linalg.LinAlgError:
            vifs[col] = float("inf")

    return vifs
