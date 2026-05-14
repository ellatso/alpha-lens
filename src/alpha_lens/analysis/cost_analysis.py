"""Transaction cost sensitivity and capacity analysis.

A backtest with 0bp costs is fantasy. This module sweeps over plausible
cost levels and reports:

* **Cost sensitivity:** Sharpe at each cost level.
* **Break-even cost:** the cost level at which Sharpe drops to zero.
* **Annualized turnover:** how aggressively the strategy trades, if
  positions are provided.

Capacity estimation (how much AUM can the strategy support before
market impact destroys edge) is highly market- and execution-specific
and is not attempted here — but the turnover number is the input most
practitioners feed into a capacity model.
"""

from __future__ import annotations

import pandas as pd

from alpha_lens.analysis.statistics import (
    sharpe_ratio,
    turnover_from_positions,
)
from alpha_lens.core.config import TRADING_DAYS_PER_YEAR, CostConfig
from alpha_lens.core.types import CostAnalysis

__all__ = ["analyze_costs", "apply_cost", "find_breakeven_cost"]


def analyze_costs(
    returns: pd.Series,
    *,
    positions: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
    risk_free_rate_annual: float = 0.0,
    config: CostConfig | None = None,
) -> CostAnalysis:
    """Sweep transaction costs and find the breakeven level.

    Either ``positions`` or ``turnover`` must be supplied to compute a
    realistic cost impact. If neither is supplied, we assume a uniform
    daily turnover of 100% (one full portfolio turn per day) as a
    pessimistic baseline.

    Args:
        returns: Strategy returns.
        positions: Optional positions DataFrame. Used to derive turnover.
        turnover: Optional pre-computed per-period turnover.
        risk_free_rate_annual: Annualized risk-free rate.
        config: Cost configuration.

    Returns:
        :class:`CostAnalysis`.
    """
    config = config or CostConfig()

    # Determine per-period turnover.
    if turnover is None and positions is not None:
        turnover = turnover_from_positions(positions)
    if turnover is None:
        # Assume 1.0 turnover per period as a worst-case baseline.
        # Mark annualized_turnover as None so the report can note this.
        turnover_used = pd.Series(1.0, index=returns.index)
        annualized_turnover: float | None = None
    else:
        turnover_used = turnover.reindex(returns.index).fillna(0.0)
        annualized_turnover = float(turnover_used.mean() * TRADING_DAYS_PER_YEAR)

    # Sweep costs.
    sensitivity: dict[float, float] = {}
    for cost_bps in config.cost_levels_bps:
        adjusted = apply_cost(returns, turnover_used, cost_bps_per_unit=cost_bps)
        sensitivity[cost_bps] = sharpe_ratio(
            adjusted, risk_free_rate_annual=risk_free_rate_annual
        )

    breakeven = find_breakeven_cost(
        returns,
        turnover=turnover_used,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    return CostAnalysis(
        cost_sensitivity=sensitivity,
        breakeven_cost_bps=breakeven,
        annualized_turnover=annualized_turnover,
    )


def apply_cost(
    returns: pd.Series,
    turnover: pd.Series,
    *,
    cost_bps_per_unit: float,
) -> pd.Series:
    """Subtract per-period transaction cost from returns.

    Cost per period = cost_bps_per_unit / 10000 * turnover_period.

    Args:
        returns: Gross returns.
        turnover: Per-period turnover series.
        cost_bps_per_unit: Cost in basis points per unit of turnover.

    Returns:
        Net returns.
    """
    cost_decimal = cost_bps_per_unit / 10_000.0
    aligned_turnover = turnover.reindex(returns.index).fillna(0.0)
    return returns - cost_decimal * aligned_turnover


def find_breakeven_cost(
    returns: pd.Series,
    *,
    turnover: pd.Series,
    risk_free_rate_annual: float = 0.0,
    max_bps: float = 1000.0,
    tolerance_bps: float = 0.5,
) -> float | None:
    """Find the cost level at which the strategy's Sharpe drops to zero.

    Uses bisection. Returns None if Sharpe is already non-positive at
    zero cost, or if the strategy is still positive at ``max_bps``.

    Args:
        returns: Strategy returns.
        turnover: Per-period turnover.
        risk_free_rate_annual: Annualized risk-free rate.
        max_bps: Upper bound for the search.
        tolerance_bps: Bisection tolerance.

    Returns:
        Break-even cost in basis points, or None.
    """
    sr_at_zero = sharpe_ratio(returns, risk_free_rate_annual=risk_free_rate_annual)
    if sr_at_zero <= 0:
        return None

    sr_at_max = sharpe_ratio(
        apply_cost(returns, turnover, cost_bps_per_unit=max_bps),
        risk_free_rate_annual=risk_free_rate_annual,
    )
    if sr_at_max > 0:
        return None  # Even at max_bps the strategy is profitable.

    lo, hi = 0.0, max_bps
    while hi - lo > tolerance_bps:
        mid = (lo + hi) / 2.0
        sr_mid = sharpe_ratio(
            apply_cost(returns, turnover, cost_bps_per_unit=mid),
            risk_free_rate_annual=risk_free_rate_annual,
        )
        if sr_mid > 0:
            lo = mid
        else:
            hi = mid

    return float((lo + hi) / 2.0)
