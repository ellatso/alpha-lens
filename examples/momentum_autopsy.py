"""Momentum strategy autopsy with full input set.

Builds a simple time-series momentum strategy on a synthetic universe,
then runs the full autopsy with benchmark, factor returns, factor
values for IC analysis, and positions for cost analysis.

Run::

    python examples/momentum_autopsy.py

Output: ``out/momentum_autopsy.html``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lens import autopsy


def _generate_universe(n_days: int, n_assets: int, seed: int = 7) -> pd.DataFrame:
    """A synthetic equity universe with realistic cross-sectional structure."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-01-01", periods=n_days, freq="B")

    # Each asset has a small idiosyncratic drift and beta to a market.
    market = rng.normal(0.0003, 0.011, n_days)
    drifts = rng.normal(0.0001, 0.0003, n_assets)
    betas = rng.uniform(0.6, 1.4, n_assets)
    idio_vol = 0.012

    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        returns[:, i] = drifts[i] + betas[i] * market + rng.normal(0, idio_vol, n_days)

    return pd.DataFrame(returns, index=dates, columns=[f"A{i:02d}" for i in range(n_assets)])


def _build_momentum_signal(returns: pd.DataFrame, *, lookback: int = 126) -> pd.DataFrame:
    """Cross-sectional 6-month momentum signal."""
    signal = returns.rolling(lookback).sum()
    # Cross-sectional rank, normalized to [-1, 1].
    ranks = signal.rank(axis=1, pct=True)
    return ranks * 2.0 - 1.0


def _signal_to_positions(signal: pd.DataFrame, *, top_k: int = 5) -> pd.DataFrame:
    """Long top-k, short bottom-k, equal weight, market-neutral."""
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date, row in signal.iterrows():
        if row.isna().any():
            continue
        ranked = row.sort_values()
        shorts = ranked.iloc[:top_k].index
        longs = ranked.iloc[-top_k:].index
        positions.loc[date, longs] = 1.0 / top_k
        positions.loc[date, shorts] = -1.0 / top_k
    return positions


def main() -> None:
    # 1. Universe and benchmark.
    universe = _generate_universe(n_days=1750, n_assets=30)
    benchmark = universe.mean(axis=1)  # equal-weighted "market"
    benchmark.name = "EW Market"

    # 2. Build momentum signal and positions.
    signal_df = _build_momentum_signal(universe, lookback=126)
    positions = _signal_to_positions(signal_df, top_k=6)

    # 3. Strategy returns = sum(position_t-1 * return_t).
    strategy_returns = (positions.shift(1) * universe).sum(axis=1).dropna()
    strategy_returns.name = "Momentum L/S"

    # 4. Factor returns: market + a synthetic size and quality factor.
    rng = np.random.default_rng(99)
    smb = pd.Series(rng.normal(0, 0.005, len(universe)), index=universe.index, name="SMB")
    qmj = pd.Series(rng.normal(0, 0.004, len(universe)), index=universe.index, name="QMJ")
    factors = pd.concat([benchmark.rename("MKT"), smb, qmj], axis=1)

    # 5. Factor value series for IC analysis: average rank momentum at each date.
    factor_values = signal_df.mean(axis=1).rename("avg_mom_rank")

    print("Running autopsy with full input set ...\n")
    report = autopsy(
        strategy_returns,
        benchmark_returns=benchmark,
        factors=factors,
        factor_values=factor_values,
        positions=positions,
        n_trials_assumed=500,  # Pessimistic — we probably tried more parameters than we admit.
    )

    r = report.readiness
    print(f"  Score:   {r.overall_score:.1f} / 100")
    print(f"  Verdict: {r.verdict.value.upper()}")
    print("  Components:")
    for c in r.components:
        marker = {"pass": "✓", "warn": "·", "fail": "✗"}[c.status]
        print(f"    {marker} {c.name:30s} {c.score:>5.1f}  {c.value}")
    print()
    print(f"  Recommendation: {r.recommendation}")

    out = Path("out/momentum_autopsy.html")
    report.save(str(out))
    print(f"\nReport written to {out.resolve()}")


if __name__ == "__main__":
    main()
