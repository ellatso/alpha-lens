"""Compare a 'real' strategy with a 'lucky noise' strategy.

A common failure mode: a researcher runs N parameter sweeps, picks the
best Sharpe, and reports it as if it were the strategy. ``alpha-lens``
catches this through PBO and Deflated Sharpe.

This example builds two strategies:
    1. A genuine signal with moderate Sharpe.
    2. A best-of-N noise strategy that looks just as good by raw Sharpe.

Both are passed to ``autopsy()`` with ``strategy_variants`` so PBO is
computed. The 'lucky noise' strategy should score much worse.

Run::

    python examples/compare_real_vs_lucky.py

Outputs: ``out/real_strategy.html`` and ``out/lucky_strategy.html``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lens import autopsy


def main() -> None:
    rng = np.random.default_rng(2024)
    n = 2000
    dates = pd.date_range("2017-01-01", periods=n, freq="B")

    # --- Real strategy: genuine drift + noise.
    real = pd.Series(rng.normal(0.0008, 0.013, n), index=dates, name="Real")

    # --- Lucky noise: simulate 200 random strategies, keep the best.
    n_trials = 200
    bag = pd.DataFrame(
        rng.normal(0, 0.013, (n, n_trials)),
        index=dates,
        columns=[f"sim_{i}" for i in range(n_trials)],
    )
    in_sample_sharpes = bag.mean() / bag.std() * np.sqrt(252)
    best_col = str(in_sample_sharpes.idxmax())
    lucky = bag[best_col].rename("LuckyNoise")

    # All variants we tried, for PBO via CSCV.
    variants = bag.copy()

    print("== REAL STRATEGY ==")
    report_real = autopsy(real, strategy_variants=variants)
    _summarize(report_real)
    report_real.save("out/real_strategy.html")

    print("\n== LUCKY NOISE (best of 200 random strategies) ==")
    report_lucky = autopsy(lucky, strategy_variants=variants)
    _summarize(report_lucky)
    report_lucky.save("out/lucky_strategy.html")

    print("\nBoth reports written to ./out/")


def _summarize(report) -> None:
    r = report.readiness
    print(f"  Raw Sharpe:  {report.statistics.sharpe_ratio:.2f}")
    print(f"  DSR p-value: {report.overfitting.deflated_sharpe_pvalue:.3f}")
    print(f"  PBO:         {report.overfitting.probability_of_backtest_overfitting}")
    print(f"  Score:       {r.overall_score:.1f}  |  {r.verdict.value.upper()}")


if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    main()
