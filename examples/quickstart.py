"""Quickstart: a 20-line autopsy on a synthetic strategy.

This example shows the typical pattern: a strategy with a respectable
Sharpe (~1.2) that nonetheless does NOT pass production readiness
without further work. The autopsy report explains why — and what to
fix before deploying.

Run from the repo root::

    python examples/quickstart.py

It writes ``out/quickstart.html`` — open it in any browser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lens import autopsy


def main() -> None:
    # Synthesize a strategy with a strong but believable Sharpe (~1.6).
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", periods=1500, freq="B")
    # Mean ~22% annual, vol ~14%. Finite-sample Sharpe lands around 1.5.
    daily_returns = rng.normal(0.00088, 0.0088, len(dates))

    # Inject a moderate 6% drawdown around 2020-Q1.
    crash_start = 540
    daily_returns[crash_start : crash_start + 8] = rng.normal(-0.008, 0.011, 8)

    strategy = pd.Series(daily_returns, index=dates, name="my_strategy")

    print("Running autopsy()...")
    report = autopsy(strategy)
    r = report.readiness
    print(f"  Sharpe:  {report.statistics.sharpe_ratio:.2f}")
    print(f"  Score:   {r.overall_score:.1f} / 100")
    print(f"  Verdict: {r.verdict.value.upper()}")
    for risk in r.top_risks[:3]:
        print(f"   - {risk}")

    out = Path("out/quickstart.html")
    report.save(str(out))
    print(f"\nReport written to {out.resolve()}")


if __name__ == "__main__":
    main()
