# CLAUDE.md

This file gives an AI coding assistant (Claude, Copilot, etc.) the
context to make safe, idiomatic changes to `alpha-lens`. Read it
before suggesting edits.

## Repository orientation

```
src/alpha_lens/
  __init__.py            # Public API.
  core/
    types.py             # Pydantic models for every result type.
    config.py            # Frozen dataclass configs.
    validator.py         # Input validation and cleaning.
  analysis/              # Pure functions, no I/O, fully tested.
    statistics.py        # Sharpe, drawdown, IC, etc.
    regime.py            # Rule-based and HMM regime detection.
    drawdown.py          # Drawdown event detection + regime tagging.
    attribution.py       # OLS factor attribution.
    decay.py             # IC decay and half-life estimation.
    correlation.py       # Factor correlation and VIF.
    overfitting.py       # DSR, PBO, MinBTL.
    validation.py        # IS/OOS split, walk-forward.
    robustness.py        # Bootstrap CI, subsample stability.
    cost_analysis.py     # Cost sensitivity sweep and break-even.
    scoring.py           # Production Readiness Score aggregator.
  report/
    generator.py         # The `autopsy()` orchestrator.
    html_report.py       # Standalone HTML renderer.
    sections.py          # Per-tab content builders.
    template.py          # HTML/CSS/JS template (one string).
  viz/
    charts.py            # 11 Plotly chart builders.
    styles.py            # Color tokens, fonts, base_layout dict.

tests/                   # 85 tests, all under 6 seconds.
examples/                # 3 runnable examples.
docs/                    # concepts.md, api.md, img/
```

## Invariants (do not violate)

1. **Decimal returns only.** `0.01` means 1%, not 1. The validator
   warns on percent-encoded input; do not "fix" the warning by
   accepting both.

2. **Always √252 annualization.** Sharpe ratios use `np.sqrt(252)`,
   not `np.sqrt(365)`. If a function needs a different period count,
   it accepts a `periods_per_year` argument with the default 252.

3. **Forward-shifted returns for IC.** The factor at time `t` should
   correlate with returns at `t + h`, NEVER with returns at `t`.
   `rank_information_coefficient` enforces this with a `forward_periods`
   argument; the tests include a regression check.

4. **No interpolation.** NaNs in returns are forward-filled briefly
   and then dropped. We never invent return values.

5. **All analysis functions are pure.** No I/O, no randomness without
   a seed, no logging side effects, no global state. The boundary
   between pure and impure code is the `report/` and `data/` modules;
   `analysis/` is always pure.

6. **Pydantic v2 models.** All result types are pydantic v2 models
   with `model_config = ConfigDict(arbitrary_types_allowed=True)`.
   Adding a new result type? Inherit from `AlphaLensModel` in
   `core/types.py` so the config is uniform.

7. **No re-introduction of look-ahead.** Any new diagnostic that
   uses forward returns must explicitly shift them and have a test
   that fails if look-ahead is present (use the "factor = today's
   return" trick from `test_statistics.py`).

8. **Configurable, opinionated.** Default weights and thresholds are
   defensible choices documented in `docs/concepts.md`. Override via
   `ScoringConfig`. Do not soften the defaults to make scores easier
   to pass; that defeats the purpose of the library.

## How to add a new diagnostic

1. Add a result type in `core/types.py`.
2. Add the analysis function in `analysis/<module>.py` &mdash; pure,
   typed, no I/O. Add to `analysis/__init__.py` exports.
3. Add tests with **synthetic data of known ground truth**. The
   pattern: construct data with a property you can verify (known IC,
   known half-life, known regime structure), then check the estimator
   recovers it within a tolerance. Look at `conftest.py` for
   generators.
4. If it should affect the score, add a scorer function in
   `analysis/scoring.py` and a weight in `core/config.ScoringConfig`.
   Weights must sum to 1.0; the dataclass validates this in
   `__post_init__`.
5. If it should appear in the report, add a chart in `viz/charts.py`
   and surface it in the appropriate `_*_section` function in
   `report/sections.py`.

## Testing

```bash
cd alpha-lens
pip install -e ".[dev]"
pytest                              # 85 tests, ~6 seconds
pytest --cov=alpha_lens             # coverage report
```

Tests live in `tests/` and are organized one file per module. The
`conftest.py` fixtures are the canonical way to generate synthetic
data. If you add a fixture, document its ground truth in the docstring.

## What NOT to do

- Do not add new dependencies without compelling reason. Current core
  deps: numpy, pandas, scipy, scikit-learn, plotly, statsmodels,
  pydantic. Anything beyond these goes behind an optional extra
  (`[hmm]`, `[data]`, `[llm]`).
- Do not call `np.random` directly. Always use a `Generator` from
  `np.random.default_rng(seed)` so tests are reproducible.
- Do not reach for matplotlib. The viz layer is Plotly; mixing
  introduces font and theme inconsistency.
- Do not "improve" the scoring weights to make a particular strategy
  pass. The weights are the product, not a parameter to be tuned.
- Do not introduce a HTTP/network call in `analysis/`. Network calls
  go in `data/` (and are opt-in via an optional extra).
- Do not store secrets, API keys, or large data files in the repo.
