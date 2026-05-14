# API Reference

This is a concise reference for the public API of `alpha-lens`. For the
methodology behind these functions, see [`concepts.md`](concepts.md).

## The entry point

### `alpha_lens.autopsy()`

```python
def autopsy(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    factors: pd.DataFrame | None = None,
    factor_values: pd.Series | pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    strategy_variants: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    regime_method: str = "rule_based",      # or "hmm"
    config: AutopsyConfig | None = None,
    n_trials_assumed: int | None = None,
) -> AutopsyReport
```

The main and usually only function you need. The only required input
is `returns`. Every other input enables an additional dimension of
analysis:

- **`benchmark_returns`** &rarr; enables CAPM-style attribution and a
  benchmark overlay on the equity curve.
- **`factors`** &rarr; enables multi-factor attribution (Fama-French
  style) and correlation/VIF analysis.
- **`factor_values`** &rarr; enables IC and alpha-decay analysis. This
  is the *value of the signal* at each date, not the return of a
  factor portfolio.
- **`positions`** &rarr; enables accurate turnover and cost analysis.
- **`strategy_variants`** &rarr; enables PBO via CSCV. Pass every
  variant of the strategy you tried; PBO catches "best-of-N" overfitting.
- **`n_trials_assumed`** &rarr; how many things you actually tested
  during research. Defaults to 100. Override with your real number.

Returns an `AutopsyReport`. Call `.save("path.html")` to render the
standalone HTML.

---

## The report

### `AutopsyReport`

```python
class AutopsyReport(pydantic.BaseModel):
    metadata: dict[str, Any]
    statistics: CoreStatistics
    regime: RegimeAnalysis | None
    attribution: FactorAttribution | None
    drawdowns: DrawdownAnalysis
    decay: DecayMetrics | None
    correlation: CorrelationAnalysis | None
    overfitting: OverfittingDiagnostics
    validation: ValidationResults
    robustness: RobustnessResults
    costs: CostAnalysis | None
    readiness: ProductionReadinessScore

    def save(self, path: str) -> str: ...
```

Pydantic v2 model with `arbitrary_types_allowed=True` (so pandas
objects are valid attribute values). Every field is typed; introspect
with `report.model_dump()` for serialization-friendly output.

### `ProductionReadinessScore` and `ReadinessVerdict`

```python
class ProductionReadinessScore(pydantic.BaseModel):
    overall_score: float                       # 0-100
    verdict: ReadinessVerdict                  # READY|CONDITIONAL|NOT_READY|REJECT
    components: list[ReadinessComponent]       # one per scoring dimension
    recommendation: str                        # plain-English action
    top_risks: list[str]                       # weakest 1-3 components

class ReadinessComponent(pydantic.BaseModel):
    name: str
    score: float       # 0-100
    weight: float      # normalized
    status: str        # "pass" | "warn" | "fail"
    value: str         # human-readable value
    detail: str        # what this measures
```

---

## Individual analysis types

These are returned as fields of `AutopsyReport` but can also be
imported and used directly.

### `CoreStatistics`

```python
class CoreStatistics(pydantic.BaseModel):
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    n_observations: int
    start_date: datetime
    end_date: datetime
    years: float
```

### `OverfittingDiagnostics`

```python
class OverfittingDiagnostics(pydantic.BaseModel):
    deflated_sharpe_ratio: float
    deflated_sharpe_pvalue: float
    probability_of_backtest_overfitting: float | None
    minimum_backtest_length_years: float
    actual_backtest_length_years: float
    minimum_length_satisfied: bool
```

### `ValidationResults`

```python
class ValidationResults(pydantic.BaseModel):
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    degradation_ratio: float
    walk_forward_sharpes: list[float]
    walk_forward_consistency: float
```

### `RobustnessResults`

```python
class RobustnessResults(pydantic.BaseModel):
    sharpe_confidence_interval: tuple[float, float]
    sharpe_std_bootstrap: float
    subsample_sharpes: list[float]
    subsample_positive_fraction: float
```

### `CostAnalysis`

```python
class CostAnalysis(pydantic.BaseModel):
    cost_sensitivity: dict[float, float]      # cost_bps -> resulting Sharpe
    breakeven_cost_bps: float | None
    annualized_turnover: float | None
```

### `RegimeAnalysis`

```python
class RegimeAnalysis(pydantic.BaseModel):
    method: RegimeMethod                      # RULE_BASED | HMM
    labels: pd.Series                         # date -> regime string
    summaries: list[RegimeSummary]
    transition_matrix: pd.DataFrame
    robustness_score: float                   # 0-1
```

### `DrawdownAnalysis` and `DrawdownEvent`

```python
class DrawdownAnalysis(pydantic.BaseModel):
    events: list[DrawdownEvent]
    n_drawdowns: int
    avg_depth: float
    avg_duration_days: float
    avg_recovery_days: float | None
    regime_concentration: dict[str, float]

class DrawdownEvent(pydantic.BaseModel):
    peak_date: datetime
    trough_date: datetime
    recovery_date: datetime | None
    depth: float
    duration_days: int
    recovery_days: int | None
    dominant_regime: RegimeLabel | None
    worst_days: list[tuple[datetime, float]]
```

### `FactorAttribution`

```python
class FactorAttribution(pydantic.BaseModel):
    factor_names: list[str]
    betas: dict[str, float]
    t_stats: dict[str, float]
    alpha_annualized: float
    alpha_t_stat: float
    r_squared: float
    rolling_betas: pd.DataFrame | None
    uniqueness_score: float                   # 0-1
```

### `DecayMetrics`

```python
class DecayMetrics(pydantic.BaseModel):
    ic_by_horizon: dict[int, float]
    ic_pvalue_by_horizon: dict[int, float]
    estimated_half_life_days: float | None
    rolling_ic: pd.Series | None
    ic_stability: float                       # 0-1
```

---

## Configuration

### `AutopsyConfig`

```python
@dataclass(frozen=True)
class AutopsyConfig:
    regime: RegimeConfig = RegimeConfig()
    overfitting: OverfittingConfig = OverfittingConfig()
    robustness: RobustnessConfig = RobustnessConfig()
    scoring: ScoringConfig = ScoringConfig()
    costs: CostConfig = CostConfig()
    rolling_window: int = 63                  # for rolling stats
    ic_horizons: tuple[int, ...] = (1, 5, 10, 20, 60)
```

All sub-configs are frozen dataclasses; mutate them with
`dataclasses.replace()`. The most useful overrides:

```python
from dataclasses import replace
from alpha_lens import AutopsyConfig, OverfittingConfig, ScoringConfig

config = AutopsyConfig(
    overfitting=OverfittingConfig(
        n_trials_assumed=500,         # be honest about how many things you tried
        cscv_n_partitions=12,          # fewer partitions = faster, more variance
    ),
    scoring=ScoringConfig(
        overfitting_weight=0.30,       # weight overfitting risk more heavily
        performance_weight=0.05,       # ... and raw Sharpe less
        ready_threshold=85.0,          # require a higher score for READY
    ),
)
report = autopsy(returns, config=config)
```

---

## Low-level analysis functions

If you only want one diagnostic, you can call the underlying functions
directly. They live in `alpha_lens.analysis`:

```python
from alpha_lens.analysis import (
    sharpe_ratio, sortino_ratio, max_drawdown,         # statistics
    rank_information_coefficient,                       # IC
    analyze_regimes,                                    # regime
    analyze_drawdowns,                                  # drawdowns
    attribute_returns,                                  # OLS attribution
    analyze_decay,                                      # IC decay + half-life
    analyze_overfitting,
    deflated_sharpe_ratio, probability_of_backtest_overfitting,
    minimum_backtest_length,
    analyze_validation, train_test_split, walk_forward_sharpes,
    analyze_robustness, bootstrap_sharpe,
    analyze_costs, find_breakeven_cost,
    compute_readiness_score,
)
```

All take typed inputs and return typed outputs. Type hints are
checked under `mypy --strict`.
