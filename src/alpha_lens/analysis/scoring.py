"""Production Readiness Score: the headline number a PM looks at first.

Aggregates every other diagnostic into a single 0-100 score with a
verdict (READY / CONDITIONAL / NOT_READY / REJECT) and a list of the
top risks. Each component contributes a 0-100 sub-score weighted
according to :class:`ScoringConfig`.

Design philosophy:

* The score must be *interpretable*. Every sub-score should map to a
  specific intuition like "the strategy is statistically distinguishable
  from chance" or "OOS performance is close to IS performance."

* The weights reflect what *actually* predicts production failure.
  Overfitting and OOS degradation matter most because they directly
  answer "will this work in production." Raw performance matters less
  because anyone can produce a high-Sharpe backtest by trying enough
  variants.

* Scoring is opinionated. A defensible default is more useful than a
  configurable abstraction nobody can interpret. Users can override
  weights via :class:`ScoringConfig`.
"""

from __future__ import annotations

import math

from alpha_lens.core.config import ScoringConfig
from alpha_lens.core.types import (
    CoreStatistics,
    CostAnalysis,
    DecayMetrics,
    OverfittingDiagnostics,
    ProductionReadinessScore,
    ReadinessComponent,
    ReadinessVerdict,
    RegimeAnalysis,
    RobustnessResults,
    ValidationResults,
)

__all__ = ["compute_readiness_score"]


def compute_readiness_score(
    *,
    statistics: CoreStatistics,
    overfitting: OverfittingDiagnostics,
    validation: ValidationResults,
    robustness: RobustnessResults,
    regime: RegimeAnalysis | None = None,
    decay: DecayMetrics | None = None,
    costs: CostAnalysis | None = None,
    config: ScoringConfig | None = None,
) -> ProductionReadinessScore:
    """Build the readiness score from analysis outputs.

    Args:
        statistics: Core performance metrics.
        overfitting: Overfitting diagnostics.
        validation: OOS validation results.
        robustness: Bootstrap/subsample robustness.
        regime: Optional regime analysis.
        decay: Optional decay metrics.
        costs: Optional cost analysis.
        config: Scoring weights and thresholds.

    Returns:
        :class:`ProductionReadinessScore` with overall score, verdict,
        component breakdown, recommendation, and top risks.
    """
    config = config or ScoringConfig()

    components: list[ReadinessComponent] = []

    # 1. Overfitting score.
    components.append(_score_overfitting(overfitting, config.overfitting_weight))

    # 2. Validation score.
    components.append(_score_validation(validation, config.validation_weight))

    # 3. Robustness score.
    components.append(_score_robustness(robustness, config.robustness_weight))

    # 4. Regime score (if available).
    if regime is not None:
        components.append(_score_regime(regime, config.regime_weight))

    # 5. Decay score (if available).
    if decay is not None:
        components.append(_score_decay(decay, config.decay_weight))

    # 6. Cost score (if available).
    if costs is not None:
        components.append(_score_costs(costs, config.cost_weight))

    # 7. Performance score (always available).
    components.append(_score_performance(statistics, config.performance_weight))

    # Re-normalize weights if some components weren't computed.
    total_weight = sum(c.weight for c in components)
    if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
        for c in components:
            c.weight = c.weight / total_weight

    # Overall score: weighted average.
    overall = sum(c.score * c.weight for c in components)

    verdict = _verdict(overall, config)
    top_risks = _identify_top_risks(components)
    recommendation = _generate_recommendation(verdict, top_risks)

    return ProductionReadinessScore(
        overall_score=round(overall, 1),
        verdict=verdict,
        components=components,
        recommendation=recommendation,
        top_risks=top_risks,
    )


# ----------------------------------------------------------------------------
# Component scorers
# ----------------------------------------------------------------------------


def _score_overfitting(diag: OverfittingDiagnostics, weight: float) -> ReadinessComponent:
    """Score overfitting risk.

    Combines DSR p-value, PBO (if available), and MinBTL.
    """
    # DSR contribution: p-value < 0.05 is a pass.
    dsr_score = _sigmoid_score(
        1.0 - diag.deflated_sharpe_pvalue,
        midpoint=0.95,
        slope=12.0,
    )

    # PBO contribution: < 0.5 is good, > 0.5 is bad.
    if diag.probability_of_backtest_overfitting is not None:
        pbo_score = _sigmoid_score(
            1.0 - diag.probability_of_backtest_overfitting,
            midpoint=0.5,
            slope=8.0,
        )
        combined = (dsr_score + pbo_score) / 2.0
    else:
        combined = dsr_score

    # MinBTL: heavy penalty if not satisfied.
    if not diag.minimum_length_satisfied:
        combined *= 0.6

    status = _bucket_status(combined)
    value = f"DSR p={diag.deflated_sharpe_pvalue:.2f}"
    if diag.probability_of_backtest_overfitting is not None:
        value += f", PBO={diag.probability_of_backtest_overfitting:.2f}"
    if not diag.minimum_length_satisfied:
        value += " (insufficient data)"

    return ReadinessComponent(
        name="Overfitting Risk",
        score=combined,
        weight=weight,
        status=status,
        value=value,
        detail=(
            "Probability that the observed Sharpe survives statistical correction for "
            "multiple testing, non-normality, and (where available) backtest selection bias."
        ),
    )


def _score_validation(val: ValidationResults, weight: float) -> ReadinessComponent:
    """Score out-of-sample validation.

    Combines OOS Sharpe level, degradation ratio, and walk-forward consistency.
    """
    # OOS Sharpe level (cap at SR=2.5 → 100).
    oos_level = _linear_score(val.out_of_sample_sharpe, min_v=0.0, max_v=2.0)

    # Degradation ratio: 1.0 = perfect retention. 0.5 = halved. <0 = sign flip.
    if val.in_sample_sharpe <= 0:
        # If IS Sharpe is non-positive, degradation isn't meaningful.
        # Use OOS level only.
        degradation_score = oos_level
    else:
        degradation_score = _sigmoid_score(
            val.degradation_ratio,
            midpoint=0.7,
            slope=5.0,
        )

    # Walk-forward consistency: fraction of positive-Sharpe windows.
    wf_score = _linear_score(val.walk_forward_consistency, min_v=0.5, max_v=1.0)

    combined = 0.4 * oos_level + 0.4 * degradation_score + 0.2 * wf_score
    status = _bucket_status(combined)

    return ReadinessComponent(
        name="Out-of-Sample Validation",
        score=combined,
        weight=weight,
        status=status,
        value=(
            f"IS Sharpe={val.in_sample_sharpe:.2f}, OOS={val.out_of_sample_sharpe:.2f}, "
            f"degradation={val.degradation_ratio:.2f}"
        ),
        detail=(
            "Compares performance on the first 70% of data (in-sample) against the held-out "
            "last 30% (out-of-sample). Walk-forward sub-windows test temporal consistency."
        ),
    )


def _score_robustness(rob: RobustnessResults, weight: float) -> ReadinessComponent:
    """Score robustness from bootstrap CI and subsample stability."""
    lo, hi = rob.sharpe_confidence_interval

    # The bootstrap CI should be far from zero.
    if lo > 0.5:
        ci_score = 100.0
    elif lo > 0:
        ci_score = 70.0 + (lo / 0.5) * 30.0
    elif hi > 0:
        # CI straddles zero. Score by fraction of CI above zero.
        ci_score = max(0.0, (hi / (hi - lo)) * 50.0)
    else:
        ci_score = 0.0

    sub_score = _linear_score(rob.subsample_positive_fraction, min_v=0.5, max_v=1.0)
    combined = 0.6 * ci_score + 0.4 * sub_score
    status = _bucket_status(combined)

    return ReadinessComponent(
        name="Robustness",
        score=combined,
        weight=weight,
        status=status,
        value=(
            f"Sharpe 95% CI = [{lo:.2f}, {hi:.2f}], "
            f"positive in {rob.subsample_positive_fraction:.0%} of subsamples"
        ),
        detail=(
            "Bootstrap confidence interval for the Sharpe ratio combined with the "
            "fraction of time-subsamples where the strategy made money."
        ),
    )


def _score_regime(regime: RegimeAnalysis, weight: float) -> ReadinessComponent:
    """Score consistency of performance across market regimes."""
    score = regime.robustness_score * 100.0
    status = _bucket_status(score)

    # Build a short description of regime performance.
    parts = []
    for s in regime.summaries:
        parts.append(f"{s.regime.value}: SR={s.sharpe_ratio:+.2f}")
    value = " | ".join(parts[:4])

    return ReadinessComponent(
        name="Regime Robustness",
        score=score,
        weight=weight,
        status=status,
        value=value,
        detail=(
            "Whether the strategy works across multiple market regimes, weighted by "
            "time spent in each regime and penalized for high Sharpe variance."
        ),
    )


def _score_decay(decay: DecayMetrics, weight: float) -> ReadinessComponent:
    """Score alpha decay quality.

    Higher score if:
        * IC at horizon 1 is meaningfully different from zero.
        * Rolling IC is stable (same sign as full-sample IC).
        * Half-life is long enough that rebalancing makes sense.
    """
    ic_1d = decay.ic_by_horizon.get(1, 0.0)
    # IC scoring: |IC| of 0.02 is publishable, 0.05 is great, 0.10 is rare.
    ic_score = _linear_score(abs(ic_1d), min_v=0.01, max_v=0.06)

    stability_score = _linear_score(decay.ic_stability, min_v=0.5, max_v=0.9)

    # Half-life: very short half-life (< 5 days) is hard to trade.
    if decay.estimated_half_life_days is None:
        hl_score = 50.0
    elif decay.estimated_half_life_days < 2:
        hl_score = 20.0
    elif decay.estimated_half_life_days < 5:
        hl_score = 50.0
    else:
        hl_score = 100.0

    combined = 0.5 * ic_score + 0.3 * stability_score + 0.2 * hl_score
    status = _bucket_status(combined)

    hl_str = (
        f"{decay.estimated_half_life_days:.0f}d"
        if decay.estimated_half_life_days is not None
        else "n/a"
    )
    return ReadinessComponent(
        name="Alpha Decay",
        score=combined,
        weight=weight,
        status=status,
        value=f"IC(1d)={ic_1d:+.3f}, stability={decay.ic_stability:.0%}, half-life={hl_str}",
        detail=(
            "Strength and stability of the factor's predictive power, and how fast it "
            "decays. A short half-life means more frequent rebalancing and higher costs."
        ),
    )


def _score_costs(costs: CostAnalysis, weight: float) -> ReadinessComponent:
    """Score robustness to transaction costs."""
    if costs.breakeven_cost_bps is None:
        # Either always profitable (good) or already unprofitable (bad).
        # Use the Sharpe at cost=0 as a proxy.
        zero_sr = costs.cost_sensitivity.get(0.0, 0.0)
        if zero_sr > 0:
            score = 100.0
            value = ">1000 bps breakeven"
        else:
            score = 0.0
            value = "Unprofitable at zero cost"
    else:
        # 10bps is a typical retail cost. 20bps is high-frequency-style execution.
        # 50bps is a comfortable margin for most institutional strategies.
        score = _linear_score(costs.breakeven_cost_bps, min_v=5.0, max_v=50.0)
        value = f"breakeven at {costs.breakeven_cost_bps:.1f} bps"
    if costs.annualized_turnover is not None:
        value += f", turnover={costs.annualized_turnover:.1f}x/yr"

    status = _bucket_status(score)
    return ReadinessComponent(
        name="Cost Sensitivity",
        score=score,
        weight=weight,
        status=status,
        value=value,
        detail=(
            "How much transaction cost the strategy can absorb before becoming "
            "unprofitable. Calculated by sweeping cost levels and bisecting for zero Sharpe."
        ),
    )


def _score_performance(stats: CoreStatistics, weight: float) -> ReadinessComponent:
    """Score raw (full-sample) performance.

    Worth the least of any component — anyone can produce a high-Sharpe
    backtest by trying enough variants. The other components stress-test
    this number.
    """
    sr_score = _linear_score(stats.sharpe_ratio, min_v=0.0, max_v=2.0)
    # Calmar matters too — Sharpe alone hides drawdown risk.
    calmar_score = _linear_score(stats.calmar_ratio, min_v=0.0, max_v=2.0)
    combined = 0.6 * sr_score + 0.4 * calmar_score
    status = _bucket_status(combined)

    return ReadinessComponent(
        name="Performance",
        score=combined,
        weight=weight,
        status=status,
        value=(
            f"Sharpe={stats.sharpe_ratio:.2f}, Calmar={stats.calmar_ratio:.2f}, "
            f"max DD={stats.max_drawdown:.1%}"
        ),
        detail=(
            "Raw backtest performance. The lowest-weighted component because raw "
            "performance is the easiest metric to overfit."
        ),
    )


# ----------------------------------------------------------------------------
# Score-to-grade conversions
# ----------------------------------------------------------------------------


def _verdict(score: float, config: ScoringConfig) -> ReadinessVerdict:
    """Map an overall score to a verdict."""
    if score >= config.ready_threshold:
        return ReadinessVerdict.READY
    if score >= config.conditional_threshold:
        return ReadinessVerdict.CONDITIONAL
    if score >= config.not_ready_threshold:
        return ReadinessVerdict.NOT_READY
    return ReadinessVerdict.REJECT


def _bucket_status(score: float) -> str:
    """Map a 0-100 sub-score to {pass, warn, fail}."""
    if score >= 70:
        return "pass"
    if score >= 40:
        return "warn"
    return "fail"


def _linear_score(value: float, *, min_v: float, max_v: float) -> float:
    """Linear mapping: min_v → 0, max_v → 100, with clipping."""
    if max_v == min_v:
        return 50.0
    raw = (value - min_v) / (max_v - min_v)
    return float(max(0.0, min(100.0, raw * 100.0)))


def _sigmoid_score(value: float, *, midpoint: float, slope: float) -> float:
    """Sigmoid mapping: smoother than linear, with midpoint=50."""
    z = (value - midpoint) * slope
    try:
        sig = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        sig = 0.0 if z < 0 else 1.0
    return float(sig * 100.0)


# ----------------------------------------------------------------------------
# Recommendation generator
# ----------------------------------------------------------------------------


def _identify_top_risks(components: list[ReadinessComponent]) -> list[str]:
    """Pull out the 1-3 weakest components as risks."""
    failing = sorted(
        [c for c in components if c.status in ("fail", "warn")],
        key=lambda c: c.score,
    )
    return [f"{c.name}: {c.value}" for c in failing[:3]]


def _generate_recommendation(
    verdict: ReadinessVerdict,
    risks: list[str],
) -> str:
    """Generate a plain-language recommendation from the verdict and risks."""
    base = {
        ReadinessVerdict.READY: (
            "The strategy passes the major statistical and out-of-sample checks. "
            "Recommended action: deploy with normal monitoring."
        ),
        ReadinessVerdict.CONDITIONAL: (
            "The strategy shows genuine signal but has notable concerns. "
            "Recommended action: paper trade for 3+ months while addressing the "
            "issues below, then revisit for live deployment."
        ),
        ReadinessVerdict.NOT_READY: (
            "Substantial concerns about robustness or overfitting. "
            "Recommended action: iterate on the strategy or expand the validation "
            "set before considering deployment."
        ),
        ReadinessVerdict.REJECT: (
            "The strategy fails core statistical and out-of-sample tests. "
            "Recommended action: do not deploy. The backtest is likely an artifact "
            "of selection bias or overfitting."
        ),
    }[verdict]

    if risks:
        risks_str = "; ".join(risks[:2])
        base += f" Top concerns: {risks_str}."

    return base
