"""alpha-lens: alpha factor autopsy and production-readiness validation.

A quantitative tool for stress-testing backtests before deployment.
Where alphalens reports IC and turnover, alpha-lens reports
*overfitting probability*, *out-of-sample degradation*, and a single
**Production Readiness Score**.

Quick start::

    import pandas as pd
    from alpha_lens import autopsy

    returns = pd.Series(...)  # daily returns of your strategy
    report = autopsy(returns)
    print(report.readiness.verdict)        # READY / CONDITIONAL / NOT_READY / REJECT
    print(report.readiness.overall_score)  # 0-100
    report.save("autopsy.html")            # standalone HTML
"""

from __future__ import annotations

from alpha_lens.core.config import (
    AutopsyConfig,
    CostConfig,
    OverfittingConfig,
    RegimeConfig,
    RobustnessConfig,
    ScoringConfig,
)
from alpha_lens.core.types import (
    AutopsyReport,
    CoreStatistics,
    CorrelationAnalysis,
    CostAnalysis,
    DecayMetrics,
    DrawdownAnalysis,
    DrawdownEvent,
    FactorAttribution,
    OverfittingDiagnostics,
    ProductionReadinessScore,
    ReadinessComponent,
    ReadinessVerdict,
    RegimeAnalysis,
    RegimeLabel,
    RegimeMethod,
    RobustnessResults,
    ValidationResults,
)
from alpha_lens.report.generator import autopsy

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "autopsy",
    "AutopsyReport",
    "ProductionReadinessScore",
    "ReadinessComponent",
    "ReadinessVerdict",
    "CoreStatistics",
    "RegimeAnalysis",
    "RegimeLabel",
    "RegimeMethod",
    "FactorAttribution",
    "DrawdownAnalysis",
    "DrawdownEvent",
    "DecayMetrics",
    "CorrelationAnalysis",
    "OverfittingDiagnostics",
    "ValidationResults",
    "RobustnessResults",
    "CostAnalysis",
    "AutopsyConfig",
    "OverfittingConfig",
    "RegimeConfig",
    "RobustnessConfig",
    "ScoringConfig",
    "CostConfig",
]
