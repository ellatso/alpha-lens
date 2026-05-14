"""Core data types, configuration, and validation."""

from alpha_lens.core.config import (
    TRADING_DAYS_PER_YEAR,
    AutopsyConfig,
    CostConfig,
    OverfittingConfig,
    RegimeConfig,
    RobustnessConfig,
    ScoringConfig,
)
from alpha_lens.core.types import (
    AlphaData,
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
    RegimeSummary,
    RobustnessResults,
    ValidationResults,
)
from alpha_lens.core.validator import (
    InputValidationError,
    check_data_quality,
    validate_aligned,
    validate_returns,
)

__all__ = [
    # Types
    "AlphaData",
    "AutopsyReport",
    "CoreStatistics",
    "CorrelationAnalysis",
    "CostAnalysis",
    "DecayMetrics",
    "DrawdownAnalysis",
    "DrawdownEvent",
    "FactorAttribution",
    "OverfittingDiagnostics",
    "ProductionReadinessScore",
    "ReadinessComponent",
    "ReadinessVerdict",
    "RegimeAnalysis",
    "RegimeLabel",
    "RegimeMethod",
    "RegimeSummary",
    "RobustnessResults",
    "ValidationResults",
    # Config
    "AutopsyConfig",
    "CostConfig",
    "OverfittingConfig",
    "RegimeConfig",
    "RobustnessConfig",
    "ScoringConfig",
    "TRADING_DAYS_PER_YEAR",
    # Validation
    "InputValidationError",
    "check_data_quality",
    "validate_aligned",
    "validate_returns",
]
