"""Transparent, rule-based market-regime classification."""

from crypto_analyzer.regimes.config import TREND_RULE_COUNT, RegimeSettings
from crypto_analyzer.regimes.detector import DIAGNOSTIC_COLUMNS, RegimeDetector
from crypto_analyzer.regimes.exceptions import (
    RegimeConfigurationError,
    RegimeError,
    RegimeInputError,
)
from crypto_analyzer.regimes.types import (
    UNKNOWN_REGIME,
    RegimeEvidence,
    RegimeResult,
    TrendRegime,
    VolatilityRegime,
)

__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "TREND_RULE_COUNT",
    "UNKNOWN_REGIME",
    "RegimeConfigurationError",
    "RegimeDetector",
    "RegimeError",
    "RegimeEvidence",
    "RegimeInputError",
    "RegimeResult",
    "RegimeSettings",
    "TrendRegime",
    "VolatilityRegime",
]
