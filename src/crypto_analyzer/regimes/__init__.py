"""Transparent, rule-based market-regime classification."""

from crypto_analyzer.evidence import Evidence
from crypto_analyzer.regimes.config import TREND_RULE_COUNT, RegimeSettings
from crypto_analyzer.regimes.detector import DIAGNOSTIC_COLUMNS, RegimeDetector
from crypto_analyzer.regimes.exceptions import (
    RegimeConfigurationError,
    RegimeError,
    RegimeInputError,
)
from crypto_analyzer.regimes.types import (
    UNKNOWN_REGIME,
    RegimeResult,
    TrendRegime,
    VolatilityRegime,
)

__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "TREND_RULE_COUNT",
    "UNKNOWN_REGIME",
    "Evidence",
    "RegimeConfigurationError",
    "RegimeDetector",
    "RegimeError",
    "RegimeInputError",
    "RegimeResult",
    "RegimeSettings",
    "TrendRegime",
    "VolatilityRegime",
]
