"""Transparent, rule-based swing setup detection."""

from crypto_analyzer.setups.config import SetupSettings
from crypto_analyzer.setups.detector import (
    SetupDetector,
    breakout_evidence,
    continuation_evidence,
    reversion_evidence,
)
from crypto_analyzer.setups.exceptions import (
    SetupConfigurationError,
    SetupError,
    SetupInputError,
)
from crypto_analyzer.setups.types import (
    Direction,
    PriceContext,
    SetupScan,
    SetupSignal,
    SetupType,
)

__all__ = [
    "Direction",
    "PriceContext",
    "SetupConfigurationError",
    "SetupDetector",
    "SetupError",
    "SetupInputError",
    "SetupScan",
    "SetupSettings",
    "SetupSignal",
    "SetupType",
    "breakout_evidence",
    "continuation_evidence",
    "reversion_evidence",
]
