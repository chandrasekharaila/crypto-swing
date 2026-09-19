"""Explicit exception categories for regime detection."""

from crypto_analyzer.exceptions import AnalyzerError


class RegimeError(AnalyzerError):
    """Base class for expected regime-detection failures."""


class RegimeConfigurationError(RegimeError):
    """Raised when regime configuration is unusable."""


class RegimeInputError(RegimeError):
    """Raised when a feature set cannot support regime detection."""
