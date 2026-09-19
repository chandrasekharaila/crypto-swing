"""Explicit exception categories for swing setup detection."""

from crypto_analyzer.exceptions import AnalyzerError


class SetupError(AnalyzerError):
    """Base class for expected setup-detection failures."""


class SetupConfigurationError(SetupError):
    """Raised when setup configuration is unusable."""


class SetupInputError(SetupError):
    """Raised when the inputs cannot support setup detection."""
