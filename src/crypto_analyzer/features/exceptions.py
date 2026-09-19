"""Explicit exception categories for feature engineering."""

from crypto_analyzer.exceptions import AnalyzerError


class FeatureEngineeringError(AnalyzerError):
    """Base class for expected feature-engineering failures."""


class FeatureConfigurationError(FeatureEngineeringError):
    """Raised when feature configuration or registration is unusable."""


class FeatureInputError(FeatureEngineeringError):
    """Raised when a source dataset cannot be used for feature generation."""


class FeatureComputationError(FeatureEngineeringError):
    """Raised when a feature does not produce an aligned, usable result."""
