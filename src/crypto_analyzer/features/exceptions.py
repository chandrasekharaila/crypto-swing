"""Explicit exception categories for Phase 2 feature engineering."""


class FeatureEngineeringError(Exception):
    """Base class for expected feature-engineering failures."""


class FeatureConfigurationError(FeatureEngineeringError):
    """Raised when feature configuration or registration is unusable."""


class FeatureInputError(FeatureEngineeringError):
    """Raised when a source dataset cannot be used for feature generation."""


class FeatureComputationError(FeatureEngineeringError):
    """Raised when a feature does not produce an aligned, usable result."""
