"""Deterministic, point-in-time feature engineering."""

from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.engine import FeatureEngine
from crypto_analyzer.features.exceptions import (
    FeatureComputationError,
    FeatureConfigurationError,
    FeatureEngineeringError,
    FeatureInputError,
)
from crypto_analyzer.features.registry import FeatureRegistry, build_default_registry
from crypto_analyzer.features.types import (
    FeatureDefinition,
    FeatureGroup,
    FeatureMetadata,
    FeatureSet,
)

__all__ = [
    "FeatureComputationError",
    "FeatureConfigurationError",
    "FeatureDefinition",
    "FeatureEngine",
    "FeatureEngineeringError",
    "FeatureGroup",
    "FeatureInputError",
    "FeatureMetadata",
    "FeatureRegistry",
    "FeatureSet",
    "FeatureSettings",
    "build_default_registry",
]
