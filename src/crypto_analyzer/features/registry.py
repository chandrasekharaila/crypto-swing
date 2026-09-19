"""Deterministic registration of feature definitions."""

from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.exceptions import FeatureConfigurationError
from crypto_analyzer.features.momentum import register_momentum_features
from crypto_analyzer.features.price import register_price_features
from crypto_analyzer.features.returns import register_returns_features
from crypto_analyzer.features.structure import register_structure_features
from crypto_analyzer.features.trend import register_trend_features
from crypto_analyzer.features.types import (
    FeatureDefinition,
    FeatureGroup,
    FeatureMetadata,
)
from crypto_analyzer.features.volatility import register_volatility_features
from crypto_analyzer.features.volume import register_volume_features


class FeatureRegistry:
    """Ordered, duplicate-free collection of feature definitions."""

    def __init__(self) -> None:
        self._definitions: dict[str, FeatureDefinition] = {}

    def register(self, definition: FeatureDefinition) -> None:
        """Add one definition, rejecting a duplicate column name."""
        if definition.name in self._definitions:
            raise FeatureConfigurationError(
                f"feature already registered: {definition.name}"
            )
        self._definitions[definition.name] = definition

    @property
    def definitions(self) -> tuple[FeatureDefinition, ...]:
        """Return definitions in registration order."""
        return tuple(self._definitions.values())

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered feature names in registration order."""
        return tuple(self._definitions)

    @property
    def metadata(self) -> tuple[FeatureMetadata, ...]:
        """Return the metadata of every registered feature."""
        return tuple(definition.metadata for definition in self._definitions.values())

    def for_group(self, group: FeatureGroup) -> tuple[FeatureDefinition, ...]:
        """Return the definitions belonging to one feature group."""
        return tuple(
            definition
            for definition in self._definitions.values()
            if definition.group is group
        )

    @property
    def max_lookback(self) -> int:
        """Return the candles required before every feature has a valid value."""
        return max(
            (definition.lookback for definition in self._definitions.values()),
            default=0,
        )

    def __len__(self) -> int:
        return len(self._definitions)


def build_default_registry(settings: FeatureSettings) -> FeatureRegistry:
    """Build the registry implied by the configured groups and windows."""
    registry = FeatureRegistry()
    if FeatureGroup.PRICE in settings.enabled_groups:
        register_price_features(registry)
    if FeatureGroup.RETURNS in settings.enabled_groups:
        register_returns_features(registry, settings)
    if FeatureGroup.TREND in settings.enabled_groups:
        register_trend_features(registry, settings)
    if FeatureGroup.MOMENTUM in settings.enabled_groups:
        register_momentum_features(registry, settings)
    if FeatureGroup.VOLATILITY in settings.enabled_groups:
        register_volatility_features(registry, settings)
    if FeatureGroup.VOLUME in settings.enabled_groups:
        register_volume_features(registry, settings)
    if FeatureGroup.STRUCTURE in settings.enabled_groups:
        register_structure_features(registry, settings)
    return registry
