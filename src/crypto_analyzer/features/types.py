"""Core value types for deterministic, point-in-time feature generation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from crypto_analyzer.features.exceptions import FeatureConfigurationError


class FeatureGroup(StrEnum):
    """Logical family that a generated feature belongs to."""

    PRICE = "price"
    RETURNS = "returns"
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    STRUCTURE = "structure"


FeatureCompute = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Self-describing record of one generated feature column."""

    name: str
    group: FeatureGroup
    description: str
    lookback: int
    parameters: tuple[tuple[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize for experiment records and structured research input."""
        return {
            "name": self.name,
            "group": self.group.value,
            "description": self.description,
            "lookback": self.lookback,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """A registered feature: how to compute it and everything needed to describe it."""

    name: str
    group: FeatureGroup
    description: str
    lookback: int
    compute: FeatureCompute = field(compare=False, repr=False)
    parameters: tuple[tuple[str, object], ...] = ()

    @property
    def metadata(self) -> FeatureMetadata:
        """Return the serializable description of this feature."""
        return FeatureMetadata(
            name=self.name,
            group=self.group,
            description=self.description,
            lookback=self.lookback,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Feature columns computed for one symbol and timeframe.

    ``frame`` is keyed by ``open_time`` and every row's values are drawn from
    candles up to and including that candle. Those values therefore become
    available at the candle's ``close_time``, which ``available_at`` carries
    explicitly so no downstream phase has to reconstruct the boundary.
    """

    symbol: str
    timeframe: str
    frame: pd.DataFrame
    metadata: tuple[FeatureMetadata, ...]
    available_at: pd.Series

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return generated feature column names in deterministic order."""
        return tuple(item.name for item in self.metadata)

    def metadata_for(self, name: str) -> FeatureMetadata:
        """Return the metadata describing one generated feature column."""
        for item in self.metadata:
            if item.name == name:
                return item
        raise FeatureConfigurationError(f"unknown feature: {name}")
