"""Trailing market-structure features.

Two rules keep these features honest about time:

* Breakout features compare the current close against a reference level formed
  from candles strictly *before* the current one, so a candle can never define
  its own breakout threshold.
* Swing labels compare the most recent window extreme against the extreme of the
  preceding, non-overlapping window. Both windows are entirely in the past, so a
  label describes the structure visible at that timestamp.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.primitives import (
    rolling_high,
    rolling_low,
    safe_divide,
)
from crypto_analyzer.features.types import FeatureDefinition, FeatureGroup

if TYPE_CHECKING:
    from crypto_analyzer.features.registry import FeatureRegistry


def recent_high(frame: pd.DataFrame, window: int) -> pd.Series:
    """Highest high over the trailing ``window`` candles."""
    return rolling_high(frame["high"], window)


def recent_low(frame: pd.DataFrame, window: int) -> pd.Series:
    """Lowest low over the trailing ``window`` candles."""
    return rolling_low(frame["low"], window)


def high_distance(frame: pd.DataFrame, window: int) -> pd.Series:
    """Close relative to the trailing high, negative when close is below it."""
    return safe_divide(frame["close"], recent_high(frame, window)).sub(1.0)


def low_distance(frame: pd.DataFrame, window: int) -> pd.Series:
    """Close relative to the trailing low, positive when close is above it."""
    return safe_divide(frame["close"], recent_low(frame, window)).sub(1.0)


def consolidation_range(frame: pd.DataFrame, window: int) -> pd.Series:
    """Trailing high-low span relative to close, a range-compression measure."""
    span = recent_high(frame, window) - recent_low(frame, window)
    return safe_divide(span, frame["close"])


def breakout_up(frame: pd.DataFrame, window: int) -> pd.Series:
    """Whether close exceeds the highest high of the preceding ``window`` candles."""
    reference = rolling_high(frame["high"].shift(1), window)
    breached = (frame["close"] > reference).astype("float64")
    return breached.where(reference.notna())


def breakout_down(frame: pd.DataFrame, window: int) -> pd.Series:
    """Whether close falls below the lowest low of the preceding ``window`` candles."""
    reference = rolling_low(frame["low"].shift(1), window)
    breached = (frame["close"] < reference).astype("float64")
    return breached.where(reference.notna())


def _swing_label(current: pd.Series, previous: pd.Series, *, rising: bool) -> pd.Series:
    comparison = current > previous if rising else current < previous
    return comparison.astype("float64").where(previous.notna())


def higher_high(frame: pd.DataFrame, window: int) -> pd.Series:
    """Whether the latest window high exceeds the preceding window's high."""
    current = recent_high(frame, window)
    return _swing_label(current, current.shift(window), rising=True)


def higher_low(frame: pd.DataFrame, window: int) -> pd.Series:
    """Whether the latest window low sits above the preceding window's low."""
    current = recent_low(frame, window)
    return _swing_label(current, current.shift(window), rising=True)


def lower_high(frame: pd.DataFrame, window: int) -> pd.Series:
    """Whether the latest window high sits below the preceding window's high."""
    current = recent_high(frame, window)
    return _swing_label(current, current.shift(window), rising=False)


def lower_low(frame: pd.DataFrame, window: int) -> pd.Series:
    """Whether the latest window low sits below the preceding window's low."""
    current = recent_low(frame, window)
    return _swing_label(current, current.shift(window), rising=False)


def register_structure_features(
    registry: FeatureRegistry, settings: FeatureSettings
) -> None:
    """Register the configured market-structure features."""
    window = settings.structure_lookback
    proximity = settings.sr_proximity_window
    consolidation = settings.consolidation_window
    window_parameters = (("window", window),)
    proximity_parameters = (("window", proximity),)
    consolidation_parameters = (("window", consolidation),)

    definitions = (
        FeatureDefinition(
            name=f"structure_recent_high_{window}",
            group=FeatureGroup.STRUCTURE,
            description=f"Highest high over the trailing {window} candles.",
            lookback=window,
            compute=partial(recent_high, window=window),
            parameters=window_parameters,
        ),
        FeatureDefinition(
            name=f"structure_recent_low_{window}",
            group=FeatureGroup.STRUCTURE,
            description=f"Lowest low over the trailing {window} candles.",
            lookback=window,
            compute=partial(recent_low, window=window),
            parameters=window_parameters,
        ),
        FeatureDefinition(
            name=f"structure_breakout_up_{window}",
            group=FeatureGroup.STRUCTURE,
            description=(
                f"Whether close exceeds the highest high of the preceding "
                f"{window} candles."
            ),
            lookback=window + 1,
            compute=partial(breakout_up, window=window),
            parameters=window_parameters,
        ),
        FeatureDefinition(
            name=f"structure_breakout_down_{window}",
            group=FeatureGroup.STRUCTURE,
            description=(
                f"Whether close falls below the lowest low of the preceding "
                f"{window} candles."
            ),
            lookback=window + 1,
            compute=partial(breakout_down, window=window),
            parameters=window_parameters,
        ),
        FeatureDefinition(
            name=f"structure_higher_high_{window}",
            group=FeatureGroup.STRUCTURE,
            description=(
                f"Whether the latest {window}-candle high exceeds the high of the "
                f"preceding {window} candles."
            ),
            lookback=window * 2,
            compute=partial(higher_high, window=window),
            parameters=window_parameters,
        ),
        FeatureDefinition(
            name=f"structure_higher_low_{window}",
            group=FeatureGroup.STRUCTURE,
            description=(
                f"Whether the latest {window}-candle low sits above the low of the "
                f"preceding {window} candles."
            ),
            lookback=window * 2,
            compute=partial(higher_low, window=window),
            parameters=window_parameters,
        ),
        FeatureDefinition(
            name=f"structure_lower_high_{window}",
            group=FeatureGroup.STRUCTURE,
            description=(
                f"Whether the latest {window}-candle high sits below the high of the "
                f"preceding {window} candles."
            ),
            lookback=window * 2,
            compute=partial(lower_high, window=window),
            parameters=window_parameters,
        ),
        FeatureDefinition(
            name=f"structure_lower_low_{window}",
            group=FeatureGroup.STRUCTURE,
            description=(
                f"Whether the latest {window}-candle low sits below the low of the "
                f"preceding {window} candles."
            ),
            lookback=window * 2,
            compute=partial(lower_low, window=window),
            parameters=window_parameters,
        ),
        FeatureDefinition(
            name=f"structure_high_distance_{proximity}",
            group=FeatureGroup.STRUCTURE,
            description=f"Close relative to the trailing {proximity}-candle high.",
            lookback=proximity,
            compute=partial(high_distance, window=proximity),
            parameters=proximity_parameters,
        ),
        FeatureDefinition(
            name=f"structure_low_distance_{proximity}",
            group=FeatureGroup.STRUCTURE,
            description=f"Close relative to the trailing {proximity}-candle low.",
            lookback=proximity,
            compute=partial(low_distance, window=proximity),
            parameters=proximity_parameters,
        ),
        FeatureDefinition(
            name=f"structure_consolidation_range_{consolidation}",
            group=FeatureGroup.STRUCTURE,
            description=(
                f"High-low span over {consolidation} candles relative to close."
            ),
            lookback=consolidation,
            compute=partial(consolidation_range, window=consolidation),
            parameters=consolidation_parameters,
        ),
    )
    for definition in definitions:
        registry.register(definition)
