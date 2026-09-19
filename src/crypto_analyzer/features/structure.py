"""Trailing market-structure features.

Breakout features compare the current close against a reference level formed
from candles strictly *before* the current one, so a candle can never define
its own breakout threshold.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.primitives import safe_divide
from crypto_analyzer.features.types import FeatureDefinition, FeatureGroup

if TYPE_CHECKING:
    from crypto_analyzer.features.registry import FeatureRegistry


def recent_high(frame: pd.DataFrame, window: int) -> pd.Series:
    """Highest high over the trailing ``window`` candles."""
    return frame["high"].rolling(window, min_periods=window).max()


def recent_low(frame: pd.DataFrame, window: int) -> pd.Series:
    """Lowest low over the trailing ``window`` candles."""
    return frame["low"].rolling(window, min_periods=window).min()


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
    reference = frame["high"].shift(1).rolling(window, min_periods=window).max()
    breached = (frame["close"] > reference).astype("float64")
    return breached.where(reference.notna())


def breakout_down(frame: pd.DataFrame, window: int) -> pd.Series:
    """Whether close falls below the lowest low of the preceding ``window`` candles."""
    reference = frame["low"].shift(1).rolling(window, min_periods=window).min()
    breached = (frame["close"] < reference).astype("float64")
    return breached.where(reference.notna())


def register_structure_features(
    registry: FeatureRegistry, settings: FeatureSettings
) -> None:
    """Register the configured market-structure features."""
    lookback = settings.structure_lookback
    parameters = (("window", lookback),)
    for name, description, compute in (
        (
            "structure_recent_high",
            f"Highest high over the trailing {lookback} candles.",
            recent_high,
        ),
        (
            "structure_recent_low",
            f"Lowest low over the trailing {lookback} candles.",
            recent_low,
        ),
        (
            "structure_breakout_up",
            (
                f"Whether close exceeds the highest high of the preceding "
                f"{lookback} candles."
            ),
            breakout_up,
        ),
        (
            "structure_breakout_down",
            (
                f"Whether close falls below the lowest low of the preceding "
                f"{lookback} candles."
            ),
            breakout_down,
        ),
    ):
        is_breakout = name.startswith("structure_breakout")
        registry.register(
            FeatureDefinition(
                name=f"{name}_{lookback}",
                group=FeatureGroup.STRUCTURE,
                description=description,
                lookback=lookback + 1 if is_breakout else lookback,
                compute=partial(compute, window=lookback),
                parameters=parameters,
            )
        )

    proximity = settings.sr_proximity_window
    registry.register(
        FeatureDefinition(
            name=f"structure_high_distance_{proximity}",
            group=FeatureGroup.STRUCTURE,
            description=(f"Close relative to the trailing {proximity}-candle high."),
            lookback=proximity,
            compute=partial(high_distance, window=proximity),
            parameters=(("window", proximity),),
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"structure_low_distance_{proximity}",
            group=FeatureGroup.STRUCTURE,
            description=(f"Close relative to the trailing {proximity}-candle low."),
            lookback=proximity,
            compute=partial(low_distance, window=proximity),
            parameters=(("window", proximity),),
        )
    )

    consolidation = settings.consolidation_window
    registry.register(
        FeatureDefinition(
            name=f"structure_consolidation_range_{consolidation}",
            group=FeatureGroup.STRUCTURE,
            description=(
                f"High-low span over {consolidation} candles relative to close."
            ),
            lookback=consolidation,
            compute=partial(consolidation_range, window=consolidation),
            parameters=(("window", consolidation),),
        )
    )
