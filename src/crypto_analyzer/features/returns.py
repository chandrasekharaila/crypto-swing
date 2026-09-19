"""Trailing return features."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.types import FeatureDefinition, FeatureGroup

if TYPE_CHECKING:
    from crypto_analyzer.features.registry import FeatureRegistry


def simple_returns(frame: pd.DataFrame, window: int) -> pd.Series:
    """Fractional change in close over ``window`` candles."""
    return frame["close"].pct_change(window)


def log_returns(frame: pd.DataFrame, window: int) -> pd.Series:
    """Natural-log return of close over ``window`` candles."""
    ratio = frame["close"].div(frame["close"].shift(window))
    return ratio.map(math.log)


def register_returns_features(
    registry: FeatureRegistry, settings: FeatureSettings
) -> None:
    """Register the configured return windows."""
    for window in settings.return_windows:
        registry.register(
            FeatureDefinition(
                name=f"returns_simple_{window}",
                group=FeatureGroup.RETURNS,
                description=f"Fractional change in close over {window} candles.",
                lookback=window + 1,
                compute=partial(simple_returns, window=window),
                parameters=(("window", window),),
            )
        )
        registry.register(
            FeatureDefinition(
                name=f"returns_log_{window}",
                group=FeatureGroup.RETURNS,
                description=f"Natural-log close return over {window} candles.",
                lookback=window + 1,
                compute=partial(log_returns, window=window),
                parameters=(("window", window),),
            )
        )
