"""Trailing volume features."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.primitives import safe_divide, signed_direction
from crypto_analyzer.features.types import FeatureDefinition, FeatureGroup

if TYPE_CHECKING:
    from crypto_analyzer.features.registry import FeatureRegistry


def volume_ma(frame: pd.DataFrame, window: int) -> pd.Series:
    """Simple moving average of volume over ``window`` candles."""
    return frame["volume"].rolling(window, min_periods=window).mean()


def volume_change(frame: pd.DataFrame, period: int) -> pd.Series:
    """Fractional change in volume over ``period`` candles."""
    return frame["volume"].pct_change(period)


def relative_volume(frame: pd.DataFrame, window: int) -> pd.Series:
    """Volume relative to its own moving average."""
    return safe_divide(frame["volume"], volume_ma(frame, window))


def signed_volume_mean(frame: pd.DataFrame, window: int) -> pd.Series:
    """Average volume over ``window`` candles, signed by candle direction."""
    signed = frame["volume"] * signed_direction(frame)
    return signed.rolling(window, min_periods=window).mean()


def volume_close_corr(frame: pd.DataFrame, window: int) -> pd.Series:
    """Rolling correlation between close and volume."""
    return frame["close"].rolling(window, min_periods=window).corr(frame["volume"])


def register_volume_features(
    registry: FeatureRegistry, settings: FeatureSettings
) -> None:
    """Register the configured volume measures."""
    for window in settings.volume_ma_windows:
        registry.register(
            FeatureDefinition(
                name=f"volume_ma_{window}",
                group=FeatureGroup.VOLUME,
                description=f"Simple moving average of volume over {window} candles.",
                lookback=window,
                compute=partial(volume_ma, window=window),
                parameters=(("window", window),),
            )
        )
        registry.register(
            FeatureDefinition(
                name=f"volume_signed_mean_{window}",
                group=FeatureGroup.VOLUME,
                description=(
                    f"Average volume over {window} candles, signed by candle direction."
                ),
                lookback=window,
                compute=partial(signed_volume_mean, window=window),
                parameters=(("window", window),),
            )
        )
        registry.register(
            FeatureDefinition(
                name=f"volume_close_corr_{window}",
                group=FeatureGroup.VOLUME,
                description=(
                    f"Rolling correlation between close and volume over {window} "
                    "candles."
                ),
                lookback=window,
                compute=partial(volume_close_corr, window=window),
                parameters=(("window", window),),
            )
        )

    registry.register(
        FeatureDefinition(
            name=f"volume_relative_{settings.relative_volume_window}",
            group=FeatureGroup.VOLUME,
            description=(
                f"Volume relative to its {settings.relative_volume_window}-candle "
                "moving average."
            ),
            lookback=settings.relative_volume_window,
            compute=partial(relative_volume, window=settings.relative_volume_window),
            parameters=(("window", settings.relative_volume_window),),
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"volume_change_{settings.volume_change_period}",
            group=FeatureGroup.VOLUME,
            description=(
                f"Fractional volume change over {settings.volume_change_period} "
                "candles."
            ),
            lookback=settings.volume_change_period + 1,
            compute=partial(volume_change, period=settings.volume_change_period),
            parameters=(("period", settings.volume_change_period),),
        )
    )
