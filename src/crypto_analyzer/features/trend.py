"""Trailing trend features built from moving averages."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.primitives import (
    ema,
    ema_spread,
    moving_average,
    safe_divide,
)
from crypto_analyzer.features.types import FeatureDefinition, FeatureGroup

if TYPE_CHECKING:
    from crypto_analyzer.features.registry import FeatureRegistry


def sma(frame: pd.DataFrame, window: int) -> pd.Series:
    """Simple moving average of close over ``window`` candles."""
    return moving_average(frame["close"], window)


def exponential_ma(frame: pd.DataFrame, span: int) -> pd.Series:
    """Exponential moving average of close over ``span`` candles."""
    return ema(frame["close"], span)


def ema_slope(frame: pd.DataFrame, span: int, slope_window: int) -> pd.Series:
    """Relative change of an EMA across ``slope_window`` candles."""
    values = ema(frame["close"], span)
    previous = values.shift(slope_window)
    return safe_divide(values - previous, slope_window * previous)


def price_vs_sma(frame: pd.DataFrame, window: int) -> pd.Series:
    """Close relative to its own simple moving average."""
    average = sma(frame, window)
    return safe_divide(frame["close"] - average, frame["close"])


def price_vs_ema(frame: pd.DataFrame, span: int) -> pd.Series:
    """Close relative to its own exponential moving average."""
    average = ema(frame["close"], span)
    return safe_divide(frame["close"] - average, frame["close"])


def ema_spread_pct(frame: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Fast EMA relative to the slow EMA, normalized by close."""
    return safe_divide(
        ema_spread(frame["close"], fast, slow),
        frame["close"],
    )


def register_trend_features(
    registry: FeatureRegistry, settings: FeatureSettings
) -> None:
    """Register the configured trend windows and the EMA spread."""
    for window in settings.sma_windows:
        registry.register(
            FeatureDefinition(
                name=f"trend_sma_{window}",
                group=FeatureGroup.TREND,
                description=f"Simple moving average of close over {window} candles.",
                lookback=window,
                compute=partial(sma, window=window),
                parameters=(("window", window),),
            )
        )

    for span in settings.ema_windows:
        registry.register(
            FeatureDefinition(
                name=f"trend_ema_{span}",
                group=FeatureGroup.TREND,
                description=f"Exponential moving average of close over {span} candles.",
                lookback=span,
                compute=partial(exponential_ma, span=span),
                parameters=(("span", span),),
            )
        )
        registry.register(
            FeatureDefinition(
                name=f"trend_ema_slope_{span}",
                group=FeatureGroup.TREND,
                description=(
                    f"Relative change of the {span}-candle EMA across "
                    f"{settings.ema_slope_window} candles."
                ),
                lookback=span + settings.ema_slope_window,
                compute=partial(
                    ema_slope, span=span, slope_window=settings.ema_slope_window
                ),
                parameters=(
                    ("span", span),
                    ("slope_window", settings.ema_slope_window),
                ),
            )
        )
        registry.register(
            FeatureDefinition(
                name=f"trend_price_vs_ema_{span}",
                group=FeatureGroup.TREND,
                description=(
                    f"Close relative to its {span}-candle exponential moving average."
                ),
                lookback=span,
                compute=partial(price_vs_ema, span=span),
                parameters=(("span", span),),
            )
        )

    for window in settings.ma_distance_windows:
        registry.register(
            FeatureDefinition(
                name=f"trend_price_vs_sma_{window}",
                group=FeatureGroup.TREND,
                description=(
                    f"Close relative to its {window}-candle simple moving average."
                ),
                lookback=window,
                compute=partial(price_vs_sma, window=window),
                parameters=(("window", window),),
            )
        )

    registry.register(
        FeatureDefinition(
            name="trend_ema_spread_pct",
            group=FeatureGroup.TREND,
            description=(
                f"Fast EMA ({settings.macd_fast}) relative to the slow EMA "
                f"({settings.macd_slow}), normalized by close."
            ),
            lookback=settings.macd_slow,
            compute=partial(
                ema_spread_pct, fast=settings.macd_fast, slow=settings.macd_slow
            ),
            parameters=(("fast", settings.macd_fast), ("slow", settings.macd_slow)),
        )
    )
