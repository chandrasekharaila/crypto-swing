"""Trailing volatility features."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.primitives import (
    fractional_change,
    moving_average,
    safe_divide,
    true_range,
    wilder_smoothing,
)
from crypto_analyzer.features.types import FeatureDefinition, FeatureGroup

if TYPE_CHECKING:
    from crypto_analyzer.features.registry import FeatureRegistry


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """Average true range over ``period`` candles."""
    return wilder_smoothing(true_range(frame), period)


def atr_pct(frame: pd.DataFrame, period: int) -> pd.Series:
    """Average true range as a fraction of the close price."""
    return safe_divide(atr(frame, period), frame["close"])


def return_std(frame: pd.DataFrame, window: int) -> pd.Series:
    """Standard deviation of one-candle returns over ``window`` candles."""
    return (
        fractional_change(frame["close"], 1).rolling(window, min_periods=window).std()
    )


def bollinger_middle(frame: pd.DataFrame, window: int) -> pd.Series:
    """Middle Bollinger band, the trailing simple moving average."""
    return moving_average(frame["close"], window)


def bollinger_deviation(frame: pd.DataFrame, window: int) -> pd.Series:
    """Trailing standard deviation of close, the Bollinger band half-width."""
    return frame["close"].rolling(window, min_periods=window).std()


def bollinger_upper(frame: pd.DataFrame, window: int, num_std: float) -> pd.Series:
    """Upper Bollinger band."""
    return bollinger_middle(frame, window) + num_std * bollinger_deviation(
        frame, window
    )


def bollinger_lower(frame: pd.DataFrame, window: int, num_std: float) -> pd.Series:
    """Lower Bollinger band."""
    return bollinger_middle(frame, window) - num_std * bollinger_deviation(
        frame, window
    )


def bollinger_width(frame: pd.DataFrame, window: int, num_std: float) -> pd.Series:
    """Band width relative to the middle band.

    The span between the bands is twice the deviation, so the bands never need to
    be materialized here. A flat window collapses the bands to zero width.
    """
    deviation = bollinger_deviation(frame, window)
    return safe_divide(
        2.0 * num_std * deviation,
        bollinger_middle(frame, window),
        zero_result=0.0,
    )


def bollinger_position(frame: pd.DataFrame, window: int, num_std: float) -> pd.Series:
    """Where close sits between the lower and upper Bollinger bands.

    Collapsed bands have no interior, so the neutral midpoint of 0.5 is returned
    rather than reporting the close at an extreme.
    """
    middle = bollinger_middle(frame, window)
    deviation = bollinger_deviation(frame, window)
    lower = middle - num_std * deviation
    upper = middle + num_std * deviation
    return safe_divide(frame["close"] - lower, upper - lower, zero_result=0.5)


def register_volatility_features(
    registry: FeatureRegistry, settings: FeatureSettings
) -> None:
    """Register the configured volatility measures and Bollinger bands."""
    registry.register(
        FeatureDefinition(
            name=f"volatility_atr_{settings.atr_period}",
            group=FeatureGroup.VOLATILITY,
            description=f"Average true range over {settings.atr_period} candles.",
            lookback=settings.atr_period,
            compute=partial(atr, period=settings.atr_period),
            parameters=(("period", settings.atr_period),),
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"volatility_atr_pct_{settings.atr_period}",
            group=FeatureGroup.VOLATILITY,
            description=(
                f"Average true range over {settings.atr_period} candles as a "
                "fraction of close."
            ),
            lookback=settings.atr_period,
            compute=partial(atr_pct, period=settings.atr_period),
            parameters=(("period", settings.atr_period),),
        )
    )

    for window in settings.rolling_std_windows:
        registry.register(
            FeatureDefinition(
                name=f"volatility_return_std_{window}",
                group=FeatureGroup.VOLATILITY,
                description=(
                    f"Standard deviation of one-candle returns over {window} candles."
                ),
                lookback=window + 1,
                compute=partial(return_std, window=window),
                parameters=(("window", window),),
            )
        )

    period = settings.bollinger_period
    num_std = settings.bollinger_num_std
    parameters = (("window", period), ("num_std", num_std))
    band_features = (
        ("volatility_bollinger_upper", "Upper Bollinger band.", bollinger_upper),
        ("volatility_bollinger_lower", "Lower Bollinger band.", bollinger_lower),
        (
            "volatility_bollinger_width",
            "Bollinger band width relative to the middle band; 0.0 when the "
            "window has no dispersion.",
            bollinger_width,
        ),
        (
            "volatility_bollinger_position",
            "Position of close between the lower and upper Bollinger bands; "
            "0.5 when the bands have collapsed.",
            bollinger_position,
        ),
    )
    for name, description, compute in band_features:
        registry.register(
            FeatureDefinition(
                name=f"{name}_{period}",
                group=FeatureGroup.VOLATILITY,
                description=(
                    f"{description} Computed over {period} candles at "
                    f"{num_std} standard deviations."
                ),
                lookback=period,
                compute=partial(compute, window=period, num_std=num_std),
                parameters=parameters,
            )
        )
