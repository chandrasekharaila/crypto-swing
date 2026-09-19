"""Trailing momentum oscillators."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.primitives import (
    ema,
    ema_spread,
    fractional_change,
    moving_average,
    rolling_high,
    rolling_low,
    safe_divide,
    wilder_smoothing,
)
from crypto_analyzer.features.types import FeatureDefinition, FeatureGroup

if TYPE_CHECKING:
    from crypto_analyzer.features.registry import FeatureRegistry


def rsi(frame: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's Relative Strength Index over ``period`` candles.

    A window with no losses yields 100 and one with no gains yields 0, the limits
    of the ratio. A window with no movement at all has no relative strength, so
    the conventional neutral reading of 50 is returned rather than an undefined
    ratio.
    """
    change = frame["close"].diff()
    average_gain = wilder_smoothing(change.clip(lower=0.0), period)
    average_loss = wilder_smoothing((-change).clip(lower=0.0), period)
    index = 100.0 - (100.0 / (1.0 + average_gain / average_loss))
    moved = (average_gain != 0.0) | (average_loss != 0.0)
    return index.where(moved, 50.0)


def macd_line(frame: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Difference between the fast and slow exponential moving averages."""
    return ema_spread(frame["close"], fast, slow)


def macd_signal(frame: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
    """Exponential moving average of the MACD line."""
    return ema(macd_line(frame, fast, slow), signal)


def macd_histogram(frame: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
    """MACD line minus its signal line, using a single MACD computation."""
    line = macd_line(frame, fast, slow)
    return line - ema(line, signal)


def rate_of_change(frame: pd.DataFrame, window: int) -> pd.Series:
    """Percentage change in close over ``window`` candles."""
    return fractional_change(frame["close"], window) * 100.0


def stochastic_k(frame: pd.DataFrame, period: int) -> pd.Series:
    """Position of close within the trailing high-low range, as a percentage.

    A window with no high-low range has no position within it, so the
    conventional midpoint of 50 is returned rather than a fabricated extreme.
    """
    lowest = rolling_low(frame["low"], period)
    highest = rolling_high(frame["high"], period)
    position = safe_divide(frame["close"] - lowest, highest - lowest, zero_result=0.5)
    return position * 100.0


def stochastic_d(frame: pd.DataFrame, period: int, smooth: int) -> pd.Series:
    """Moving average of the stochastic oscillator."""
    return moving_average(stochastic_k(frame, period), smooth)


def register_momentum_features(
    registry: FeatureRegistry, settings: FeatureSettings
) -> None:
    """Register the configured momentum oscillators."""
    registry.register(
        FeatureDefinition(
            name=f"momentum_rsi_{settings.rsi_period}",
            group=FeatureGroup.MOMENTUM,
            description=(
                f"Wilder RSI over {settings.rsi_period} candles; 50 when the "
                "window has no movement."
            ),
            lookback=settings.rsi_period + 1,
            compute=partial(rsi, period=settings.rsi_period),
            parameters=(("period", settings.rsi_period),),
        )
    )

    registry.register(
        FeatureDefinition(
            name="momentum_macd",
            group=FeatureGroup.MOMENTUM,
            description=(
                f"MACD line: EMA {settings.macd_fast} minus EMA {settings.macd_slow}."
            ),
            lookback=settings.macd_slow,
            compute=partial(
                macd_line, fast=settings.macd_fast, slow=settings.macd_slow
            ),
            parameters=(("fast", settings.macd_fast), ("slow", settings.macd_slow)),
        )
    )
    registry.register(
        FeatureDefinition(
            name="momentum_macd_signal",
            group=FeatureGroup.MOMENTUM,
            description=(f"EMA {settings.macd_signal} of the MACD line."),
            lookback=settings.macd_slow + settings.macd_signal - 1,
            compute=partial(
                macd_signal,
                fast=settings.macd_fast,
                slow=settings.macd_slow,
                signal=settings.macd_signal,
            ),
            parameters=(
                ("fast", settings.macd_fast),
                ("slow", settings.macd_slow),
                ("signal", settings.macd_signal),
            ),
        )
    )
    registry.register(
        FeatureDefinition(
            name="momentum_macd_hist",
            group=FeatureGroup.MOMENTUM,
            description="MACD line minus its signal line.",
            lookback=settings.macd_slow + settings.macd_signal - 1,
            compute=partial(
                macd_histogram,
                fast=settings.macd_fast,
                slow=settings.macd_slow,
                signal=settings.macd_signal,
            ),
            parameters=(
                ("fast", settings.macd_fast),
                ("slow", settings.macd_slow),
                ("signal", settings.macd_signal),
            ),
        )
    )

    for window in settings.roc_windows:
        registry.register(
            FeatureDefinition(
                name=f"momentum_roc_{window}",
                group=FeatureGroup.MOMENTUM,
                description=f"Percentage close change over {window} candles.",
                lookback=window + 1,
                compute=partial(rate_of_change, window=window),
                parameters=(("window", window),),
            )
        )

    registry.register(
        FeatureDefinition(
            name=f"momentum_stoch_k_{settings.stoch_k_period}",
            group=FeatureGroup.MOMENTUM,
            description=(
                f"Close position within the trailing {settings.stoch_k_period}-candle "
                "high-low range, as a percentage; 50 when the window has no range."
            ),
            lookback=settings.stoch_k_period,
            compute=partial(stochastic_k, period=settings.stoch_k_period),
            parameters=(("period", settings.stoch_k_period),),
        )
    )
    registry.register(
        FeatureDefinition(
            name=f"momentum_stoch_d_{settings.stoch_d_period}",
            group=FeatureGroup.MOMENTUM,
            description=(
                f"{settings.stoch_d_period}-candle moving average of the "
                "stochastic oscillator."
            ),
            lookback=settings.stoch_k_period + settings.stoch_d_period - 1,
            compute=partial(
                stochastic_d,
                period=settings.stoch_k_period,
                smooth=settings.stoch_d_period,
            ),
            parameters=(
                ("period", settings.stoch_k_period),
                ("smooth", settings.stoch_d_period),
            ),
        )
    )
