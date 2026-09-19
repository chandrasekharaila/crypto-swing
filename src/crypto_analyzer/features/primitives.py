"""Shared causal series primitives used by more than one feature group.

Every helper here uses only the current and preceding candles. Nothing in this
module may shift backwards, center a window, or reduce across the whole series.

Each group module defines a thin frame-taking wrapper and delegates its
arithmetic here, so a given calculation exists in exactly one place.
"""

import math

import pandas as pd

_UNDEFINED = math.nan


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    zero_result: float = _UNDEFINED,
) -> pd.Series:
    """Divide elementwise, substituting ``zero_result`` where the denominator is zero.

    An undefined ratio -- either ``0 / 0`` or ``x / 0`` -- must not masquerade as a
    real measurement. Unless the caller supplies a documented neutral value the
    result is missing, so a degenerate window stays distinguishable from a genuine
    reading. A missing denominator propagates as missing.
    """
    return numerator.div(denominator).where(denominator != 0, zero_result)


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Return a trailing simple moving average over ``window`` candles."""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Return a trailing exponential moving average over ``span`` candles."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def ema_spread(series: pd.Series, fast: int, slow: int) -> pd.Series:
    """Return the difference between a fast and a slow exponential average."""
    return ema(series, fast) - ema(series, slow)


def fractional_change(series: pd.Series, window: int) -> pd.Series:
    """Return the fractional change of a series over ``window`` candles."""
    return series.pct_change(window)


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    """Return the highest value over the trailing ``window`` candles."""
    return series.rolling(window, min_periods=window).max()


def rolling_low(series: pd.Series, window: int) -> pd.Series:
    """Return the lowest value over the trailing ``window`` candles."""
    return series.rolling(window, min_periods=window).min()


def signed_direction(frame: pd.DataFrame) -> pd.Series:
    """Return -1.0, 0.0, or 1.0 for each candle's close-versus-open direction."""
    change = frame["close"] - frame["open"]
    return change.div(change.abs()).where(change != 0, 0.0)


def true_range(frame: pd.DataFrame) -> pd.Series:
    """Return the trailing true range of each candle.

    The first candle in a series has no predecessor, so only its own high-to-low
    span is available and that is what is returned.
    """
    previous_close = frame["close"].shift(1)
    components = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return components.max(axis=1)


def wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    """Apply Wilder's smoothing, an exponentially weighted mean with alpha 1/period."""
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
