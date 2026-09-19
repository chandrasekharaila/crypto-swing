"""Shared causal series primitives used by more than one feature group.

Every helper here uses only the current and preceding candles. Nothing in this
module may shift backwards, center a window, or reduce across the whole series.
"""

import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide elementwise, yielding 0.0 wherever the denominator is zero.

    Missing denominators propagate as missing rather than becoming zero, so
    warmup positions are preserved.
    """
    return numerator.div(denominator).where(denominator != 0, 0.0)


def signed_direction(frame: pd.DataFrame) -> pd.Series:
    """Return -1.0, 0.0, or 1.0 for each candle's close-versus-open direction."""
    change = frame["close"] - frame["open"]
    return change.div(change.abs()).where(change != 0, 0.0)


def true_range(frame: pd.DataFrame) -> pd.Series:
    """Return the trailing true range of each candle.

    The final candle in a series has no predecessor, so only its own high-to-low
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


def ema(series: pd.Series, span: int) -> pd.Series:
    """Return a trailing exponential moving average over ``span`` candles."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()
