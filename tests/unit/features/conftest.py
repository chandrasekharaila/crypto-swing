"""Shared deterministic fixtures for feature tests.

Synthetic frames are built arithmetically so expected feature values can be
derived by hand. No random generator is used anywhere in this suite.
"""

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import pandas as pd
import pytest

START = datetime(2024, 1, 1, tzinfo=UTC)
TIMEFRAME = "1h"
WAVE_ROWS = 300


def _frame_from_candles(
    candles: Sequence[tuple[float, float, float, float, float]],
) -> pd.DataFrame:
    open_time = pd.date_range(START, periods=len(candles), freq=TIMEFRAME)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "close_time": open_time
            + pd.Timedelta(TIMEFRAME)
            - pd.Timedelta(milliseconds=1),
            "open": [candle[0] for candle in candles],
            "high": [candle[1] for candle in candles],
            "low": [candle[2] for candle in candles],
            "close": [candle[3] for candle in candles],
            "volume": [candle[4] for candle in candles],
        }
    )


def _frame_from_close(close: Sequence[float], span: float = 2.0) -> pd.DataFrame:
    open_prices = [close[0], *close[:-1]]
    candles = [
        (
            open_price,
            max(open_price, value) + span,
            min(open_price, value) - span,
            value,
            100.0 + index,
        )
        for index, (open_price, value) in enumerate(
            zip(open_prices, close, strict=False)
        )
    ]
    return _frame_from_candles(candles)


def _leading_missing(series: pd.Series) -> int:
    valid = series.notna()
    if not valid.any():
        return len(series)
    return int(valid.to_numpy().argmax())


@pytest.fixture
def make_frame() -> Callable[..., pd.DataFrame]:
    """Build a frame from explicit (open, high, low, close, volume) rows."""
    return _frame_from_candles


@pytest.fixture
def make_close_frame() -> Callable[..., pd.DataFrame]:
    """Build a frame from a close-price series, with open at the prior close."""
    return _frame_from_close


@pytest.fixture
def leading_missing() -> Callable[[pd.Series], int]:
    """Count the leading missing values of a computed feature."""
    return _leading_missing


@pytest.fixture
def wave_frame() -> pd.DataFrame:
    """Return a long, non-degenerate frame suitable for warmup and causality sweeps."""
    close = [
        100.0 + 0.05 * index + 5.0 * math.sin(index / 6) for index in range(WAVE_ROWS)
    ]
    return _frame_from_close(close)
