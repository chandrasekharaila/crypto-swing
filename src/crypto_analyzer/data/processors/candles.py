"""Conversions between typed candles and canonical tabular data."""

from collections.abc import Iterable

import pandas as pd

from crypto_analyzer.data.models import Candle

CANONICAL_COLUMNS = (
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def candles_to_frame(candles: Iterable[Candle]) -> pd.DataFrame:
    """Convert typed candles to the canonical UTC OHLCV DataFrame schema."""
    records = [
        {
            "open_time": candle.open_time,
            "close_time": candle.close_time,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in candles
    ]
    frame = pd.DataFrame.from_records(records, columns=CANONICAL_COLUMNS)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    return frame
