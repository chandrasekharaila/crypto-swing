"""Tests for conversion from typed candles to canonical tabular data."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from crypto_analyzer.data.models import Candle
from crypto_analyzer.data.processors import candles_to_frame


def test_converts_candles_to_utc_frame() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candle = Candle(
        open_time=start,
        close_time=start + timedelta(hours=1) - timedelta(milliseconds=1),
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal("105"),
        volume=Decimal("12.5"),
    )

    frame = candles_to_frame([candle])

    assert list(frame.columns) == [
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert str(frame["open_time"].dtype) == "datetime64[ns, UTC]"
    assert frame.loc[0, "close"] == Decimal("105")
