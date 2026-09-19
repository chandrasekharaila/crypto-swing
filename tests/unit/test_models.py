"""Tests for shared Phase 1 data models."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from crypto_analyzer.data.models import Candle


def test_candle_is_immutable() -> None:
    open_time = datetime(2024, 1, 1, tzinfo=UTC)
    candle = Candle(
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal("105"),
        volume=Decimal("12.5"),
    )

    assert candle.open_time == open_time
    assert candle.close == Decimal("105")
