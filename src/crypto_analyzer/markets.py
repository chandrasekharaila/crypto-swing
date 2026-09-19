"""Binance interval durations and candle boundary arithmetic.

This is exchange metadata rather than user configuration, so it lives at the top
level: it is fixed by Binance, never overridden, and is needed by the data layer
before any settings exist. Keeping it here also keeps importing a duration from
pulling in the configuration package, which imports every component's settings.
"""

from datetime import UTC, datetime, timedelta

BINANCE_TIMEFRAME_DURATIONS: dict[str, timedelta] = {
    "1s": timedelta(seconds=1),
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "1w": timedelta(weeks=1),
}

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def is_timeframe_boundary(value: datetime, timeframe: str) -> bool:
    """Return whether ``value`` is an exact Binance UTC candle boundary."""
    if value.tzinfo is None or timeframe not in BINANCE_TIMEFRAME_DURATIONS:
        return False
    value_utc = value.astimezone(UTC)
    if timeframe == "1w":
        return (
            value_utc.weekday() == 0
            and value_utc.hour == 0
            and value_utc.minute == 0
            and value_utc.second == 0
            and value_utc.microsecond == 0
        )
    return (value_utc - _EPOCH) % BINANCE_TIMEFRAME_DURATIONS[timeframe] == timedelta(0)


def floor_to_timeframe(value: datetime, timeframe: str) -> datetime:
    """Floor an aware datetime to its current Binance UTC candle boundary."""
    if value.tzinfo is None:
        raise ValueError("value must be timezone-aware")
    duration = BINANCE_TIMEFRAME_DURATIONS[timeframe]
    value_utc = value.astimezone(UTC)
    if timeframe == "1w":
        start_of_day = value_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_day - timedelta(days=start_of_day.weekday())
    return value_utc - (value_utc - _EPOCH) % duration
