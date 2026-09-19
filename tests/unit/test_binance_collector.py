"""Unit tests for the Binance public OHLCV collector."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
import pytest

from crypto_analyzer.config import AppSettings
from crypto_analyzer.data.collectors import BinanceOHLCVCollector
from crypto_analyzer.data.exceptions import (
    ConfigurationError,
    DataCollectionError,
    DataValidationError,
)

NOW = datetime(2024, 1, 2, tzinfo=UTC)
HOUR_MS = 60 * 60 * 1_000


def _milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _kline(open_time: datetime, *, close_offset_ms: int = HOUR_MS - 1) -> list[Any]:
    open_ms = _milliseconds(open_time)
    return [
        open_ms,
        "100.0",
        "110.0",
        "90.0",
        "105.0",
        "12.5",
        open_ms + close_offset_ms,
        "0",
        1,
        "0",
        "0",
        "0",
    ]


def _collector(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    settings: AppSettings | None = None,
    sleeps: list[float] | None = None,
) -> BinanceOHLCVCollector:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return BinanceOHLCVCollector(
        settings or AppSettings(),
        client=client,
        sleep=(sleeps if sleeps is not None else []).append,
        clock=lambda: NOW,
    )


def test_fetches_public_klines_and_normalizes_values() -> None:
    start = NOW - timedelta(hours=2)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/klines"
        assert request.url.params["symbol"] == "BTCUSDT"
        assert request.url.params["interval"] == "1h"
        assert "apiKey" not in request.url.params
        return httpx.Response(200, json=[_kline(start)], request=request)

    candles = _collector(handler).fetch_historical("BTC/USDT", "1h", start, NOW)

    assert len(candles) == 1
    assert candles[0].open_time == start
    assert candles[0].open_time.tzinfo is UTC
    assert candles[0].close == Decimal("105.0")
    assert candles[0].volume == Decimal("12.5")


def test_paginates_until_requested_range_is_complete() -> None:
    start = NOW - timedelta(hours=3)
    rows = [_kline(start + timedelta(hours=offset)) for offset in range(3)]
    requested_starts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_start = int(request.url.params["startTime"])
        requested_starts.append(requested_start)
        available = [row for row in rows if int(row[0]) >= requested_start]
        return httpx.Response(200, json=available[:2], request=request)

    settings = AppSettings(request_limit=2)
    candles = _collector(handler, settings=settings).fetch_historical(
        "BTC/USDT", "1h", start, NOW
    )

    assert [candle.open_time for candle in candles] == [
        start,
        start + timedelta(hours=1),
        start + timedelta(hours=2),
    ]
    assert requested_starts == [
        _milliseconds(start),
        _milliseconds(start + timedelta(hours=1)) + 1,
    ]


def test_excludes_candle_that_is_not_closed_at_clock_snapshot() -> None:
    closed_start = NOW - timedelta(hours=1)
    current_start = NOW

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[_kline(closed_start), _kline(current_start)],
            request=request,
        )

    candles = _collector(handler).fetch_historical(
        "BTC/USDT", "1h", closed_start, NOW + timedelta(hours=2)
    )

    assert [candle.open_time for candle in candles] == [closed_start]


def test_retries_network_failure_with_exponential_backoff() -> None:
    start = NOW - timedelta(hours=1)
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("temporary failure", request=request)
        return httpx.Response(200, json=[_kline(start)], request=request)

    settings = AppSettings(max_retries=2, retry_backoff_seconds=0.5)
    candles = _collector(handler, settings=settings, sleeps=sleeps).fetch_historical(
        "BTC/USDT", "1h", start, NOW
    )

    assert len(candles) == 1
    assert attempts == 3
    assert sleeps == [0.5, 1.0]


def test_honors_retry_after_on_rate_limit_response() -> None:
    start = NOW - timedelta(hours=1)
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429, headers={"Retry-After": "3"}, json={}, request=request
            )
        return httpx.Response(200, json=[_kline(start)], request=request)

    candles = _collector(handler, sleeps=sleeps).fetch_historical(
        "BTC/USDT", "1h", start, NOW
    )

    assert len(candles) == 1
    assert sleeps == [3.0]


def test_does_not_retry_non_transient_api_error() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"code": -1121}, request=request)

    with pytest.raises(DataCollectionError, match="HTTP 400"):
        _collector(handler).fetch_historical(
            "BTC/USDT", "1h", NOW - timedelta(hours=1), NOW
        )

    assert attempts == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"not": "a list"},
        [[_milliseconds(NOW), "100"]],
        [[_milliseconds(NOW), "invalid", "110", "90", "105", "1", 0]],
    ],
)
def test_rejects_malformed_api_payload(payload: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    with pytest.raises(DataValidationError):
        _collector(handler).fetch_historical(
            "BTC/USDT", "1h", NOW - timedelta(hours=1), NOW
        )


def test_rejects_unconfigured_market() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("request should not be made")

    with pytest.raises(ConfigurationError, match="symbol is not configured"):
        _collector(handler).fetch_historical(
            "ADA/USDT", "1h", NOW - timedelta(hours=1), NOW
        )
