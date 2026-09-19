"""Binance public REST collector for historical OHLCV candles."""

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from itertools import pairwise
import logging
import time
from typing import Any

import httpx

from crypto_analyzer.config import AppSettings
from crypto_analyzer.data.exceptions import (
    ConfigurationError,
    DataCollectionError,
    DataValidationError,
)
from crypto_analyzer.data.models import Candle

logger = logging.getLogger(__name__)

Sleep = Callable[[float], None]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BinanceOHLCVCollector:
    """Collect closed historical candles from Binance's unauthenticated API."""

    _KLINES_PATH = "/api/v3/klines"
    _RETRYABLE_STATUS_CODES = frozenset({418, 429, 500, 502, 503, 504})

    def __init__(
        self,
        settings: AppSettings,
        *,
        client: httpx.Client | None = None,
        sleep: Sleep = time.sleep,
        clock: Clock = _utc_now,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._clock = clock
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=settings.request_timeout_seconds,
            headers={"User-Agent": "crypto-swing-analyzer/0.1"},
        )

    def __enter__(self) -> "BinanceOHLCVCollector":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the internally created HTTP client."""
        if self._owns_client:
            self._client.close()

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> list[Candle]:
        """Fetch closed candles in the half-open range ``[start_time, end_time)``.

        If ``end_time`` is omitted, the collector uses one UTC clock snapshot as
        the exclusive upper bound. Results are ordered by opening timestamp.
        """
        self._validate_request(symbol, timeframe, start_time, end_time)

        observed_at = self._clock()
        if observed_at.tzinfo is None:
            raise ConfigurationError("collector clock must return a timezone-aware datetime")
        observed_at = observed_at.astimezone(UTC)
        requested_end = (end_time or observed_at).astimezone(UTC)
        effective_end = min(requested_end, observed_at)

        start_utc = start_time.astimezone(UTC)
        if start_utc >= effective_end:
            return []

        cursor_ms = _to_milliseconds(start_utc)
        end_ms = _to_milliseconds(effective_end)
        candles_by_open_time: dict[datetime, Candle] = {}

        while cursor_ms < end_ms:
            rows = self._request_page(
                symbol=symbol,
                timeframe=timeframe,
                start_ms=cursor_ms,
                end_ms=end_ms - 1,
            )
            if not rows:
                break

            page = [self._parse_kline(row) for row in rows]
            self._validate_page_order(page)

            for candle in page:
                if not (start_utc <= candle.open_time < requested_end):
                    continue
                if candle.close_time >= observed_at:
                    continue
                existing = candles_by_open_time.get(candle.open_time)
                if existing is not None and existing != candle:
                    raise DataValidationError(
                        f"conflicting candles at {candle.open_time.isoformat()}"
                    )
                candles_by_open_time[candle.open_time] = candle

            last_open_ms = _to_milliseconds(page[-1].open_time)
            next_cursor_ms = last_open_ms + 1
            if next_cursor_ms <= cursor_ms:
                raise DataValidationError("Binance pagination did not advance")
            cursor_ms = next_cursor_ms

            if len(rows) < self._settings.request_limit:
                break

        return sorted(candles_by_open_time.values(), key=lambda candle: candle.open_time)

    def _validate_request(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime | None,
    ) -> None:
        if symbol not in self._settings.symbols:
            raise ConfigurationError(f"symbol is not configured: {symbol}")
        if timeframe not in self._settings.timeframes:
            raise ConfigurationError(f"timeframe is not configured: {timeframe}")
        if start_time.tzinfo is None:
            raise ConfigurationError("start_time must be timezone-aware")
        if end_time is not None:
            if end_time.tzinfo is None:
                raise ConfigurationError("end_time must be timezone-aware")
            if start_time >= end_time:
                raise ConfigurationError("start_time must be earlier than end_time")

    def _request_page(
        self,
        *,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
    ) -> list[Any]:
        url = f"{str(self._settings.binance_base_url).rstrip('/')}{self._KLINES_PATH}"
        params: dict[str, str | int] = {
            "symbol": symbol.replace("/", "").upper(),
            "interval": timeframe,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": self._settings.request_limit,
        }

        for attempt in range(self._settings.max_retries + 1):
            try:
                response = self._client.get(url, params=params)
            except httpx.RequestError as error:
                if attempt >= self._settings.max_retries:
                    raise DataCollectionError(
                        f"Binance request failed after {attempt + 1} attempts"
                    ) from error
                self._wait_before_retry(attempt, None)
                continue

            if response.status_code in self._RETRYABLE_STATUS_CODES:
                if attempt >= self._settings.max_retries:
                    raise DataCollectionError(
                        f"Binance returned retryable HTTP {response.status_code} "
                        f"after {attempt + 1} attempts"
                    )
                self._wait_before_retry(attempt, response.headers.get("Retry-After"))
                continue

            if response.is_error:
                detail = response.text[:500]
                raise DataCollectionError(
                    f"Binance returned HTTP {response.status_code}: {detail}"
                )

            try:
                payload = response.json()
            except ValueError as error:
                raise DataValidationError("Binance returned invalid JSON") from error
            if not isinstance(payload, list):
                raise DataValidationError("Binance kline response must be a list")
            return payload

        raise DataCollectionError("Binance request exhausted retry attempts")

    def _wait_before_retry(self, attempt: int, retry_after: str | None) -> None:
        delay = self._settings.retry_backoff_seconds * (2**attempt)
        if retry_after is not None:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=UTC)
                    delay = max(
                        delay,
                        (
                            retry_at.astimezone(UTC)
                            - self._clock().astimezone(UTC)
                        ).total_seconds(),
                    )
                except (TypeError, ValueError, OverflowError):
                    logger.warning(
                        "Ignoring invalid Binance Retry-After header: %s", retry_after
                    )
        delay = max(0.0, delay)
        logger.warning("Retrying Binance request in %.2f seconds", delay)
        self._sleep(delay)

    @staticmethod
    def _parse_kline(row: Any) -> Candle:
        if not isinstance(row, list) or len(row) < 7:
            raise DataValidationError("Binance kline row must contain at least 7 fields")
        try:
            open_ms = int(row[0])
            close_ms = int(row[6])
            prices = tuple(Decimal(str(value)) for value in row[1:5])
            volume = Decimal(str(row[5]))
        except (TypeError, ValueError, InvalidOperation) as error:
            raise DataValidationError("Binance kline row contains invalid values") from error

        open_price, high, low, close = prices
        numeric_values = (*prices, volume)
        if not all(value.is_finite() for value in numeric_values):
            raise DataValidationError("Binance kline values must be finite")
        if min(prices) <= 0 or volume < 0:
            raise DataValidationError("Binance prices must be positive and volume non-negative")
        if high < max(open_price, close, low) or low > min(open_price, close, high):
            raise DataValidationError("Binance kline violates OHLC price bounds")
        if close_ms < open_ms:
            raise DataValidationError("Binance kline closes before it opens")

        return Candle(
            open_time=_from_milliseconds(open_ms),
            close_time=_from_milliseconds(close_ms),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )

    @staticmethod
    def _validate_page_order(page: list[Candle]) -> None:
        if any(
            current.open_time <= previous.open_time
            for previous, current in pairwise(page)
        ):
            raise DataValidationError("Binance klines must have increasing open times")


def _to_milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _from_milliseconds(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000, tz=UTC)
