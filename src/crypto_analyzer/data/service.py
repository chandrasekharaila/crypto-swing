"""Orchestration for the Phase 1 collection-to-storage pipeline."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from crypto_analyzer.config import AppSettings
from crypto_analyzer.data.exceptions import ConfigurationError
from crypto_analyzer.data.models import Candle
from crypto_analyzer.data.processors import candles_to_frame
from crypto_analyzer.data.storage import ParquetMarketDataStore, TimeRange
from crypto_analyzer.data.validators import OHLCVValidator

logger = logging.getLogger(__name__)


class HistoricalCollector(Protocol):
    """Collector interface required by the data pipeline."""

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> Sequence[Candle]: ...


@dataclass(frozen=True, slots=True)
class DataUpdateResult:
    """Summary of one idempotent market-data update."""

    symbol: str
    timeframe: str
    requested_ranges: tuple[TimeRange, ...]
    unresolved_ranges: tuple[TimeRange, ...]
    downloaded_candles: int
    stored_candles: int
    path: Path

    @property
    def cache_hit(self) -> bool:
        return not self.requested_ranges

    @property
    def is_complete(self) -> bool:
        """Whether every candle in the requested range is now cached."""
        return not self.unresolved_ranges


class MarketDataPipeline:
    """Coordinate cache inspection, collection, validation, and raw storage."""

    def __init__(
        self,
        settings: AppSettings,
        collector: HistoricalCollector,
        store: ParquetMarketDataStore,
        *,
        validator: OHLCVValidator | None = None,
    ) -> None:
        self._settings = settings
        self._collector = collector
        self._store = store
        self._validator = validator or OHLCVValidator()

    def update(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> DataUpdateResult:
        """Download only missing ranges and merge validated closed candles."""
        self._validate_market(symbol, timeframe)
        missing_ranges = tuple(
            self._store.missing_ranges(symbol, timeframe, start, end)
        )
        path = self._store.path_for_raw(symbol, timeframe)

        if not missing_ranges:
            stored = self._store.load_raw(symbol, timeframe)
            logger.info(
                "Market data already current; no API request needed",
                extra={"symbol": symbol, "timeframe": timeframe},
            )
            return DataUpdateResult(symbol, timeframe, (), (), 0, len(stored), path)

        downloaded_count = 0
        logger.info(
            "Updating %s %s across %d missing range(s)",
            symbol,
            timeframe,
            len(missing_ranges),
        )
        for missing in missing_ranges:
            logger.info(
                "Downloading %s %s from %s to %s",
                symbol,
                timeframe,
                missing.start.isoformat(),
                missing.end.isoformat(),
            )
            candles = self._collector.fetch_historical(
                symbol, timeframe, missing.start, missing.end
            )
            if not candles:
                logger.warning(
                    "No candles returned for %s %s from %s to %s",
                    symbol,
                    timeframe,
                    missing.start.isoformat(),
                    missing.end.isoformat(),
                )
                continue

            frame = candles_to_frame(candles)
            report = self._validator.validate(frame, timeframe)
            for issue in report.warnings:
                logger.warning("Validation warning [%s]: %s", issue.code, issue.message)
            report.raise_for_errors()
            self._store.update_raw(symbol, timeframe, frame)
            downloaded_count += len(frame)

        stored = self._store.load_raw(symbol, timeframe)
        unresolved_ranges = tuple(
            self._store.missing_ranges(symbol, timeframe, start, end)
        )
        if unresolved_ranges:
            logger.error(
                "Update incomplete for %s %s: %d range(s) remain unresolved",
                symbol,
                timeframe,
                len(unresolved_ranges),
            )
        logger.info(
            "Update complete for %s %s: downloaded=%d stored=%d path=%s",
            symbol,
            timeframe,
            downloaded_count,
            len(stored),
            path,
        )
        return DataUpdateResult(
            symbol,
            timeframe,
            missing_ranges,
            unresolved_ranges,
            downloaded_count,
            len(stored),
            path,
        )

    def _validate_market(self, symbol: str, timeframe: str) -> None:
        if symbol not in self._settings.symbols:
            raise ConfigurationError(f"symbol is not configured: {symbol}")
        if timeframe not in self._settings.timeframes:
            raise ConfigurationError(f"timeframe is not configured: {timeframe}")
