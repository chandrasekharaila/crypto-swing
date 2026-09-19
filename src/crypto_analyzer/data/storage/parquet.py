"""Atomic, collector-independent Parquet storage for market data."""

import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from crypto_analyzer.data.exceptions import DataStorageError, DataValidationError
from crypto_analyzer.data.processors.candles import CANONICAL_COLUMNS
from crypto_analyzer.data.validators import OHLCVValidator, timeframe_duration
from crypto_analyzer.markets import is_timeframe_boundary

_MARKET_COMPONENT = re.compile(r"^[A-Za-z0-9]+$")
_TIMEFRAME_COMPONENT = re.compile(r"^[1-9][0-9]*[smhdwM]$")

logger = logging.getLogger(__name__)


class MarketDataLayer(StrEnum):
    """Physical data layers that must remain isolated."""

    RAW = "raw"
    PROCESSED = "processed"


@dataclass(frozen=True, slots=True)
class TimeRange:
    """A half-open UTC range requiring collection."""

    start: datetime
    end: datetime


class ParquetMarketDataStore:
    """Persist canonical OHLCV frames in deterministic Parquet locations."""

    def __init__(self, data_directory: Path, *, exchange: str = "binance") -> None:
        if not _MARKET_COMPONENT.fullmatch(exchange):
            raise DataStorageError("exchange must contain only letters and numbers")
        self._data_directory = data_directory
        self._exchange = exchange.lower()
        self._validator = OHLCVValidator()

    def path_for(self, layer: MarketDataLayer, symbol: str, timeframe: str) -> Path:
        """Return a stable path without creating files or directories."""
        base, quote = self._parse_symbol(symbol)
        if not _TIMEFRAME_COMPONENT.fullmatch(timeframe):
            raise DataStorageError(f"invalid timeframe path component: {timeframe}")
        market = f"{base.upper()}-{quote.upper()}"
        return (
            self._data_directory
            / layer.value
            / self._exchange
            / market
            / f"{timeframe}.parquet"
        )

    def load_raw(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load existing raw data, or an empty canonical frame if absent."""
        return self._load(MarketDataLayer.RAW, symbol, timeframe)

    def path_for_raw(self, symbol: str, timeframe: str) -> Path:
        """Return the deterministic raw dataset path."""
        return self.path_for(MarketDataLayer.RAW, symbol, timeframe)

    def load_processed(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load existing processed data, or an empty canonical frame if absent."""
        return self._load(MarketDataLayer.PROCESSED, symbol, timeframe)

    def update_raw(
        self, symbol: str, timeframe: str, downloaded: pd.DataFrame
    ) -> pd.DataFrame:
        """Atomically merge downloaded candles while preserving stored history."""
        self._validate_frame(downloaded, timeframe, context="downloaded raw data")
        path = self.path_for(MarketDataLayer.RAW, symbol, timeframe)
        existing = self._load(MarketDataLayer.RAW, symbol, timeframe)

        if downloaded.empty:
            return existing
        if path.exists() and tuple(existing.columns) != tuple(downloaded.columns):
            raise DataStorageError("stored and downloaded raw schemas do not match")

        merged = self._merge_without_conflicts(existing, downloaded)
        self._validate_frame(merged, timeframe, context="merged raw data")
        self._atomic_write(path, merged)
        return merged

    def write_processed(
        self, symbol: str, timeframe: str, processed: pd.DataFrame
    ) -> None:
        """Atomically write processed data without touching its raw source."""
        self._validate_frame(processed, timeframe, context="processed data")
        path = self.path_for(MarketDataLayer.PROCESSED, symbol, timeframe)
        self._atomic_write(path, processed)

    def missing_ranges(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[TimeRange]:
        """Find absent candle ranges so a collector can avoid redundant requests."""
        if start.tzinfo is None or end.tzinfo is None:
            raise DataStorageError("cache range timestamps must be timezone-aware")
        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        if start_utc >= end_utc:
            raise DataStorageError("cache range start must be earlier than end")

        duration = timeframe_duration(timeframe)
        if not is_timeframe_boundary(start_utc, timeframe) or not is_timeframe_boundary(
            end_utc, timeframe
        ):
            raise DataStorageError(
                f"cache range must align to {timeframe} Binance candle boundaries"
            )

        existing = self.load_raw(symbol, timeframe)
        available = {
            timestamp.to_pydatetime()
            for timestamp in existing["open_time"]
            if start_utc <= timestamp.to_pydatetime() < end_utc
        }

        missing: list[TimeRange] = []
        missing_start: datetime | None = None
        expected = start_utc
        while expected < end_utc:
            if expected not in available and missing_start is None:
                missing_start = expected
            elif expected in available and missing_start is not None:
                missing.append(TimeRange(missing_start, expected))
                missing_start = None
            expected += duration
        if missing_start is not None:
            missing.append(TimeRange(missing_start, end_utc))
        return missing

    def _load(
        self, layer: MarketDataLayer, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        path = self.path_for(layer, symbol, timeframe)
        if not path.exists():
            return _empty_frame()
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
        except Exception as error:
            raise DataStorageError(
                f"failed to read Parquet data from {path}"
            ) from error
        self._validate_frame(frame, timeframe, context=f"stored {layer.value} data")
        return frame

    def _validate_frame(
        self, frame: pd.DataFrame, timeframe: str, *, context: str
    ) -> None:
        report = self._validator.validate(frame, timeframe)
        for issue in report.warnings:
            logger.warning(
                "Validation warning for %s [%s]: %s", context, issue.code, issue.message
            )
        try:
            report.raise_for_errors()
        except DataValidationError as error:
            raise DataStorageError(f"invalid {context}: {error}") from error

    @staticmethod
    def _merge_without_conflicts(
        existing: pd.DataFrame, downloaded: pd.DataFrame
    ) -> pd.DataFrame:
        if existing.empty:
            return downloaded.copy(deep=True)

        combined = pd.concat([existing, downloaded], ignore_index=True)
        duplicate_rows = combined[combined["open_time"].duplicated(keep=False)]
        for open_time, group in duplicate_rows.groupby("open_time", sort=False):
            reference = group.iloc[0]
            is_conflicting = any(
                not reference.equals(group.iloc[position])
                for position in range(1, len(group))
            )
            if is_conflicting:
                raise DataStorageError(f"conflicting raw candles at {open_time}")

        merged = combined.drop_duplicates(subset=["open_time"], keep="first")
        return merged.sort_values("open_time", kind="stable").reset_index(drop=True)

    @staticmethod
    def _atomic_write(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.stem}-",
                suffix=".parquet",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            frame.to_parquet(temporary_path, engine="pyarrow", index=False)
            os.replace(temporary_path, path)
        except Exception as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise DataStorageError(f"failed to write Parquet data to {path}") from error

    @staticmethod
    def _parse_symbol(symbol: str) -> tuple[str, str]:
        parts = symbol.split("/")
        if len(parts) != 2 or not all(
            _MARKET_COMPONENT.fullmatch(part) for part in parts
        ):
            raise DataStorageError(f"symbol must use safe BASE/QUOTE format: {symbol}")
        return parts[0], parts[1]


def _empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=CANONICAL_COLUMNS)
    frame["open_time"] = pd.Series(dtype="datetime64[ns, UTC]")
    frame["close_time"] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame
