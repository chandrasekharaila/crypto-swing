"""Non-mutating validation for canonical OHLCV datasets."""

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from numbers import Real
from typing import Any

import pandas as pd

from crypto_analyzer.data.exceptions import ConfigurationError
from crypto_analyzer.data.validators.report import (
    IssueSeverity,
    ValidationIssue,
    ValidationReport,
)
from crypto_analyzer.markets import BINANCE_TIMEFRAME_DURATIONS

REQUIRED_COLUMNS = frozenset(
    {"open_time", "close_time", "open", "high", "low", "close", "volume"}
)
PRICE_COLUMNS = ("open", "high", "low", "close")
NUMERIC_COLUMNS = (*PRICE_COLUMNS, "volume")


def timeframe_duration(timeframe: str) -> timedelta:
    """Return the fixed duration represented by a supported Binance timeframe."""
    try:
        return BINANCE_TIMEFRAME_DURATIONS[timeframe]
    except KeyError as error:
        raise ConfigurationError(
            f"timeframe cannot be validated as a fixed interval: {timeframe}"
        ) from error


class OHLCVValidator:
    """Validate canonical candle frames without sorting or repairing them."""

    def validate(
        self,
        frame: pd.DataFrame,
        timeframe: str,
        *,
        as_of: datetime | None = None,
    ) -> ValidationReport:
        """Return all detectable errors and warnings for ``frame``.

        Internal gaps are warnings because the validator cannot know whether an
        exchange outage or an illiquid market caused them. No values are changed.
        """
        duration = timeframe_duration(timeframe)
        validation_time = as_of or datetime.now(UTC)
        if validation_time.tzinfo is None:
            raise ConfigurationError("as_of must be timezone-aware")
        validation_time = validation_time.astimezone(UTC)

        issues: list[ValidationIssue] = []
        missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        unexpected_columns = sorted(set(frame.columns).difference(REQUIRED_COLUMNS))
        if missing_columns:
            issues.append(
                self._error(
                    "missing_columns",
                    f"required columns are missing: {', '.join(missing_columns)}",
                )
            )
            return ValidationReport(tuple(issues))
        if unexpected_columns:
            unexpected = ", ".join(unexpected_columns)
            issues.append(
                self._error(
                    "unexpected_columns",
                    f"canonical dataset contains unexpected columns: {unexpected}",
                )
            )

        if frame.empty:
            issues.append(self._warning("empty_dataset", "dataset contains no candles"))
            return ValidationReport(tuple(issues))

        issues.extend(self._validate_missing_values(frame))
        timestamps_valid, timestamp_issues = self._validate_timestamp_types(frame)
        issues.extend(timestamp_issues)
        numeric_valid, numeric_issues = self._validate_numeric_types(frame)
        issues.extend(numeric_issues)

        if timestamps_valid:
            issues.extend(self._validate_timestamps(frame, duration, validation_time))
        if numeric_valid:
            issues.extend(self._validate_prices_and_volume(frame))

        return ValidationReport(tuple(issues))

    def _validate_missing_values(self, frame: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for column in sorted(REQUIRED_COLUMNS):
            missing = frame[column].isna()
            if missing.any():
                rows = self._row_labels(frame, missing)
                issues.append(
                    self._error(
                        "missing_values",
                        f"column '{column}' contains missing values",
                        rows,
                    )
                )
        return issues

    def _validate_timestamp_types(
        self, frame: pd.DataFrame
    ) -> tuple[bool, list[ValidationIssue]]:
        issues: list[ValidationIssue] = []
        for column in ("open_time", "close_time"):
            dtype = frame[column].dtype
            if not isinstance(dtype, pd.DatetimeTZDtype):
                issues.append(
                    self._error(
                        "invalid_timestamp_type",
                        f"column '{column}' must be timezone-aware datetime64",
                    )
                )
            elif str(dtype.tz) != "UTC":
                issues.append(
                    self._error(
                        "invalid_timestamp_timezone",
                        f"column '{column}' must use UTC, found {dtype.tz}",
                    )
                )
        return not issues and not frame[
            ["open_time", "close_time"]
        ].isna().any().any(), issues

    def _validate_numeric_types(
        self, frame: pd.DataFrame
    ) -> tuple[bool, list[ValidationIssue]]:
        issues: list[ValidationIssue] = []
        all_valid = True
        for column in NUMERIC_COLUMNS:
            invalid = ~frame[column].map(_is_finite_number)
            invalid &= ~frame[column].isna()
            if invalid.any():
                all_valid = False
                issues.append(
                    self._error(
                        "invalid_numeric_type",
                        f"column '{column}' must contain finite numeric values",
                        self._row_labels(frame, invalid),
                    )
                )
            if frame[column].isna().any():
                all_valid = False
        return all_valid, issues

    def _validate_timestamps(
        self,
        frame: pd.DataFrame,
        duration: timedelta,
        as_of: datetime,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        open_times = frame["open_time"]
        close_times = frame["close_time"]

        duplicates = open_times.duplicated(keep=False)
        if duplicates.any():
            issues.append(
                self._error(
                    "duplicate_candles",
                    "multiple candles have the same open_time",
                    self._row_labels(frame, duplicates),
                )
            )

        ordered = open_times.is_monotonic_increasing
        if not ordered:
            issues.append(
                self._error(
                    "non_chronological_order",
                    "candles must be ordered by increasing open_time",
                )
            )

        invalid_bounds = close_times < open_times
        if invalid_bounds.any():
            issues.append(
                self._error(
                    "invalid_timestamp_bounds",
                    "close_time must not precede open_time",
                    self._row_labels(frame, invalid_bounds),
                )
            )

        expected_close = open_times + duration - timedelta(milliseconds=1)
        invalid_duration = close_times != expected_close
        if invalid_duration.any():
            issues.append(
                self._error(
                    "invalid_candle_duration",
                    f"candle boundaries must match the {duration} timeframe",
                    self._row_labels(frame, invalid_duration),
                )
            )

        incomplete = close_times >= pd.Timestamp(as_of)
        if incomplete.any():
            issues.append(
                self._error(
                    "incomplete_candles",
                    "dataset contains candles that were not closed at validation time",
                    self._row_labels(frame, incomplete),
                )
            )

        if ordered and not duplicates.any():
            issues.extend(self._validate_intervals(frame, duration))
        return issues

    def _validate_intervals(
        self, frame: pd.DataFrame, duration: timedelta
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        expected = pd.Timedelta(duration)
        for position in range(1, len(frame)):
            difference = (
                frame["open_time"].iloc[position]
                - frame["open_time"].iloc[position - 1]
            )
            if difference == expected:
                continue
            row = (str(frame.index[position]),)
            if difference > expected and difference % expected == pd.Timedelta(0):
                missing_count = int(difference / expected) - 1
                issues.append(
                    self._warning(
                        "missing_candles",
                        f"detected {missing_count} missing candle(s) before {row[0]}",
                        row,
                    )
                )
            else:
                issues.append(
                    self._error(
                        "irregular_candle_interval",
                        f"open_time interval before {row[0]} does not match {duration}",
                        row,
                    )
                )
        return issues

    def _validate_prices_and_volume(self, frame: pd.DataFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for index, row in frame.iterrows():
            row_label = (str(index),)
            prices = [row[column] for column in PRICE_COLUMNS]
            if any(value <= 0 for value in prices):
                issues.append(
                    self._error(
                        "invalid_price",
                        "OHLC prices must be positive",
                        row_label,
                    )
                )
            if row["high"] < max(prices) or row["low"] > min(prices):
                issues.append(
                    self._error(
                        "invalid_ohlc_relationship",
                        "high must be the greatest OHLC value and low the smallest",
                        row_label,
                    )
                )
            if row["volume"] < 0:
                issues.append(
                    self._error(
                        "invalid_volume",
                        "volume must be non-negative",
                        row_label,
                    )
                )
        return issues

    @staticmethod
    def _row_labels(frame: pd.DataFrame, mask: pd.Series) -> tuple[str, ...]:
        return tuple(str(index) for index in frame.index[mask])

    @staticmethod
    def _error(code: str, message: str, rows: tuple[str, ...] = ()) -> ValidationIssue:
        return ValidationIssue(IssueSeverity.ERROR, code, message, rows)

    @staticmethod
    def _warning(
        code: str, message: str, rows: tuple[str, ...] = ()
    ) -> ValidationIssue:
        return ValidationIssue(IssueSeverity.WARNING, code, message, rows)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Decimal):
        return value.is_finite()
    if not isinstance(value, Real):
        return False
    return math.isfinite(value)
