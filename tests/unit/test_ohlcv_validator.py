"""Tests for canonical OHLCV dataset validation."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from crypto_analyzer.data.exceptions import ConfigurationError, DataValidationError
from crypto_analyzer.data.validators import OHLCVValidator, ValidationReport

START = datetime(2024, 1, 1, tzinfo=UTC)
AS_OF = START + timedelta(hours=10)


def _valid_frame(rows: int = 3) -> pd.DataFrame:
    open_times = pd.date_range(START, periods=rows, freq="1h")
    return pd.DataFrame(
        {
            "open_time": open_times,
            "close_time": open_times + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "open": [100.0 + offset for offset in range(rows)],
            "high": [110.0 + offset for offset in range(rows)],
            "low": [90.0 + offset for offset in range(rows)],
            "close": [105.0 + offset for offset in range(rows)],
            "volume": [10.0 + offset for offset in range(rows)],
        }
    )


def _codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.issues}


def test_valid_dataset_has_no_issues_and_is_not_modified() -> None:
    frame = _valid_frame()
    original = frame.copy(deep=True)

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert report.is_valid
    assert report.issues == ()
    pd.testing.assert_frame_equal(frame, original)


def test_reports_missing_required_columns() -> None:
    frame = _valid_frame().drop(columns=["volume", "high"])

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert _codes(report) == {"missing_columns"}
    assert "high, volume" in report.errors[0].message


def test_reports_missing_values_with_rows() -> None:
    frame = _valid_frame()
    frame.loc[1, "close"] = None

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert "missing_values" in _codes(report)
    issue = next(issue for issue in report.errors if issue.code == "missing_values")
    assert issue.rows == ("1",)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
def test_rejects_non_numeric_market_values(column: str) -> None:
    frame = _valid_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[1, column] = "not-a-number"

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert "invalid_numeric_type" in _codes(report)


def test_rejects_naive_timestamp_columns() -> None:
    frame = _valid_frame()
    frame["open_time"] = frame["open_time"].dt.tz_localize(None)

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert "invalid_timestamp_type" in _codes(report)


def test_reports_duplicate_candles() -> None:
    frame = _valid_frame()
    frame.loc[1, ["open_time", "close_time"]] = frame.loc[
        0, ["open_time", "close_time"]
    ].values

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert "duplicate_candles" in _codes(report)


def test_reports_non_chronological_data() -> None:
    frame = _valid_frame().iloc[[1, 0, 2]].reset_index(drop=True)

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert "non_chronological_order" in _codes(report)


@pytest.mark.parametrize(
    ("column", "value", "expected_code"),
    [
        ("high", 99.0, "invalid_ohlc_relationship"),
        ("low", 106.0, "invalid_ohlc_relationship"),
        ("open", 0.0, "invalid_price"),
        ("volume", -1.0, "invalid_volume"),
    ],
)
def test_reports_invalid_market_values(
    column: str, value: float, expected_code: str
) -> None:
    frame = _valid_frame()
    frame.loc[0, column] = value

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert expected_code in _codes(report)


def test_warns_about_internal_missing_candles() -> None:
    frame = _valid_frame().drop(index=1).reset_index(drop=True)

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert report.is_valid
    assert [issue.code for issue in report.warnings] == ["missing_candles"]
    assert "1 missing candle" in report.warnings[0].message


def test_reports_incomplete_candles() -> None:
    frame = _valid_frame(rows=2)
    validation_time = frame.loc[1, "close_time"]

    report = OHLCVValidator().validate(
        frame, "1h", as_of=validation_time.to_pydatetime()
    )

    assert "incomplete_candles" in _codes(report)


def test_reports_invalid_candle_duration() -> None:
    frame = _valid_frame()
    frame.loc[1, "close_time"] -= pd.Timedelta(minutes=1)

    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    assert "invalid_candle_duration" in _codes(report)


def test_report_can_raise_clear_summary() -> None:
    frame = _valid_frame()
    frame.loc[0, "volume"] = -1
    report = OHLCVValidator().validate(frame, "1h", as_of=AS_OF)

    with pytest.raises(DataValidationError, match="invalid_volume"):
        report.raise_for_errors()


def test_rejects_non_fixed_or_unknown_timeframe() -> None:
    with pytest.raises(ConfigurationError, match="cannot be validated"):
        OHLCVValidator().validate(_valid_frame(), "1M", as_of=AS_OF)
