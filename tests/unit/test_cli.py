"""Tests for CLI configuration and historical range resolution."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_analyzer.cli import resolve_download_range
from crypto_analyzer.config import AppSettings, load_settings
from crypto_analyzer.data.exceptions import ConfigurationError


def test_default_download_range_uses_configured_candle_count() -> None:
    now = datetime(2024, 1, 1, 10, 31, tzinfo=UTC)
    settings = AppSettings(default_history_candles=3)

    start, end = resolve_download_range(
        settings, "1h", start=None, end=None, candle_count=None, now=now
    )

    assert end == datetime(2024, 1, 1, 10, tzinfo=UTC)
    assert start == end - timedelta(hours=3)


def test_loads_symbols_and_timeframes_from_json_configuration(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        '{"symbols":["ADA/USDT"],"timeframes":["1h"],"primary_timeframe":"1h"}',
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.symbols == ("ADA/USDT",)
    assert settings.timeframes == ("1h",)


def test_download_range_rejects_shifted_boundaries() -> None:
    start = datetime(2024, 1, 1, 0, 30, tzinfo=UTC)

    with pytest.raises(ConfigurationError, match="candle boundaries"):
        resolve_download_range(
            AppSettings(),
            "1h",
            start=start,
            end=start + timedelta(hours=2),
            candle_count=None,
        )


def test_weekly_default_range_uses_monday_boundary() -> None:
    settings = AppSettings(
        timeframes=("1w",), primary_timeframe="1w", default_history_candles=1
    )

    start, end = resolve_download_range(
        settings,
        "1w",
        start=None,
        end=None,
        candle_count=None,
        now=datetime(2024, 1, 10, tzinfo=UTC),
    )

    assert end == datetime(2024, 1, 8, tzinfo=UTC)
    assert start == datetime(2024, 1, 1, tzinfo=UTC)
