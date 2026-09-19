"""Tests for CLI configuration and historical range resolution."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_analyzer.cli import (
    BACKFILL_EARLIEST_START,
    build_parser,
    resolve_backfill_scope,
    resolve_download_range,
)
from crypto_analyzer.config import AppSettings, load_settings
from crypto_analyzer.data.exceptions import ConfigurationError


def test_backfill_defaults_to_the_configured_scope() -> None:
    args = build_parser().parse_args(["backfill"])

    assert args.command == "backfill"
    assert args.timeframe is None
    assert args.symbol is None
    assert args.start == BACKFILL_EARLIEST_START
    assert args.end is None


def test_backfill_accepts_repeated_timeframes_and_symbols() -> None:
    args = build_parser().parse_args(
        [
            "backfill",
            "--timeframe",
            "1d",
            "--timeframe",
            "4h",
            "--symbol",
            "BTC/USDT",
            "--symbol",
            "ETH/USDT",
        ]
    )

    assert args.timeframe == ["1d", "4h"]
    assert args.symbol == ["BTC/USDT", "ETH/USDT"]


def test_backfill_resolves_every_configured_symbol_by_default() -> None:
    settings = AppSettings()

    timeframes, symbols = resolve_backfill_scope(
        settings, timeframes=None, symbols=None
    )

    assert timeframes == (settings.primary_timeframe,)
    assert symbols == settings.symbols


def test_backfill_scope_can_be_narrowed() -> None:
    settings = AppSettings()

    timeframes, symbols = resolve_backfill_scope(
        settings, timeframes=["1d"], symbols=["BTC/USDT"]
    )

    assert timeframes == ("1d",)
    assert symbols == ("BTC/USDT",)


@pytest.mark.parametrize(
    ("timeframes", "symbols", "message"),
    [
        (["7m"], None, "timeframe is not configured"),
        (None, ["DOGE/USDT"], "symbol is not configured"),
    ],
)
def test_backfill_rejects_anything_unconfigured(timeframes, symbols, message) -> None:
    """A typo must fail before hundreds of requests are issued."""
    with pytest.raises(ConfigurationError, match=message):
        resolve_backfill_scope(AppSettings(), timeframes=timeframes, symbols=symbols)


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
