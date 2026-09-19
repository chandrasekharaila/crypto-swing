"""Tests for typed Phase 1 configuration."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from crypto_analyzer.config import AppSettings


def test_default_settings_cover_initial_market_scope() -> None:
    settings = AppSettings()

    assert settings.symbols == ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    assert settings.timeframes == ("15m", "1h", "4h", "1d")
    assert settings.primary_timeframe == "4h"
    assert settings.request_limit == 1_000


def test_settings_allow_additional_symbols() -> None:
    settings = AppSettings(symbols=("BTC/USDT", "ADA/USDT"))

    assert settings.symbols == ("BTC/USDT", "ADA/USDT")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"symbols": ()}, "at least one symbol"),
        ({"symbols": ("BTCUSDT",)}, "BASE/QUOTE"),
        ({"symbols": ("BTC/USD-T",)}, "BASE/QUOTE"),
        ({"timeframes": ("1h",), "primary_timeframe": "4h"}, "primary_timeframe"),
        ({"timeframes": ("17m",), "primary_timeframe": "17m"}, "unsupported Binance"),
    ],
)
def test_settings_reject_inconsistent_market_configuration(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        AppSettings(**override)


def test_settings_require_timezone_aware_history_bounds() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        AppSettings(history_start=datetime(2024, 1, 1))


def test_settings_require_ordered_history_bounds() -> None:
    instant = datetime(2024, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="earlier than"):
        AppSettings(history_start=instant, history_end=instant)


def test_create_data_directories_uses_configured_root(tmp_path: Path) -> None:
    settings = AppSettings(data_directory=tmp_path / "market-data")

    settings.create_data_directories()

    assert settings.raw_data_directory.is_dir()
    assert settings.processed_data_directory.is_dir()
    assert settings.features_data_directory.is_dir()
    assert settings.experiments_data_directory.is_dir()
