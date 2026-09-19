"""Tests for local Parquet market-data storage and caching."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from crypto_analyzer.data.exceptions import DataStorageError
from crypto_analyzer.data.models import Candle
from crypto_analyzer.data.processors import candles_to_frame
from crypto_analyzer.data.storage import (
    MarketDataLayer,
    ParquetMarketDataStore,
    TimeRange,
)

START = datetime(2024, 1, 1, tzinfo=UTC)


def _frame(offsets: list[int]) -> pd.DataFrame:
    open_times = pd.DatetimeIndex([START + timedelta(hours=value) for value in offsets])
    return pd.DataFrame(
        {
            "open_time": open_times,
            "close_time": open_times + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "open": [100.0 + value for value in offsets],
            "high": [110.0 + value for value in offsets],
            "low": [90.0 + value for value in offsets],
            "close": [105.0 + value for value in offsets],
            "volume": [10.0 + value for value in offsets],
        }
    )


def test_builds_deterministic_layer_paths(tmp_path: Path) -> None:
    store = ParquetMarketDataStore(tmp_path)

    assert store.path_for(MarketDataLayer.RAW, "BTC/USDT", "4h") == (
        tmp_path / "raw" / "binance" / "BTC-USDT" / "4h.parquet"
    )
    assert store.path_for(MarketDataLayer.PROCESSED, "BTC/USDT", "4h") == (
        tmp_path / "processed" / "binance" / "BTC-USDT" / "4h.parquet"
    )


def test_updates_and_loads_existing_history_without_duplicates(tmp_path: Path) -> None:
    store = ParquetMarketDataStore(tmp_path)
    store.update_raw("BTC/USDT", "1h", _frame([0, 1]))

    merged = store.update_raw("BTC/USDT", "1h", _frame([1, 2]))
    loaded = store.load_raw("BTC/USDT", "1h")

    assert list(merged["open_time"]) == list(pd.date_range(START, periods=3, freq="1h"))
    assert not merged["open_time"].duplicated().any()
    pd.testing.assert_frame_equal(loaded, merged)


def test_conflicting_update_is_rejected_and_raw_history_is_preserved(
    tmp_path: Path,
) -> None:
    store = ParquetMarketDataStore(tmp_path)
    original = _frame([0, 1])
    store.update_raw("BTC/USDT", "1h", original)
    conflict = _frame([1, 2])
    conflict.loc[0, "close"] = 107.0

    with pytest.raises(DataStorageError, match="conflicting raw candles"):
        store.update_raw("BTC/USDT", "1h", conflict)

    pd.testing.assert_frame_equal(store.load_raw("BTC/USDT", "1h"), original)


def test_processed_write_does_not_change_raw_data(tmp_path: Path) -> None:
    store = ParquetMarketDataStore(tmp_path)
    raw = _frame([0, 1])
    processed = raw.copy(deep=True)
    processed["close"] += 1.0
    store.update_raw("ETH/USDT", "1h", raw)

    store.write_processed("ETH/USDT", "1h", processed)

    pd.testing.assert_frame_equal(store.load_raw("ETH/USDT", "1h"), raw)
    pd.testing.assert_frame_equal(store.load_processed("ETH/USDT", "1h"), processed)


def test_missing_ranges_identify_only_absent_intervals(tmp_path: Path) -> None:
    store = ParquetMarketDataStore(tmp_path)
    store.update_raw("SOL/USDT", "1h", _frame([0, 2]))

    missing = store.missing_ranges(
        "SOL/USDT", "1h", START, START + timedelta(hours=4)
    )

    assert missing == [
        TimeRange(START + timedelta(hours=1), START + timedelta(hours=2)),
        TimeRange(START + timedelta(hours=3), START + timedelta(hours=4)),
    ]


def test_complete_cache_requires_no_download(tmp_path: Path) -> None:
    store = ParquetMarketDataStore(tmp_path)
    store.update_raw("BTC/USDT", "1h", _frame([0, 1, 2]))

    assert (
        store.missing_ranges(
            "BTC/USDT", "1h", START, START + timedelta(hours=3)
        )
        == []
    )


def test_absent_dataset_returns_whole_requested_range(tmp_path: Path) -> None:
    store = ParquetMarketDataStore(tmp_path)

    assert store.missing_ranges(
        "BTC/USDT", "1h", START, START + timedelta(hours=2)
    ) == [TimeRange(START, START + timedelta(hours=2))]


def test_typed_candle_flows_through_frame_conversion_and_parquet(tmp_path: Path) -> None:
    store = ParquetMarketDataStore(tmp_path)
    candle = Candle(
        open_time=START,
        close_time=START + timedelta(hours=1) - timedelta(milliseconds=1),
        open=Decimal("100.0"),
        high=Decimal("110.0"),
        low=Decimal("90.0"),
        close=Decimal("105.0"),
        volume=Decimal("12.5"),
    )

    store.update_raw("BTC/USDT", "1h", candles_to_frame([candle]))
    loaded = store.load_raw("BTC/USDT", "1h")

    assert loaded.loc[0, "open_time"].to_pydatetime() == START
    assert loaded.loc[0, "close"] == Decimal("105.0")
