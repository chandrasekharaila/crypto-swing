"""Tests for the integrated Phase 1 data pipeline."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from crypto_analyzer.config import AppSettings
from crypto_analyzer.data.models import Candle
from crypto_analyzer.data.service import MarketDataPipeline
from crypto_analyzer.data.storage import ParquetMarketDataStore

START = datetime(2024, 1, 1, tzinfo=UTC)


class RecordingCollector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, datetime, datetime | None]] = []

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> list[Candle]:
        self.calls.append((symbol, timeframe, start_time, end_time))
        assert end_time is not None
        candles: list[Candle] = []
        current = start_time
        while current < end_time:
            candles.append(
                Candle(
                    open_time=current,
                    close_time=current + timedelta(hours=1) - timedelta(milliseconds=1),
                    open=Decimal("100"),
                    high=Decimal("110"),
                    low=Decimal("90"),
                    close=Decimal("105"),
                    volume=Decimal("10"),
                )
            )
            current += timedelta(hours=1)
        return candles


def test_pipeline_downloads_validates_and_stores_missing_data(tmp_path: Path) -> None:
    settings = AppSettings(data_directory=tmp_path)
    collector = RecordingCollector()
    store = ParquetMarketDataStore(tmp_path)
    pipeline = MarketDataPipeline(settings, collector, store)
    end = START + timedelta(hours=3)

    result = pipeline.update("BTC/USDT", "1h", START, end)

    assert result.downloaded_candles == 3
    assert result.stored_candles == 3
    assert result.path.exists()
    assert collector.calls == [("BTC/USDT", "1h", START, end)]


def test_pipeline_is_idempotent_and_skips_unnecessary_api_calls(tmp_path: Path) -> None:
    settings = AppSettings(data_directory=tmp_path)
    collector = RecordingCollector()
    pipeline = MarketDataPipeline(
        settings, collector, ParquetMarketDataStore(tmp_path)
    )
    end = START + timedelta(hours=2)
    pipeline.update("BTC/USDT", "1h", START, end)
    collector.calls.clear()

    result = pipeline.update("BTC/USDT", "1h", START, end)

    assert result.cache_hit
    assert result.downloaded_candles == 0
    assert collector.calls == []
