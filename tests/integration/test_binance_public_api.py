"""Opt-in contract test for Binance's unauthenticated kline endpoint."""

import os
from datetime import UTC, datetime, timedelta

import pytest

from crypto_analyzer.config import AppSettings
from crypto_analyzer.data.collectors import BinanceOHLCVCollector

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_BINANCE_INTEGRATION") != "1",
        reason="set RUN_BINANCE_INTEGRATION=1 to access Binance",
    ),
]


def test_binance_public_kline_contract() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)

    with BinanceOHLCVCollector(AppSettings()) as collector:
        candles = collector.fetch_historical("BTC/USDT", "1h", start, end)

    assert len(candles) == 1
    assert candles[0].open_time == start
    assert candles[0].close_time == end - timedelta(milliseconds=1)
