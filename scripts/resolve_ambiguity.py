"""Fetch finer candles for the bars a backtest could not resolve.

A coarse candle whose range contains both the stop and the target does not say
which came first. Guessing swings the Phase 4 result by half a unit of R, so this
script replaces the guess with a measurement: it runs the backtest, collects every
ambiguous bar, fetches the finer candles covering it, and caches them for later
runs.

    python scripts/resolve_ambiguity.py [--timeframe 4h] [--refine 1m]

The cache is written under the processed data directory and is read by
``IntrabarWindows.from_parquet``. Running the script again only fetches windows
that are not already cached.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from crypto_analyzer.backtesting import (
    BacktestEngine,
    BacktestSettings,
    IntrabarWindows,
)
from crypto_analyzer.config import AppSettings, load_settings
from crypto_analyzer.data.collectors import BinanceOHLCVCollector
from crypto_analyzer.data.processors import candles_to_frame
from crypto_analyzer.data.storage import ParquetMarketDataStore
from crypto_analyzer.features import FeatureEngine
from crypto_analyzer.regimes import RegimeDetector
from crypto_analyzer.setups import SetupDetector

logger = logging.getLogger("resolve_ambiguity")

DEVELOPMENT_START = datetime(2022, 1, 1, tzinfo=UTC)
HOLDOUT_START = datetime(2026, 1, 1, tzinfo=UTC)
CACHE_NAME = "intrabar_windows.parquet"


def ambiguous_bars(settings: AppSettings, timeframe: str) -> set[tuple[str, datetime]]:
    """Return every bar where a stop and a target were both touched."""
    store = ParquetMarketDataStore(settings.data_directory)
    feature_engine = FeatureEngine()
    regimes = RegimeDetector()
    setups = SetupDetector()

    candles: dict[str, pd.DataFrame] = {}
    scans = []
    for symbol in settings.symbols:
        frame = store.load_raw(symbol, timeframe)
        features = feature_engine.compute(frame, symbol=symbol, timeframe=timeframe)
        scans.append(setups.detect(features, frame, regimes.detect(features)))
        candles[symbol] = frame

    found: set[tuple[str, datetime]] = set()
    for start, end in ((DEVELOPMENT_START, HOLDOUT_START), (HOLDOUT_START, None)):
        config = BacktestSettings(window_start=start, window_end=end)
        result = BacktestEngine(config).run(candles, scans)
        for trade in result.trades:
            if trade.ambiguous and trade.exit_time is not None:
                found.add((trade.symbol, trade.exit_time.to_pydatetime()))
    return found


def cached_keys(path: Path) -> set[tuple[str, datetime]]:
    """Return the windows already on disk."""
    if not path.exists():
        return set()
    frame = pd.read_parquet(path, engine="pyarrow")
    pairs = frame[["symbol", "bar_open_time"]].drop_duplicates()
    return {
        (str(symbol), pd.Timestamp(stamp).to_pydatetime())
        for symbol, stamp in zip(
            pairs["symbol"].tolist(), pairs["bar_open_time"].tolist(), strict=True
        )
    }


def merge_into_cache(
    path: Path, frames: dict[tuple[str, datetime], pd.DataFrame]
) -> IntrabarWindows:
    """Append newly fetched windows to whatever is already cached."""
    fresh = IntrabarWindows.from_frames(frames).to_frame()
    if path.exists():
        existing = pd.read_parquet(path, engine="pyarrow")
        fresh = pd.concat([existing, fresh], ignore_index=True)
    return IntrabarWindows(fresh)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve ambiguous bars with finer candles"
    )
    parser.add_argument("--config", type=Path, default=Path("config/universe.json"))
    parser.add_argument("--timeframe", default="4h", help="the coarse candle interval")
    parser.add_argument("--refine", default="1m", help="the finer interval to fetch")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = load_settings(args.config)
    cache_path = settings.processed_data_directory / CACHE_NAME
    known = cached_keys(cache_path)

    wanted = ambiguous_bars(settings, args.timeframe)
    pending = sorted(wanted - known)
    logger.info(
        "ambiguous bars: %d, already cached: %d, to fetch: %d",
        len(wanted),
        len(wanted) - len(pending),
        len(pending),
    )
    if not pending:
        logger.info("nothing to fetch")
        return 0

    if args.refine not in settings.timeframes:
        settings = settings.model_copy(
            update={"timeframes": (*settings.timeframes, args.refine)}
        )
    span = timedelta(hours=4) if args.timeframe == "4h" else timedelta(days=1)

    frames: dict[tuple[str, datetime], pd.DataFrame] = {}
    failures = 0
    with BinanceOHLCVCollector(settings) as collector:
        for position, (symbol, bar_open) in enumerate(pending, start=1):
            try:
                candles = collector.fetch_historical(
                    symbol, args.refine, bar_open, bar_open + span
                )
            except Exception as error:
                failures += 1
                logger.error("%s %s could not be fetched: %s", symbol, bar_open, error)
                continue
            if candles:
                frames[(symbol, bar_open)] = candles_to_frame(candles)
            if position % 100 == 0:
                logger.info("fetched %d/%d", position, len(pending))

    if not frames:
        logger.error("no windows could be fetched")
        return 1

    merged = merge_into_cache(cache_path, frames)
    merged.write_parquet(cache_path)
    logger.info(
        "cached %d window(s) at %s, %d fetch failure(s)",
        len(merged),
        cache_path,
        failures,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
