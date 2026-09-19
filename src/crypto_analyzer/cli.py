"""Command-line interface for Phase 1 market-data operations."""

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path

from crypto_analyzer.config import AppSettings, load_settings
from crypto_analyzer.data.collectors import BinanceOHLCVCollector
from crypto_analyzer.data.exceptions import DataFoundationError, ConfigurationError
from crypto_analyzer.data.service import MarketDataPipeline
from crypto_analyzer.data.storage import ParquetMarketDataStore
from crypto_analyzer.data.validators import timeframe_duration

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the project command parser."""
    parser = argparse.ArgumentParser(prog="crypto_analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser(
        "download", help="download or update closed Binance OHLCV candles"
    )
    download.add_argument("--symbol", required=True, help="configured BASE/QUOTE symbol")
    download.add_argument("--timeframe", required=True, help="configured Binance interval")
    download.add_argument("--start", type=_parse_datetime, help="inclusive ISO-8601 UTC time")
    download.add_argument("--end", type=_parse_datetime, help="exclusive ISO-8601 UTC time")
    download.add_argument(
        "--candles",
        type=_positive_int,
        help="number of recent closed candles when --start is omitted",
    )
    download.add_argument("--config", type=Path, help="optional JSON settings file")
    download.add_argument(
        "--data-directory", type=Path, help="override the configured data directory"
    )
    download.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = load_settings(args.config)
        if args.data_directory is not None:
            settings = settings.model_copy(update={"data_directory": args.data_directory})
        start, end = resolve_download_range(
            settings,
            args.timeframe,
            start=args.start,
            end=args.end,
            candle_count=args.candles,
        )
        store = ParquetMarketDataStore(settings.data_directory)
        with BinanceOHLCVCollector(settings) as collector:
            result = MarketDataPipeline(settings, collector, store).update(
                args.symbol, args.timeframe, start, end
            )
        if result.cache_hit:
            logger.info("Cache hit: %s", result.path)
        else:
            logger.info(
                "Saved %d total candle(s) to %s",
                result.stored_candles,
                result.path,
            )
        return 0
    except DataFoundationError as error:
        logger.error("Download failed: %s", error)
        return 1
    except KeyboardInterrupt:
        logger.error("Download interrupted")
        return 130
    except Exception:
        logger.exception("Unexpected download failure")
        return 1


def resolve_download_range(
    settings: AppSettings,
    timeframe: str,
    *,
    start: datetime | None,
    end: datetime | None,
    candle_count: int | None,
    now: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Resolve a complete, timeframe-aligned historical request range."""
    duration = timeframe_duration(timeframe)
    current = (now or datetime.now(UTC)).astimezone(UTC)
    resolved_end = end or settings.history_end or _floor_time(current, duration)
    resolved_end = resolved_end.astimezone(UTC)
    count = candle_count or settings.default_history_candles
    resolved_start = start or settings.history_start or resolved_end - count * duration
    resolved_start = resolved_start.astimezone(UTC)

    if resolved_start >= resolved_end:
        raise ConfigurationError("download start must be earlier than end")
    if not _is_aligned(resolved_start, duration) or not _is_aligned(
        resolved_end, duration
    ):
        raise ConfigurationError(
            f"download range must align to {timeframe} candle boundaries"
        )
    return resolved_start, resolved_end


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO-8601 datetime: {value}") from error
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime must include a UTC offset")
    return parsed.astimezone(UTC)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _floor_time(value: datetime, duration: timedelta) -> datetime:
    duration_seconds = int(duration.total_seconds())
    timestamp = int(value.timestamp())
    return datetime.fromtimestamp(
        timestamp - timestamp % duration_seconds,
        tz=UTC,
    )


def _is_aligned(value: datetime, duration: timedelta) -> bool:
    duration_seconds = int(duration.total_seconds())
    return value.microsecond == 0 and int(value.timestamp()) % duration_seconds == 0
