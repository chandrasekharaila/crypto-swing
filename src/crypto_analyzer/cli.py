"""Command-line interface for market-data operations."""

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from crypto_analyzer.config import AppSettings, load_settings
from crypto_analyzer.data.collectors import BinanceOHLCVCollector
from crypto_analyzer.data.exceptions import ConfigurationError
from crypto_analyzer.data.service import MarketDataPipeline
from crypto_analyzer.data.storage import ParquetMarketDataStore
from crypto_analyzer.data.validators import OHLCVValidator, timeframe_duration
from crypto_analyzer.exceptions import AnalyzerError
from crypto_analyzer.markets import floor_to_timeframe, is_timeframe_boundary

logger = logging.getLogger(__name__)

BACKFILL_EARLIEST_START = datetime(2017, 1, 1, tzinfo=UTC)
"""Earliest start a backfill will request.

Binance listed its first USDT pairs in 2017, and requesting earlier simply
returns nothing, so an earlier date would only add empty requests.
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the project command parser."""
    parser = argparse.ArgumentParser(prog="crypto_analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser(
        "download", help="download or update closed Binance OHLCV candles"
    )
    download.add_argument(
        "--symbol", required=True, help="configured BASE/QUOTE symbol"
    )
    download.add_argument(
        "--timeframe", required=True, help="configured Binance interval"
    )
    download.add_argument(
        "--start", type=_parse_datetime, help="inclusive ISO-8601 UTC time"
    )
    download.add_argument(
        "--end", type=_parse_datetime, help="exclusive ISO-8601 UTC time"
    )
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

    backfill = subparsers.add_parser(
        "backfill",
        help="download the configured universe, rebuilding the dataset from scratch",
    )
    backfill.add_argument(
        "--timeframe",
        action="append",
        help="configured interval to backfill, repeatable (default: primary timeframe)",
    )
    backfill.add_argument(
        "--symbol",
        action="append",
        help="configured symbol to backfill, repeatable (default: every symbol)",
    )
    backfill.add_argument(
        "--start",
        type=_parse_datetime,
        default=BACKFILL_EARLIEST_START,
        help=(
            f"inclusive ISO-8601 UTC start (default: {BACKFILL_EARLIEST_START.date()})"
        ),
    )
    backfill.add_argument(
        "--end", type=_parse_datetime, help="exclusive ISO-8601 UTC time"
    )
    backfill.add_argument("--config", type=Path, help="optional JSON settings file")
    backfill.add_argument(
        "--data-directory", type=Path, help="override the configured data directory"
    )
    backfill.add_argument(
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
            settings = settings.model_copy(
                update={"data_directory": args.data_directory}
            )
        if args.command == "backfill":
            return run_backfill(args, settings)
        return run_download(args, settings)
    except AnalyzerError as error:
        logger.error("%s failed: %s", args.command, error)
        return 1
    except KeyboardInterrupt:
        logger.error("%s interrupted", args.command)
        return 130
    except Exception:
        logger.exception("Unexpected %s failure", args.command)
        return 1


def run_download(args: argparse.Namespace, settings: AppSettings) -> int:
    """Download or update one configured market."""
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
    if not result.is_complete:
        logger.error(
            "Download remains incomplete: %d unresolved range(s)",
            len(result.unresolved_ranges),
        )
        return 1
    return 0


def resolve_backfill_scope(
    settings: AppSettings,
    *,
    timeframes: Sequence[str] | None,
    symbols: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve which datasets a backfill covers, rejecting anything unconfigured."""
    resolved_timeframes = (
        tuple(timeframes) if timeframes else (settings.primary_timeframe,)
    )
    resolved_symbols = tuple(symbols) if symbols else settings.symbols
    for timeframe in resolved_timeframes:
        if timeframe not in settings.timeframes:
            raise ConfigurationError(f"timeframe is not configured: {timeframe}")
    for symbol in resolved_symbols:
        if symbol not in settings.symbols:
            raise ConfigurationError(f"symbol is not configured: {symbol}")
    return resolved_timeframes, resolved_symbols


def run_backfill(args: argparse.Namespace, settings: AppSettings) -> int:
    """Backfill every dataset in scope, reporting coverage for each.

    A backfill is deliberately not held to the completeness bar that ``download``
    applies. Ranges before a pair's listing, and outages inside its history, have
    nothing to fetch, so an incomplete dataset is the normal outcome rather than
    a failure. The run fails only if a market cannot be collected or a stored
    dataset does not validate.
    """
    timeframes, symbols = resolve_backfill_scope(
        settings, timeframes=args.timeframe, symbols=args.symbol
    )
    start = args.start.astimezone(UTC)
    store = ParquetMarketDataStore(settings.data_directory)
    validator = OHLCVValidator()
    invalid = 0
    uncollectable = 0

    with BinanceOHLCVCollector(settings) as collector:
        pipeline = MarketDataPipeline(settings, collector, store)
        for timeframe in timeframes:
            end = args.end or floor_to_timeframe(datetime.now(UTC), timeframe)
            end = end.astimezone(UTC)
            for boundary, label in ((start, "start"), (end, "end")):
                if not is_timeframe_boundary(boundary, timeframe):
                    raise ConfigurationError(
                        f"backfill {label} must align to {timeframe} candle boundaries"
                    )
            for symbol in symbols:
                try:
                    pipeline.update(symbol, timeframe, start, end)
                except AnalyzerError as error:
                    uncollectable += 1
                    logger.error(
                        "%s %s could not be collected: %s", symbol, timeframe, error
                    )
                    continue
                frame = store.load_raw(symbol, timeframe)
                report = validator.validate(frame, timeframe)
                if report.errors:
                    invalid += 1
                    summary = "; ".join(
                        f"{issue.code}: {issue.message}" for issue in report.errors
                    )
                    logger.error(
                        "%s %s failed validation: %s", symbol, timeframe, summary
                    )
                first = frame["open_time"].min() if len(frame) else None
                logger.info(
                    "%s %s: %d candle(s)%s",
                    symbol,
                    timeframe,
                    len(frame),
                    f", earliest {first}" if first is not None else "",
                )

    logger.info(
        "Backfill finished: %d dataset(s), %d invalid, %d uncollectable",
        len(timeframes) * len(symbols),
        invalid,
        uncollectable,
    )
    return 1 if invalid or uncollectable else 0


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
    resolved_end = end or settings.history_end or floor_to_timeframe(current, timeframe)
    resolved_end = resolved_end.astimezone(UTC)
    count = candle_count or settings.default_history_candles
    resolved_start = start or settings.history_start or resolved_end - count * duration
    resolved_start = resolved_start.astimezone(UTC)

    if resolved_start >= resolved_end:
        raise ConfigurationError("download start must be earlier than end")
    if not is_timeframe_boundary(
        resolved_start, timeframe
    ) or not is_timeframe_boundary(resolved_end, timeframe):
        raise ConfigurationError(
            f"download range must align to {timeframe} candle boundaries"
        )
    return resolved_start, resolved_end


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid ISO-8601 datetime: {value}"
        ) from error
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
