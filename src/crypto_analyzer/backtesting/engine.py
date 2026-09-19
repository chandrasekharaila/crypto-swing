"""Orchestration: candles and detected signals in, trades and metrics out.

The engine never re-derives a signal and never searches for a bar by timestamp
inside the simulation. Each symbol's candles are converted to positional arrays
once, signals are resolved to an index once, and the simulation then reads only
positions it has already been told are available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import pandas as pd

from crypto_analyzer.backtesting.config import BacktestSettings
from crypto_analyzer.backtesting.exceptions import BacktestInputError
from crypto_analyzer.backtesting.intrabar import IntrabarResolver
from crypto_analyzer.backtesting.metrics import bars_per_year, compute_metrics
from crypto_analyzer.backtesting.portfolio import build_equity_curve
from crypto_analyzer.backtesting.simulator import CandleSeries, simulate_trade
from crypto_analyzer.backtesting.types import (
    BacktestResult,
    SkippedSignal,
    SkipReason,
    TradeRecord,
)
from crypto_analyzer.setups.types import SetupScan, SetupSignal


def _last_usable_index(series: CandleSeries, window_end: datetime | None) -> int:
    if window_end is None:
        return len(series) - 1
    last = -1
    for index, stamp in enumerate(series.open_time):
        if stamp < window_end:
            last = index
        else:
            break
    return last


class BacktestEngine:
    """Simulate detected setups against historical candles."""

    def __init__(self, settings: BacktestSettings | None = None) -> None:
        self._settings = settings or BacktestSettings()

    @property
    def settings(self) -> BacktestSettings:
        """Return the configuration this engine applies."""
        return self._settings

    def run(
        self,
        candles_by_symbol: Mapping[str, pd.DataFrame],
        scans: Sequence[SetupScan],
        *,
        intrabar: IntrabarResolver | None = None,
    ) -> BacktestResult:
        """Return trades, skipped signals, an equity curve, and metrics.

        Signals whose timestamp falls outside the configured window produce no
        trade and are reported as skipped. The simulation is additionally bounded
        by the last bar inside the window, so a run never reads the period it is
        holding out.
        """
        settings = self._settings
        if not scans:
            raise BacktestInputError("at least one setup scan is required")

        symbols = [scan.symbol for scan in scans]
        duplicates = sorted({symbol for symbol in symbols if symbols.count(symbol) > 1})
        if duplicates:
            raise BacktestInputError(
                "each symbol may appear in only one scan: " + ", ".join(duplicates)
            )
        timeframes = {scan.timeframe for scan in scans}
        if len(timeframes) != 1:
            raise BacktestInputError(
                "every scan must share one timeframe: " + ", ".join(sorted(timeframes))
            )
        missing = sorted(
            symbol for symbol in symbols if symbol not in candles_by_symbol
        )
        if missing:
            raise BacktestInputError("no candles supplied for: " + ", ".join(missing))

        trades: list[TradeRecord] = []
        skipped: list[SkippedSignal] = []
        prices: dict[str, pd.Series] = {}

        for scan in scans:
            frame = candles_by_symbol[scan.symbol]
            series = CandleSeries.from_frame(frame)
            last_index = _last_usable_index(series, settings.window_end)
            prices[scan.symbol] = frame.set_index("open_time")["close"].astype(
                "float64"
            )
            index_by_time = {
                pd.Timestamp(stamp): index
                for index, stamp in enumerate(series.open_time)
            }

            for signal in scan.signals:
                stamp = pd.Timestamp(signal.timestamp)
                if settings.window_start is not None and stamp < pd.Timestamp(
                    settings.window_start
                ):
                    skipped.append(_skip_signal(signal, SkipReason.OUTSIDE_WINDOW))
                    continue
                if settings.window_end is not None and stamp >= pd.Timestamp(
                    settings.window_end
                ):
                    skipped.append(_skip_signal(signal, SkipReason.OUTSIDE_WINDOW))
                    continue
                index = index_by_time.get(stamp)
                if index is None:
                    raise BacktestInputError(
                        f"signal at {stamp} does not match a candle of {scan.symbol}"
                    )
                outcome = simulate_trade(
                    signal,
                    index,
                    series,
                    settings,
                    last_index=last_index,
                    intrabar=intrabar,
                )
                if isinstance(outcome, TradeRecord):
                    trades.append(outcome)
                else:
                    skipped.append(outcome)

        trades.sort(key=lambda trade: (trade.entry_time, trade.symbol))
        equity, portfolio_skips = build_equity_curve(trades, prices, settings)
        skipped.extend(portfolio_skips)
        skipped.sort(key=lambda item: (item.signal_time, item.symbol))

        timeframe = next(iter(timeframes))
        return BacktestResult(
            settings=settings,
            timeframe=timeframe,
            window_start=settings.window_start,
            window_end=settings.window_end,
            trades=tuple(trades),
            skipped=tuple(skipped),
            equity=equity,
            metrics=compute_metrics(
                trades,
                equity,
                skipped_signals=len(skipped),
                periods_per_year=bars_per_year(timeframe),
            ),
        )


def _skip_signal(signal: SetupSignal, reason: SkipReason) -> SkippedSignal:
    return SkippedSignal(
        symbol=signal.symbol,
        setup=signal.setup,
        direction=signal.direction,
        signal_time=signal.timestamp,
        reason=reason,
    )
