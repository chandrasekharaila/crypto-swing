"""Resolving a bar's internal path with finer candles.

A four-hour candle whose range contains both the stop and the target does not say
which came first. Guessing was the single largest source of uncertainty in the
backtest, so where finer candles exist the guess is replaced by a measurement.

The unit of storage is a *window*: the finer candles covering one coarse bar, for
one symbol. Windows are cached rather than fetched during simulation, so a run
never touches the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from crypto_analyzer.backtesting.exceptions import BacktestInputError
from crypto_analyzer.backtesting.types import ExitReason, direction_sign
from crypto_analyzer.taxonomy import Direction

WINDOW_COLUMNS = ("symbol", "bar_open_time", "minute_open_time", "high", "low")

Resolution = ExitReason | None


class IntrabarResolver(Protocol):
    """Decides which level came first, when finer data allows it.

    Returning ``None`` means the finer data could not decide either, and the
    caller should fall back to its configured assumption.
    """

    def resolve(
        self,
        symbol: str,
        bar_open_time: datetime,
        direction: Direction,
        stop: float,
        target: float,
    ) -> Resolution: ...


def resolution_from_minutes(
    highs: Sequence[float],
    lows: Sequence[float],
    *,
    direction: Direction,
    stop: float,
    target: float,
) -> Resolution:
    """Return the level the finer candles reached first.

    ``None`` means the sequence could not decide: either it never reached a level,
    or one finer candle contained both, which is the same ambiguity one scale
    down.
    """
    sign = direction_sign(direction)
    for high, low in zip(highs, lows, strict=True):
        stop_hit = low <= stop if sign > 0 else high >= stop
        target_hit = high >= target if sign > 0 else low <= target
        if stop_hit and target_hit:
            return None
        if stop_hit:
            return ExitReason.STOP_LOSS
        if target_hit:
            return ExitReason.TAKE_PROFIT
    return None


@dataclass(frozen=True, slots=True)
class _Window:
    highs: tuple[float, ...]
    lows: tuple[float, ...]


class IntrabarWindows:
    """Cached finer candles for the bars a backtest could not resolve."""

    def __init__(self, windows: pd.DataFrame) -> None:
        self._frame = windows.copy(deep=True)
        self._windows = self._index(self._frame)
        self._resolved = 0
        self._attempted = 0

    @staticmethod
    def _index(windows: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], _Window]:
        if windows.empty:
            return {}
        missing = [name for name in WINDOW_COLUMNS if name not in windows.columns]
        if missing:
            raise BacktestInputError(
                "intrabar windows are missing columns: " + ", ".join(missing)
            )
        ordered = windows.sort_values(["symbol", "bar_open_time", "minute_open_time"])
        grouped: dict[tuple[str, pd.Timestamp], tuple[list[float], list[float]]] = {}
        for symbol, bar_open, high, low in zip(
            ordered["symbol"].tolist(),
            ordered["bar_open_time"].tolist(),
            ordered["high"].tolist(),
            ordered["low"].tolist(),
            strict=True,
        ):
            key = (str(symbol), pd.Timestamp(bar_open))
            highs, lows = grouped.setdefault(key, ([], []))
            highs.append(float(high))
            lows.append(float(low))
        return {
            key: _Window(highs=tuple(highs), lows=tuple(lows))
            for key, (highs, lows) in grouped.items()
        }

    @classmethod
    def from_parquet(cls, path: Path) -> IntrabarWindows:
        """Load cached windows, or an empty resolver when nothing is cached."""
        if not path.exists():
            return cls(pd.DataFrame(columns=list(WINDOW_COLUMNS)))
        return cls(pd.read_parquet(path, engine="pyarrow"))

    @classmethod
    def from_frames(
        cls, frames: dict[tuple[str, datetime], pd.DataFrame]
    ) -> IntrabarWindows:
        """Build windows from finer candle frames, for tests and for the fetcher."""
        records = [
            {
                "symbol": symbol,
                "bar_open_time": pd.Timestamp(bar_open),
                "minute_open_time": minute,
                "high": float(high),
                "low": float(low),
            }
            for (symbol, bar_open), frame in frames.items()
            for minute, high, low in zip(
                frame["open_time"].tolist(),
                frame["high"].tolist(),
                frame["low"].tolist(),
                strict=True,
            )
        ]
        return cls(pd.DataFrame.from_records(records, columns=list(WINDOW_COLUMNS)))

    def write_parquet(self, path: Path) -> None:
        """Persist the windows so later runs need no network access.

        The frame is written as loaded, minute ordering included: the resolution
        depends on the sequence of finer candles, so dropping the timestamps would
        silently change the answer on the next read.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self._frame.to_parquet(path, engine="pyarrow", index=False)

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the cached windows, for inspection or merging."""
        return self._frame.copy(deep=True)

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def attempts(self) -> int:
        """How many ambiguity resolutions were requested."""
        return self._attempted

    @property
    def resolved(self) -> int:
        """How many of those the finer candles actually decided."""
        return self._resolved

    def resolve(
        self,
        symbol: str,
        bar_open_time: datetime,
        direction: Direction,
        stop: float,
        target: float,
    ) -> Resolution:
        """Return the level the finer candles reached first, if it is known."""
        window = self._windows.get((symbol, pd.Timestamp(bar_open_time)))
        if window is None:
            return None
        self._attempted += 1
        outcome = resolution_from_minutes(
            window.highs, window.lows, direction=direction, stop=stop, target=target
        )
        if outcome is not None:
            self._resolved += 1
        return outcome
