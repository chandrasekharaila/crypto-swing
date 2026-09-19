"""Value types for event-driven backtesting.

A trade record is written so that the fill can be audited without reading the
simulation: it carries both the signal time and the entry time, and both the
unslipped and slipped outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd

from crypto_analyzer.taxonomy import Direction, SetupType

if TYPE_CHECKING:
    from crypto_analyzer.backtesting.config import BacktestSettings
    from crypto_analyzer.backtesting.metrics import MetricsReport


def direction_sign(direction: Direction) -> float:
    """Return +1 for a long and -1 for a short."""
    return 1.0 if direction is Direction.LONG else -1.0


class ExitReason(StrEnum):
    """Why a simulated position was closed."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_HOLDING = "max_holding"
    CENSORED = "censored"


class SkipReason(StrEnum):
    """Why a signal produced no trade."""

    NO_ENTRY_BAR = "no_entry_bar"
    OUTSIDE_WINDOW = "outside_window"
    INVALID_STOP = "invalid_stop"
    POSITION_LIMIT = "position_limit"
    SYMBOL_BUSY = "symbol_busy"
    ACCOUNT_RUINED = "account_ruined"


class AmbiguousExitPolicy(StrEnum):
    """How to resolve a bar whose range contains both the stop and the target.

    OHLC data does not reveal the path a price took inside a bar, so one of the
    two levels must be chosen. ``STOP_FIRST`` is the pessimistic reading and the
    default, because a backtest that assumes the best case is not evidence.
    """

    STOP_FIRST = "stop_first"
    TARGET_FIRST = "target_first"


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """One simulated round trip, with everything needed to audit the fill.

    ``gross_return`` uses the unslipped prices and ``net_return`` uses the
    slipped ones after fees, both as a fraction of entry notional, so
    ``gross_return - net_return`` is exactly the cost of the assumptions.
    """

    symbol: str
    timeframe: str
    setup: SetupType
    direction: Direction

    signal_time: datetime
    signal_index: int
    entry_time: datetime
    entry_index: int
    exit_time: datetime | None
    exit_index: int | None

    signal_price: float
    entry_price: float
    exit_price: float | None
    stop_price: float
    target_price: float
    risk_per_unit: float

    holding_bars: int
    exit_reason: ExitReason
    gross_return: float
    net_return: float
    r_multiple: float
    fees: float
    slippage_cost: float
    mae: float
    mfe: float
    ambiguous: bool

    @property
    def is_usable(self) -> bool:
        """Whether the trade closed inside the simulated window."""
        return self.exit_reason is not ExitReason.CENSORED

    @property
    def is_win(self) -> bool:
        """Whether the trade finished ahead after costs."""
        return self.is_usable and self.net_return > 0.0


@dataclass(frozen=True, slots=True)
class SkippedSignal:
    """A signal that produced no trade, kept so nothing is dropped silently."""

    symbol: str
    setup: SetupType
    direction: Direction
    signal_time: datetime
    reason: SkipReason


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Trades, skipped signals, equity curve, and metrics for one run.

    ``equity`` is indexed by candle timestamp and carries the mark-to-market
    equity, the running peak, the drawdown from that peak, and the number of open
    positions at each bar.
    """

    settings: BacktestSettings
    timeframe: str
    window_start: datetime | None
    window_end: datetime | None
    trades: tuple[TradeRecord, ...]
    skipped: tuple[SkippedSignal, ...]
    equity: pd.DataFrame
    metrics: MetricsReport

    @property
    def usable_trades(self) -> tuple[TradeRecord, ...]:
        """Return the trades that closed inside the window."""
        return tuple(trade for trade in self.trades if trade.is_usable)

    def for_setup(self, setup: SetupType) -> tuple[TradeRecord, ...]:
        """Return the usable trades belonging to one setup family."""
        return tuple(trade for trade in self.usable_trades if trade.setup is setup)

    def for_direction(self, direction: Direction) -> tuple[TradeRecord, ...]:
        """Return the usable trades on one side of the market."""
        return tuple(
            trade for trade in self.usable_trades if trade.direction is direction
        )
