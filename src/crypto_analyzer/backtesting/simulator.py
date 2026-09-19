"""Single-trade fill simulation.

These rules decide whether a backtest is evidence or wishful thinking, so they
are conservative on purpose and stated in one place:

* the fill happens at the open of a bar strictly later than the decision, so the
  price that triggered the signal can never be the price we get;
* a bar that opens beyond a level fills at that open, not at the level, because a
  gap is a real and unfavourable fact;
* slippage and fees are charged on every fill rather than assumed away;
* when a single bar contains both the stop and the target, the pessimistic
  reading is taken and the case is counted, because OHLC does not reveal which
  came first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from crypto_analyzer.backtesting.config import BacktestSettings
from crypto_analyzer.backtesting.exceptions import BacktestLeakageError
from crypto_analyzer.backtesting.intrabar import IntrabarResolver
from crypto_analyzer.backtesting.types import (
    AmbiguousExitPolicy,
    ExitReason,
    SkippedSignal,
    SkipReason,
    TradeRecord,
    direction_sign,
)
from crypto_analyzer.setups.types import SetupSignal


@dataclass(frozen=True, slots=True)
class CandleSeries:
    """Positional OHLC arrays for one symbol.

    The simulation indexes these instead of searching for timestamps, which
    keeps a bar from ever being located by anything but its position.
    """

    open_time: tuple[datetime, ...]
    open: tuple[float, ...]
    high: tuple[float, ...]
    low: tuple[float, ...]
    close: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.open)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> CandleSeries:
        """Build positional arrays from a canonical candle frame."""
        return cls(
            open_time=tuple(frame["open_time"].tolist()),
            open=tuple(frame["open"].astype("float64").tolist()),
            high=tuple(frame["high"].astype("float64").tolist()),
            low=tuple(frame["low"].astype("float64").tolist()),
            close=tuple(frame["close"].astype("float64").tolist()),
        )


def _skip(signal: SetupSignal, reason: SkipReason) -> SkippedSignal:
    return SkippedSignal(
        symbol=signal.symbol,
        setup=signal.setup,
        direction=signal.direction,
        signal_time=signal.timestamp,
        reason=reason,
    )


def simulate_trade(
    signal: SetupSignal,
    signal_index: int,
    series: CandleSeries,
    settings: BacktestSettings,
    *,
    last_index: int,
    intrabar: IntrabarResolver | None = None,
) -> TradeRecord | SkippedSignal:
    """Simulate one signal as a single trade, or report why it produced none.

    ``last_index`` is the final bar the simulation is allowed to read, inclusive.
    Nothing beyond it is touched, which is what stops a windowed run from reading
    the period it is meant to hold out.
    """
    sign = direction_sign(signal.direction)
    entry_index = signal_index + settings.execution_delay_bars
    if entry_index >= len(series):
        return _skip(signal, SkipReason.NO_ENTRY_BAR)
    if entry_index > last_index:
        return _skip(signal, SkipReason.OUTSIDE_WINDOW)

    entry_time = series.open_time[entry_index]
    if entry_time <= signal.available_at:
        raise BacktestLeakageError(
            f"entry at {entry_time.isoformat()} is not after the signal became "
            f"known at {signal.available_at.isoformat()}"
        )

    raw_entry = series.open[entry_index]
    stop = signal.invalidation_context.level
    if stop is None or (stop >= raw_entry if sign > 0 else stop <= raw_entry):
        return _skip(signal, SkipReason.INVALID_STOP)

    entry_price = raw_entry * (1.0 + sign * settings.slippage_rate)
    risk_per_unit = abs(entry_price - stop)
    target = entry_price + sign * settings.take_profit_r_multiple * risk_per_unit

    timeout_index = entry_index + settings.max_holding_bars
    censored = timeout_index > last_index
    walk_end = min(timeout_index, last_index)

    exit_index: int | None = None
    exit_reason: ExitReason | None = None
    raw_exit: float | None = None
    ambiguous = False
    resolved_intrabar = False
    lowest = series.low[entry_index]
    highest = series.high[entry_index]

    for index in range(entry_index, walk_end + 1):
        bar_open = series.open[index]
        bar_high = series.high[index]
        bar_low = series.low[index]
        lowest = min(lowest, bar_low)
        highest = max(highest, bar_high)

        # A gap is checked before any level touch, and never fires on the entry
        # bar because the entry price was validated to sit beyond the stop.
        if (sign > 0 and bar_open <= stop) or (sign < 0 and bar_open >= stop):
            exit_index, exit_reason, raw_exit = index, ExitReason.STOP_LOSS, bar_open
            break
        if (sign > 0 and bar_open >= target) or (sign < 0 and bar_open <= target):
            exit_index, exit_reason, raw_exit = index, ExitReason.TAKE_PROFIT, bar_open
            break

        stop_hit = bar_low <= stop if sign > 0 else bar_high >= stop
        target_hit = bar_high >= target if sign > 0 else bar_low <= target
        if stop_hit and target_hit:
            ambiguous = True
            decided = (
                intrabar.resolve(
                    signal.symbol,
                    series.open_time[index],
                    signal.direction,
                    stop,
                    target,
                )
                if intrabar is not None
                else None
            )
            if decided is ExitReason.STOP_LOSS:
                exit_index, exit_reason, raw_exit = index, ExitReason.STOP_LOSS, stop
                resolved_intrabar = True
            elif decided is ExitReason.TAKE_PROFIT:
                exit_index, exit_reason, raw_exit = (
                    index,
                    ExitReason.TAKE_PROFIT,
                    target,
                )
                resolved_intrabar = True
            elif settings.ambiguous_exit_policy is AmbiguousExitPolicy.STOP_FIRST:
                exit_index, exit_reason, raw_exit = index, ExitReason.STOP_LOSS, stop
            else:
                exit_index, exit_reason, raw_exit = (
                    index,
                    ExitReason.TAKE_PROFIT,
                    target,
                )
            break
        if stop_hit:
            exit_index, exit_reason, raw_exit = index, ExitReason.STOP_LOSS, stop
            break
        if target_hit:
            exit_index, exit_reason, raw_exit = index, ExitReason.TAKE_PROFIT, target
            break

    if exit_reason is None:
        exit_index = walk_end
        raw_exit = series.close[walk_end]
        exit_reason = ExitReason.CENSORED if censored else ExitReason.MAX_HOLDING

    assert exit_index is not None and raw_exit is not None

    slipped_exit = raw_exit * (1.0 - sign * settings.slippage_rate)
    gross_return = sign * (raw_exit - raw_entry) / raw_entry
    slipped_return = sign * (slipped_exit - entry_price) / entry_price
    fees = 2.0 * settings.fee_rate
    net_return = slipped_return - fees
    risk_fraction_of_price = risk_per_unit / entry_price

    if sign > 0:
        mae = lowest / entry_price - 1.0
        mfe = highest / entry_price - 1.0
    else:
        mae = 1.0 - highest / entry_price
        mfe = 1.0 - lowest / entry_price

    return TradeRecord(
        symbol=signal.symbol,
        timeframe=signal.timeframe,
        setup=signal.setup,
        direction=signal.direction,
        signal_time=signal.timestamp,
        signal_index=signal_index,
        entry_time=entry_time,
        entry_index=entry_index,
        exit_time=series.open_time[exit_index],
        exit_index=exit_index,
        signal_price=signal.entry_context.level or raw_entry,
        entry_price=entry_price,
        exit_price=slipped_exit,
        stop_price=stop,
        target_price=target,
        risk_per_unit=risk_per_unit,
        holding_bars=exit_index - entry_index + 1,
        exit_reason=exit_reason,
        gross_return=gross_return,
        net_return=net_return,
        r_multiple=net_return / risk_fraction_of_price,
        fees=fees,
        slippage_cost=gross_return - slipped_return,
        mae=mae,
        mfe=mfe,
        ambiguous=ambiguous,
        resolved_intrabar=resolved_intrabar,
    )
