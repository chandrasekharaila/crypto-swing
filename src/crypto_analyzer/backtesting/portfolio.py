"""Portfolio assembly: sizing, the equity curve, and drawdown.

Each trade risks the same fraction of equity at entry, so the R multiple from the
simulator stays comparable across setups and across symbols. Equity is marked to
market every bar, which is what makes the drawdown and the Sharpe ratio reflect
open positions rather than only closed ones.

This is a portfolio proxy, not an account. It assumes the modelled fills, and it
ignores margin, funding, borrow, and any limit on notional -- only risk is capped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from crypto_analyzer.backtesting.config import BacktestSettings
from crypto_analyzer.backtesting.types import (
    SkippedSignal,
    SkipReason,
    TradeRecord,
    direction_sign,
)

CURVE_COLUMNS = ("equity", "peak", "drawdown", "open_positions")


@dataclass(frozen=True, slots=True)
class _Position:
    """An open position and the size it was given at entry."""

    trade: TradeRecord
    units: float


def _skip(trade: TradeRecord, reason: SkipReason) -> SkippedSignal:
    return SkippedSignal(
        symbol=trade.symbol,
        setup=trade.setup,
        direction=trade.direction,
        signal_time=trade.signal_time,
        reason=reason,
    )


def _empty_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "equity": pd.Series(dtype="float64"),
            "peak": pd.Series(dtype="float64"),
            "drawdown": pd.Series(dtype="float64"),
            "open_positions": pd.Series(dtype="int64"),
        },
        index=pd.DatetimeIndex([], tz="UTC"),
    )


def build_equity_curve(
    trades: Sequence[TradeRecord],
    prices: Mapping[str, pd.Series],
    settings: BacktestSettings,
) -> tuple[pd.DataFrame, tuple[SkippedSignal, ...]]:
    """Assemble usable trades into an equity curve, reporting anything skipped.

    ``prices`` maps a symbol to its close series indexed by candle open time. A
    symbol with no bar at a timestamp is marked at its last known close.
    """
    usable = [trade for trade in trades if trade.is_usable]
    if not usable:
        return _empty_curve(), ()

    price_maps: dict[str, dict[pd.Timestamp, float]] = {
        symbol: {
            pd.Timestamp(stamp): float(value)
            for stamp, value in zip(series.index.tolist(), series.tolist(), strict=True)
        }
        for symbol, series in prices.items()
    }
    entries: dict[pd.Timestamp, list[TradeRecord]] = {}
    exits: dict[pd.Timestamp, list[TradeRecord]] = {}
    for trade in usable:
        entries.setdefault(pd.Timestamp(trade.entry_time), []).append(trade)
        if trade.exit_time is not None:
            exits.setdefault(pd.Timestamp(trade.exit_time), []).append(trade)

    timeline = sorted(
        set().union(
            *(set(series) for series in price_maps.values()),
            entries.keys(),
            exits.keys(),
        )
    )
    first, last = min(entries), max(exits)
    timeline = [stamp for stamp in timeline if first <= stamp <= last]

    realized = 0.0
    open_positions: dict[str, _Position] = {}
    last_price: dict[str, float] = {}
    skipped: list[SkippedSignal] = []
    rows: list[tuple[pd.Timestamp, float, float, float, int]] = []
    peak = settings.initial_equity

    def equity_now() -> float:
        unrealized = 0.0
        for symbol, position in open_positions.items():
            mark = last_price.get(symbol, position.trade.entry_price)
            sign = direction_sign(position.trade.direction)
            unrealized += position.units * (mark - position.trade.entry_price) * sign
        return settings.initial_equity + realized + unrealized

    for stamp in timeline:
        for trade in exits.get(stamp, ()):
            position = open_positions.pop(trade.symbol, None)
            if position is None or trade.exit_price is None:
                continue
            sign = direction_sign(trade.direction)
            gross = position.units * (trade.exit_price - trade.entry_price) * sign
            realized += gross - position.units * trade.entry_price * trade.fees

        equity_at_entry = equity_now()
        for trade in entries.get(stamp, ()):
            if equity_at_entry <= 0.0:
                # A position sized when the account was larger can realize a loss
                # bigger than what is left. An account cannot trade through zero,
                # so entries stop and the ruin is reported rather than drawn.
                skipped.append(_skip(trade, SkipReason.ACCOUNT_RUINED))
                continue
            if len(open_positions) >= settings.max_concurrent_positions:
                skipped.append(_skip(trade, SkipReason.POSITION_LIMIT))
                continue
            if settings.one_position_per_symbol and trade.symbol in open_positions:
                skipped.append(_skip(trade, SkipReason.SYMBOL_BUSY))
                continue
            units_by_risk = (
                equity_at_entry * settings.risk_fraction
            ) / trade.risk_per_unit
            units_by_notional = (
                equity_at_entry * settings.max_notional_multiple
            ) / trade.entry_price
            open_positions[trade.symbol] = _Position(
                trade, min(units_by_risk, units_by_notional)
            )

        for symbol, series in price_maps.items():
            price = series.get(stamp)
            if price is not None:
                last_price[symbol] = price

        equity = equity_now()
        peak = max(peak, equity)
        rows.append((stamp, equity, peak, equity / peak - 1.0, len(open_positions)))

    curve = pd.DataFrame(rows, columns=["timestamp", *CURVE_COLUMNS]).set_index(
        "timestamp"
    )
    curve["open_positions"] = curve["open_positions"].astype("int64")
    return curve, tuple(skipped)
