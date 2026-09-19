"""Look-ahead guards for the backtester.

The feature, regime, and setup layers each have their own causality sweep. This
covers the layer that decides fills, which is where a backtest most often leaks:
reading a bar the decision could not have reached, or letting a windowed run see
past the end of its window.
"""

import pandas as pd
import pytest

from crypto_analyzer.backtesting import BacktestEngine, BacktestSettings, ExitReason
from crypto_analyzer.setups.types import SetupScan

from .conftest import DELTA, START, TIMEFRAME

QUIET = (100.0, 101.0, 99.0, 100.0)


def _scan(signals):
    return SetupScan(symbol="BTC/USDT", timeframe=TIMEFRAME, signals=tuple(signals))


def _settings(**overrides) -> BacktestSettings:
    base = {"fee_bps": 10.0, "slippage_bps": 0.0, "max_holding_bars": 4}
    base.update(overrides)
    return BacktestSettings(**base)


def _corrupt_after(
    frame: pd.DataFrame, cut: int, sentinel: float = 5.0
) -> pd.DataFrame:
    corrupted = frame.copy(deep=True)
    columns = list(corrupted.columns)
    for column in ("open", "high", "low", "close"):
        corrupted.iloc[cut:, columns.index(column)] = sentinel
    return corrupted


def test_trades_do_not_change_when_the_future_changes(make_frame, make_signal) -> None:
    """A trade that opened and closed before the cut cannot depend on it.

    Trades still open at the cut are excluded on purpose: their outcome genuinely
    depends on later bars, and that is the honest behaviour rather than a leak.
    """
    frame = make_frame([QUIET] * 10)
    scan = _scan([make_signal(0, stop=95.0), make_signal(1, stop=95.0)])
    settings = _settings(max_holding_bars=2)
    cut = 5

    baseline = BacktestEngine(settings).run({"BTC/USDT": frame}, [scan])
    altered = BacktestEngine(settings).run(
        {"BTC/USDT": _corrupt_after(frame, cut)}, [scan]
    )

    earlier = [
        trade
        for trade in baseline.trades
        if trade.exit_index is not None and trade.exit_index < cut
    ]
    compared = [
        trade
        for trade in altered.trades
        if trade.exit_index is not None and trade.exit_index < cut
    ]

    assert len(earlier) == 2
    assert earlier == compared


def test_the_corruption_actually_changes_later_outcomes(
    make_frame, make_signal
) -> None:
    """Guard against a vacuous sweep: the altered bars must matter."""
    frame = make_frame([QUIET] * 8)
    scan = _scan([make_signal(0, stop=95.0)])

    baseline = BacktestEngine(_settings()).run({"BTC/USDT": frame}, [scan])
    altered = BacktestEngine(_settings()).run(
        {"BTC/USDT": _corrupt_after(frame, 3)}, [scan]
    )

    assert baseline.trades[0].exit_reason is ExitReason.MAX_HOLDING
    assert altered.trades[0].exit_reason is ExitReason.STOP_LOSS


def test_every_entry_strictly_follows_the_signal(make_frame, make_signal) -> None:
    frame = make_frame([QUIET] * 6)
    scan = _scan([make_signal(index, stop=95.0) for index in range(4)])

    result = BacktestEngine(_settings()).run({"BTC/USDT": frame}, [scan])

    assert result.trades
    for trade in result.trades:
        assert trade.entry_time > trade.signal_time
        assert trade.entry_index > trade.signal_index


def test_no_trade_exits_before_it_enters(make_frame, make_signal) -> None:
    frame = make_frame([QUIET] * 8)
    scan = _scan([make_signal(0, stop=95.0), make_signal(2, stop=95.0)])

    result = BacktestEngine(_settings()).run({"BTC/USDT": frame}, [scan])

    for trade in result.trades:
        assert trade.exit_index is not None
        assert trade.exit_index >= trade.entry_index
        assert trade.holding_bars >= 1


def test_a_signal_on_the_final_bar_cannot_be_traded(make_frame, make_signal) -> None:
    frame = make_frame([QUIET] * 4)
    scan = _scan([make_signal(3, stop=95.0)])

    result = BacktestEngine(_settings()).run({"BTC/USDT": frame}, [scan])

    assert result.trades == ()
    assert result.skipped[0].reason.value == "no_entry_bar"


def test_the_window_bounds_what_the_simulation_can_read(
    make_frame, make_signal
) -> None:
    """A stop that only triggers after the window must not close the trade.

    This is the purging guarantee: without it a development run would quietly
    read the holdout to decide how a trade ended.
    """
    frame = make_frame([*[QUIET] * 3, (100.0, 101.0, 50.0, 60.0)])
    scan = _scan([make_signal(0, stop=95.0)])
    end = START + 3 * DELTA

    bounded = BacktestEngine(_settings(window_end=end, max_holding_bars=20)).run(
        {"BTC/USDT": frame}, [scan]
    )
    unbounded = BacktestEngine(_settings(max_holding_bars=20)).run(
        {"BTC/USDT": frame}, [scan]
    )

    assert bounded.trades[0].exit_reason is ExitReason.CENSORED
    assert unbounded.trades[0].exit_reason is ExitReason.STOP_LOSS


def test_a_censored_trade_is_excluded_from_metrics(make_frame, make_signal) -> None:
    frame = make_frame([*[QUIET] * 3, (100.0, 101.0, 50.0, 60.0)])
    scan = _scan([make_signal(0, stop=95.0)])

    result = BacktestEngine(
        _settings(window_end=START + 3 * DELTA, max_holding_bars=20)
    ).run({"BTC/USDT": frame}, [scan])

    assert result.metrics.total_trades == 0
    assert result.metrics.censored_trades == 1
    assert result.usable_trades == ()


def test_the_fill_price_comes_from_a_later_bar_than_the_signal(
    make_frame, make_signal
) -> None:
    """The signal bar's own prices must be absent from the entry."""
    signal_bar = (100.0, 110.0, 90.0, 105.0)
    frame = make_frame([signal_bar, (200.0, 210.0, 190.0, 205.0)])
    scan = _scan([make_signal(0, stop=95.0, signal_price=105.0)])

    result = BacktestEngine(
        BacktestSettings(fee_bps=0.0, slippage_bps=0.0, execution_delay_bars=1)
    ).run({"BTC/USDT": frame}, [scan])

    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(200.0)
    assert trade.signal_price == pytest.approx(105.0)
    assert trade.entry_price not in signal_bar
