"""Orchestration tests for the backtest engine."""

import pandas as pd
import pytest

from crypto_analyzer.backtesting import (
    BacktestEngine,
    BacktestInputError,
    BacktestSettings,
    ExitReason,
    SkipReason,
)
from crypto_analyzer.setups.types import SetupScan
from crypto_analyzer.taxonomy import SetupType

from .conftest import DELTA, START, TIMEFRAME

QUIET = (100.0, 101.0, 99.0, 100.0)


def _scan(symbol, signals):
    return SetupScan(symbol=symbol, timeframe=TIMEFRAME, signals=tuple(signals))


def _settings(**overrides) -> BacktestSettings:
    base = {"fee_bps": 10.0, "slippage_bps": 0.0, "take_profit_r_multiple": 2.0}
    base.update(overrides)
    return BacktestSettings(**base)


def test_one_signal_produces_one_trade(make_frame, make_signal) -> None:
    frame = make_frame([QUIET, QUIET, (100.0, 112.0, 99.0, 111.0), QUIET])
    signal = make_signal(0, stop=95.0)

    result = BacktestEngine(_settings()).run(
        {"BTC/USDT": frame}, [_scan("BTC/USDT", [signal])]
    )

    assert len(result.trades) == 1
    assert result.metrics.total_trades == 1
    assert result.trades[0].exit_reason is ExitReason.TAKE_PROFIT
    assert result.trades[0].r_multiple == pytest.approx(1.96)
    assert result.skipped == ()


def test_a_signal_before_the_window_is_skipped(make_frame, make_signal) -> None:
    frame = make_frame([QUIET] * 6)
    signal = make_signal(0, stop=95.0)

    result = BacktestEngine(_settings(window_start=START + 2 * DELTA)).run(
        {"BTC/USDT": frame}, [_scan("BTC/USDT", [signal])]
    )

    assert result.trades == ()
    assert result.skipped[0].reason is SkipReason.OUTSIDE_WINDOW


def test_a_signal_after_the_window_is_skipped(make_frame, make_signal) -> None:
    frame = make_frame([QUIET] * 6)
    signal = make_signal(4, stop=95.0)

    result = BacktestEngine(_settings(window_end=START + 2 * DELTA)).run(
        {"BTC/USDT": frame}, [_scan("BTC/USDT", [signal])]
    )

    assert result.trades == ()
    assert result.skipped[0].reason is SkipReason.OUTSIDE_WINDOW


def test_the_window_is_recorded_on_the_result(make_frame, make_signal) -> None:
    frame = make_frame([QUIET] * 6)
    start = START + DELTA
    end = START + 5 * DELTA

    result = BacktestEngine(_settings(window_start=start, window_end=end)).run(
        {"BTC/USDT": frame}, [_scan("BTC/USDT", [make_signal(1, stop=95.0)])]
    )

    assert result.window_start == start
    assert result.window_end == end


def test_trades_can_be_filtered(make_frame, make_signal) -> None:
    target_bar = (100.0, 112.0, 99.0, 111.0)
    frame = make_frame([QUIET, QUIET, target_bar, QUIET, target_bar, QUIET])
    signals = [
        make_signal(0, stop=95.0, setup=SetupType.BREAKOUT),
        make_signal(3, stop=95.0, setup=SetupType.MEAN_REVERSION),
    ]

    result = BacktestEngine(_settings()).run(
        {"BTC/USDT": frame}, [_scan("BTC/USDT", signals)]
    )

    assert len(result.for_setup(SetupType.BREAKOUT)) == 1
    assert len(result.for_setup(SetupType.MEAN_REVERSION)) == 1


def test_no_scans_is_rejected() -> None:
    with pytest.raises(BacktestInputError, match="at least one"):
        BacktestEngine().run({}, [])


def test_a_duplicated_symbol_is_rejected(make_frame, make_signal) -> None:
    frame = make_frame([QUIET] * 4)
    scans = [
        _scan("BTC/USDT", [make_signal(0, stop=95.0)]),
        _scan("BTC/USDT", [make_signal(1, stop=95.0)]),
    ]

    with pytest.raises(BacktestInputError, match="only one scan"):
        BacktestEngine().run({"BTC/USDT": frame}, scans)


def test_mixed_timeframes_are_rejected(make_frame, make_signal) -> None:
    frame = make_frame([QUIET] * 4)
    other = SetupScan(
        symbol="ETH/USDT", timeframe="1h", signals=(make_signal(0, stop=95.0),)
    )

    with pytest.raises(BacktestInputError, match="one timeframe"):
        BacktestEngine().run(
            {"BTC/USDT": frame, "ETH/USDT": frame},
            [_scan("BTC/USDT", [make_signal(0, stop=95.0)]), other],
        )


def test_missing_candles_are_reported(make_frame, make_signal) -> None:
    with pytest.raises(BacktestInputError, match="no candles"):
        BacktestEngine().run({}, [_scan("BTC/USDT", [make_signal(0, stop=95.0)])])


def test_a_signal_with_no_matching_candle_is_reported(make_frame, make_signal) -> None:
    """A signal must line up with a bar rather than being silently placed."""
    frame = make_frame([QUIET] * 2)
    stray = make_signal(5, stop=95.0)

    with pytest.raises(BacktestInputError, match="does not match a candle"):
        BacktestEngine().run({"BTC/USDT": frame}, [_scan("BTC/USDT", [stray])])


def test_the_run_is_deterministic(make_frame, make_signal) -> None:
    frame = make_frame([QUIET, QUIET, (100.0, 112.0, 99.0, 111.0), QUIET])
    scan = _scan("BTC/USDT", [make_signal(0, stop=95.0)])

    first = BacktestEngine(_settings()).run({"BTC/USDT": frame}, [scan])
    second = BacktestEngine(_settings()).run({"BTC/USDT": frame}, [scan])

    assert first.trades == second.trades
    pd.testing.assert_frame_equal(first.equity, second.equity)
    assert first.metrics == second.metrics


def test_multiple_symbols_share_one_curve(make_frame, make_signal) -> None:
    frame = make_frame([QUIET, QUIET, (100.0, 112.0, 99.0, 111.0), QUIET])
    candles = {"BTC/USDT": frame, "ETH/USDT": frame}
    scans = [
        _scan("BTC/USDT", [make_signal(0, stop=95.0, symbol="BTC/USDT")]),
        _scan("ETH/USDT", [make_signal(0, stop=95.0, symbol="ETH/USDT")]),
    ]

    result = BacktestEngine(_settings()).run(candles, scans)

    assert result.metrics.total_trades == 2
    assert result.metrics.skipped_signals == 0
