"""Tests for resolving a bar's internal path with finer candles."""

from datetime import timedelta

import pandas as pd
import pytest

from crypto_analyzer.backtesting import (
    BacktestEngine,
    BacktestSettings,
    ExitReason,
    IntrabarWindows,
    resolution_from_minutes,
)
from crypto_analyzer.backtesting.simulator import simulate_trade
from crypto_analyzer.backtesting.types import AmbiguousExitPolicy
from crypto_analyzer.setups.types import SetupScan
from crypto_analyzer.taxonomy import Direction

from .conftest import DELTA, START, TIMEFRAME

QUIET = (100.0, 101.0, 99.0, 100.0)
AMBIGUOUS = (100.0, 112.0, 94.0, 105.0)


def _minutes(rows):
    index = pd.date_range(START, periods=len(rows), freq="1min")
    return pd.DataFrame(
        {
            "open_time": index,
            "high": [row[0] for row in rows],
            "low": [row[1] for row in rows],
        }
    )


class _StubResolver:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def resolve(self, symbol, bar_open_time, direction, stop, target):
        self.calls.append((symbol, bar_open_time, direction, stop, target))
        return self.outcome


def test_the_stop_is_reported_when_the_finer_candles_reach_it_first() -> None:
    """The low dips to the stop before the high reaches the target."""
    highs = [101.0, 106.0, 108.0]
    lows = [99.0, 95.0, 100.0]

    outcome = resolution_from_minutes(
        highs, lows, direction=Direction.LONG, stop=95.0, target=110.0
    )

    assert outcome is ExitReason.STOP_LOSS


def test_the_target_is_reported_when_the_finer_candles_reach_it_first() -> None:
    highs = [101.0, 111.0, 108.0]
    lows = [99.0, 97.0, 94.0]

    outcome = resolution_from_minutes(
        highs, lows, direction=Direction.LONG, stop=95.0, target=110.0
    )

    assert outcome is ExitReason.TAKE_PROFIT


def test_a_single_finer_candle_holding_both_levels_is_still_ambiguous() -> None:
    """The same problem one scale down, so the assumption still applies."""
    highs = [112.0]
    lows = [94.0]

    outcome = resolution_from_minutes(
        highs, lows, direction=Direction.LONG, stop=95.0, target=110.0
    )

    assert outcome is None


def test_a_short_is_resolved_the_same_way() -> None:
    highs = [104.0, 104.5, 103.0]
    lows = [99.0, 98.0, 90.0]

    outcome = resolution_from_minutes(
        highs, lows, direction=Direction.SHORT, stop=105.0, target=90.0
    )

    assert outcome is ExitReason.TAKE_PROFIT


def test_nothing_reached_reports_no_decision() -> None:
    outcome = resolution_from_minutes(
        [101.0], [99.0], direction=Direction.LONG, stop=95.0, target=110.0
    )

    assert outcome is None


def test_windows_resolve_a_cached_bar() -> None:
    bar_open = START + DELTA
    windows = IntrabarWindows.from_frames(
        {
            ("BTC/USDT", bar_open): _minutes([(101.0, 99.0), (106.0, 95.0)]),
        }
    )

    outcome = windows.resolve(
        "BTC/USDT", bar_open, Direction.LONG, stop=95.0, target=110.0
    )

    assert outcome is ExitReason.STOP_LOSS
    assert (windows.attempts, windows.resolved) == (1, 1)


def test_an_uncached_bar_reports_no_decision() -> None:
    windows = IntrabarWindows.from_frames({})

    outcome = windows.resolve(
        "BTC/USDT", START, Direction.LONG, stop=95.0, target=110.0
    )

    assert outcome is None
    assert windows.attempts == 0


def test_windows_round_trip_through_parquet(tmp_path) -> None:
    """Minute ordering must survive the round trip, or the answer changes."""
    bar_open = START + DELTA
    windows = IntrabarWindows.from_frames(
        {("BTC/USDT", bar_open): _minutes([(101.0, 94.0), (112.0, 99.0)])}
    )
    path = tmp_path / "intrabar.parquet"

    windows.write_parquet(path)
    reloaded = IntrabarWindows.from_parquet(path)

    assert len(reloaded) == 1
    assert (
        reloaded.resolve("BTC/USDT", bar_open, Direction.LONG, stop=95.0, target=110.0)
        is ExitReason.STOP_LOSS
    )


def test_a_missing_cache_file_is_not_an_error(tmp_path) -> None:
    windows = IntrabarWindows.from_parquet(tmp_path / "absent.parquet")

    assert len(windows) == 0


def test_the_simulator_uses_the_resolver_on_an_ambiguous_bar(
    make_frame, make_signal, series_of
) -> None:
    frame = make_frame([QUIET, QUIET, AMBIGUOUS])
    signal = make_signal(0, stop=95.0)
    resolver = _StubResolver(ExitReason.TAKE_PROFIT)

    trade = simulate_trade(
        signal,
        0,
        series_of(frame),
        BacktestSettings(fee_bps=0.0, slippage_bps=0.0),
        last_index=2,
        intrabar=resolver,
    )

    assert trade.exit_reason is ExitReason.TAKE_PROFIT
    assert trade.ambiguous
    assert trade.resolved_intrabar
    assert len(resolver.calls) == 1


def test_the_simulator_falls_back_when_the_resolver_cannot_decide(
    make_frame, make_signal, series_of
) -> None:
    frame = make_frame([QUIET, QUIET, AMBIGUOUS])
    signal = make_signal(0, stop=95.0)
    resolver = _StubResolver(None)

    trade = simulate_trade(
        signal,
        0,
        series_of(frame),
        BacktestSettings(
            fee_bps=0.0,
            slippage_bps=0.0,
            ambiguous_exit_policy=AmbiguousExitPolicy.STOP_FIRST,
        ),
        last_index=2,
        intrabar=resolver,
    )

    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.ambiguous
    assert not trade.resolved_intrabar


def test_an_unresolved_bar_is_policy_driven_when_no_resolver_is_supplied(
    make_frame, make_signal, series_of
) -> None:
    frame = make_frame([QUIET, QUIET, AMBIGUOUS])
    signal = make_signal(0, stop=95.0)

    trade = simulate_trade(
        signal,
        0,
        series_of(frame),
        BacktestSettings(fee_bps=0.0, slippage_bps=0.0),
        last_index=2,
    )

    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert not trade.resolved_intrabar


def test_the_policy_still_moves_the_result_without_finer_data(
    make_frame, make_signal, series_of
) -> None:
    """The assumption remains the fallback, and it is still the thing to avoid."""
    frame = make_frame([QUIET, QUIET, AMBIGUOUS])
    signal = make_signal(0, stop=95.0)

    def run(policy):
        return simulate_trade(
            signal,
            0,
            series_of(frame),
            BacktestSettings(
                fee_bps=0.0, slippage_bps=0.0, ambiguous_exit_policy=policy
            ),
            last_index=2,
        )

    assert run(AmbiguousExitPolicy.STOP_FIRST).exit_reason is ExitReason.STOP_LOSS
    assert run(AmbiguousExitPolicy.TARGET_FIRST).exit_reason is ExitReason.TAKE_PROFIT


def test_the_engine_passes_the_resolver_through(make_frame, make_signal) -> None:
    frame = make_frame([QUIET, QUIET, AMBIGUOUS, QUIET])
    scan = SetupScan(
        symbol="BTC/USDT",
        timeframe=TIMEFRAME,
        signals=(make_signal(0, stop=95.0),),
    )
    resolver = _StubResolver(ExitReason.TAKE_PROFIT)

    result = BacktestEngine(BacktestSettings(fee_bps=0.0, slippage_bps=0.0)).run(
        {"BTC/USDT": frame}, [scan], intrabar=resolver
    )

    assert result.trades[0].exit_reason is ExitReason.TAKE_PROFIT
    assert result.trades[0].resolved_intrabar


def test_the_resolver_never_sees_a_bar_after_the_window(
    make_frame, make_signal
) -> None:
    frame = make_frame([QUIET, QUIET, AMBIGUOUS])
    scan = SetupScan(
        symbol="BTC/USDT",
        timeframe=TIMEFRAME,
        signals=(make_signal(0, stop=95.0),),
    )
    resolver = _StubResolver(ExitReason.TAKE_PROFIT)

    BacktestEngine(
        BacktestSettings(fee_bps=0.0, slippage_bps=0.0, window_end=START + 2 * DELTA)
    ).run({"BTC/USDT": frame}, [scan], intrabar=resolver)

    assert resolver.calls == []


def test_windows_reject_a_frame_without_the_expected_columns() -> None:
    from crypto_analyzer.backtesting import BacktestInputError

    with pytest.raises(BacktestInputError, match="missing columns"):
        IntrabarWindows(pd.DataFrame({"symbol": ["BTC/USDT"]}))


def test_resolution_is_deterministic() -> None:
    highs = [101.0, 112.0]
    lows = [99.0, 94.0]
    first = resolution_from_minutes(
        highs, lows, direction=Direction.LONG, stop=95.0, target=110.0
    )
    second = resolution_from_minutes(
        highs, lows, direction=Direction.LONG, stop=95.0, target=110.0
    )

    assert first is second


def test_a_one_bar_window_shorter_than_expected_still_works() -> None:
    """A window with a single finer candle is a legitimate, if coarse, answer."""
    windows = IntrabarWindows.from_frames(
        {("BTC/USDT", START): _minutes([(101.0, 99.0)])}
    )

    assert (
        windows.resolve("BTC/USDT", START, Direction.LONG, stop=95.0, target=110.0)
        is None
    )


def test_bar_open_time_matching_tolerates_timestamp_types() -> None:
    """The cache is keyed by timestamp, and pandas and datetime must agree."""
    bar_open = START + DELTA
    windows = IntrabarWindows.from_frames(
        {("BTC/USDT", bar_open): _minutes([(101.0, 94.0)])}
    )

    outcome = windows.resolve(
        "BTC/USDT",
        bar_open + timedelta(0),
        Direction.LONG,
        stop=95.0,
        target=110.0,
    )

    assert outcome is ExitReason.STOP_LOSS
