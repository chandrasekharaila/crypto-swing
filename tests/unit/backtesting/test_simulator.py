"""Fill-model tests.

Every number here is worked out by hand from the configured costs and the
explicit candles, so a change in behaviour shows up as a failing expectation
rather than as a silently different result.
"""

import pytest

from crypto_analyzer.backtesting.config import BacktestSettings
from crypto_analyzer.backtesting.simulator import simulate_trade
from crypto_analyzer.backtesting.types import (
    AmbiguousExitPolicy,
    ExitReason,
    SkippedSignal,
    SkipReason,
    TradeRecord,
)
from crypto_analyzer.taxonomy import Direction

QUIET = [(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)]


def _trade(outcome) -> TradeRecord:
    assert isinstance(outcome, TradeRecord), outcome
    return outcome


def _skipped(outcome) -> SkippedSignal:
    assert isinstance(outcome, SkippedSignal), outcome
    return outcome


def test_entry_fills_at_the_next_bar_open(
    make_frame, make_signal, free_settings, series_of
) -> None:
    """The price that triggered the signal must never be the price we get."""
    frame = make_frame([(100.0, 110.0, 90.0, 105.0), (200.0, 210.0, 190.0, 205.0)])
    signal = make_signal(0, stop=95.0, signal_price=105.0)

    trade = _trade(
        simulate_trade(signal, 0, series_of(frame), free_settings, last_index=1)
    )

    assert trade.entry_price == pytest.approx(200.0)
    assert trade.entry_price not in {105.0, 110.0, 90.0}
    assert trade.entry_time == frame["open_time"].iloc[1]


def test_entry_is_always_after_the_signal_was_known(
    make_frame, make_signal, free_settings, series_of
) -> None:
    frame = make_frame(QUIET)
    signal = make_signal(0, stop=95.0)

    trade = _trade(
        simulate_trade(signal, 0, series_of(frame), free_settings, last_index=1)
    )

    assert trade.entry_time > signal.available_at
    assert trade.entry_index > trade.signal_index


def test_stop_touch_fills_at_the_stop(
    make_frame, make_signal, priced_settings, series_of
) -> None:
    frame = make_frame([*QUIET, (100.0, 101.0, 94.0, 98.0)])
    signal = make_signal(0, stop=95.0)

    trade = _trade(
        simulate_trade(signal, 0, series_of(frame), priced_settings, last_index=2)
    )

    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_price == pytest.approx(95.0)
    # gross -5%, net of a 10bps round trip, per unit of notional
    assert trade.gross_return == pytest.approx(-0.05)
    assert trade.fees == pytest.approx(0.002)
    assert trade.net_return == pytest.approx(-0.052)
    # risk was 5% of price, so a stop-out is a little worse than -1R
    assert trade.r_multiple == pytest.approx(-1.04)
    assert trade.holding_bars == 2


def test_target_touch_fills_at_the_target(
    make_frame, make_signal, priced_settings, series_of
) -> None:
    frame = make_frame([*QUIET, (100.0, 112.0, 99.0, 111.0)])
    signal = make_signal(0, stop=95.0)

    trade = _trade(
        simulate_trade(signal, 0, series_of(frame), priced_settings, last_index=2)
    )

    assert trade.exit_reason is ExitReason.TAKE_PROFIT
    assert trade.target_price == pytest.approx(110.0)
    assert trade.gross_return == pytest.approx(0.10)
    assert trade.net_return == pytest.approx(0.098)
    assert trade.r_multiple == pytest.approx(1.96)


def test_a_gap_through_the_stop_fills_at_the_open(
    make_frame, make_signal, priced_settings, series_of
) -> None:
    """The single most common way a backtest overstates performance."""
    frame = make_frame([*QUIET, (80.0, 82.0, 75.0, 78.0)])
    signal = make_signal(0, stop=95.0)

    trade = _trade(
        simulate_trade(signal, 0, series_of(frame), priced_settings, last_index=2)
    )

    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_price == pytest.approx(80.0)
    assert trade.exit_price < trade.stop_price
    assert trade.r_multiple < -1.0


def test_both_levels_in_one_bar_resolves_pessimistically(
    make_frame, make_signal, series_of
) -> None:
    frame = make_frame([*QUIET, (100.0, 112.0, 94.0, 105.0)])
    signal = make_signal(0, stop=95.0)

    stop_first = _trade(
        simulate_trade(
            signal,
            0,
            series_of(frame),
            BacktestSettings(fee_bps=0.0, slippage_bps=0.0),
            last_index=2,
        )
    )
    target_first = _trade(
        simulate_trade(
            signal,
            0,
            series_of(frame),
            BacktestSettings(
                fee_bps=0.0,
                slippage_bps=0.0,
                ambiguous_exit_policy=AmbiguousExitPolicy.TARGET_FIRST,
            ),
            last_index=2,
        )
    )

    assert stop_first.exit_reason is ExitReason.STOP_LOSS
    assert stop_first.ambiguous
    assert target_first.exit_reason is ExitReason.TAKE_PROFIT
    assert stop_first.exit_price < target_first.exit_price


def test_a_stop_on_the_wrong_side_is_rejected(
    make_frame, make_signal, free_settings, series_of
) -> None:
    frame = make_frame(QUIET)
    signal = make_signal(0, stop=105.0)

    outcome = simulate_trade(signal, 0, series_of(frame), free_settings, last_index=1)

    assert _skipped(outcome).reason is SkipReason.INVALID_STOP


def test_a_missing_stop_is_rejected(
    make_frame, make_signal, free_settings, series_of
) -> None:
    frame = make_frame(QUIET)
    signal = make_signal(0, stop=None)

    outcome = simulate_trade(signal, 0, series_of(frame), free_settings, last_index=1)

    assert _skipped(outcome).reason is SkipReason.INVALID_STOP


def test_a_signal_on_the_final_bar_produces_no_trade(
    make_frame, make_signal, free_settings, series_of
) -> None:
    frame = make_frame(QUIET)
    signal = make_signal(1, stop=95.0)

    outcome = simulate_trade(signal, 1, series_of(frame), free_settings, last_index=1)

    assert _skipped(outcome).reason is SkipReason.NO_ENTRY_BAR


def test_the_holding_limit_closes_at_the_close(
    make_frame, make_signal, free_settings, series_of
) -> None:
    frame = make_frame([(100.0, 101.0, 99.0, 100.0)] * 6)
    signal = make_signal(0, stop=95.0)
    settings = BacktestSettings(
        fee_bps=0.0, slippage_bps=0.0, max_holding_bars=2, take_profit_r_multiple=2.0
    )

    trade = _trade(simulate_trade(signal, 0, series_of(frame), settings, last_index=5))

    assert trade.exit_reason is ExitReason.MAX_HOLDING
    assert trade.exit_index == 3
    assert trade.holding_bars == 3


def test_a_trade_still_open_at_the_window_end_is_censored(
    make_frame, make_signal, free_settings, series_of
) -> None:
    frame = make_frame([(100.0, 101.0, 99.0, 100.0)] * 6)
    signal = make_signal(0, stop=95.0)

    trade = _trade(
        simulate_trade(signal, 0, series_of(frame), free_settings, last_index=2)
    )

    assert trade.exit_reason is ExitReason.CENSORED
    assert not trade.is_usable
    assert not trade.is_win


def test_slippage_is_adverse_on_both_sides(make_frame, make_signal, series_of) -> None:
    frame = make_frame([*QUIET, (105.0, 120.0, 104.0, 118.0)])
    signal = make_signal(0, stop=95.0)
    settings = BacktestSettings(
        fee_bps=0.0, slippage_bps=100.0, take_profit_r_multiple=2.0
    )

    trade = _trade(simulate_trade(signal, 0, series_of(frame), settings, last_index=2))

    # entry slips up, the target is recomputed from the slipped fill, and the
    # exit slips back down, so the shortfall against the target compounds.
    assert trade.entry_price == pytest.approx(101.0)
    assert trade.risk_per_unit == pytest.approx(6.0)
    assert trade.target_price == pytest.approx(113.0)
    assert trade.exit_price == pytest.approx(113.0 * 0.99)
    assert trade.exit_price < trade.target_price
    assert trade.slippage_cost > 0.0
    assert trade.net_return < trade.gross_return


def test_a_short_mirrors_the_long_geometry(
    make_frame, make_signal, priced_settings, series_of
) -> None:
    frame = make_frame([*QUIET, (94.0, 95.0, 88.0, 90.0)])
    signal = make_signal(0, direction=Direction.SHORT, stop=105.0)

    trade = _trade(
        simulate_trade(signal, 0, series_of(frame), priced_settings, last_index=2)
    )

    assert trade.target_price == pytest.approx(90.0)
    assert trade.exit_reason is ExitReason.TAKE_PROFIT
    assert trade.gross_return == pytest.approx(0.10)
    assert trade.r_multiple > 0.0


def test_excursions_are_recorded(
    make_frame, make_signal, priced_settings, series_of
) -> None:
    frame = make_frame([*QUIET, (100.0, 112.0, 94.0, 111.0)])
    signal = make_signal(0, stop=95.0)

    trade = _trade(
        simulate_trade(signal, 0, series_of(frame), priced_settings, last_index=2)
    )

    assert trade.mae == pytest.approx(94.0 / 100.0 - 1.0)
    assert trade.mfe == pytest.approx(112.0 / 100.0 - 1.0)
