"""Portfolio assembly tests against hand-calculated equity and drawdown."""

import pandas as pd
import pytest

from crypto_analyzer.backtesting.config import BacktestSettings
from crypto_analyzer.backtesting.portfolio import CURVE_COLUMNS, build_equity_curve
from crypto_analyzer.backtesting.types import ExitReason, SkipReason
from crypto_analyzer.taxonomy import Direction, SetupType

FLAT = [(100.0, 101.0, 99.0, 100.0)] * 10


def _prices(make_frame, symbols=("BTC/USDT",)):
    frame = make_frame(FLAT)
    series = frame.set_index("open_time")["close"].astype("float64")
    return dict.fromkeys(symbols, series)


def _settings(**overrides) -> BacktestSettings:
    base = {
        "initial_equity": 100_000.0,
        "risk_fraction": 0.01,
        "max_notional_multiple": 1.0,
        "max_concurrent_positions": 5,
        "fee_bps": 10.0,
        "slippage_bps": 0.0,
        "execution_delay_bars": 1,
    }
    base.update(overrides)
    return BacktestSettings(**base)


def test_hand_calculated_equity_after_one_win(make_frame, make_trade) -> None:
    """Risk 1% with a 5 point stop sizes 200 units on a 100 price."""
    trade = make_trade(entry_index=1, exit_index=3, net_return=0.098, r_multiple=1.96)

    curve, skipped = build_equity_curve([trade], _prices(make_frame), _settings())

    # 200 units x 10 points = 2000 gross, less 200 x 100 x 0.002 fees = 1960
    assert curve["equity"].iloc[-1] == pytest.approx(101_960.0)
    assert skipped == ()
    assert tuple(curve.columns) == CURVE_COLUMNS


def test_the_notional_cap_bounds_a_very_tight_stop(make_frame, make_trade) -> None:
    """Without the cap a 0.1 point stop would size 10,000 units instead of 1,000."""
    trade = make_trade(
        entry_index=1,
        exit_index=3,
        entry_price=100.0,
        exit_price=110.0,
        stop_price=99.9,
        risk_per_unit=0.1,
        fees=0.002,
    )

    curve, _ = build_equity_curve([trade], _prices(make_frame), _settings())

    # 1,000 units x 10 points = 10,000 gross, less 1,000 x 100 x 0.002 = 200
    assert curve["equity"].iloc[-1] == pytest.approx(109_800.0)


def test_the_concurrency_cap_skips_and_records(make_frame, make_trade) -> None:
    symbols = [f"SYM{index}/USDT" for index in range(6)]
    trades = [
        make_trade(symbol=symbol, entry_index=1, exit_index=3) for symbol in symbols
    ]

    curve, skipped = build_equity_curve(
        trades, _prices(make_frame, symbols), _settings(max_concurrent_positions=5)
    )

    assert len(skipped) == 1
    assert skipped[0].reason is SkipReason.POSITION_LIMIT
    assert curve["open_positions"].max() == 5


def test_one_position_per_symbol_is_enforced(make_frame, make_trade) -> None:
    first = make_trade(entry_index=1, exit_index=4)
    overlapping = make_trade(entry_index=2, exit_index=5)

    _, skipped = build_equity_curve(
        [first, overlapping], _prices(make_frame), _settings()
    )

    assert len(skipped) == 1
    assert skipped[0].reason is SkipReason.SYMBOL_BUSY


def test_overlapping_trades_are_allowed_when_configured(make_frame, make_trade) -> None:
    first = make_trade(entry_index=1, exit_index=4)
    overlapping = make_trade(entry_index=2, exit_index=5)

    _, skipped = build_equity_curve(
        [first, overlapping],
        _prices(make_frame),
        _settings(one_position_per_symbol=False),
    )

    assert skipped == ()


def test_entries_stop_once_the_account_is_ruined(make_frame, make_trade) -> None:
    wipeout = make_trade(
        entry_index=1,
        exit_index=2,
        entry_price=1_000.0,
        exit_price=0.0,
        stop_price=999.0,
        risk_per_unit=1.0,
        fees=0.0,
    )
    later = make_trade(entry_index=4, exit_index=6)

    curve, skipped = build_equity_curve(
        [wipeout, later], _prices(make_frame), _settings()
    )

    assert skipped[0].reason is SkipReason.ACCOUNT_RUINED
    assert curve["equity"].min() == pytest.approx(0.0)


def test_drawdown_tracks_the_running_peak(make_frame, make_trade) -> None:
    """The curve is driven by prices, so the loss has to be expressed in prices."""
    winner = make_trade(
        entry_index=1,
        exit_index=3,
        entry_price=100.0,
        exit_price=110.0,
        stop_price=95.0,
        risk_per_unit=5.0,
        fees=0.0,
    )
    loser = make_trade(
        entry_index=4,
        exit_index=6,
        entry_price=100.0,
        exit_price=20.0,
        stop_price=99.0,
        risk_per_unit=1.0,
        fees=0.0,
    )

    curve, _ = build_equity_curve([winner, loser], _prices(make_frame), _settings())

    assert curve["peak"].is_monotonic_increasing
    assert (curve["drawdown"] <= 0.0).all()
    assert curve["peak"].iloc[-1] == pytest.approx(curve["peak"].max())
    # the peak is reached on the exit bar, so no drawdown is visible before it
    assert curve["drawdown"].iloc[:2].eq(0.0).all()
    assert curve["drawdown"].min() < -0.5


def test_censored_trades_are_left_out_of_the_curve(make_frame, make_trade) -> None:
    censored = make_trade(entry_index=1, exit_index=3, exit_reason=ExitReason.CENSORED)

    curve, skipped = build_equity_curve([censored], _prices(make_frame), _settings())

    assert curve.empty
    assert skipped == ()


def test_the_curve_is_deterministic(make_frame, make_trade) -> None:
    trades = [
        make_trade(symbol=f"SYM{index}/USDT", entry_index=1, exit_index=3)
        for index in range(3)
    ]
    prices = _prices(make_frame, [f"SYM{index}/USDT" for index in range(3)])

    first, _ = build_equity_curve(trades, prices, _settings())
    second, _ = build_equity_curve(trades, prices, _settings())

    pd.testing.assert_frame_equal(first, second)


def test_direction_affects_the_pnl_sign(make_frame, make_trade) -> None:
    short = make_trade(
        direction=Direction.SHORT,
        setup=SetupType.MEAN_REVERSION,
        entry_index=1,
        exit_index=3,
        entry_price=100.0,
        exit_price=90.0,
    )

    curve, _ = build_equity_curve([short], _prices(make_frame), _settings())

    assert curve["equity"].iloc[-1] > 100_000.0
