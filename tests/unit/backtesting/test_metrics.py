"""Metric tests against values computed by hand."""

import math

import pandas as pd
import pytest

from crypto_analyzer.backtesting.metrics import bars_per_year, compute_metrics
from crypto_analyzer.backtesting.types import ExitReason
from crypto_analyzer.taxonomy import Direction, SetupType

PERIODS = 6 * 365.25  # 4h bars in a year


def _equity(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="4h", tz="UTC")
    equity = pd.Series(values, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "equity": equity,
            "peak": equity.cummax(),
            "drawdown": equity / equity.cummax() - 1.0,
            "open_positions": pd.Series([0] * len(values), index=index),
        }
    )


def _report(trades, values):
    return compute_metrics(
        trades, _equity(values), skipped_signals=0, periods_per_year=PERIODS
    )


def test_hand_calculated_trade_statistics(make_trade) -> None:
    trades = [
        make_trade(entry_index=1, exit_index=2, net_return=0.02, r_multiple=0.4),
        make_trade(entry_index=3, exit_index=4, net_return=-0.01, r_multiple=-0.2),
        make_trade(entry_index=5, exit_index=6, net_return=0.03, r_multiple=0.6),
        make_trade(entry_index=7, exit_index=8, net_return=-0.01, r_multiple=-0.2),
    ]

    report = _report(trades, [100.0, 100.0])

    assert report.total_trades == 4
    assert report.win_rate == pytest.approx(0.5)
    assert report.average_return == pytest.approx(0.0075)
    # sorted returns are -0.01, -0.01, 0.02, 0.03
    assert report.median_return == pytest.approx(0.005)
    # gains 0.05 against losses 0.02
    assert report.profit_factor == pytest.approx(2.5)
    assert report.expectancy == pytest.approx(0.15)
    assert report.average_r_multiple == pytest.approx(0.15)


def test_profit_factor_is_infinite_without_losses(make_trade) -> None:
    trades = [make_trade(entry_index=1, exit_index=2, net_return=0.02)]

    report = _report(trades, [100.0, 101.0])

    assert report.profit_factor == math.inf
    assert report.win_rate == 1.0


def test_no_trades_produces_zeros(make_trade) -> None:
    report = _report([], [100.0, 100.0])

    assert report.total_trades == 0
    assert report.win_rate == 0.0
    assert report.profit_factor == 0.0
    assert report.expectancy == 0.0


def test_censored_trades_are_excluded(make_trade) -> None:
    trades = [
        make_trade(entry_index=1, exit_index=2, net_return=0.02),
        make_trade(
            entry_index=3,
            exit_index=4,
            net_return=-0.50,
            exit_reason=ExitReason.CENSORED,
        ),
    ]

    report = _report(trades, [100.0, 102.0])

    assert report.total_trades == 1
    assert report.censored_trades == 1
    assert report.average_return == pytest.approx(0.02)
    assert dict(report.exit_counts)["censored"] == 1


def test_hand_calculated_drawdown_and_return(make_trade) -> None:
    trades = [make_trade(entry_index=1, exit_index=2)]

    report = _report(trades, [100.0, 110.0, 99.0, 120.0])

    # the only below-peak bar is 99 against a peak of 110
    assert report.max_drawdown == pytest.approx(-0.1)
    assert report.max_drawdown_bars == 1
    assert report.total_return == pytest.approx(0.20)


def test_consecutive_runs_are_counted(make_trade) -> None:
    outcomes = [0.01, 0.01, 0.01, -0.01, -0.01, 0.01, -0.01]
    trades = [
        make_trade(
            entry_index=2 * index + 1,
            exit_index=2 * index + 2,
            net_return=value,
            r_multiple=value * 20,
        )
        for index, value in enumerate(outcomes)
    ]

    report = _report(trades, [100.0, 100.0])

    assert report.max_consecutive_wins == 3
    assert report.max_consecutive_losses == 2


def test_sharpe_is_zero_for_a_flat_curve(make_trade) -> None:
    report = _report([], [100.0] * 5)

    assert report.sharpe_ratio == 0.0


def test_ruin_is_reported(make_trade) -> None:
    report = _report([make_trade(entry_index=1, exit_index=2)], [100.0, -5.0])

    assert report.account_ruined


def test_ambiguous_exits_are_counted(make_trade) -> None:
    trades = [
        make_trade(entry_index=1, exit_index=2, ambiguous=True),
        make_trade(entry_index=3, exit_index=4),
    ]

    assert _report(trades, [100.0, 100.0]).ambiguous_exits == 1


def test_bars_per_year_matches_the_timeframe() -> None:
    assert bars_per_year("4h") == pytest.approx(6 * 365.25)
    assert bars_per_year("1d") == pytest.approx(365.25)


def test_average_holding_period_is_reported(make_trade) -> None:
    trades = [
        make_trade(entry_index=1, exit_index=3),  # 3 bars
        make_trade(entry_index=5, exit_index=5),  # 1 bar
    ]

    assert _report(trades, [100.0, 100.0]).average_holding_bars == pytest.approx(2.0)


def test_report_serializes_for_experiment_records(make_trade) -> None:
    import json

    payload = _report([make_trade()], [100.0, 110.0]).to_dict()

    json.dumps(payload)
    assert payload["total_trades"] == 1
    assert "account_ruined" in payload


def test_setup_and_direction_filters(make_trade) -> None:
    trades = [
        make_trade(entry_index=1, exit_index=2, setup=SetupType.BREAKOUT),
        make_trade(
            entry_index=3,
            exit_index=4,
            setup=SetupType.MEAN_REVERSION,
            direction=Direction.SHORT,
        ),
    ]

    report = _report(trades, [100.0, 100.0])

    assert report.total_trades == 2
