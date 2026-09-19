"""Tests for trailing momentum oscillators."""

import pandas as pd
import pytest

from crypto_analyzer.features import momentum, trend


def test_rsi_reaches_100_when_every_candle_gains(make_close_frame) -> None:
    frame = make_close_frame([100.0 + index for index in range(20)])

    series = momentum.rsi(frame, 14)

    assert pd.isna(series.iloc[13])
    assert series.iloc[-1] == pytest.approx(100.0)


def test_rsi_reaches_0_when_every_candle_loses(make_close_frame) -> None:
    frame = make_close_frame([100.0 - index for index in range(20)])

    series = momentum.rsi(frame, 14)

    assert pd.isna(series.iloc[13])
    assert series.iloc[-1] == pytest.approx(0.0)


def test_rsi_stays_within_its_bounds_on_mixed_input(make_close_frame) -> None:
    close = [100.0 + (5.0 if index % 3 else -4.0) for index in range(1, 60)]
    frame = make_close_frame([100.0, *close])

    series = momentum.rsi(frame, 14).dropna()

    assert series.between(0.0, 100.0).all()


def test_macd_line_is_the_difference_of_the_two_averages(make_close_frame) -> None:
    frame = make_close_frame([100.0 + index for index in range(40)])

    series = momentum.macd_line(frame, 12, 26)

    expected = (
        trend.exponential_ma(frame, 12).iloc[-1]
        - trend.exponential_ma(frame, 26).iloc[-1]
    )
    assert series.iloc[-1] == pytest.approx(expected)


def test_macd_histogram_is_the_line_minus_its_signal(make_close_frame) -> None:
    frame = make_close_frame([100.0 + index + (index % 5) for index in range(60)])

    line = momentum.macd_line(frame, 12, 26)
    signal = momentum.macd_signal(frame, 12, 26, 9)
    histogram = momentum.macd_histogram(frame, 12, 26, 9)

    pd.testing.assert_series_equal(histogram, line - signal)


def test_rate_of_change_is_expressed_as_a_percentage(make_close_frame) -> None:
    frame = make_close_frame([100.0, 110.0])

    series = momentum.rate_of_change(frame, 1)

    assert pd.isna(series.iloc[0])
    assert series.iloc[1] == pytest.approx(10.0)


def test_stochastic_k_compares_close_to_the_window_extremes(make_frame) -> None:
    frame = make_frame(
        [
            (9.0, 10.0, 8.0, 9.5, 1.0),
            (9.5, 11.0, 9.0, 10.5, 1.0),
            (10.5, 13.0, 10.0, 12.0, 1.0),
        ]
    )

    series = momentum.stochastic_k(frame, 3)

    assert pd.isna(series.iloc[1])
    assert series.iloc[2] == pytest.approx((12.0 - 8.0) / (13.0 - 8.0) * 100.0)


def test_stochastic_k_is_flat_at_the_midpoint_when_range_is_zero(make_frame) -> None:
    frame = make_frame([(10.0, 10.0, 10.0, 10.0, 1.0)] * 4)

    series = momentum.stochastic_k(frame, 3)

    assert series.iloc[3] == 0.0


def test_stochastic_d_smooths_the_oscillator(make_close_frame) -> None:
    frame = make_close_frame([100.0 + (index % 7) for index in range(40)])

    oscillator = momentum.stochastic_k(frame, 5)
    smoothed = momentum.stochastic_d(frame, 5, 3)

    assert pd.isna(smoothed.iloc[:6]).all()
    assert smoothed.iloc[-1] == pytest.approx(oscillator.iloc[-3:].mean())
