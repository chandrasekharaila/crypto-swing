"""Tests for trailing trend features."""

import pandas as pd
import pytest

from crypto_analyzer.features import trend


def test_sma_is_the_trailing_mean_of_the_window(make_close_frame) -> None:
    frame = make_close_frame([1.0, 2.0, 3.0, 4.0])

    series = trend.sma(frame, 3)

    assert pd.isna(series.iloc[:2]).all()
    assert series.iloc[2] == pytest.approx(2.0)
    assert series.iloc[3] == pytest.approx(3.0)


def test_ema_follows_the_recursive_definition(make_close_frame) -> None:
    frame = make_close_frame([10.0, 20.0, 30.0, 40.0])
    alpha = 2.0 / 3.0

    series = trend.exponential_ma(frame, 2)

    assert pd.isna(series.iloc[0])
    seeded = alpha * 20.0 + (1.0 - alpha) * 10.0
    assert series.iloc[1] == pytest.approx(seeded)
    assert series.iloc[2] == pytest.approx(alpha * 30.0 + (1.0 - alpha) * seeded)


def test_price_vs_sma_is_the_relative_distance(make_close_frame) -> None:
    frame = make_close_frame([10.0, 20.0, 30.0])

    series = trend.price_vs_sma(frame, 2)

    assert series.iloc[2] == pytest.approx((30.0 - 25.0) / 30.0)


def test_ema_slope_is_zero_for_a_constant_series(make_close_frame) -> None:
    frame = make_close_frame([50.0] * 12)

    series = trend.ema_slope(frame, 3, 2)

    assert series.dropna().eq(0.0).all()
    assert series.dropna().size > 0


def test_ema_spread_is_positive_when_an_uptrend_lifts_the_fast_average(
    make_close_frame,
) -> None:
    frame = make_close_frame([100.0 + 5.0 * index for index in range(40)])

    series = trend.ema_spread_pct(frame, 3, 10)

    assert series.iloc[-1] > 0.0


def test_trend_features_never_read_beyond_the_current_row(make_close_frame) -> None:
    frame = make_close_frame([10.0, 20.0, 30.0, 40.0, 50.0])

    series = trend.sma(frame, 2)

    assert len(series) == len(frame)
    assert series.index.equals(frame.index)
