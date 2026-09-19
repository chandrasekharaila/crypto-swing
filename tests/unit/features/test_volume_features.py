"""Tests for trailing volume features."""

import pandas as pd
import pytest

from crypto_analyzer.features import volume

THREE_CANDLES = [
    (10.0, 11.0, 9.0, 10.0, 100.0),
    (10.0, 11.0, 9.0, 10.0, 200.0),
    (10.0, 11.0, 9.0, 10.0, 300.0),
]


def test_volume_ma_is_the_trailing_mean(make_frame) -> None:
    frame = make_frame(THREE_CANDLES)

    series = volume.volume_ma(frame, 3)

    assert pd.isna(series.iloc[1])
    assert series.iloc[2] == pytest.approx(200.0)


def test_relative_volume_compares_volume_to_its_average(make_frame) -> None:
    frame = make_frame(THREE_CANDLES)

    series = volume.relative_volume(frame, 3)

    assert pd.isna(series.iloc[1])
    assert series.iloc[2] == pytest.approx(1.5)


def test_volume_change_is_a_fractional_change(make_frame) -> None:
    frame = make_frame(
        [
            (10.0, 11.0, 9.0, 10.0, 100.0),
            (10.0, 11.0, 9.0, 10.0, 150.0),
        ]
    )

    series = volume.volume_change(frame, 1)

    assert pd.isna(series.iloc[0])
    assert series.iloc[1] == pytest.approx(0.5)


def test_signed_volume_negates_bearish_candles(make_frame) -> None:
    frame = make_frame(
        [
            (9.0, 11.0, 8.0, 10.0, 100.0),
            (11.0, 12.0, 9.0, 10.0, 300.0),
        ]
    )

    series = volume.signed_volume_mean(frame, 2)

    assert series.iloc[1] == pytest.approx(-100.0)


def test_signed_volume_is_zero_for_a_doji(make_frame) -> None:
    frame = make_frame(
        [
            (10.0, 11.0, 9.0, 10.0, 100.0),
            (10.0, 11.0, 9.0, 10.0, 300.0),
        ]
    )

    series = volume.signed_volume_mean(frame, 2)

    assert series.iloc[1] == pytest.approx(0.0)


def test_volume_close_correlation_is_one_when_both_rise_together(
    make_close_frame,
) -> None:
    frame = make_close_frame([10.0 + index for index in range(10)])

    series = volume.volume_close_corr(frame, 5)

    assert series.iloc[-1] == pytest.approx(1.0)
