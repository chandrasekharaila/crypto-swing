"""Tests for trailing volatility features."""

import pandas as pd
import pytest

from crypto_analyzer.features import volatility
from crypto_analyzer.features.primitives import true_range


def test_true_range_uses_the_previous_close(make_frame) -> None:
    frame = make_frame(
        [
            (10.0, 12.0, 9.0, 11.0, 1.0),
            (11.0, 15.0, 10.0, 14.0, 1.0),
        ]
    )

    series = true_range(frame)

    assert series.iloc[0] == pytest.approx(3.0)
    assert series.iloc[1] == pytest.approx(5.0)


def test_atr_averages_the_true_range_over_its_period(make_close_frame) -> None:
    frame = make_close_frame([11.0] * 6)

    series = volatility.atr(frame, 3)

    assert pd.isna(series.iloc[:2]).all()
    assert series.iloc[2:].eq(4.0).all()


def test_atr_pct_is_atr_relative_to_close(make_close_frame) -> None:
    frame = make_close_frame([11.0] * 6)

    absolute = volatility.atr(frame, 3)
    relative = volatility.atr_pct(frame, 3)

    assert relative.iloc[-1] == pytest.approx(absolute.iloc[-1] / 11.0)


def test_return_std_needs_a_return_before_it_warms_up(make_close_frame) -> None:
    frame = make_close_frame([100.0, 105.0, 100.0, 110.0, 100.0])

    series = volatility.return_std(frame, 2)

    assert pd.isna(series.iloc[1])
    assert (series.iloc[2:] > 0.0).all()


def test_bollinger_bands_have_positive_width_once_warm(make_close_frame) -> None:
    frame = make_close_frame([10.0, 12.0, 14.0, 16.0, 18.0])

    width = volatility.bollinger_width(frame, 3, 2.0)

    assert pd.isna(width.iloc[:2]).all()
    assert (width.iloc[2:] > 0.0).all()


def test_bollinger_position_is_half_when_close_sits_on_the_middle_band(
    make_close_frame,
) -> None:
    frame = make_close_frame([10.0, 20.0, 10.0, 15.0])

    position = volatility.bollinger_position(frame, 3, 2.0)

    assert position.iloc[3] == pytest.approx(0.5)


def test_bollinger_width_collapses_for_a_constant_series(make_close_frame) -> None:
    frame = make_close_frame([10.0] * 5)

    width = volatility.bollinger_width(frame, 3, 2.0)

    assert width.iloc[-1] == 0.0
