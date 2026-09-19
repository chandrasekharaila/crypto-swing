"""Tests for single-candle price-shape features."""

import pandas as pd
import pytest

from crypto_analyzer.features import price

BULLISH = (100.0, 110.0, 90.0, 105.0, 10.0)


def test_body_ratio_is_body_over_range(make_frame) -> None:
    frame = make_frame([BULLISH])

    assert price.body_ratio(frame).iloc[0] == pytest.approx(5.0 / 20.0)


def test_close_location_is_close_position_within_the_range(make_frame) -> None:
    frame = make_frame([BULLISH])

    assert price.close_location(frame).iloc[0] == pytest.approx(15.0 / 20.0)


def test_wick_ratios_measure_the_parts_outside_the_body(make_frame) -> None:
    frame = make_frame([BULLISH])

    assert price.upper_wick_ratio(frame).iloc[0] == pytest.approx(5.0 / 20.0)
    assert price.lower_wick_ratio(frame).iloc[0] == pytest.approx(10.0 / 20.0)


def test_body_and_wicks_partition_the_range(make_frame) -> None:
    frame = make_frame([BULLISH])

    total = (
        price.body_ratio(frame).iloc[0]
        + price.upper_wick_ratio(frame).iloc[0]
        + price.lower_wick_ratio(frame).iloc[0]
    )

    assert total == pytest.approx(1.0)


def test_range_pct_is_range_over_close(make_frame) -> None:
    frame = make_frame([BULLISH])

    assert price.range_pct(frame).iloc[0] == pytest.approx(20.0 / 105.0)


@pytest.mark.parametrize(
    "compute",
    [
        price.body_ratio,
        price.range_pct,
        price.upper_wick_ratio,
        price.lower_wick_ratio,
    ],
)
def test_flat_candle_reports_zero_for_magnitude_ratios(make_frame, compute) -> None:
    frame = make_frame([(100.0, 100.0, 100.0, 100.0, 5.0)])

    value = compute(frame).iloc[0]

    assert not pd.isna(value)
    assert value == 0.0


def test_flat_candle_reports_the_midpoint_for_close_location(make_frame) -> None:
    """A candle with no range has no position, so the neutral midpoint applies."""
    frame = make_frame([(100.0, 100.0, 100.0, 100.0, 5.0)])

    value = price.close_location(frame).iloc[0]

    assert not pd.isna(value)
    assert value == 0.5


def test_gap_pct_compares_open_to_the_previous_close(make_frame) -> None:
    frame = make_frame(
        [
            (100.0, 105.0, 95.0, 100.0, 1.0),
            (110.0, 115.0, 105.0, 112.0, 1.0),
        ]
    )

    series = price.gap_pct(frame)

    assert pd.isna(series.iloc[0])
    assert series.iloc[1] == pytest.approx(0.10)


def test_price_features_preserve_the_input_index(make_frame) -> None:
    frame = make_frame([BULLISH] * 4)

    series = price.body_ratio(frame)

    assert series.index.equals(frame.index)
    assert len(series) == len(frame)
