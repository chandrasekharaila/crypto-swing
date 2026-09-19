"""Tests for trailing market-structure features."""

import pandas as pd
import pytest

from crypto_analyzer.features import structure


def test_recent_high_and_low_are_trailing_extremes(make_frame) -> None:
    frame = make_frame(
        [
            (10.0, 12.0, 9.0, 11.0, 1.0),
            (11.0, 15.0, 10.0, 14.0, 1.0),
            (14.0, 16.0, 13.0, 15.0, 1.0),
        ]
    )

    assert structure.recent_high(frame, 2).iloc[1] == pytest.approx(15.0)
    assert structure.recent_low(frame, 2).iloc[2] == pytest.approx(10.0)


def test_breakout_up_uses_only_candles_before_the_current_one(make_frame) -> None:
    frame = make_frame(
        [
            (10.0, 12.0, 9.0, 11.0, 1.0),
            (11.0, 13.0, 10.0, 12.0, 1.0),
            (12.0, 20.0, 11.0, 19.0, 1.0),
        ]
    )

    series = structure.breakout_up(frame, 2)

    assert pd.isna(series.iloc[1])
    # The reference is max(12, 13) = 13, not the current candle's own high of 20.
    assert series.iloc[2] == 1.0


def test_breakout_up_stays_zero_when_close_remains_below_prior_highs(
    make_frame,
) -> None:
    frame = make_frame(
        [
            (10.0, 20.0, 9.0, 11.0, 1.0),
            (11.0, 21.0, 10.0, 12.0, 1.0),
            (12.0, 13.0, 11.0, 12.5, 1.0),
        ]
    )

    series = structure.breakout_up(frame, 2)

    assert series.iloc[2] == 0.0


def test_breakout_down_uses_only_candles_before_the_current_one(make_frame) -> None:
    frame = make_frame(
        [
            (10.0, 12.0, 9.0, 11.0, 1.0),
            (11.0, 13.0, 10.0, 12.0, 1.0),
            (12.0, 13.0, 5.0, 6.0, 1.0),
        ]
    )

    series = structure.breakout_down(frame, 2)

    # The reference is min(9, 10) = 9, not the current candle's own low of 5.
    assert series.iloc[2] == 1.0


def test_breakout_features_are_missing_during_warmup(make_frame) -> None:
    frame = make_frame([(10.0, 11.0, 9.0, 10.0, 1.0)] * 6)

    series = structure.breakout_up(frame, 3)

    assert pd.isna(series.iloc[:3]).all()
    assert series.iloc[3:].notna().all()


def test_high_and_low_distance_are_relative_to_the_trailing_extremes(
    make_frame,
) -> None:
    frame = make_frame([(10.0, 12.0, 8.0, 10.0, 1.0)])

    assert structure.high_distance(frame, 1).iloc[0] == pytest.approx(10.0 / 12.0 - 1.0)
    assert structure.low_distance(frame, 1).iloc[0] == pytest.approx(10.0 / 8.0 - 1.0)


def test_consolidation_range_is_the_span_relative_to_close(make_frame) -> None:
    frame = make_frame([(10.0, 12.0, 8.0, 10.0, 1.0)])

    series = structure.consolidation_range(frame, 1)

    assert series.iloc[0] == pytest.approx(4.0 / 10.0)
