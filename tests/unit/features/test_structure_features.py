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


def test_higher_high_flags_a_rising_window_extreme(make_frame) -> None:
    frame = make_frame(
        [
            (10.0, 12.0, 9.0, 11.0, 1.0),
            (11.0, 13.0, 10.0, 12.0, 1.0),
            (12.0, 20.0, 11.0, 19.0, 1.0),
            (19.0, 21.0, 18.0, 20.0, 1.0),
        ]
    )

    series = structure.higher_high(frame, 2)

    # Window highs are 13 (rows 0-1) and 21 (rows 2-3), so the latest is higher.
    assert pd.isna(series.iloc[:3]).all()
    assert series.iloc[3] == 1.0


def test_higher_low_flags_a_rising_window_trough(make_frame) -> None:
    frame = make_frame(
        [
            (10.0, 12.0, 9.0, 11.0, 1.0),
            (11.0, 13.0, 10.0, 12.0, 1.0),
            (12.0, 20.0, 11.0, 19.0, 1.0),
            (19.0, 21.0, 18.0, 20.0, 1.0),
        ]
    )

    series = structure.higher_low(frame, 2)

    # Window lows are 9 (rows 0-1) and 11 (rows 2-3).
    assert series.iloc[3] == 1.0


def test_lower_low_flags_a_falling_window_trough(make_frame) -> None:
    frame = make_frame(
        [
            (20.0, 21.0, 18.0, 19.0, 1.0),
            (19.0, 20.0, 15.0, 16.0, 1.0),
            (16.0, 17.0, 14.0, 15.0, 1.0),
            (15.0, 16.0, 8.0, 9.0, 1.0),
        ]
    )

    series = structure.lower_low(frame, 2)

    # Window lows are 15 (rows 0-1) and 8 (rows 2-3).
    assert series.iloc[3] == 1.0


def test_lower_high_flags_a_falling_window_extreme(make_frame) -> None:
    frame = make_frame(
        [
            (20.0, 25.0, 18.0, 19.0, 1.0),
            (19.0, 24.0, 15.0, 16.0, 1.0),
            (16.0, 22.0, 14.0, 15.0, 1.0),
            (15.0, 21.0, 8.0, 9.0, 1.0),
        ]
    )

    series = structure.lower_high(frame, 2)

    # Window highs are 25 (rows 0-1) and 22 (rows 2-3).
    assert series.iloc[3] == 1.0


def test_swing_labels_need_two_complete_windows(make_frame) -> None:
    frame = make_frame([(10.0, 11.0, 9.0, 10.0, 1.0)] * 8)

    series = structure.higher_high(frame, 3)

    assert pd.isna(series.iloc[:5]).all()
    assert series.iloc[5:].notna().all()


def test_swing_labels_are_defined_rather_than_missing_on_flat_input(
    make_frame,
) -> None:
    """Equal windows compare False, which is a real answer, not a missing one."""
    frame = make_frame([(10.0, 11.0, 9.0, 10.0, 1.0)] * 8)

    series = structure.lower_low(frame, 3)

    assert series.iloc[5:].eq(0.0).all()


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
