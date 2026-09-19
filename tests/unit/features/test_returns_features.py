"""Tests for trailing return features."""

import math

import pandas as pd
import pytest

from crypto_analyzer.features import returns


def test_simple_returns_are_fractional_close_changes(make_close_frame) -> None:
    frame = make_close_frame([100.0, 110.0, 121.0])

    series = returns.simple_returns(frame, 1)

    assert pd.isna(series.iloc[0])
    assert series.iloc[1] == pytest.approx(0.10)
    assert series.iloc[2] == pytest.approx(0.10)


def test_log_returns_are_natural_logs(make_close_frame) -> None:
    frame = make_close_frame([100.0, 110.0])

    series = returns.log_returns(frame, 1)

    assert pd.isna(series.iloc[0])
    assert series.iloc[1] == pytest.approx(math.log(1.1))


def test_multi_candle_returns_use_the_shifted_window(make_close_frame) -> None:
    frame = make_close_frame([100.0, 105.0, 110.0, 120.0])

    series = returns.simple_returns(frame, 2)

    assert pd.isna(series.iloc[1])
    assert series.iloc[2] == pytest.approx(0.10)
    assert series.iloc[3] == pytest.approx(120.0 / 105.0 - 1.0)


def test_log_returns_preserve_missing_values_during_warmup(make_close_frame) -> None:
    frame = make_close_frame([100.0, 105.0, 110.0])

    series = returns.log_returns(frame, 2)

    assert pd.isna(series.iloc[:2]).all()
    assert series.iloc[2] == pytest.approx(math.log(1.10))
