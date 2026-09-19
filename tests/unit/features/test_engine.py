"""Tests for the feature engine's assembly, validation, and determinism."""

import logging
from decimal import Decimal

import pandas as pd
import pytest

from crypto_analyzer.features import (
    FeatureConfigurationError,
    FeatureEngine,
    FeatureGroup,
    FeatureInputError,
    FeatureSettings,
)

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "close_time": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )


def test_frame_starts_with_open_time_then_every_registered_feature(wave_frame) -> None:
    engine = FeatureEngine()

    result = engine.compute(wave_frame, symbol=SYMBOL, timeframe=TIMEFRAME)

    assert result.frame.columns[0] == "open_time"
    assert tuple(result.frame.columns[1:]) == engine.registry.names
    assert len(result.frame) == len(wave_frame)


def test_available_at_is_the_candle_close_boundary(wave_frame) -> None:
    result = FeatureEngine().compute(wave_frame, symbol=SYMBOL, timeframe=TIMEFRAME)

    pd.testing.assert_series_equal(
        result.available_at, wave_frame["close_time"], check_names=False
    )
    assert (result.available_at > result.frame["open_time"]).all()


def test_engine_does_not_modify_the_input_frame(wave_frame) -> None:
    original = wave_frame.copy(deep=True)

    FeatureEngine().compute(wave_frame, symbol=SYMBOL, timeframe=TIMEFRAME)

    pd.testing.assert_frame_equal(wave_frame, original)


def test_engine_accepts_exact_decimal_price_columns(make_close_frame) -> None:
    frame = make_close_frame([100.0, 101.0, 102.0, 103.0])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].map(Decimal)
    assert frame["close"].dtype == object

    result = FeatureEngine().compute(frame, symbol=SYMBOL, timeframe=TIMEFRAME)

    assert result.frame["price_range_pct"].notna().all()


def test_engine_is_deterministic(wave_frame) -> None:
    engine = FeatureEngine()

    first = engine.compute(wave_frame, symbol=SYMBOL, timeframe=TIMEFRAME)
    second = engine.compute(wave_frame, symbol=SYMBOL, timeframe=TIMEFRAME)

    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_only_configured_groups_are_computed(wave_frame) -> None:
    settings = FeatureSettings(enabled_groups=(FeatureGroup.VOLATILITY,))

    result = FeatureEngine(settings).compute(
        wave_frame, symbol=SYMBOL, timeframe=TIMEFRAME
    )

    assert result.metadata
    assert all(meta.group is FeatureGroup.VOLATILITY for meta in result.metadata)


def test_engine_rejects_a_frame_missing_required_columns(wave_frame) -> None:
    broken = wave_frame.drop(columns=["volume"])

    with pytest.raises(FeatureInputError, match="missing_columns"):
        FeatureEngine().compute(broken, symbol=SYMBOL, timeframe=TIMEFRAME)


def test_engine_rejects_unordered_candles(wave_frame) -> None:
    reversed_frame = wave_frame.iloc[::-1].reset_index(drop=True)

    with pytest.raises(FeatureInputError, match="non_chronological_order"):
        FeatureEngine().compute(reversed_frame, symbol=SYMBOL, timeframe=TIMEFRAME)


def test_engine_warns_when_there_is_not_enough_history(
    make_close_frame, caplog
) -> None:
    frame = make_close_frame([100.0 + index for index in range(10)])

    with caplog.at_level(logging.WARNING):
        FeatureEngine().compute(frame, symbol=SYMBOL, timeframe=TIMEFRAME)

    assert "required before every feature has a value" in caplog.text


def test_empty_frame_produces_an_empty_feature_set() -> None:
    engine = FeatureEngine()

    result = engine.compute(_empty_frame(), symbol=SYMBOL, timeframe=TIMEFRAME)

    assert result.frame.empty
    assert len(result.frame.columns) == len(engine.registry.names) + 1
    assert result.available_at.empty


def test_feature_set_exposes_metadata_by_name(wave_frame) -> None:
    result = FeatureEngine().compute(wave_frame, symbol=SYMBOL, timeframe=TIMEFRAME)

    metadata = result.metadata_for("trend_sma_20")

    assert metadata.group is FeatureGroup.TREND
    assert metadata.lookback == 20
    assert result.feature_names == tuple(result.frame.columns[1:])


def test_unknown_feature_lookup_fails(wave_frame) -> None:
    result = FeatureEngine().compute(wave_frame, symbol=SYMBOL, timeframe=TIMEFRAME)

    with pytest.raises(FeatureConfigurationError, match="unknown feature"):
        result.metadata_for("not_a_feature")
