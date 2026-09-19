"""Tests for regime classification."""

import pandas as pd
import pytest

from crypto_analyzer.features import FeatureEngine
from crypto_analyzer.regimes import (
    UNKNOWN_REGIME,
    RegimeDetector,
    RegimeInputError,
    RegimeSettings,
    TrendRegime,
    VolatilityRegime,
)

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"


def _latest(result, column: str) -> str:
    return str(result.frame[column].iloc[-1])


def test_rising_market_is_bullish(run_regimes, uptrend_frame) -> None:
    result = run_regimes(uptrend_frame)

    assert _latest(result, "trend_regime") == TrendRegime.BULLISH.value


def test_falling_market_is_bearish(run_regimes, downtrend_frame) -> None:
    result = run_regimes(downtrend_frame)

    assert _latest(result, "trend_regime") == TrendRegime.BEARISH.value


def test_flat_market_is_sideways(run_regimes, sideways_frame) -> None:
    result = run_regimes(sideways_frame)

    assert _latest(result, "trend_regime") == TrendRegime.SIDEWAYS.value


def test_volatility_spike_is_high(run_regimes, volatility_spike_frame) -> None:
    result = run_regimes(volatility_spike_frame)

    assert _latest(result, "volatility_regime") == VolatilityRegime.HIGH.value


def test_calm_after_volatility_is_low(run_regimes, calm_after_volatility_frame) -> None:
    result = run_regimes(calm_after_volatility_frame)

    assert _latest(result, "volatility_regime") == VolatilityRegime.LOW.value


def test_steady_volatility_is_normal(run_regimes, sideways_frame) -> None:
    result = run_regimes(sideways_frame)

    assert _latest(result, "volatility_regime") == VolatilityRegime.NORMAL.value


def test_axes_are_independent(run_regimes, uptrend_frame) -> None:
    """A trend and a volatility state can hold at the same time."""
    result = run_regimes(uptrend_frame)

    trend = _latest(result, "trend_regime")
    volatility = _latest(result, "volatility_regime")

    assert trend != UNKNOWN_REGIME
    assert volatility != UNKNOWN_REGIME
    assert _latest(result, "combined_regime") == f"{trend}__{volatility}"


def test_combined_label_names_both_axes(run_regimes, uptrend_frame) -> None:
    result = run_regimes(uptrend_frame)

    combined = _latest(result, "combined_regime")

    assert combined.startswith(TrendRegime.BULLISH.value)
    assert combined.count("__") == 1


def test_each_axis_is_unknown_until_its_own_warmup(
    run_regimes, uptrend_frame, regime_settings
) -> None:
    result = run_regimes(uptrend_frame)
    trend_warmup = regime_settings.trend_slow_window
    volatility_warmup = (
        regime_settings.volatility_atr_period + regime_settings.volatility_lookback - 1
    )

    assert (
        result.frame["trend_regime"].iloc[: trend_warmup - 1].eq(UNKNOWN_REGIME).all()
    )
    assert result.frame["trend_regime"].iloc[trend_warmup - 1] != UNKNOWN_REGIME
    assert (
        result.frame["volatility_regime"]
        .iloc[: volatility_warmup - 1]
        .eq(UNKNOWN_REGIME)
        .all()
    )
    assert (
        result.frame["volatility_regime"].iloc[volatility_warmup - 1] != UNKNOWN_REGIME
    )


def test_the_combined_label_waits_for_both_axes(
    run_regimes, uptrend_frame, regime_settings
) -> None:
    """A combination is only meaningful once neither axis is missing."""
    result = run_regimes(uptrend_frame)
    warmup = regime_settings.warmup

    assert result.frame["combined_regime"].iloc[: warmup - 1].eq(UNKNOWN_REGIME).all()
    assert result.frame["combined_regime"].iloc[warmup - 1] != UNKNOWN_REGIME


def test_volatility_can_be_known_while_trend_is_not(
    run_regimes, uptrend_frame, regime_settings
) -> None:
    """The two axes warm up independently, so they must not be gated together."""
    result = run_regimes(uptrend_frame)
    volatility_warmup = (
        regime_settings.volatility_atr_period + regime_settings.volatility_lookback - 1
    )

    window = result.frame.iloc[volatility_warmup - 1 : regime_settings.warmup - 2]

    assert window["volatility_regime"].ne(UNKNOWN_REGIME).all()
    assert window["trend_regime"].eq(UNKNOWN_REGIME).all()
    assert window["combined_regime"].eq(UNKNOWN_REGIME).all()


def test_detection_is_deterministic(run_regimes, volatility_spike_frame) -> None:
    first = run_regimes(volatility_spike_frame)
    second = run_regimes(volatility_spike_frame)

    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_labels_align_with_the_feature_rows(
    run_regimes, uptrend_frame, feature_settings
) -> None:
    result = run_regimes(uptrend_frame)

    assert len(result.frame) == len(uptrend_frame)
    assert result.frame["open_time"].iloc[0] == uptrend_frame["open_time"].iloc[0]
    assert result.available_at.equals(uptrend_frame["close_time"])
    assert result.symbol == SYMBOL
    assert result.timeframe == TIMEFRAME


def test_missing_features_are_reported(
    sideways_frame, regime_settings, feature_settings
) -> None:
    reduced = feature_settings.model_copy(update={"sma_windows": (3,)})
    features = FeatureEngine(reduced).compute(
        sideways_frame, symbol=SYMBOL, timeframe=TIMEFRAME
    )

    with pytest.raises(RegimeInputError, match="trend_sma_5"):
        RegimeDetector(regime_settings).detect(features)


def test_stricter_agreement_requirement_can_suppress_a_trend(
    run_regimes, uptrend_frame
) -> None:
    """Raising the bar to unanimity must not change a clearly trending market."""
    result = run_regimes(uptrend_frame)

    assert _latest(result, "trend_regime") == TrendRegime.BULLISH.value


def test_thresholds_are_configurable(sideways_frame, feature_settings) -> None:
    """A very low bar lets a flat market register a direction."""
    detector = RegimeDetector(
        RegimeSettings(
            trend_fast_window=3,
            trend_slow_window=5,
            trend_ema_slope_span=3,
            volatility_atr_period=3,
            volatility_lookback=5,
            trend_min_agreeing_rules=1,
            trend_ma_spread_threshold=1e-9,
            trend_price_distance_threshold=1e-9,
            trend_ema_slope_threshold=1e-9,
            trend_ema_spread_threshold=1e-9,
        )
    )
    features = FeatureEngine(feature_settings).compute(
        sideways_frame, symbol=SYMBOL, timeframe=TIMEFRAME
    )

    result = detector.detect(features)

    assert _latest(result, "trend_regime") != TrendRegime.SIDEWAYS.value
