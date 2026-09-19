"""Rule tests for each setup family, driven by exact feature values."""

import pytest

from crypto_analyzer.features.types import FeatureSet
from crypto_analyzer.setups import (
    Direction,
    SetupDetector,
    SetupInputError,
    SetupSettings,
    SetupType,
)


def _continuation_inputs(build_inputs, regime: str, rows: int = 6):
    return build_inputs(
        trend_regimes=[regime] * rows,
        overrides={
            "trend_price_vs_sma_3": [-0.02] * rows,
            "trend_ema_slope_3": [0.001] * rows,
            "momentum_rsi_3": [50.0] * rows,
            "volume_relative_3": [1.2] * rows,
        },
    )


def test_pullback_in_an_uptrend_is_a_long_continuation(detector, build_inputs) -> None:
    candles, features, regimes = _continuation_inputs(build_inputs, "bullish_trend")

    scan = detector.detect(features, candles, regimes)
    signals = scan.for_setup(SetupType.TREND_CONTINUATION)

    assert len(signals) == 6
    assert all(signal.direction is Direction.LONG for signal in signals)


def test_the_same_pullback_is_ignored_outside_a_trend(detector, build_inputs) -> None:
    candles, features, regimes = _continuation_inputs(
        build_inputs, "sideways_consolidation"
    )

    scan = detector.detect(features, candles, regimes)

    assert scan.for_setup(SetupType.TREND_CONTINUATION) == ()


def test_continuation_needs_a_pullback(detector, build_inputs) -> None:
    """Price extended above its average is not a pullback, so it is not a setup."""
    rows = 6
    candles, features, regimes = build_inputs(
        trend_regimes=["bullish_trend"] * rows,
        overrides={
            "trend_price_vs_sma_3": [0.08] * rows,
            "trend_ema_slope_3": [0.001] * rows,
            "momentum_rsi_3": [50.0] * rows,
            "volume_relative_3": [1.2] * rows,
        },
    )

    scan = detector.detect(features, candles, regimes)

    assert scan.for_setup(SetupType.TREND_CONTINUATION) == ()


def test_continuation_needs_volume(detector, build_inputs) -> None:
    rows = 6
    candles, features, regimes = build_inputs(
        trend_regimes=["bullish_trend"] * rows,
        overrides={
            "trend_price_vs_sma_3": [-0.02] * rows,
            "trend_ema_slope_3": [0.001] * rows,
            "momentum_rsi_3": [50.0] * rows,
            "volume_relative_3": [0.5] * rows,
        },
    )

    scan = detector.detect(features, candles, regimes)

    assert scan.for_setup(SetupType.TREND_CONTINUATION) == ()


def test_a_bearish_rally_is_a_short_continuation(detector, build_inputs) -> None:
    rows = 6
    candles, features, regimes = build_inputs(
        trend_regimes=["bearish_trend"] * rows,
        overrides={
            "trend_price_vs_sma_3": [0.02] * rows,
            "trend_ema_slope_3": [-0.001] * rows,
            "momentum_rsi_3": [50.0] * rows,
            "volume_relative_3": [1.2] * rows,
        },
    )

    scan = detector.detect(features, candles, regimes)
    signals = scan.for_setup(SetupType.TREND_CONTINUATION)

    assert len(signals) == 6
    assert all(signal.direction is Direction.SHORT for signal in signals)


BREAKOUT_WIDTH = [0.10] * 5 + [0.02] * 5


def _breakout_inputs(build_inputs, *, volume: float = 2.0, width=None):
    rows = 10
    return build_inputs(
        trend_regimes=["sideways_consolidation"] * rows,
        overrides={
            "volatility_bollinger_width_3": width or BREAKOUT_WIDTH,
            "structure_breakout_up_3": [0.0] * 9 + [1.0],
            "volume_relative_3": [1.0] * 9 + [volume],
        },
        close=[100.0] * 9 + [111.1],
    )


def test_squeeze_breakout_with_volume_is_a_long_breakout(
    detector, build_inputs
) -> None:
    candles, features, regimes = _breakout_inputs(build_inputs)

    scan = detector.detect(features, candles, regimes)
    signals = scan.for_setup(SetupType.BREAKOUT)

    assert len(signals) == 1
    assert signals[0].direction is Direction.LONG
    assert signals[0].timestamp == features.frame["open_time"].iloc[9].to_pydatetime()


def test_breakout_needs_volume_expansion(detector, build_inputs) -> None:
    candles, features, regimes = _breakout_inputs(build_inputs, volume=1.0)

    scan = detector.detect(features, candles, regimes)

    assert scan.for_setup(SetupType.BREAKOUT) == ()


def test_breakout_needs_a_squeeze(detector, build_inputs) -> None:
    """A break from an already-wide band is not the setup being described."""
    candles, features, regimes = _breakout_inputs(build_inputs, width=[0.10] * 10)

    scan = detector.detect(features, candles, regimes)

    assert scan.for_setup(SetupType.BREAKOUT) == ()


def test_breakout_is_rejected_when_price_is_already_extended(
    detector, build_inputs
) -> None:
    rows = 10
    candles, features, regimes = build_inputs(
        trend_regimes=["sideways_consolidation"] * rows,
        overrides={
            "volatility_bollinger_width_3": BREAKOUT_WIDTH,
            "structure_breakout_up_3": [0.0] * 9 + [1.0],
            "volume_relative_3": [1.0] * 9 + [2.0],
        },
        close=[100.0] * 9 + [130.0],
    )

    scan = detector.detect(features, candles, regimes)

    assert scan.for_setup(SetupType.BREAKOUT) == ()


def _reversion_inputs(build_inputs, regime: str, *, distance, rsi, band, rows: int = 5):
    return build_inputs(
        trend_regimes=[regime] * rows,
        overrides={
            "trend_price_vs_sma_3": [distance] * rows,
            "momentum_rsi_3": [rsi] * rows,
            "volatility_bollinger_position_3": [band] * rows,
        },
    )


def test_stretched_low_in_a_range_is_a_long_reversion(detector, build_inputs) -> None:
    candles, features, regimes = _reversion_inputs(
        build_inputs, "sideways_consolidation", distance=-0.05, rsi=25.0, band=-0.2
    )

    scan = detector.detect(features, candles, regimes)
    signals = scan.for_setup(SetupType.MEAN_REVERSION)

    assert len(signals) == 5
    assert all(signal.direction is Direction.LONG for signal in signals)


def test_stretched_high_in_a_range_is_a_short_reversion(detector, build_inputs) -> None:
    candles, features, regimes = _reversion_inputs(
        build_inputs, "sideways_consolidation", distance=0.05, rsi=75.0, band=1.2
    )

    scan = detector.detect(features, candles, regimes)
    signals = scan.for_setup(SetupType.MEAN_REVERSION)

    assert len(signals) == 5
    assert all(signal.direction is Direction.SHORT for signal in signals)


def test_reversion_is_ignored_inside_a_trend(detector, build_inputs) -> None:
    candles, features, regimes = _reversion_inputs(
        build_inputs, "bearish_trend", distance=-0.05, rsi=25.0, band=-0.2
    )

    scan = detector.detect(features, candles, regimes)

    assert scan.for_setup(SetupType.MEAN_REVERSION) == ()


def test_reversion_can_be_allowed_outside_a_range(build_inputs) -> None:
    detector = SetupDetector(
        SetupSettings(
            trend_fast_window=3,
            ema_slope_span=3,
            rsi_period=3,
            volume_window=3,
            structure_lookback=3,
            bollinger_period=3,
            breakout_squeeze_lookback=5,
            reversion_require_sideways=False,
        )
    )
    candles, features, regimes = _reversion_inputs(
        build_inputs, "bearish_trend", distance=-0.05, rsi=25.0, band=-0.2
    )

    scan = detector.detect(features, candles, regimes)

    assert len(scan.for_setup(SetupType.MEAN_REVERSION)) == 5


def test_disabling_a_setup_removes_its_signals(build_inputs) -> None:
    detector = SetupDetector(
        SetupSettings(
            trend_fast_window=3,
            ema_slope_span=3,
            rsi_period=3,
            volume_window=3,
            structure_lookback=3,
            bollinger_period=3,
            breakout_squeeze_lookback=5,
            enabled_setups=(SetupType.TREND_CONTINUATION,),
        )
    )
    candles, features, regimes = _continuation_inputs(build_inputs, "bullish_trend")

    scan = detector.detect(features, candles, regimes)

    assert scan.for_setup(SetupType.TREND_CONTINUATION)
    assert scan.for_setup(SetupType.BREAKOUT) == ()
    assert scan.for_setup(SetupType.MEAN_REVERSION) == ()


def test_missing_features_are_reported(build_inputs, detector) -> None:
    candles, features, regimes = _continuation_inputs(build_inputs, "bullish_trend")
    reduced = features.frame.drop(columns=["momentum_rsi_3"])

    trimmed = FeatureSet(
        symbol=features.symbol,
        timeframe=features.timeframe,
        frame=reduced,
        metadata=tuple(
            item for item in features.metadata if item.name != "momentum_rsi_3"
        ),
        available_at=features.available_at,
    )

    with pytest.raises(SetupInputError, match="momentum_rsi_3"):
        detector.detect(trimmed, candles, regimes)


def test_misaligned_inputs_are_rejected(build_inputs, detector) -> None:
    candles, features, regimes = _continuation_inputs(build_inputs, "bullish_trend")

    with pytest.raises(SetupInputError, match="same rows"):
        detector.detect(features, candles.iloc[:-1], regimes)


def test_detection_is_deterministic(detector, build_inputs) -> None:
    candles, features, regimes = _continuation_inputs(build_inputs, "bullish_trend")

    first = detector.detect(features, candles, regimes)
    second = detector.detect(features, candles, regimes)

    assert first.signals == second.signals
