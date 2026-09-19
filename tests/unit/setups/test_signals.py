"""Tests that a detected setup carries the full structured record."""

import pytest

from crypto_analyzer.setups import Direction, SetupType

ROWS = 6


def _scan(detector, build_inputs):
    candles, features, regimes = build_inputs(
        trend_regimes=["bullish_trend"] * ROWS,
        overrides={
            "trend_price_vs_sma_3": [-0.02] * ROWS,
            "trend_ema_slope_3": [0.001] * ROWS,
            "momentum_rsi_3": [50.0] * ROWS,
            "volume_relative_3": [1.2] * ROWS,
        },
        close=[100.0 + index for index in range(ROWS)],
    )
    return detector.detect(features, candles, regimes)


def test_every_required_field_is_populated(detector, build_inputs) -> None:
    signal = _scan(detector, build_inputs).signals[-1]

    assert signal.symbol == "BTC/USDT"
    assert signal.timeframe == "1h"
    assert signal.timestamp is not None
    assert signal.direction is Direction.LONG
    assert signal.setup is SetupType.TREND_CONTINUATION
    assert signal.evidence
    assert signal.entry_context.detail
    assert signal.invalidation_context.detail
    assert signal.features


def test_a_signal_is_known_only_after_its_candle_closes(detector, build_inputs) -> None:
    signal = _scan(detector, build_inputs).signals[-1]

    assert signal.available_at > signal.timestamp


def test_timestamp_and_entry_reference_the_signal_candle(
    detector, build_inputs
) -> None:
    signal = _scan(detector, build_inputs).signals[-1]

    expected = 100.0 + ROWS - 1
    assert signal.entry_context.level == pytest.approx(expected)


def test_invalidation_is_the_trailing_low_for_a_long(detector, build_inputs) -> None:
    signal = _scan(detector, build_inputs).signals[-1]

    assert signal.invalidation_context.level == pytest.approx(90.0)
    assert "below" in signal.invalidation_context.detail


def test_relevant_features_are_recorded_and_readable(detector, build_inputs) -> None:
    signal = _scan(detector, build_inputs).signals[-1]

    names = {name for name, _ in signal.features}
    assert "momentum_rsi_3" in names
    assert signal.feature("momentum_rsi_3") == pytest.approx(50.0)


def test_unknown_feature_lookup_fails(detector, build_inputs) -> None:
    signal = _scan(detector, build_inputs).signals[-1]

    with pytest.raises(KeyError):
        signal.feature("not_a_feature")


def test_evidence_carries_the_measurement_and_the_threshold(
    detector, build_inputs
) -> None:
    signal = _scan(detector, build_inputs).signals[-1]

    by_rule = {item.rule: item for item in signal.evidence}

    assert by_rule["pullback_zone"].value == pytest.approx(-0.02)
    assert by_rule["volume_confirmation"].threshold == pytest.approx(1.0)
    assert "requires between" in by_rule["pullback_zone"].detail
    assert "requires at least" in by_rule["volume_confirmation"].detail


def test_every_evidence_entry_points_at_the_signal_direction(
    detector, build_inputs
) -> None:
    signal = _scan(detector, build_inputs).signals[-1]

    assert all(item.supports == Direction.LONG.value for item in signal.evidence)


def test_signals_are_ordered_by_time(detector, build_inputs) -> None:
    scan = _scan(detector, build_inputs)

    timestamps = [signal.timestamp for signal in scan.signals]

    assert timestamps == sorted(timestamps)


def test_scan_can_be_filtered(detector, build_inputs) -> None:
    scan = _scan(detector, build_inputs)

    assert len(scan.for_direction(Direction.LONG)) == len(scan.signals)
    assert scan.for_direction(Direction.SHORT) == ()
