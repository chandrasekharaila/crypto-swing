"""Detected setups must depend only on the current and past rows.

The feature and regime layers each have their own causality sweep against real
candles. This covers the layer above them, where two things could still leak:
reading a feature row ahead of the signal, and the trailing squeeze norm that the
breakout setup computes itself.

Each helper below returns its arguments already in ``detect`` order.
"""

from crypto_analyzer.setups import SetupType

ROWS = 40
BREAKOUT_ROWS = (7, 8, 9)


def _continuation_args(build_inputs, cut: int, *, tail_is_bullish: bool):
    regimes = ["bullish_trend"] * ROWS
    if not tail_is_bullish:
        regimes = ["bullish_trend"] * cut + ["sideways_consolidation"] * (ROWS - cut)
    candles, features, regime_result = build_inputs(
        trend_regimes=regimes,
        overrides={
            "trend_price_vs_sma_3": [-0.02] * ROWS,
            "trend_ema_slope_3": [0.001] * ROWS,
            "momentum_rsi_3": [50.0] * ROWS,
            "volume_relative_3": [1.2] * ROWS,
        },
    )
    return features, candles, regime_result


def _breakout_args(build_inputs, width):
    candles, features, regime_result = build_inputs(
        trend_regimes=["sideways_consolidation"] * ROWS,
        overrides={
            "volatility_bollinger_width_3": width,
            "structure_breakout_up_3": [
                1.0 if index in BREAKOUT_ROWS else 0.0 for index in range(ROWS)
            ],
            "volume_relative_3": [
                2.0 if index in BREAKOUT_ROWS else 1.0 for index in range(ROWS)
            ],
        },
        close=[111.1] * ROWS,
    )
    return features, candles, regime_result


def _timestamps(scan, before):
    return [signal.timestamp for signal in scan.signals if signal.timestamp < before]


def _cut_time(features, cut: int):
    return features.frame["open_time"].iloc[cut]


def test_future_regime_changes_do_not_affect_earlier_signals(
    detector, build_inputs
) -> None:
    cut = 20
    features, candles, regimes = _continuation_args(
        build_inputs, cut, tail_is_bullish=True
    )
    cut_time = _cut_time(features, cut)

    baseline = detector.detect(features, candles, regimes)
    altered = detector.detect(
        *_continuation_args(build_inputs, cut, tail_is_bullish=False)
    )

    assert _timestamps(baseline, cut_time) == _timestamps(altered, cut_time)
    assert len(_timestamps(baseline, cut_time)) == cut


def test_the_regime_perturbation_actually_matters(detector, build_inputs) -> None:
    """Guard against a vacuous sweep: the change must alter later signals."""
    cut = 20
    features, candles, regimes = _continuation_args(
        build_inputs, cut, tail_is_bullish=True
    )
    cut_time = _cut_time(features, cut)

    baseline = detector.detect(features, candles, regimes)
    altered = detector.detect(
        *_continuation_args(build_inputs, cut, tail_is_bullish=False)
    )

    assert len(baseline.signals) == ROWS
    assert len(altered.signals) == cut
    assert len(_timestamps(baseline, cut_time)) > 0


SQUEEZED_WIDTH = [0.10] * 5 + [0.02] * (ROWS - 5)


def test_future_band_widths_do_not_affect_earlier_breakouts(
    detector, build_inputs
) -> None:
    cut = 20
    features, candles, regimes = _breakout_args(build_inputs, SQUEEZED_WIDTH)
    cut_time = _cut_time(features, cut)

    baseline = detector.detect(features, candles, regimes)
    altered = detector.detect(
        *_breakout_args(build_inputs, SQUEEZED_WIDTH[:cut] + [5.0] * (ROWS - cut))
    )

    assert _timestamps(baseline, cut_time) == _timestamps(altered, cut_time)
    assert len(_timestamps(baseline, cut_time)) == len(BREAKOUT_ROWS)


def test_the_breakout_setup_really_reads_band_width(detector, build_inputs) -> None:
    """Guard the sweep: remove the compression and the breakouts must vanish."""
    squeezed = detector.detect(*_breakout_args(build_inputs, SQUEEZED_WIDTH))
    flattened = detector.detect(*_breakout_args(build_inputs, [0.02] * ROWS))

    assert len(squeezed.for_setup(SetupType.BREAKOUT)) == len(BREAKOUT_ROWS)
    assert flattened.for_setup(SetupType.BREAKOUT) == ()
