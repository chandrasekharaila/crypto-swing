"""Regime labels must depend only on the current and past candles.

The feature layer has its own causality sweep; this covers the layer above it,
because a detector could still leak by reading a future feature row, or by
reducing across the series when computing a norm.
"""

import pandas as pd

from crypto_analyzer.regimes.detector import DIAGNOSTIC_COLUMNS

LABELS = ("trend_regime", "volatility_regime", "combined_regime")
SENTINEL = 1.0e6


def _corrupt_after(frame: pd.DataFrame, cut: int) -> pd.DataFrame:
    corrupted = frame.copy(deep=True)
    columns = list(corrupted.columns)
    for column in ("open", "high", "low", "close", "volume"):
        corrupted.iloc[cut:, columns.index(column)] = SENTINEL
    return corrupted


def test_labels_ignore_every_future_candle(run_regimes, volatility_spike_frame) -> None:
    cut = len(volatility_spike_frame) - 20
    baseline = run_regimes(volatility_spike_frame)

    altered = run_regimes(_corrupt_after(volatility_spike_frame, cut))

    assert (
        baseline.frame[list(LABELS)]
        .iloc[:cut]
        .equals(altered.frame[list(LABELS)].iloc[:cut])
    )


def test_the_perturbation_actually_changes_later_labels(
    run_regimes, volatility_spike_frame
) -> None:
    """Guard against a vacuous test: the corruption must matter after the cut."""
    cut = len(volatility_spike_frame) - 20
    baseline = run_regimes(volatility_spike_frame)

    altered = run_regimes(_corrupt_after(volatility_spike_frame, cut))

    assert (
        not baseline.frame[list(LABELS)]
        .iloc[cut:]
        .equals(altered.frame[list(LABELS)].iloc[cut:])
    )


def test_rule_measurements_ignore_every_future_candle(
    run_regimes, volatility_spike_frame
) -> None:
    cut = len(volatility_spike_frame) - 20

    baseline = run_regimes(volatility_spike_frame)
    altered = run_regimes(_corrupt_after(volatility_spike_frame, cut))

    pd.testing.assert_frame_equal(
        baseline.frame[list(DIAGNOSTIC_COLUMNS)].iloc[:cut],
        altered.frame[list(DIAGNOSTIC_COLUMNS)].iloc[:cut],
    )


def test_prefix_stability_across_several_cut_points(
    run_regimes, volatility_spike_frame
) -> None:
    baseline = run_regimes(volatility_spike_frame)

    for cut in (10, 40, len(volatility_spike_frame) - 5):
        altered = run_regimes(_corrupt_after(volatility_spike_frame, cut))
        assert (
            baseline.frame[list(LABELS)]
            .iloc[:cut]
            .equals(altered.frame[list(LABELS)].iloc[:cut])
        ), f"labels changed before cut {cut}"
