"""Look-ahead guards applied to every registered feature.

Any feature added to the registry is covered here automatically, so the sweep
cannot silently skip a new definition.
"""

import pandas as pd
import pytest

from crypto_analyzer.features import (
    FeatureComputationError,
    FeatureDefinition,
    FeatureEngine,
    FeatureGroup,
    FeatureSettings,
)
from crypto_analyzer.features.causality import (
    assert_future_independent,
    assert_prefix_stable,
)

DEFINITIONS = FeatureEngine(FeatureSettings()).registry.definitions
IDS = [definition.name for definition in DEFINITIONS]
CUTS = (3, 120, 250)


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("definition", DEFINITIONS, ids=IDS)
def test_feature_is_prefix_stable(definition, cut, wave_frame) -> None:
    assert_prefix_stable(definition, wave_frame, cut)


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("definition", DEFINITIONS, ids=IDS)
def test_feature_ignores_every_future_row(definition, cut, wave_frame) -> None:
    assert_future_independent(definition, wave_frame, cut)


def test_guard_rejects_a_feature_that_reads_the_next_candle(wave_frame) -> None:
    def leaky(frame: pd.DataFrame) -> pd.Series:
        return frame["close"].shift(-1)

    definition = FeatureDefinition(
        name="leaky_forward_shift",
        group=FeatureGroup.PRICE,
        description="Deliberately peeks at the following candle.",
        lookback=1,
        compute=leaky,
    )

    with pytest.raises(AssertionError):
        assert_prefix_stable(definition, wave_frame, 100)
    with pytest.raises(AssertionError):
        assert_future_independent(definition, wave_frame, 100)


def test_guard_rejects_a_feature_normalized_over_the_whole_series(
    wave_frame,
) -> None:
    def leaky(frame: pd.DataFrame) -> pd.Series:
        return (frame["close"] - frame["close"].mean()) / frame["close"].std()

    definition = FeatureDefinition(
        name="leaky_global_zscore",
        group=FeatureGroup.PRICE,
        description="Deliberately normalizes using the full series.",
        lookback=1,
        compute=leaky,
    )

    with pytest.raises(AssertionError):
        assert_prefix_stable(definition, wave_frame, 100)


def test_guard_rejects_a_cut_outside_the_frame(wave_frame) -> None:
    definition = DEFINITIONS[0]

    with pytest.raises(FeatureComputationError, match="cut must fall inside"):
        assert_prefix_stable(definition, wave_frame, len(wave_frame))
