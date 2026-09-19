"""Guards that prove a feature depends only on the current and past candles.

A feature is causal when its value at row ``t`` is unchanged by any later row.
These helpers assert that property directly instead of relying on inspection.

Feature code must never:

* shift backwards (``shift(-n)``),
* centre a window (``rolling(center=True)``),
* fill backwards or interpolate,
* reduce over the whole series (global min, max, mean, std, rank, or z-score).

Normalization and scaling are deliberately absent from this phase: they must be
fit on training data only, which is a Phase 5 concern.
"""

from __future__ import annotations

import pandas as pd
from pandas.testing import assert_series_equal

from crypto_analyzer.features.exceptions import FeatureComputationError
from crypto_analyzer.features.types import FeatureDefinition

_FUTURE_SENTINEL = 1.0e9
_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def assert_prefix_stable(
    definition: FeatureDefinition, frame: pd.DataFrame, cut: int
) -> None:
    """Assert rows before ``cut`` do not depend on rows at or after ``cut``."""
    if not 0 < cut < len(frame):
        raise FeatureComputationError(
            f"cut must fall inside the frame: got {cut} for {len(frame)} rows"
        )
    on_full = definition.compute(frame).iloc[:cut]
    on_prefix = definition.compute(frame.iloc[:cut])
    assert_series_equal(
        on_full,
        on_prefix,
        check_names=False,
        obj=f"{definition.name} prefix",
    )


def assert_future_independent(
    definition: FeatureDefinition, frame: pd.DataFrame, cut: int
) -> None:
    """Assert rows before ``cut`` are unchanged when every later row is replaced.

    This is stronger than truncation: a leak that only reads a future value
    without changing alignment still moves the value here.
    """
    if not 0 < cut < len(frame):
        raise FeatureComputationError(
            f"cut must fall inside the frame: got {cut} for {len(frame)} rows"
        )
    baseline = definition.compute(frame).iloc[:cut]

    corrupted = frame.copy(deep=True)
    column_names = list(corrupted.columns)
    for column in _OHLCV_COLUMNS:
        corrupted.iloc[cut:, column_names.index(column)] = _FUTURE_SENTINEL
    after = definition.compute(corrupted).iloc[:cut]

    assert_series_equal(
        baseline,
        after,
        check_names=False,
        obj=f"{definition.name} future independence",
    )
