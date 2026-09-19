"""Deterministic computation of point-in-time feature frames."""

import logging
from datetime import datetime

import pandas as pd

from crypto_analyzer.data.validators import OHLCVValidator
from crypto_analyzer.data.validators.ohlcv import NUMERIC_COLUMNS
from crypto_analyzer.features.config import FeatureSettings
from crypto_analyzer.features.exceptions import (
    FeatureComputationError,
    FeatureInputError,
)
from crypto_analyzer.features.registry import FeatureRegistry, build_default_registry
from crypto_analyzer.features.types import FeatureDefinition, FeatureSet

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Compute configured features from a canonical OHLCV frame."""

    def __init__(
        self,
        settings: FeatureSettings | None = None,
        *,
        registry: FeatureRegistry | None = None,
        validator: OHLCVValidator | None = None,
    ) -> None:
        self._settings = settings or FeatureSettings()
        self._registry = registry or build_default_registry(self._settings)
        self._validator = validator or OHLCVValidator()

    @property
    def registry(self) -> FeatureRegistry:
        """Return the registry of features this engine computes."""
        return self._registry

    def compute(
        self,
        frame: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        as_of: datetime | None = None,
    ) -> FeatureSet:
        """Return point-in-time features for ``frame``.

        Every returned value is derived from candles up to and including its own
        row, so the value becomes available at that candle's ``close_time``.

        Feature values never depend on the clock. ``as_of`` is forwarded to the
        input validator, which defaults to the current time when it is omitted;
        that default makes the *validation* outcome time-dependent, because a
        candle that is still forming now would be rejected. Pass an explicit
        instant when a result must be reproducible independently of when it runs.
        """
        self._validate_input(frame, timeframe, as_of=as_of)
        self._warn_when_history_is_insufficient(frame, symbol, timeframe)

        working = self._numeric_view(frame)
        columns = {
            definition.name: self._compute_aligned(definition, working)
            for definition in self._registry.definitions
        }
        features = pd.DataFrame(columns)
        features.insert(0, "open_time", frame["open_time"].to_numpy())
        return FeatureSet(
            symbol=symbol,
            timeframe=timeframe,
            frame=features,
            metadata=self._registry.metadata,
            available_at=frame["close_time"],
        )

    def _validate_input(
        self, frame: pd.DataFrame, timeframe: str, *, as_of: datetime | None
    ) -> None:
        report = self._validator.validate(frame, timeframe, as_of=as_of)
        if report.is_valid:
            return
        summary = "; ".join(f"{issue.code}: {issue.message}" for issue in report.errors)
        raise FeatureInputError(
            f"input frame is not a valid canonical OHLCV dataset: {summary}"
        )

    def _warn_when_history_is_insufficient(
        self, frame: pd.DataFrame, symbol: str, timeframe: str
    ) -> None:
        required = self._registry.max_lookback
        if len(frame) >= required:
            return
        logger.warning(
            "Only %d candle(s) supplied for %s %s; %d are required before every "
            "feature has a value",
            len(frame),
            symbol,
            timeframe,
            required,
        )

    @staticmethod
    def _numeric_view(frame: pd.DataFrame) -> pd.DataFrame:
        """Return a float64 view of the OHLCV columns for window arithmetic.

        Stored candles carry exact Decimal values that trailing-window
        operations cannot use directly. The caller's frame is never modified.
        """
        working = frame.copy(deep=True)
        for column in NUMERIC_COLUMNS:
            working[column] = working[column].astype("float64")
        return working

    @staticmethod
    def _compute_aligned(
        definition: FeatureDefinition, frame: pd.DataFrame
    ) -> pd.Series:
        result = definition.compute(frame)
        if not isinstance(result, pd.Series):
            raise FeatureComputationError(
                f"feature {definition.name} did not return a pandas Series"
            )
        if len(result) != len(frame) or not result.index.equals(frame.index):
            raise FeatureComputationError(
                f"feature {definition.name} must preserve the input index and length"
            )
        return result.astype("float64")
