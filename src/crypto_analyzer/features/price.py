"""Single-candle price-shape features.

Zero-range candles need one deliberate decision per feature, because the only
undefined case is a candle whose high equals its low.

* Magnitude ratios (a body, a wick, a range) are genuinely zero in that case and
  report 0.0.
* Position ratios (where the close sits inside the range) have no position to
  report, so they fall back to the neutral midpoint of 0.5 instead of claiming
  the close sat at the extreme.

Neither choice can hide a degenerate window the way a blanket 0.0 would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from crypto_analyzer.features.primitives import safe_divide
from crypto_analyzer.features.types import FeatureDefinition, FeatureGroup

if TYPE_CHECKING:
    from crypto_analyzer.features.registry import FeatureRegistry


def body_ratio(frame: pd.DataFrame) -> pd.Series:
    """Candle body as a fraction of its high-to-low range."""
    return safe_divide(
        frame["close"] - frame["open"],
        frame["high"] - frame["low"],
        zero_result=0.0,
    )


def range_pct(frame: pd.DataFrame) -> pd.Series:
    """High-to-low range as a fraction of the close price."""
    return safe_divide(frame["high"] - frame["low"], frame["close"])


def close_location(frame: pd.DataFrame) -> pd.Series:
    """Where the close sits within the candle's high-to-low range."""
    return safe_divide(
        frame["close"] - frame["low"],
        frame["high"] - frame["low"],
        zero_result=0.5,
    )


def upper_wick_ratio(frame: pd.DataFrame) -> pd.Series:
    """Upper wick as a fraction of the candle's high-to-low range."""
    body_top = frame[["open", "close"]].max(axis=1)
    return safe_divide(
        frame["high"] - body_top,
        frame["high"] - frame["low"],
        zero_result=0.0,
    )


def lower_wick_ratio(frame: pd.DataFrame) -> pd.Series:
    """Lower wick as a fraction of the candle's high-to-low range."""
    body_bottom = frame[["open", "close"]].min(axis=1)
    return safe_divide(
        body_bottom - frame["low"],
        frame["high"] - frame["low"],
        zero_result=0.0,
    )


def gap_pct(frame: pd.DataFrame) -> pd.Series:
    """Open price relative to the previous candle's close."""
    return frame["open"].div(frame["close"].shift(1)).sub(1.0)


def register_price_features(registry: FeatureRegistry) -> None:
    """Register the windowless price-shape features."""
    definitions = (
        FeatureDefinition(
            name="price_body_ratio",
            group=FeatureGroup.PRICE,
            description=(
                "Body (close minus open) as a fraction of the high-low range; "
                "0.0 when the candle has no range."
            ),
            lookback=1,
            compute=body_ratio,
        ),
        FeatureDefinition(
            name="price_range_pct",
            group=FeatureGroup.PRICE,
            description="High-low range as a fraction of the close price.",
            lookback=1,
            compute=range_pct,
        ),
        FeatureDefinition(
            name="price_close_location",
            group=FeatureGroup.PRICE,
            description=(
                "Position of the close within the candle's high-low range; "
                "0.5 when the candle has no range."
            ),
            lookback=1,
            compute=close_location,
        ),
        FeatureDefinition(
            name="price_upper_wick_ratio",
            group=FeatureGroup.PRICE,
            description=(
                "Upper wick as a fraction of the candle's high-low range; "
                "0.0 when the candle has no range."
            ),
            lookback=1,
            compute=upper_wick_ratio,
        ),
        FeatureDefinition(
            name="price_lower_wick_ratio",
            group=FeatureGroup.PRICE,
            description=(
                "Lower wick as a fraction of the candle's high-low range; "
                "0.0 when the candle has no range."
            ),
            lookback=1,
            compute=lower_wick_ratio,
        ),
        FeatureDefinition(
            name="price_gap_pct",
            group=FeatureGroup.PRICE,
            description="Open price relative to the previous candle's close.",
            lookback=2,
            compute=gap_pct,
        ),
    )
    for definition in definitions:
        registry.register(definition)
