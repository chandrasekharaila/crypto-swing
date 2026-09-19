"""Transparent, rule-based market-regime detection.

Trend and volatility are classified independently from features that already
exist in the feature set, so no new indicator is computed here and every value
inherits the causality guarantees of the feature layer.

Each trend rule compares a measurement against a configured threshold and votes
bullish, bearish, or neutral. The trend regime follows only when enough rules
agree and they outweigh the opposing votes, which keeps a single noisy rule from
flipping the classification. Volatility is classified by comparing current
average true range with its own trailing norm, so the bands adapt to the
instrument instead of assuming an absolute volatility level.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from crypto_analyzer.features.primitives import safe_divide
from crypto_analyzer.features.types import FeatureSet
from crypto_analyzer.regimes.config import RegimeSettings
from crypto_analyzer.regimes.exceptions import RegimeInputError
from crypto_analyzer.regimes.types import (
    UNKNOWN_REGIME,
    RegimeEvidence,
    RegimeResult,
    TrendRegime,
    VolatilityRegime,
)

MA_SPREAD_COLUMN = "rule_ma_spread"
PRICE_DISTANCE_COLUMN = "rule_price_distance"
EMA_SLOPE_COLUMN = "rule_ema_slope"
EMA_SPREAD_COLUMN = "rule_ema_spread"
VOLATILITY_RATIO_COLUMN = "rule_volatility_ratio"

DIAGNOSTIC_COLUMNS = (
    MA_SPREAD_COLUMN,
    PRICE_DISTANCE_COLUMN,
    EMA_SLOPE_COLUMN,
    EMA_SPREAD_COLUMN,
    VOLATILITY_RATIO_COLUMN,
)


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _direction(
    value: Any, threshold: float, *, bullish: str, bearish: str, neutral: str
) -> str:
    if value is None or pd.isna(value):
        return UNKNOWN_REGIME
    if value > threshold:
        return bullish
    if value < -threshold:
        return bearish
    return neutral


def _detail(label: str, value: Any, threshold: float) -> str:
    if value is None or pd.isna(value):
        return f"{label} is not available yet"
    return f"{label} is {float(value):+.5f} against threshold {threshold:.5f}"


def _trend_rules(settings: RegimeSettings) -> tuple[tuple[str, str, str, float], ...]:
    """Return (column, rule name, label, threshold) for each trend rule."""
    return (
        (
            MA_SPREAD_COLUMN,
            "trend_ma_spread",
            "fast SMA relative to slow SMA",
            settings.trend_ma_spread_threshold,
        ),
        (
            PRICE_DISTANCE_COLUMN,
            "trend_price_distance",
            "close relative to its fast SMA",
            settings.trend_price_distance_threshold,
        ),
        (
            EMA_SLOPE_COLUMN,
            "trend_ema_slope",
            "fast EMA slope",
            settings.trend_ema_slope_threshold,
        ),
        (
            EMA_SPREAD_COLUMN,
            "trend_ema_spread",
            "fast EMA relative to slow EMA",
            settings.trend_ema_spread_threshold,
        ),
    )


def _volatility_support(ratio: Any, settings: RegimeSettings) -> str:
    if ratio is None or pd.isna(ratio):
        return UNKNOWN_REGIME
    if ratio > settings.volatility_high_ratio:
        return VolatilityRegime.HIGH.value
    if ratio < settings.volatility_low_ratio:
        return VolatilityRegime.LOW.value
    return VolatilityRegime.NORMAL.value


def build_evidence(
    row: pd.Series, settings: RegimeSettings
) -> tuple[RegimeEvidence, ...]:
    """Return the per-rule reasoning behind one classified row."""
    bullish = TrendRegime.BULLISH.value
    bearish = TrendRegime.BEARISH.value
    neutral = TrendRegime.SIDEWAYS.value

    evidence: list[RegimeEvidence] = []
    bullish_votes = 0
    bearish_votes = 0
    for column, rule, label, threshold in _trend_rules(settings):
        value = row[column]
        supports = _direction(
            value, threshold, bullish=bullish, bearish=bearish, neutral=neutral
        )
        bullish_votes += supports == bullish
        bearish_votes += supports == bearish
        evidence.append(
            RegimeEvidence(
                rule=rule,
                supports=supports,
                detail=_detail(label, value, threshold),
                value=_number(value),
                threshold=threshold,
            )
        )

    evidence.append(
        RegimeEvidence(
            rule="trend_aggregate",
            supports=str(row["trend_regime"]),
            detail=(
                f"{bullish_votes} rule(s) support bullish and {bearish_votes} support "
                f"bearish; {settings.trend_min_agreeing_rules} must agree and "
                "outweigh the opposing votes"
            ),
        )
    )

    ratio = row[VOLATILITY_RATIO_COLUMN]
    evidence.append(
        RegimeEvidence(
            rule="volatility_ratio",
            supports=_volatility_support(ratio, settings),
            detail=(
                "average true range is not comparable to its norm yet"
                if _number(ratio) is None
                else (
                    f"average true range is {float(ratio):.4f}x its trailing "
                    f"{settings.volatility_lookback}-candle norm against bands "
                    f"{settings.volatility_low_ratio:.2f} and "
                    f"{settings.volatility_high_ratio:.2f}"
                )
            ),
            value=_number(ratio),
        )
    )
    return tuple(evidence)


class RegimeDetector:
    """Classify trend and volatility from an existing feature set."""

    def __init__(self, settings: RegimeSettings | None = None) -> None:
        self._settings = settings or RegimeSettings()

    @property
    def settings(self) -> RegimeSettings:
        """Return the configuration this detector applies."""
        return self._settings

    @property
    def required_features(self) -> tuple[str, ...]:
        """Return the feature columns this configuration consumes."""
        return self._settings.required_features

    def detect(self, features: FeatureSet) -> RegimeResult:
        """Return regime labels for every row of ``features``.

        Every label is derived from features at or before its own row, so a
        regime at row ``t`` is knowable at that candle's ``close_time``.
        """
        self._require_features(features)
        settings = self._settings
        source = features.frame

        ma_spread = safe_divide(
            source[f"trend_sma_{settings.trend_fast_window}"],
            source[f"trend_sma_{settings.trend_slow_window}"],
        ).sub(1.0)
        price_distance = source[f"trend_price_vs_sma_{settings.trend_fast_window}"]
        ema_slope = source[f"trend_ema_slope_{settings.trend_ema_slope_span}"]
        ema_spread = source["trend_ema_spread_pct"]

        atr_pct = source[f"volatility_atr_pct_{settings.volatility_atr_period}"]
        norm = atr_pct.rolling(
            settings.volatility_lookback,
            min_periods=settings.volatility_lookback,
        ).mean()
        volatility_ratio = safe_divide(atr_pct, norm)

        trend_known = (
            ma_spread.notna()
            & price_distance.notna()
            & ema_slope.notna()
            & ema_spread.notna()
        )
        volatility_known = volatility_ratio.notna()

        bullish_votes = (
            (ma_spread > settings.trend_ma_spread_threshold).astype("int64")
            + (price_distance > settings.trend_price_distance_threshold).astype("int64")
            + (ema_slope > settings.trend_ema_slope_threshold).astype("int64")
            + (ema_spread > settings.trend_ema_spread_threshold).astype("int64")
        )
        bearish_votes = (
            (ma_spread < -settings.trend_ma_spread_threshold).astype("int64")
            + (price_distance < -settings.trend_price_distance_threshold).astype(
                "int64"
            )
            + (ema_slope < -settings.trend_ema_slope_threshold).astype("int64")
            + (ema_spread < -settings.trend_ema_spread_threshold).astype("int64")
        )

        minimum = settings.trend_min_agreeing_rules
        is_bullish = (bullish_votes >= minimum) & (bullish_votes > bearish_votes)
        is_bearish = (bearish_votes >= minimum) & (bearish_votes > bullish_votes)

        trend = pd.Series(
            TrendRegime.SIDEWAYS.value, index=source.index, dtype="object"
        )
        trend = trend.mask(is_bearish, TrendRegime.BEARISH.value)
        trend = trend.mask(is_bullish, TrendRegime.BULLISH.value)
        trend = trend.mask(~trend_known, TrendRegime.UNKNOWN.value)

        volatility = pd.Series(
            VolatilityRegime.NORMAL.value, index=source.index, dtype="object"
        )
        volatility = volatility.mask(
            volatility_ratio < settings.volatility_low_ratio,
            VolatilityRegime.LOW.value,
        )
        volatility = volatility.mask(
            volatility_ratio > settings.volatility_high_ratio,
            VolatilityRegime.HIGH.value,
        )
        volatility = volatility.mask(~volatility_known, VolatilityRegime.UNKNOWN.value)

        combined = trend.str.cat(volatility, sep="__")
        combined = combined.mask(~(trend_known & volatility_known), UNKNOWN_REGIME)

        frame = pd.DataFrame(
            {
                "open_time": source["open_time"].to_numpy(),
                "trend_regime": trend.to_numpy(),
                "volatility_regime": volatility.to_numpy(),
                "combined_regime": combined.to_numpy(),
                MA_SPREAD_COLUMN: ma_spread.to_numpy(),
                PRICE_DISTANCE_COLUMN: price_distance.to_numpy(),
                EMA_SLOPE_COLUMN: ema_slope.to_numpy(),
                EMA_SPREAD_COLUMN: ema_spread.to_numpy(),
                VOLATILITY_RATIO_COLUMN: volatility_ratio.to_numpy(),
            }
        )
        return RegimeResult(
            symbol=features.symbol,
            timeframe=features.timeframe,
            frame=frame,
            available_at=features.available_at,
            settings=settings,
        )

    def _require_features(self, features: FeatureSet) -> None:
        available = set(features.feature_names)
        missing = [
            name for name in self._settings.required_features if name not in available
        ]
        if missing:
            raise RegimeInputError(
                "feature set is missing columns required for regime detection: "
                + ", ".join(missing)
                + "; configure FeatureSettings so these windows are generated"
            )
