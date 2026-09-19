"""Transparent, rule-based swing setup detection.

Each setup is a hypothesis about market context that can be checked against
history. It is not a claim about profitability: nothing here estimates a return,
a win rate, or a probability of profit, and no rule reads a future candle.

Every condition is evaluated from the current and preceding rows only, and every
price level comes from a feature that already obeys the same rule. The regime
layer supplies the context each setup requires, so a trend continuation cannot
fire in a market that is not trending.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from crypto_analyzer.evidence import Evidence
from crypto_analyzer.features.primitives import safe_divide
from crypto_analyzer.features.types import FeatureSet
from crypto_analyzer.regimes.types import RegimeResult, TrendRegime
from crypto_analyzer.setups.config import SetupSettings
from crypto_analyzer.setups.exceptions import SetupInputError
from crypto_analyzer.setups.types import (
    Direction,
    PriceContext,
    SetupScan,
    SetupSignal,
    SetupType,
)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _fmt(value: Any) -> str:
    number = _number(value)
    return "unavailable" if number is None else f"{number:+.5f}"


def _at_least(
    rule: str, supports: str, label: str, value: Any, threshold: float
) -> Evidence:
    return Evidence(
        rule=rule,
        supports=supports,
        detail=f"{label} is {_fmt(value)}, requires at least {threshold:.5f}",
        value=_number(value),
        threshold=threshold,
    )


def _at_most(
    rule: str, supports: str, label: str, value: Any, threshold: float
) -> Evidence:
    return Evidence(
        rule=rule,
        supports=supports,
        detail=f"{label} is {_fmt(value)}, requires at most {threshold:.5f}",
        value=_number(value),
        threshold=threshold,
    )


def _between(
    rule: str, supports: str, label: str, value: Any, low: float, high: float
) -> Evidence:
    return Evidence(
        rule=rule,
        supports=supports,
        detail=f"{label} is {_fmt(value)}, requires between {low:.5f} and {high:.5f}",
        value=_number(value),
        threshold=high,
    )


def _regime_context(supports: str, observed: str, required: str) -> Evidence:
    return Evidence(
        rule="regime_context",
        supports=supports,
        detail=f"trend regime is {observed}, this setup requires {required}",
    )


def continuation_evidence(
    direction: Direction,
    regime: str,
    distance: float,
    slope: float,
    rsi: float,
    volume: float,
    settings: SetupSettings,
) -> tuple[Evidence, ...]:
    """Return the reasoning behind one trend-continuation signal."""
    supports = direction.value
    is_long = direction is Direction.LONG
    low, high = (
        (
            -settings.continuation_max_pullback_depth,
            settings.continuation_max_pullback,
        )
        if is_long
        else (
            -settings.continuation_max_pullback,
            settings.continuation_max_pullback_depth,
        )
    )
    rsi_low, rsi_high = (
        (settings.continuation_rsi_floor, settings.continuation_rsi_ceiling)
        if is_long
        else (
            100.0 - settings.continuation_rsi_ceiling,
            100.0 - settings.continuation_rsi_floor,
        )
    )
    slope_rule = _at_least if is_long else _at_most
    slope_threshold = (
        settings.continuation_min_slope if is_long else -settings.continuation_min_slope
    )
    return (
        _regime_context(
            supports,
            regime,
            TrendRegime.BULLISH.value if is_long else TrendRegime.BEARISH.value,
        ),
        _between(
            "pullback_zone",
            supports,
            "close relative to the fast average",
            distance,
            low,
            high,
        ),
        slope_rule("ema_slope", supports, "fast EMA slope", slope, slope_threshold),
        _between("momentum_recovery", supports, "momentum RSI", rsi, rsi_low, rsi_high),
        _at_least(
            "volume_confirmation",
            supports,
            "relative volume",
            volume,
            settings.continuation_min_relative_volume,
        ),
    )


def breakout_evidence(
    direction: Direction,
    squeeze_ratio_before_break: float,
    volume: float,
    extension: float,
    settings: SetupSettings,
) -> tuple[Evidence, ...]:
    """Return the reasoning behind one breakout signal."""
    supports = direction.value
    return (
        _at_most(
            "squeeze",
            supports,
            f"band width vs its {settings.breakout_squeeze_lookback}-candle norm",
            squeeze_ratio_before_break,
            settings.breakout_squeeze_ratio,
        ),
        _at_least(
            "volume_expansion",
            supports,
            "relative volume",
            volume,
            settings.breakout_min_relative_volume,
        ),
        _at_most(
            "extension",
            supports,
            "distance beyond the broken level",
            abs(extension),
            settings.breakout_max_extension,
        ),
    )


def reversion_evidence(
    direction: Direction,
    regime: str,
    distance: float,
    rsi: float,
    band: float,
    settings: SetupSettings,
) -> tuple[Evidence, ...]:
    """Return the reasoning behind one mean-reversion signal."""
    supports = direction.value
    is_long = direction is Direction.LONG
    stretch_rule = _at_most if is_long else _at_least
    stretch_threshold = (
        -settings.reversion_min_stretch if is_long else settings.reversion_min_stretch
    )
    rsi_threshold = (
        settings.reversion_rsi_oversold
        if is_long
        else 100.0 - settings.reversion_rsi_overbought
    )
    band_rule = _at_most if is_long else _at_least
    band_threshold = (
        settings.reversion_long_band_max
        if is_long
        else settings.reversion_short_band_min
    )
    return (
        _regime_context(
            supports,
            regime,
            TrendRegime.SIDEWAYS.value
            if settings.reversion_require_sideways
            else "a classified regime",
        ),
        stretch_rule(
            "stretch",
            supports,
            "close relative to the fast average",
            distance,
            stretch_threshold,
        ),
        (_at_most if is_long else _at_least)(
            "momentum_extreme", supports, "momentum RSI", rsi, rsi_threshold
        ),
        band_rule("band_extreme", supports, "Bollinger position", band, band_threshold),
    )


class SetupDetector:
    """Detect trend-continuation, breakout, and mean-reversion setups."""

    def __init__(self, settings: SetupSettings | None = None) -> None:
        self._settings = settings or SetupSettings()

    @property
    def settings(self) -> SetupSettings:
        """Return the configuration this detector applies."""
        return self._settings

    @property
    def required_features(self) -> tuple[str, ...]:
        """Return every feature column the enabled setups read."""
        return self._settings.required_features

    def detect(
        self, features: FeatureSet, candles: pd.DataFrame, regimes: RegimeResult
    ) -> SetupScan:
        """Return every setup found across the supplied history.

        ``candles``, ``features``, and ``regimes`` must describe the same rows in
        the same order. Candles supply only the close price, which anchors the
        entry and invalidation levels a signal reports.
        """
        self._require_features(features)
        self._require_alignment(features, candles, regimes)
        prepared = self._prepare(features, candles, regimes)

        signals: list[SetupSignal] = []
        if SetupType.TREND_CONTINUATION in self._settings.enabled_setups:
            signals.extend(self._continuation(prepared, features))
        if SetupType.BREAKOUT in self._settings.enabled_setups:
            signals.extend(self._breakout(prepared, features))
        if SetupType.MEAN_REVERSION in self._settings.enabled_setups:
            signals.extend(self._mean_reversion(prepared, features))

        signals.sort(key=lambda signal: (signal.timestamp, signal.setup.value))
        return SetupScan(
            symbol=features.symbol,
            timeframe=features.timeframe,
            signals=tuple(signals),
        )

    def _continuation(
        self, prepared: dict[str, pd.Series], features: FeatureSet
    ) -> list[SetupSignal]:
        settings = self._settings
        setup = SetupType.TREND_CONTINUATION
        distance = prepared["price_distance"]
        slope = prepared["ema_slope"]
        rsi = prepared["rsi"]
        volume = prepared["volume_relative"]
        regime = prepared["trend_regime"]
        volume_ok = volume >= settings.continuation_min_relative_volume

        masks = {
            Direction.LONG: regime.eq(TrendRegime.BULLISH.value)
            & (distance <= settings.continuation_max_pullback)
            & (distance >= -settings.continuation_max_pullback_depth)
            & (slope >= settings.continuation_min_slope)
            & rsi.between(
                settings.continuation_rsi_floor, settings.continuation_rsi_ceiling
            )
            & volume_ok,
            Direction.SHORT: regime.eq(TrendRegime.BEARISH.value)
            & (distance >= -settings.continuation_max_pullback)
            & (distance <= settings.continuation_max_pullback_depth)
            & (slope <= -settings.continuation_min_slope)
            & rsi.between(
                100.0 - settings.continuation_rsi_ceiling,
                100.0 - settings.continuation_rsi_floor,
            )
            & volume_ok,
        }

        signals: list[SetupSignal] = []
        for direction, mask in masks.items():
            for position in mask.to_numpy().nonzero()[0]:
                evidence = continuation_evidence(
                    direction,
                    str(regime.iloc[position]),
                    float(distance.iloc[position]),
                    float(slope.iloc[position]),
                    float(rsi.iloc[position]),
                    float(volume.iloc[position]),
                    settings,
                )
                signals.append(
                    self._signal(
                        features,
                        position,
                        direction,
                        setup,
                        evidence,
                        entry=PriceContext(
                            _number(prepared["close"].iloc[position]),
                            "reference close after a pullback into the fast average "
                            "while the trend remains intact",
                        ),
                        invalidation=self._structure_invalidation(
                            prepared, position, direction, settings
                        ),
                    )
                )
        return signals

    def _breakout(
        self, prepared: dict[str, pd.Series], features: FeatureSet
    ) -> list[SetupSignal]:
        settings = self._settings
        setup = SetupType.BREAKOUT
        width = prepared["bollinger_width"]
        volume = prepared["volume_relative"]
        extension_up = prepared["extension_up"]
        extension_down = prepared["extension_down"]
        # Band width is measured against its own trailing norm, so "narrow" is
        # relative to the instrument rather than an absolute width.
        width_norm = width.rolling(
            settings.breakout_squeeze_lookback,
            min_periods=settings.breakout_squeeze_lookback,
        ).mean()
        squeeze_before_break = safe_divide(width, width_norm).shift(1)
        squeeze = squeeze_before_break <= settings.breakout_squeeze_ratio
        volume_ok = volume >= settings.breakout_min_relative_volume

        masks = {
            Direction.LONG: prepared["breakout_up"].eq(1.0)
            & squeeze
            & volume_ok
            & (extension_up <= settings.breakout_max_extension),
            Direction.SHORT: prepared["breakout_down"].eq(1.0)
            & squeeze
            & volume_ok
            & (extension_down >= -settings.breakout_max_extension),
        }

        signals: list[SetupSignal] = []
        for direction, mask in masks.items():
            is_long = direction is Direction.LONG
            extension = extension_up if is_long else extension_down
            level_key = "reference_high" if is_long else "reference_low"
            for position in mask.to_numpy().nonzero()[0]:
                evidence = breakout_evidence(
                    direction,
                    float(squeeze_before_break.iloc[position]),
                    float(volume.iloc[position]),
                    float(extension.iloc[position]),
                    settings,
                )
                signals.append(
                    self._signal(
                        features,
                        position,
                        direction,
                        setup,
                        evidence,
                        entry=PriceContext(
                            _number(prepared["close"].iloc[position]),
                            "reference close after closing beyond the prior "
                            f"{settings.structure_lookback}-candle "
                            f"{'high' if is_long else 'low'} following a squeeze",
                        ),
                        invalidation=PriceContext(
                            _number(prepared[level_key].iloc[position]),
                            "invalidated if price closes back "
                            f"{'below' if is_long else 'above'} the broken level",
                        ),
                    )
                )
        return signals

    def _mean_reversion(
        self, prepared: dict[str, pd.Series], features: FeatureSet
    ) -> list[SetupSignal]:
        settings = self._settings
        setup = SetupType.MEAN_REVERSION
        distance = prepared["price_distance"]
        rsi = prepared["rsi"]
        band = prepared["bollinger_position"]
        regime = prepared["trend_regime"]
        context = (
            regime.eq(TrendRegime.SIDEWAYS.value)
            if settings.reversion_require_sideways
            else regime.ne("unknown")
        )

        masks = {
            Direction.LONG: context
            & (distance <= -settings.reversion_min_stretch)
            & (rsi <= settings.reversion_rsi_oversold)
            & (band <= settings.reversion_long_band_max),
            Direction.SHORT: context
            & (distance >= settings.reversion_min_stretch)
            & (rsi >= settings.reversion_rsi_overbought)
            & (band >= settings.reversion_short_band_min),
        }

        signals: list[SetupSignal] = []
        for direction, mask in masks.items():
            for position in mask.to_numpy().nonzero()[0]:
                evidence = reversion_evidence(
                    direction,
                    str(regime.iloc[position]),
                    float(distance.iloc[position]),
                    float(rsi.iloc[position]),
                    float(band.iloc[position]),
                    settings,
                )
                signals.append(
                    self._signal(
                        features,
                        position,
                        direction,
                        setup,
                        evidence,
                        entry=PriceContext(
                            _number(prepared["close"].iloc[position]),
                            "reference close while price is stretched away from its "
                            "fast average inside a sideways regime",
                        ),
                        invalidation=self._structure_invalidation(
                            prepared, position, direction, settings
                        ),
                    )
                )
        return signals

    def _structure_invalidation(
        self,
        prepared: dict[str, pd.Series],
        position: int,
        direction: Direction,
        settings: SetupSettings,
    ) -> PriceContext:
        is_long = direction is Direction.LONG
        key = "recent_low" if is_long else "recent_high"
        return PriceContext(
            _number(prepared[key].iloc[position]),
            "invalidated if price closes "
            f"{'below' if is_long else 'above'} the trailing "
            f"{settings.structure_lookback}-candle "
            f"{'low' if is_long else 'high'}",
        )

    def _signal(
        self,
        features: FeatureSet,
        position: int,
        direction: Direction,
        setup: SetupType,
        evidence: tuple[Evidence, ...],
        *,
        entry: PriceContext,
        invalidation: PriceContext,
    ) -> SetupSignal:
        frame = features.frame
        readings: list[tuple[str, float]] = []
        for name in self._settings.features_for(setup):
            value = frame[name].iloc[position]
            if not pd.isna(value):
                readings.append((name, float(value)))
        return SetupSignal(
            symbol=features.symbol,
            timeframe=features.timeframe,
            timestamp=frame["open_time"].iloc[position].to_pydatetime(),
            available_at=features.available_at.iloc[position].to_pydatetime(),
            direction=direction,
            setup=setup,
            evidence=evidence,
            entry_context=entry,
            invalidation_context=invalidation,
            features=tuple(readings),
        )

    def _require_features(self, features: FeatureSet) -> None:
        available = set(features.feature_names)
        missing = [
            name for name in self._settings.required_features if name not in available
        ]
        if missing:
            raise SetupInputError(
                "feature set is missing columns required for setup detection: "
                + ", ".join(missing)
                + "; configure FeatureSettings so these windows are generated"
            )

    def _require_alignment(
        self, features: FeatureSet, candles: pd.DataFrame, regimes: RegimeResult
    ) -> None:
        missing = [column for column in _OHLCV_COLUMNS if column not in candles.columns]
        if missing:
            raise SetupInputError(
                "candle frame is missing columns: " + ", ".join(missing)
            )
        lengths = (len(candles), len(features.frame), len(regimes.frame))
        if len(set(lengths)) != 1:
            raise SetupInputError(
                "candles, features, and regimes must cover the same rows: "
                f"got {lengths[0]}, {lengths[1]}, and {lengths[2]}"
            )

    def _prepare(
        self, features: FeatureSet, candles: pd.DataFrame, regimes: RegimeResult
    ) -> dict[str, pd.Series]:
        settings = self._settings
        frame = features.frame
        close = candles["close"].astype("float64")
        prepared: dict[str, pd.Series] = {
            "close": close,
            "trend_regime": regimes.frame[regimes.trend_column],
        }
        required = set(settings.required_features)
        sources = (
            ("price_distance", f"trend_price_vs_sma_{settings.trend_fast_window}"),
            ("ema_slope", f"trend_ema_slope_{settings.ema_slope_span}"),
            ("rsi", f"momentum_rsi_{settings.rsi_period}"),
            ("volume_relative", f"volume_relative_{settings.volume_window}"),
            ("breakout_up", f"structure_breakout_up_{settings.structure_lookback}"),
            ("breakout_down", f"structure_breakout_down_{settings.structure_lookback}"),
            ("recent_high", f"structure_recent_high_{settings.structure_lookback}"),
            ("recent_low", f"structure_recent_low_{settings.structure_lookback}"),
            (
                "bollinger_width",
                f"volatility_bollinger_width_{settings.bollinger_period}",
            ),
            (
                "bollinger_position",
                f"volatility_bollinger_position_{settings.bollinger_period}",
            ),
        )
        for key, feature in sources:
            if feature in required:
                prepared[key] = frame[feature]

        # The broken level is the prior window's extreme, which is the trailing
        # extreme one candle earlier.
        if "recent_high" in prepared:
            prepared["reference_high"] = prepared["recent_high"].shift(1)
            prepared["extension_up"] = safe_divide(
                close, prepared["reference_high"]
            ).sub(1.0)
        if "recent_low" in prepared:
            prepared["reference_low"] = prepared["recent_low"].shift(1)
            prepared["extension_down"] = safe_divide(
                close, prepared["reference_low"]
            ).sub(1.0)
        return prepared
