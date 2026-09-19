"""Typed configuration for rule-based swing setup detection.

Every threshold, window, and enabling switch lives here so a detection can be
reproduced exactly from its configuration. Feature windows are named rather than
numbered twice: the feature columns a setup needs are derived from these values.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crypto_analyzer.setups.types import SetupType


class SetupSettings(BaseModel):
    """Central configuration for swing setup detection."""

    model_config = ConfigDict(frozen=True)

    enabled_setups: tuple[SetupType, ...] = tuple(SetupType)

    trend_fast_window: int = Field(default=50, ge=2)
    ema_slope_span: int = Field(default=50, ge=2)
    rsi_period: int = Field(default=14, ge=2)
    volume_window: int = Field(default=20, ge=1)
    structure_lookback: int = Field(default=20, ge=2)
    bollinger_period: int = Field(default=20, ge=2)

    continuation_max_pullback: float = Field(default=0.0, le=0.0)
    continuation_max_pullback_depth: float = Field(default=0.06, gt=0.0)
    continuation_min_slope: float = Field(default=0.0005, gt=0.0)
    continuation_rsi_floor: float = Field(default=40.0, ge=0.0, le=100.0)
    continuation_rsi_ceiling: float = Field(default=65.0, ge=0.0, le=100.0)
    continuation_min_relative_volume: float = Field(default=1.0, gt=0.0)

    breakout_squeeze_ratio: float = Field(default=0.6, gt=0.0)
    breakout_squeeze_lookback: int = Field(default=100, ge=2)
    breakout_min_relative_volume: float = Field(default=1.5, gt=0.0)
    breakout_max_extension: float = Field(default=0.02, gt=0.0)

    reversion_require_sideways: bool = True
    reversion_min_stretch: float = Field(default=0.03, gt=0.0)
    reversion_rsi_oversold: float = Field(default=35.0, ge=0.0, le=100.0)
    reversion_rsi_overbought: float = Field(default=65.0, ge=0.0, le=100.0)
    reversion_long_band_max: float = Field(default=0.0)
    reversion_short_band_min: float = Field(default=1.0)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Validate the relationships between thresholds."""
        if not self.enabled_setups:
            raise ValueError("at least one setup must be enabled")
        if len(set(self.enabled_setups)) != len(self.enabled_setups):
            raise ValueError("enabled_setups must be unique")
        if self.continuation_rsi_floor >= self.continuation_rsi_ceiling:
            raise ValueError(
                "continuation_rsi_floor must be below continuation_rsi_ceiling"
            )
        if self.reversion_rsi_oversold >= self.reversion_rsi_overbought:
            raise ValueError(
                "reversion_rsi_oversold must be below reversion_rsi_overbought"
            )
        if self.reversion_long_band_max >= self.reversion_short_band_min:
            raise ValueError(
                "reversion_long_band_max must be below reversion_short_band_min"
            )
        return self

    def features_for(self, setup: SetupType) -> tuple[str, ...]:
        """Return the feature columns one setup family reads."""
        if setup is SetupType.TREND_CONTINUATION:
            return (
                f"trend_price_vs_sma_{self.trend_fast_window}",
                f"trend_ema_slope_{self.ema_slope_span}",
                f"momentum_rsi_{self.rsi_period}",
                f"volume_relative_{self.volume_window}",
                f"structure_recent_low_{self.structure_lookback}",
            )
        if setup is SetupType.BREAKOUT:
            return (
                f"structure_breakout_up_{self.structure_lookback}",
                f"structure_breakout_down_{self.structure_lookback}",
                f"structure_recent_high_{self.structure_lookback}",
                f"structure_recent_low_{self.structure_lookback}",
                f"volatility_bollinger_width_{self.bollinger_period}",
                f"volume_relative_{self.volume_window}",
            )
        return (
            f"trend_price_vs_sma_{self.trend_fast_window}",
            f"momentum_rsi_{self.rsi_period}",
            f"volatility_bollinger_position_{self.bollinger_period}",
            f"structure_recent_low_{self.structure_lookback}",
            f"structure_recent_high_{self.structure_lookback}",
        )

    @property
    def required_features(self) -> tuple[str, ...]:
        """Return every feature column the enabled setups read."""
        names: list[str] = []
        for setup in self.enabled_setups:
            for name in self.features_for(setup):
                if name not in names:
                    names.append(name)
        return tuple(names)
