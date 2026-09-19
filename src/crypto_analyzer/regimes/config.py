"""Typed configuration for rule-based regime detection.

No threshold or window is duplicated inside the detector: every number that
decides a regime lives here.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TREND_RULE_COUNT = 4


class RegimeSettings(BaseModel):
    """Central configuration for regime classification."""

    model_config = ConfigDict(frozen=True)

    trend_fast_window: int = Field(default=50, ge=2)
    trend_slow_window: int = Field(default=200, ge=3)
    trend_ema_slope_span: int = Field(default=50, ge=2)
    volatility_atr_period: int = Field(default=14, ge=2)
    volatility_lookback: int = Field(default=200, ge=2)

    trend_ma_spread_threshold: float = Field(default=0.005, gt=0)
    trend_price_distance_threshold: float = Field(default=0.01, gt=0)
    trend_ema_slope_threshold: float = Field(default=0.0005, gt=0)
    trend_ema_spread_threshold: float = Field(default=0.005, gt=0)
    trend_min_agreeing_rules: int = Field(default=3, ge=1)

    volatility_high_ratio: float = Field(default=1.3, gt=1.0)
    volatility_low_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Validate the relationships between windows and thresholds."""
        if self.trend_fast_window >= self.trend_slow_window:
            raise ValueError("trend_fast_window must be shorter than trend_slow_window")
        if self.trend_min_agreeing_rules > TREND_RULE_COUNT:
            raise ValueError(
                "trend_min_agreeing_rules cannot exceed the "
                f"{TREND_RULE_COUNT} trend rules"
            )
        return self

    @property
    def required_features(self) -> tuple[str, ...]:
        """Return the feature columns this configuration consumes."""
        return (
            f"trend_sma_{self.trend_fast_window}",
            f"trend_sma_{self.trend_slow_window}",
            f"trend_price_vs_sma_{self.trend_fast_window}",
            f"trend_ema_slope_{self.trend_ema_slope_span}",
            "trend_ema_spread_pct",
            f"volatility_atr_pct_{self.volatility_atr_period}",
        )

    @property
    def warmup(self) -> int:
        """Candles needed before both axes can be classified.

        The trend axis is limited by its slow average; the volatility axis needs
        the ATR period plus the full trailing lookback before its norm exists.
        """
        trend = self.trend_slow_window
        volatility = self.volatility_atr_period + self.volatility_lookback - 1
        return max(trend, volatility)
