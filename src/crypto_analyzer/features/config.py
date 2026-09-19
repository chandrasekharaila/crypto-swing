"""Typed configuration for Phase 2 feature engineering.

Every window, period, and threshold used by feature calculation lives here so
that no literal parameter is duplicated inside the feature modules.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crypto_analyzer.features.types import FeatureGroup

_WINDOW_TUPLE_FIELDS = (
    "return_windows",
    "sma_windows",
    "ema_windows",
    "ma_distance_windows",
    "roc_windows",
    "rolling_std_windows",
    "volume_ma_windows",
)


class FeatureSettings(BaseModel):
    """Central configuration for deterministic feature generation."""

    model_config = ConfigDict(frozen=True)

    enabled_groups: tuple[FeatureGroup, ...] = tuple(FeatureGroup)

    return_windows: tuple[int, ...] = (1, 3, 6, 12, 24)

    sma_windows: tuple[int, ...] = (20, 50, 200)
    ema_windows: tuple[int, ...] = (12, 26, 50)
    ma_distance_windows: tuple[int, ...] = (20, 50)
    ema_slope_window: int = Field(default=5, ge=1)

    rsi_period: int = Field(default=14, ge=2)
    macd_fast: int = Field(default=12, ge=1)
    macd_slow: int = Field(default=26, ge=2)
    macd_signal: int = Field(default=9, ge=1)
    roc_windows: tuple[int, ...] = (6, 12, 24)
    stoch_k_period: int = Field(default=14, ge=2)
    stoch_d_period: int = Field(default=3, ge=1)

    atr_period: int = Field(default=14, ge=2)
    rolling_std_windows: tuple[int, ...] = (20, 50)
    bollinger_period: int = Field(default=20, ge=2)
    bollinger_num_std: float = Field(default=2.0, gt=0)

    volume_ma_windows: tuple[int, ...] = (20, 50)
    relative_volume_window: int = Field(default=20, ge=1)
    volume_change_period: int = Field(default=1, ge=1)

    structure_lookback: int = Field(default=20, ge=2)
    consolidation_window: int = Field(default=20, ge=2)
    sr_proximity_window: int = Field(default=50, ge=2)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Validate window collections and the relationships between periods."""
        if not self.enabled_groups:
            raise ValueError("at least one feature group must be enabled")
        if len(set(self.enabled_groups)) != len(self.enabled_groups):
            raise ValueError("enabled_groups must be unique")

        for name in _WINDOW_TUPLE_FIELDS:
            windows: tuple[int, ...] = getattr(self, name)
            if not windows:
                raise ValueError(f"{name} must not be empty")
            if len(set(windows)) != len(windows):
                raise ValueError(f"{name} must not contain duplicates")
            if any(window < 1 for window in windows):
                raise ValueError(f"{name} entries must be positive")

        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be shorter than macd_slow")
        if self.stoch_d_period > self.stoch_k_period:
            raise ValueError("stoch_d_period must not exceed stoch_k_period")
        return self
