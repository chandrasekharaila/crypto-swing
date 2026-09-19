"""Typed configuration for the Phase 1 data foundation."""

from datetime import datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class AppSettings(BaseModel):
    """Central configuration for public market-data collection and storage."""

    model_config = ConfigDict(frozen=True)

    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    timeframes: tuple[str, ...] = ("15m", "1h", "4h", "1d")
    primary_timeframe: str = "4h"
    history_start: datetime | None = None
    history_end: datetime | None = None

    binance_base_url: HttpUrl = HttpUrl("https://api.binance.com")
    request_limit: int = Field(default=1_000, ge=1, le=1_000)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)

    data_directory: Path = Path("data")

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Validate relationships and formats shared by configuration fields."""
        if not self.symbols:
            raise ValueError("at least one symbol must be configured")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be unique")
        if any(symbol.count("/") != 1 or not all(symbol.split("/")) for symbol in self.symbols):
            raise ValueError("symbols must use BASE/QUOTE format")

        if not self.timeframes:
            raise ValueError("at least one timeframe must be configured")
        if len(set(self.timeframes)) != len(self.timeframes):
            raise ValueError("timeframes must be unique")
        if self.primary_timeframe not in self.timeframes:
            raise ValueError("primary_timeframe must be included in timeframes")

        for name, value in (
            ("history_start", self.history_start),
            ("history_end", self.history_end),
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if (
            self.history_start is not None
            and self.history_end is not None
            and self.history_start >= self.history_end
        ):
            raise ValueError("history_start must be earlier than history_end")
        return self

    @property
    def raw_data_directory(self) -> Path:
        """Directory for downloaded source data."""
        return self.data_directory / "raw"

    @property
    def processed_data_directory(self) -> Path:
        """Directory for cleaned data."""
        return self.data_directory / "processed"

    @property
    def features_data_directory(self) -> Path:
        """Directory reserved for future generated features."""
        return self.data_directory / "features"

    @property
    def experiments_data_directory(self) -> Path:
        """Directory reserved for future experiment artifacts."""
        return self.data_directory / "experiments"

    def create_data_directories(self) -> None:
        """Create configured data directories if they do not exist."""
        for directory in (
            self.raw_data_directory,
            self.processed_data_directory,
            self.features_data_directory,
            self.experiments_data_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)
