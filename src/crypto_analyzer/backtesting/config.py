"""Typed configuration for backtesting.

Every number that decides a fill, a cost, or a position belongs here, so a
result can be reproduced exactly from its configuration. Costs default to
non-zero on purpose: a backtest with free, frictionless execution is not
evidence about a real market.
"""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crypto_analyzer.backtesting.types import AmbiguousExitPolicy

_BASIS_POINTS = 10_000.0


class BacktestSettings(BaseModel):
    """Central configuration for simulated execution and portfolio assembly."""

    model_config = ConfigDict(frozen=True)

    execution_delay_bars: int = Field(default=1, ge=1)
    fee_bps: float = Field(default=10.0, ge=0.0)
    slippage_bps: float = Field(default=5.0, ge=0.0)

    take_profit_r_multiple: float = Field(default=2.0, gt=0.0)
    max_holding_bars: int = Field(default=42, ge=1)
    ambiguous_exit_policy: AmbiguousExitPolicy = AmbiguousExitPolicy.STOP_FIRST

    initial_equity: float = Field(default=100_000.0, gt=0.0)
    risk_fraction: float = Field(default=0.01, gt=0.0, le=0.25)
    max_notional_multiple: float = Field(default=1.0, gt=0.0)
    """Cap on position notional as a multiple of equity.

    Risk-based sizing alone is unbounded: a structural stop sitting a few basis
    points from the entry implies an enormous position to risk a fixed fraction,
    and the round-trip fee then dwarfs the risk being taken. Capping the notional
    keeps a tight stop from turning into hidden leverage. A trade that hits the
    cap simply risks less than ``risk_fraction``.
    """

    max_concurrent_positions: int = Field(default=5, ge=1)
    one_position_per_symbol: bool = True

    window_start: datetime | None = None
    window_end: datetime | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Validate the window and its alignment."""
        for name, value in (
            ("window_start", self.window_start),
            ("window_end", self.window_end),
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if (
            self.window_start is not None
            and self.window_end is not None
            and self.window_start >= self.window_end
        ):
            raise ValueError("window_start must be earlier than window_end")
        return self

    @property
    def fee_rate(self) -> float:
        """Return the per-side fee as a fraction of notional."""
        return self.fee_bps / _BASIS_POINTS

    @property
    def slippage_rate(self) -> float:
        """Return the per-side adverse slippage as a fraction of price."""
        return self.slippage_bps / _BASIS_POINTS
