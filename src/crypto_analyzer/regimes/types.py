"""Value types for transparent, rule-based regime classification.

Trend and volatility are independent properties of a market, so they are
classified on separate axes. A market can be trending up while volatility is
high, and reporting a single label would have to discard one of those facts.
"""

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from crypto_analyzer.regimes.config import RegimeSettings

UNKNOWN_REGIME = "unknown"


class TrendRegime(StrEnum):
    """Directional state of the market."""

    BULLISH = "bullish_trend"
    BEARISH = "bearish_trend"
    SIDEWAYS = "sideways_consolidation"
    UNKNOWN = UNKNOWN_REGIME


class VolatilityRegime(StrEnum):
    """Volatility state relative to the instrument's own recent norm."""

    LOW = "low_volatility"
    NORMAL = "normal_volatility"
    HIGH = "high_volatility"
    UNKNOWN = UNKNOWN_REGIME


@dataclass(frozen=True, slots=True)
class RegimeEvidence:
    """One rule outcome, with the measurement that produced it.

    ``supports`` is the regime this rule points at, so the evidence for a
    classification can be read directly rather than inferred from the label.
    """

    rule: str
    supports: str
    detail: str
    value: float | None = None
    threshold: float | None = None


@dataclass(frozen=True, slots=True)
class RegimeResult:
    """Regime labels for one symbol and timeframe.

    ``frame`` is keyed by ``open_time`` and carries two label columns, the
    combined convenience label, and the rule measurements behind them. Every row
    is classified from features that were available at that candle's
    ``close_time``.
    """

    symbol: str
    timeframe: str
    frame: pd.DataFrame
    available_at: pd.Series
    settings: RegimeSettings

    @property
    def trend_column(self) -> str:
        """Name of the trend label column."""
        return "trend_regime"

    @property
    def volatility_column(self) -> str:
        """Name of the volatility label column."""
        return "volatility_regime"

    @property
    def combined_column(self) -> str:
        """Name of the combined label column."""
        return "combined_regime"

    @property
    def label_columns(self) -> tuple[str, ...]:
        """Return the three label columns in deterministic order."""
        return (self.trend_column, self.volatility_column, self.combined_column)

    def explain(self, position: int = -1) -> tuple[RegimeEvidence, ...]:
        """Return the per-rule reasoning for one row, newest row by default."""
        if not self.frame.empty and not -len(self.frame) <= position < len(self.frame):
            raise IndexError(f"position out of range: {position}")
        from crypto_analyzer.regimes.detector import build_evidence

        row = self.frame.iloc[position]
        return build_evidence(row, self.settings)
