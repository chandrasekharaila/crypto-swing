"""Value types for rule-based swing setup detection.

A setup is a hypothesis about market context, not a claim about profitability.
Nothing in these records estimates a return, a win rate, or a probability of
profit; they describe what was observed and what would contradict it.
"""

from dataclasses import dataclass
from datetime import datetime

from crypto_analyzer.evidence import Evidence
from crypto_analyzer.taxonomy import Direction, SetupType

__all__ = [
    "Direction",
    "PriceContext",
    "SetupScan",
    "SetupSignal",
    "SetupType",
]


@dataclass(frozen=True, slots=True)
class PriceContext:
    """A price level together with the reasoning that produced it.

    ``level`` is missing when the setup has no price-based invalidation, and
    ``detail`` always explains the condition in words.
    """

    level: float | None
    detail: str


@dataclass(frozen=True, slots=True)
class SetupSignal:
    """One detected setup on one candle.

    Every field describes what was observed at that timestamp: the reasoning
    (``evidence``), where the setup would be acted on (``entry_context``), what
    would contradict it (``invalidation_context``), and the feature readings the
    rules consumed (``features``).
    """

    symbol: str
    timeframe: str
    timestamp: datetime
    available_at: datetime
    direction: Direction
    setup: SetupType
    evidence: tuple[Evidence, ...]
    entry_context: PriceContext
    invalidation_context: PriceContext
    features: tuple[tuple[str, float], ...]

    def feature(self, name: str) -> float:
        """Return one recorded feature reading by name."""
        for key, value in self.features:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class SetupScan:
    """Every setup detected across one symbol and timeframe."""

    symbol: str
    timeframe: str
    signals: tuple[SetupSignal, ...]

    def for_setup(self, setup: SetupType) -> tuple[SetupSignal, ...]:
        """Return only the signals of one setup family."""
        return tuple(signal for signal in self.signals if signal.setup is setup)

    def for_direction(self, direction: Direction) -> tuple[SetupSignal, ...]:
        """Return only the signals on one side of the market."""
        return tuple(signal for signal in self.signals if signal.direction is direction)
