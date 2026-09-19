"""A rule outcome together with the measurement that produced it.

Regime classification and setup detection both explain themselves the same way,
so they share one evidence type rather than two that could drift apart.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Evidence:
    """One rule outcome, with the measurement behind it.

    ``supports`` names the classification the rule points at, so the reasoning
    can be read directly instead of being inferred from the label it accompanies.
    ``value`` and ``threshold`` are optional because not every rule compares
    against a number.
    """

    rule: str
    supports: str
    detail: str
    value: float | None = None
    threshold: float | None = None
