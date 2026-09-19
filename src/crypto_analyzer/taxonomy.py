"""Shared classification vocabulary.

The setup layer, the backtester, and reporting all name a direction and a setup
family, so those enums live here rather than inside one of them. A package-local
home would also work logically, but importing a leaf submodule executes its
parent package, which would drag the whole setup and feature chain into modules
that only need two strings.
"""

from enum import StrEnum


class Direction(StrEnum):
    """Side of the market a setup anticipates."""

    LONG = "long"
    SHORT = "short"


class SetupType(StrEnum):
    """Family of swing setup."""

    TREND_CONTINUATION = "trend_continuation"
    BREAKOUT = "breakout"
    MEAN_REVERSION = "mean_reversion"
