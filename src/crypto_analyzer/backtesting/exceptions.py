"""Explicit exception categories for backtesting."""

from crypto_analyzer.exceptions import AnalyzerError


class BacktestError(AnalyzerError):
    """Base class for expected backtesting failures."""


class BacktestConfigurationError(BacktestError):
    """Raised when backtest configuration is unusable."""


class BacktestInputError(BacktestError):
    """Raised when the inputs cannot support a backtest."""


class BacktestLeakageError(BacktestError):
    """Raised when a simulation would have used information it could not have had.

    This is a programming error in the simulation, not bad data, so it is raised
    rather than logged: a silently leaked backtest is worse than a failed one.
    """
