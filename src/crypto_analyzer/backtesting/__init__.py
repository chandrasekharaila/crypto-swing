"""Event-driven backtesting for detected setups."""

from crypto_analyzer.backtesting.config import BacktestSettings
from crypto_analyzer.backtesting.engine import BacktestEngine
from crypto_analyzer.backtesting.exceptions import (
    BacktestConfigurationError,
    BacktestError,
    BacktestInputError,
    BacktestLeakageError,
)
from crypto_analyzer.backtesting.intrabar import (
    WINDOW_COLUMNS,
    IntrabarResolver,
    IntrabarWindows,
    resolution_from_minutes,
)
from crypto_analyzer.backtesting.metrics import MetricsReport, bars_per_year
from crypto_analyzer.backtesting.portfolio import build_equity_curve
from crypto_analyzer.backtesting.simulator import CandleSeries, simulate_trade
from crypto_analyzer.backtesting.types import (
    AmbiguousExitPolicy,
    BacktestResult,
    ExitReason,
    SkippedSignal,
    SkipReason,
    TradeRecord,
)

__all__ = [
    "WINDOW_COLUMNS",
    "AmbiguousExitPolicy",
    "BacktestConfigurationError",
    "BacktestEngine",
    "BacktestError",
    "BacktestInputError",
    "BacktestLeakageError",
    "BacktestResult",
    "BacktestSettings",
    "CandleSeries",
    "ExitReason",
    "IntrabarResolver",
    "IntrabarWindows",
    "MetricsReport",
    "SkipReason",
    "SkippedSignal",
    "TradeRecord",
    "bars_per_year",
    "build_equity_curve",
    "resolution_from_minutes",
    "simulate_trade",
]
