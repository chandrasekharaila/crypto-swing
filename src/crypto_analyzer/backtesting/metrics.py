"""Performance metrics for a simulated run.

Censored trades are excluded from every trade statistic, because a position that
was still open at the end of the window has no outcome to measure. Skipped
signals are counted rather than ignored, so the reader can see how much of the
signal set the simulation actually acted on.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from crypto_analyzer.backtesting.types import TradeRecord
from crypto_analyzer.data.validators import timeframe_duration

_DAYS_PER_YEAR = 365.25


def bars_per_year(timeframe: str) -> float:
    """Return how many candles of ``timeframe`` elapse in a year."""
    duration = timeframe_duration(timeframe)
    return timedelta(days=_DAYS_PER_YEAR) / duration


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """Summary of one simulated run.

    ``expectancy`` is the mean outcome in risk units rather than in return
    fraction: every trade risks the same fraction of equity, so the R multiple is
    the number that is comparable across setups and symbols.
    """

    total_trades: int
    censored_trades: int
    skipped_signals: int
    win_rate: float
    average_return: float
    median_return: float
    expectancy: float
    profit_factor: float
    average_r_multiple: float
    average_holding_bars: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_return: float
    max_drawdown: float
    max_drawdown_bars: int
    sharpe_ratio: float
    ambiguous_exits: int
    account_ruined: bool
    """Whether equity reached zero, after which no further trades are taken."""

    exit_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize for experiment records."""
        return {
            "total_trades": self.total_trades,
            "censored_trades": self.censored_trades,
            "skipped_signals": self.skipped_signals,
            "win_rate": self.win_rate,
            "average_return": self.average_return,
            "median_return": self.median_return,
            "expectancy": self.expectancy,
            "profit_factor": self.profit_factor,
            "average_r_multiple": self.average_r_multiple,
            "average_holding_bars": self.average_holding_bars,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_bars": self.max_drawdown_bars,
            "sharpe_ratio": self.sharpe_ratio,
            "ambiguous_exits": self.ambiguous_exits,
            "account_ruined": self.account_ruined,
            "exit_counts": dict(self.exit_counts),
        }


def _longest_run(outcomes: Sequence[bool]) -> tuple[int, int]:
    best_win = best_loss = current_win = current_loss = 0
    for won in outcomes:
        current_win = current_win + 1 if won else 0
        current_loss = 0 if won else current_loss + 1
        best_win = max(best_win, current_win)
        best_loss = max(best_loss, current_loss)
    return best_win, best_loss


def _profit_factor(returns: Sequence[float]) -> float:
    gains = sum(value for value in returns if value > 0.0)
    losses = sum(value for value in returns if value <= 0.0)
    if losses == 0.0:
        return math.inf if gains > 0.0 else 0.0
    return gains / abs(losses)


def _drawdown_stats(equity: pd.Series) -> tuple[float, int]:
    if equity.empty:
        return 0.0, 0
    drawdown = equity / equity.cummax() - 1.0
    longest = current = 0
    for value in drawdown:
        current = current + 1 if value < 0.0 else 0
        longest = max(longest, current)
    return float(drawdown.min()), longest


def _sharpe(equity: pd.Series, periods_per_year: float) -> float:
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    if returns.empty:
        return 0.0
    deviation = float(returns.std())
    # A single return has an undefined sample deviation, and a flat curve has a
    # zero one. Neither is evidence of risk-adjusted return, so both report zero
    # rather than a NaN that would silently poison every comparison.
    if not math.isfinite(deviation) or deviation == 0.0:
        return 0.0
    return float(returns.mean()) / deviation * math.sqrt(periods_per_year)


def compute_metrics(
    trades: Sequence[TradeRecord],
    equity: pd.DataFrame,
    *,
    skipped_signals: int,
    periods_per_year: float,
) -> MetricsReport:
    """Summarise trades and an equity curve into one report."""
    usable = [trade for trade in trades if trade.is_usable]
    censored = len(trades) - len(usable)
    returns = [trade.net_return for trade in usable]
    ordered = sorted(usable, key=lambda trade: (trade.exit_time, trade.symbol))

    equity_series = (
        equity["equity"] if "equity" in equity.columns else pd.Series(dtype=float)
    )
    max_drawdown, max_drawdown_bars = _drawdown_stats(equity_series)
    total_return = (
        float(equity_series.iloc[-1]) / float(equity_series.iloc[0]) - 1.0
        if len(equity_series) > 1
        else 0.0
    )
    wins, losses = _longest_run([trade.is_win for trade in ordered])
    exit_counts: dict[str, int] = {}
    for trade in trades:
        exit_counts[trade.exit_reason.value] = (
            exit_counts.get(trade.exit_reason.value, 0) + 1
        )

    return MetricsReport(
        total_trades=len(usable),
        censored_trades=censored,
        skipped_signals=skipped_signals,
        win_rate=(
            sum(1 for trade in usable if trade.is_win) / len(usable) if usable else 0.0
        ),
        average_return=statistics.fmean(returns) if returns else 0.0,
        median_return=statistics.median(returns) if returns else 0.0,
        expectancy=(
            statistics.fmean([trade.r_multiple for trade in usable]) if usable else 0.0
        ),
        profit_factor=_profit_factor(returns),
        average_r_multiple=(
            statistics.fmean([trade.r_multiple for trade in usable]) if usable else 0.0
        ),
        average_holding_bars=(
            statistics.fmean([trade.holding_bars for trade in usable])
            if usable
            else 0.0
        ),
        max_consecutive_wins=wins,
        max_consecutive_losses=losses,
        total_return=total_return,
        max_drawdown=max_drawdown,
        max_drawdown_bars=max_drawdown_bars,
        sharpe_ratio=_sharpe(equity_series, periods_per_year),
        ambiguous_exits=sum(1 for trade in trades if trade.ambiguous),
        account_ruined=bool((equity_series <= 0.0).any()),
        exit_counts=tuple(sorted(exit_counts.items())),
    )
