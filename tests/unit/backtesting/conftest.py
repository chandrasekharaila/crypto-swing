"""Shared fixtures for backtesting tests.

Candles and signals are built explicitly so every expected fill can be worked out
by hand rather than read back off the implementation.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from crypto_analyzer.backtesting.config import BacktestSettings
from crypto_analyzer.backtesting.simulator import CandleSeries
from crypto_analyzer.backtesting.types import (
    ExitReason,
    TradeRecord,
)
from crypto_analyzer.evidence import Evidence
from crypto_analyzer.setups.types import PriceContext, SetupSignal
from crypto_analyzer.taxonomy import Direction, SetupType

START = datetime(2024, 1, 1, tzinfo=UTC)
TIMEFRAME = "4h"
DELTA = timedelta(hours=4)


def _frame(rows: Sequence[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build a candle frame from explicit (open, high, low, close) rows."""
    index = pd.date_range(START, periods=len(rows), freq=TIMEFRAME)
    return pd.DataFrame(
        {
            "open_time": index,
            "close_time": index + DELTA - timedelta(milliseconds=1),
            "open": [row[0] for row in rows],
            "high": [row[1] for row in rows],
            "low": [row[2] for row in rows],
            "close": [row[3] for row in rows],
            "volume": [1.0] * len(rows),
        }
    )


def _signal(
    index: int,
    *,
    direction: Direction = Direction.LONG,
    setup: SetupType = SetupType.BREAKOUT,
    stop: float | None = 95.0,
    signal_price: float = 100.0,
    symbol: str = "BTC/USDT",
    timeframe: str = TIMEFRAME,
) -> SetupSignal:
    stamp = START + index * DELTA
    return SetupSignal(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=stamp,
        available_at=stamp + DELTA - timedelta(milliseconds=1),
        direction=direction,
        setup=setup,
        evidence=(Evidence(rule="synthetic", supports="long", detail="test"),),
        entry_context=PriceContext(level=signal_price, detail="reference close"),
        invalidation_context=PriceContext(level=stop, detail="structural level"),
        features=(("synthetic_feature", 1.0),),
    )


def _trade(
    *,
    symbol: str = "BTC/USDT",
    setup: SetupType = SetupType.BREAKOUT,
    direction: Direction = Direction.LONG,
    entry_index: int = 1,
    exit_index: int = 3,
    entry_price: float = 100.0,
    exit_price: float = 110.0,
    stop_price: float = 95.0,
    risk_per_unit: float = 5.0,
    exit_reason: ExitReason = ExitReason.TAKE_PROFIT,
    net_return: float = 0.10,
    r_multiple: float = 1.96,
    fees: float = 0.002,
    ambiguous: bool = False,
) -> TradeRecord:
    """Build a trade record directly, for metrics and portfolio tests."""
    return TradeRecord(
        symbol=symbol,
        timeframe=TIMEFRAME,
        setup=setup,
        direction=direction,
        signal_time=START + (entry_index - 1) * DELTA,
        signal_index=entry_index - 1,
        entry_time=START + entry_index * DELTA,
        entry_index=entry_index,
        exit_time=START + exit_index * DELTA,
        exit_index=exit_index,
        signal_price=entry_price,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_price=stop_price,
        target_price=entry_price + 2 * risk_per_unit,
        risk_per_unit=risk_per_unit,
        holding_bars=exit_index - entry_index + 1,
        exit_reason=exit_reason,
        gross_return=net_return + fees,
        net_return=net_return,
        r_multiple=r_multiple,
        fees=fees,
        slippage_cost=0.0,
        mae=-0.01,
        mfe=0.02,
        ambiguous=ambiguous,
    )


@pytest.fixture
def make_frame() -> Callable[..., pd.DataFrame]:
    return _frame


@pytest.fixture
def make_trade() -> Callable[..., TradeRecord]:
    return _trade


@pytest.fixture
def make_signal() -> Callable[..., SetupSignal]:
    return _signal


@pytest.fixture
def free_settings() -> BacktestSettings:
    """Settings with no costs and no delay, for isolating the fill geometry."""
    return BacktestSettings(
        execution_delay_bars=1,
        fee_bps=0.0,
        slippage_bps=0.0,
        take_profit_r_multiple=2.0,
    )


@pytest.fixture
def priced_settings() -> BacktestSettings:
    """Settings with only a fee, so the arithmetic stays hand-checkable."""
    return BacktestSettings(
        execution_delay_bars=1,
        fee_bps=10.0,
        slippage_bps=0.0,
        take_profit_r_multiple=2.0,
    )


@pytest.fixture
def series_of() -> Callable[[pd.DataFrame], CandleSeries]:
    return CandleSeries.from_frame
