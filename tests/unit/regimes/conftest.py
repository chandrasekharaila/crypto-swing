"""Shared fixtures for regime-detection tests.

Windows are configured small so a few dozen candles exercise the same code a
long history would, and so the expected label of each fixture frame is obvious
from its construction.
"""

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import pandas as pd
import pytest

from crypto_analyzer.features import FeatureEngine, FeatureSettings
from crypto_analyzer.regimes import RegimeDetector, RegimeSettings

START = datetime(2024, 1, 1, tzinfo=UTC)
TIMEFRAME = "1h"
ROWS = 90
SPIKE_ROWS = 3

FAST_WINDOW = 3
SLOW_WINDOW = 5
EMA_SLOPE_SPAN = 3
ATR_PERIOD = 3
VOLATILITY_LOOKBACK = 5


def _small_feature_settings() -> FeatureSettings:
    return FeatureSettings(
        sma_windows=(FAST_WINDOW, SLOW_WINDOW),
        ema_windows=(EMA_SLOPE_SPAN,),
        ma_distance_windows=(FAST_WINDOW,),
        ema_slope_window=2,
        macd_fast=2,
        macd_slow=SLOW_WINDOW,
        macd_signal=2,
        atr_period=ATR_PERIOD,
        rolling_std_windows=(3,),
        volume_ma_windows=(3,),
        relative_volume_window=3,
        volume_change_period=1,
        structure_lookback=3,
        consolidation_window=3,
        sr_proximity_window=3,
        rsi_period=3,
        stoch_k_period=3,
        stoch_d_period=2,
        roc_windows=(2,),
        return_windows=(1,),
    )


def _small_regime_settings() -> RegimeSettings:
    return RegimeSettings(
        trend_fast_window=FAST_WINDOW,
        trend_slow_window=SLOW_WINDOW,
        trend_ema_slope_span=EMA_SLOPE_SPAN,
        volatility_atr_period=ATR_PERIOD,
        volatility_lookback=VOLATILITY_LOOKBACK,
    )


def _frame(close: Sequence[float], span: Sequence[float] | float) -> pd.DataFrame:
    """Build a canonical OHLCV frame whose candle span drives true range."""
    rows = len(close)
    opens = [close[0], *close[:-1]]
    spans = [span] * rows if isinstance(span, int | float) else list(span)
    open_time = pd.date_range(START, periods=rows, freq=TIMEFRAME)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "close_time": open_time
            + pd.Timedelta(TIMEFRAME)
            - pd.Timedelta(milliseconds=1),
            "open": opens,
            "high": [
                max(open_price, value) + width
                for open_price, value, width in zip(opens, close, spans, strict=True)
            ],
            "low": [
                min(open_price, value) - width
                for open_price, value, width in zip(opens, close, spans, strict=True)
            ],
            "close": list(close),
            "volume": [100.0 + index for index in range(rows)],
        }
    )


def _with_span(spike: float, base: float) -> list[float]:
    return [base] * (ROWS - SPIKE_ROWS) + [spike] * SPIKE_ROWS


@pytest.fixture
def feature_settings() -> FeatureSettings:
    return _small_feature_settings()


@pytest.fixture
def regime_settings() -> RegimeSettings:
    return _small_regime_settings()


@pytest.fixture
def run_regimes(
    feature_settings: FeatureSettings, regime_settings: RegimeSettings
) -> Callable[[pd.DataFrame], object]:
    """Return a callable that computes features and then regimes for a frame."""
    engine = FeatureEngine(feature_settings)
    detector = RegimeDetector(regime_settings)

    def run(frame: pd.DataFrame):
        features = engine.compute(frame, symbol="BTC/USDT", timeframe=TIMEFRAME)
        return detector.detect(features)

    return run


@pytest.fixture
def uptrend_frame() -> pd.DataFrame:
    """A compounding uptrend, so every distance stays above its threshold in percent."""
    return _frame([100.0 * 1.02**index for index in range(ROWS)], 0.10)


@pytest.fixture
def downtrend_frame() -> pd.DataFrame:
    return _frame([200.0 * 0.98**index for index in range(ROWS)], 0.10)


@pytest.fixture
def sideways_frame() -> pd.DataFrame:
    return _frame([100.0 + 0.05 * math.sin(index / 4) for index in range(ROWS)], 0.10)


@pytest.fixture
def volatility_spike_frame() -> pd.DataFrame:
    close = [100.0 + 0.05 * math.sin(index / 4) for index in range(ROWS)]
    return _frame(close, _with_span(spike=3.0, base=0.10))


@pytest.fixture
def calm_after_volatility_frame() -> pd.DataFrame:
    close = [100.0 + 0.05 * math.sin(index / 4) for index in range(ROWS)]
    return _frame(close, _with_span(spike=0.10, base=3.0))
