"""Shared fixtures for setup-detection tests.

Two styles of test live here. Rule tests build a feature set directly so the
detector's own logic is exercised against exact values, with no indicator in the
way. Integration and causality tests run the real feature and regime layers so
the whole pipeline is covered.
"""

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import pandas as pd
import pytest

from crypto_analyzer.config import FeatureSettings
from crypto_analyzer.features import FeatureEngine
from crypto_analyzer.features.types import FeatureGroup, FeatureMetadata, FeatureSet
from crypto_analyzer.regimes import RegimeDetector, RegimeSettings
from crypto_analyzer.regimes.types import RegimeResult
from crypto_analyzer.setups import SetupDetector, SetupSettings

START = datetime(2024, 1, 1, tzinfo=UTC)
TIMEFRAME = "1h"
SYMBOL = "BTC/USDT"

FAST = 3
SLOPE_SPAN = 3
RSI_PERIOD = 3
VOLUME_WINDOW = 3
STRUCTURE_LOOKBACK = 3
BOLLINGER_PERIOD = 3
SQUEEZE_LOOKBACK = 5

FEATURE_DEFAULTS: dict[str, float] = {
    "trend_price_vs_sma_3": 0.0,
    "trend_ema_slope_3": 0.0,
    "momentum_rsi_3": 50.0,
    "volume_relative_3": 1.0,
    "structure_breakout_up_3": 0.0,
    "structure_breakout_down_3": 0.0,
    "structure_recent_high_3": 110.0,
    "structure_recent_low_3": 90.0,
    "volatility_bollinger_width_3": 0.10,
    "volatility_bollinger_position_3": 0.5,
}


def _small_feature_settings() -> FeatureSettings:
    return FeatureSettings(
        sma_windows=(FAST, 5),
        ema_windows=(SLOPE_SPAN,),
        ma_distance_windows=(FAST,),
        ema_slope_window=2,
        macd_fast=2,
        macd_slow=5,
        macd_signal=2,
        atr_period=3,
        rolling_std_windows=(3,),
        volume_ma_windows=(3,),
        relative_volume_window=VOLUME_WINDOW,
        volume_change_period=1,
        structure_lookback=STRUCTURE_LOOKBACK,
        consolidation_window=3,
        sr_proximity_window=3,
        bollinger_period=BOLLINGER_PERIOD,
        rsi_period=RSI_PERIOD,
        stoch_k_period=3,
        stoch_d_period=2,
        roc_windows=(2,),
        return_windows=(1,),
    )


def _small_regime_settings() -> RegimeSettings:
    return RegimeSettings(
        trend_fast_window=FAST,
        trend_slow_window=5,
        trend_ema_slope_span=SLOPE_SPAN,
        volatility_atr_period=3,
        volatility_lookback=5,
    )


def _small_setup_settings() -> SetupSettings:
    return SetupSettings(
        trend_fast_window=FAST,
        ema_slope_span=SLOPE_SPAN,
        rsi_period=RSI_PERIOD,
        volume_window=VOLUME_WINDOW,
        structure_lookback=STRUCTURE_LOOKBACK,
        bollinger_period=BOLLINGER_PERIOD,
        breakout_squeeze_lookback=SQUEEZE_LOOKBACK,
    )


def _inputs(
    trend_regimes: Sequence[str],
    overrides: dict[str, Sequence[float]] | None = None,
    close: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, FeatureSet, RegimeResult]:
    """Build candles, a feature set, and regimes with exact control over values."""
    rows = len(trend_regimes)
    overrides = overrides or {}
    index = pd.date_range(START, periods=rows, freq=TIMEFRAME)
    columns = {
        name: list(overrides.get(name, [default] * rows))
        for name, default in FEATURE_DEFAULTS.items()
    }
    unknown = set(overrides).difference(columns)
    if unknown:
        raise KeyError(f"unknown feature column(s): {sorted(unknown)}")

    close_values = list(close) if close is not None else [100.0] * rows
    features = FeatureSet(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        frame=pd.DataFrame({"open_time": index, **columns}),
        metadata=tuple(
            FeatureMetadata(
                name=name,
                group=FeatureGroup.TREND,
                description="synthetic",
                lookback=1,
            )
            for name in columns
        ),
        available_at=pd.Series(
            index + pd.Timedelta(TIMEFRAME) - pd.Timedelta(milliseconds=1)
        ),
    )
    regimes = RegimeResult(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        frame=pd.DataFrame(
            {
                "open_time": index,
                "trend_regime": list(trend_regimes),
                "volatility_regime": ["normal_volatility"] * rows,
                "combined_regime": [
                    f"{regime}__normal_volatility" for regime in trend_regimes
                ],
            }
        ),
        available_at=features.available_at,
        settings=_small_regime_settings(),
    )
    candles = pd.DataFrame(
        {
            "open_time": index,
            "close_time": index
            + pd.Timedelta(TIMEFRAME)
            - pd.Timedelta(milliseconds=1),
            "open": close_values,
            "high": [value + 1.0 for value in close_values],
            "low": [value - 1.0 for value in close_values],
            "close": close_values,
            "volume": [100.0] * rows,
        }
    )
    return candles, features, regimes


def _candle_frame(
    close: Sequence[float], span: Sequence[float] | float
) -> pd.DataFrame:
    rows = len(close)
    opens = [close[0], *close[:-1]]
    spans = [span] * rows if isinstance(span, int | float) else list(span)
    index = pd.date_range(START, periods=rows, freq=TIMEFRAME)
    return pd.DataFrame(
        {
            "open_time": index,
            "close_time": index
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


@pytest.fixture
def setup_settings() -> SetupSettings:
    return _small_setup_settings()


@pytest.fixture
def detector(setup_settings: SetupSettings) -> SetupDetector:
    return SetupDetector(setup_settings)


@pytest.fixture
def build_inputs() -> Callable[..., tuple[pd.DataFrame, FeatureSet, RegimeResult]]:
    return _inputs


@pytest.fixture
def run_pipeline() -> Callable[[pd.DataFrame], object]:
    """Return a callable running features, regimes, and setups over candles."""
    engine = FeatureEngine(_small_feature_settings())
    regimes = RegimeDetector(_small_regime_settings())
    setups = SetupDetector(_small_setup_settings())

    def run(candles: pd.DataFrame):
        features = engine.compute(candles, symbol=SYMBOL, timeframe=TIMEFRAME)
        return setups.detect(features, candles, regimes.detect(features))

    return run


@pytest.fixture
def make_candles() -> Callable[..., pd.DataFrame]:
    return _candle_frame


@pytest.fixture
def trending_candles() -> pd.DataFrame:
    """A long compounding uptrend with steady volatility."""
    return _candle_frame([100.0 * 1.02**index for index in range(140)], 0.15)


@pytest.fixture
def volatile_candles() -> pd.DataFrame:
    """A wavy market whose candle span changes, so regimes and setups vary."""
    close = [100.0 + 6.0 * math.sin(index / 7) + 0.05 * index for index in range(140)]
    span = [0.2 + 0.5 * abs(math.sin(index / 11)) for index in range(140)]
    return _candle_frame(close, span)
