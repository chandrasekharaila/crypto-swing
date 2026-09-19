# Crypto Swing Trade Analyzer

A research-oriented cryptocurrency market analysis project. It does not place
trades or manage exchange accounts.

## Current phase

Phase 3 — Rule-Based Analysis. Phase 1 provides the project structure, typed
configuration, a Binance public historical OHLCV collector, non-mutating OHLCV
dataset validation, and atomic local Parquet storage with cache-aware range
detection. Phase 2 adds a deterministic, point-in-time feature engine covering
price, returns, trend, momentum, volatility, volume, and market structure.
Phase 3 adds a transparent, rule-based regime detector and a swing setup engine.
Setups are hypotheses about market context — nothing in this project claims they
are profitable.

## Requirements

- Python 3.11+

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
```

The deterministic suite skips external access. To run the Binance public API
contract test explicitly:

```bash
RUN_BINANCE_INTEGRATION=1 pytest tests/integration
```

Linting, formatting, and static type checking:

```bash
ruff check src tests
ruff format --check src tests
mypy
```

`mypy` runs in strict mode over `src` only.

## Configuration

`crypto_analyzer.config.AppSettings` centralizes Phase 1 settings. Defaults
cover BTC/USDT, ETH/USDT, and SOL/USDT on 15m, 1h, 4h, and 1d timeframes.
Settings can be overridden explicitly when constructing the object.

An optional JSON configuration file can contain any `AppSettings` fields.

## Download market data

```bash
python -m crypto_analyzer download --symbol BTC/USDT --timeframe 4h
```

By default, this updates the latest configured number of closed candles. Use
`--start` and `--end` with aligned ISO-8601 timestamps for a specific range, or
`--candles` for a smaller recent update. Existing Parquet coverage is checked
before Binance is called.

Downloaded OHLCV values become usable only after their inclusive `close_time`.
They must not be interpreted as information available at `open_time` when
building features or backtests.

## Feature engineering

```python
from crypto_analyzer.data.storage import ParquetMarketDataStore
from crypto_analyzer.features import FeatureEngine

store = ParquetMarketDataStore(settings.data_directory)
frame = store.load_raw("BTC/USDT", "4h")

features = FeatureEngine().compute(frame, symbol="BTC/USDT", timeframe="4h")

features.frame  # open_time plus one column per generated feature
features.metadata  # name, group, description, lookback, parameters
features.available_at  # close_time, when each row's values become known
```

Features are point-in-time: a row is computed only from that candle and earlier
ones, and it becomes available at that candle's `close_time`. The first
`lookback - 1` rows of each column are missing values and are never filled.
Values are raw and unscaled — normalization belongs to a later phase and must be
fit on training data only. Feature windows and periods are configured through
`AppSettings.features`.

## Regime classification

```python
from crypto_analyzer.regimes import RegimeDetector

regimes = RegimeDetector().detect(features)

regimes.frame[["open_time", "trend_regime", "volatility_regime", "combined_regime"]]

for item in regimes.explain():          # newest row by default
    print(f"{item.rule:20s} {item.supports:28s} {item.detail}")
```

Trend and volatility are classified on separate axes, so a market that is
trending up while volatility is elevated reports both. Each axis is `unknown`
until it has enough history — the two warm up independently.

`explain()` returns one entry per rule with the measured value and the threshold
it was compared against, so a classification can always be traced back to the
measurements behind it. Every threshold lives in `AppSettings.regimes`.

## Swing setups

```python
from crypto_analyzer.setups import SetupDetector

scan = SetupDetector().detect(features, candles, regimes)

for signal in scan.signals:
    print(signal.timestamp, signal.setup.value, signal.direction.value)
    for item in signal.evidence:
        print(f"   {item.rule}: {item.detail}")
    print("   entry       :", signal.entry_context.level)
    print("   invalidation:", signal.invalidation_context.level)
```

Three families are detected: trend continuation, breakout, and mean reversion.
Each requires its regime context, so a continuation needs a trend and a mean
reversion stays in a sideways market. Setups are hypotheses to be tested, not
statements about profitability — no return, win rate, or probability is claimed
anywhere in this layer.

Every signal carries its reasoning, an entry reference price, a structural
invalidation level with the reasoning behind it, and the feature readings the
rules used. Thresholds live in `AppSettings.setups`.
