# Crypto Swing Trade Analyzer

A research-oriented cryptocurrency market analysis project. It does not place
trades or manage exchange accounts.

## Current phase

Phase 1 — Data Foundation. The current implementation provides the project
structure, typed configuration, a Binance public historical OHLCV collector,
non-mutating OHLCV dataset validation, and atomic local Parquet storage with
cache-aware range detection.

## Requirements

- Python 3.11+

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
```

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
