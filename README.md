# Crypto Swing Trade Analyzer

A research-oriented cryptocurrency market analysis project. It does not place
trades or manage exchange accounts.

## Current phase

Phase 1 — Data Foundation. The current implementation provides the project
structure, typed configuration, a Binance public historical OHLCV collector,
and non-mutating OHLCV dataset validation. Storage and caching will be added
incrementally.

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
