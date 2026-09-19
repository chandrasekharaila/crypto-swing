# Architecture

## Phase 1 data flow

```text
Configuration
    ↓
Binance public collector (not yet implemented)
    ↓
OHLCV validation and cleaning (not yet implemented)
    ↓
Parquet storage and cache (not yet implemented)
```

The package uses a `src` layout. Data responsibilities are separated into
collectors, validators, processors, and storage modules. A future data service
will coordinate those components without embedding their behavior in the HTTP
client.

All timestamps will be normalized to UTC. Incomplete candles will be excluded,
missing candles will be reported rather than fabricated, and generated data
will remain separate from raw downloads.
