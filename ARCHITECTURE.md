# Architecture

## Phase 1 data flow

```text
Configuration
    ↓
Binance public collector
    ↓
OHLCV validation
    ↓
Parquet storage and cache (not yet implemented)
```

The Binance collector uses the unauthenticated Spot REST kline endpoint. It
paginates bounded UTC ranges, retries transient network and API failures, honors
numeric `Retry-After` headers, and removes candles that were not closed at the
collector's clock snapshot.

The OHLCV validator accepts canonical pandas DataFrames and returns an immutable
report of structured errors and warnings. It never sorts, fills, drops, or
otherwise repairs input data. Internal gaps are warnings; invalid structure,
types, values, ordering, and incomplete candles are errors.

The package uses a `src` layout. Data responsibilities are separated into
collectors, validators, processors, and storage modules. A future data service
will coordinate those components without embedding their behavior in the HTTP
client.

All timestamps will be normalized to UTC. Incomplete candles will be excluded,
missing candles will be reported rather than fabricated, and generated data
will remain separate from raw downloads.
