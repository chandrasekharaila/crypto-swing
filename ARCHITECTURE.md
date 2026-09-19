# Architecture

## Phase 1 data flow

```text
Configuration
    ↓
Binance public collector
    ↓
OHLCV validation
    ↓
Parquet storage and cache
```

The Binance collector uses the unauthenticated Spot REST kline endpoint. It
paginates bounded UTC ranges, retries transient network and API failures, honors
numeric or HTTP-date `Retry-After` headers, and removes candles that were not
closed at the collector's clock snapshot.

The OHLCV validator accepts canonical pandas DataFrames and returns an immutable
report of structured errors and warnings. It never sorts, fills, drops, or
otherwise repairs input data. Internal gaps are warnings; invalid structure,
types, values, ordering, and incomplete candles are errors.

Typed candles are converted to a canonical DataFrame before validation and
storage. Raw and processed Parquet datasets use separate deterministic paths:
`data/<layer>/binance/<BASE>-<QUOTE>/<timeframe>.parquet`. Raw updates are
merged atomically, exact overlaps are deduplicated, and conflicting overlaps
are rejected. Cache range inspection identifies only missing intervals for a
collector; the storage module has no Binance client dependency.

All requested ranges must begin and end on exact Binance candle boundaries.
After collection, the pipeline checks the cache again and reports unresolved
ranges explicitly. The CLI returns a failure status when an update remains
incomplete.

`MarketDataPipeline` coordinates the boundaries without coupling storage to
Binance. It checks cache coverage, invokes the collector only for missing
ranges, converts typed candles to a canonical frame, validates the frame, and
merges it into raw Parquet storage. The CLI is a thin configuration and logging
layer over this service.

The package uses a `src` layout. Data responsibilities are separated into
collectors, validators, processors, storage, and orchestration modules.

All timestamps will be normalized to UTC. Incomplete candles will be excluded,
missing candles will be reported rather than fabricated, and generated data
will remain separate from raw downloads.

## Data availability convention

Binance `close_time` is the inclusive final millisecond of a candle. A candle's
OHLCV values are available only when the evaluation time is strictly later than
`close_time`; they must never be treated as known at `open_time`. Phase 2
features must use this convention (or the equivalent next candle boundary) to
avoid look-ahead bias.
