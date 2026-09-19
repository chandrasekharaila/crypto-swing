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
features follow this convention through `FeatureSet.available_at`.

## Phase 2 feature flow

```text
Canonical OHLCV frame
    ↓
OHLCV validation (reused from Phase 1)
    ↓
Feature engine
    ↓
FeatureSet (frame, metadata, available_at)
```

Feature calculation lives in `crypto_analyzer/features/`. One module per group
(`price`, `returns`, `trend`, `momentum`, `volatility`, `volume`, `structure`)
owns its compute functions and a `register_<group>_features` function. Every
window and period comes from `FeatureSettings`, so no parameter is duplicated
inside the feature modules.

`build_default_registry` expands those settings into `FeatureDefinition` records,
each binding a pure `frame -> Series` function to its name, group, description,
lookback, and parameters. `FeatureEngine` validates the input frame, computes
every definition in registry order, and assembles a `FeatureSet`. Computation is
in memory; nothing is written to `data/features/` in this phase.

The engine reuses the Phase 1 `OHLCVValidator` rather than re-implementing
schema checks, and coerces the exact `Decimal` price columns to float64 on a
private copy so trailing-window arithmetic is exact and the caller's frame is
never modified.

### Look-ahead prevention

Features are point-in-time: a value at row `t` uses only candles up to and
including `t`. The frame stays keyed by `open_time` so it joins to raw data, and
`FeatureSet.available_at` carries the `close_time` at which the value actually
becomes known. Later phases must treat `available_at`, not `open_time`, as the
earliest usable time.

Calculation is restricted to trailing primitives: `rolling(w, min_periods=w)`,
`ewm(span=w, adjust=False)`, `diff`, `shift` with a non-negative offset, and
`pct_change`. Backward shifts, centred windows, backward filling, interpolation,
and statistics computed across a whole series are forbidden.

Two further rules make the guarantee enforceable rather than aspirational. Every
compute function must return a Series with the input's index and length, so
trimming or reindexing is rejected by the engine. And `features/causality.py`
asserts the property directly: `assert_prefix_stable` recomputes on a truncated
frame and requires the earlier values to match, while `assert_future_independent`
replaces every later candle with a sentinel and requires the earlier values to
stay put. Both run over the entire registry, so a newly registered feature is
covered automatically.

Warmup positions are missing values and are never filled. Each definition
declares the candles it requires as `lookback`, and the first `lookback - 1`
rows of that column are missing. Normalization, scaling, and feature selection
are deliberately absent: they must be fit on training data only and belong to a
later phase.
