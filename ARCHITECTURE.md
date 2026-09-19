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

Breakout and swing-label features compare against levels formed entirely from
earlier candles. A breakout uses the window immediately preceding the current
candle, and a swing label compares the latest window extreme against the
preceding, non-overlapping window, so neither can define its own reference.

Warmup positions are missing values and are never filled. Each definition
declares the candles it requires as `lookback`, and the first `lookback - 1`
rows of that column are missing.

### Undefined values

A ratio whose denominator is zero is not a measurement. Unless a feature can
name a documented neutral value, the result is missing rather than a fabricated
zero. `safe_divide` implements that default and every call site states its
choice explicitly: a zero-width candle has no position, so
`price_close_location` and `volatility_bollinger_position` report 0.5, and
`momentum_stoch_k` reports 50. Magnitudes that are genuinely absent on such a
candle -- a body, a wick, a band width -- report 0.0.

A feature that can also go missing *after* its warmup declares it through
`may_be_undefined`. That covers the rolling correlation, which has no value when
either series is constant, and the volume change and relative volume, which have
none when the base is exactly zero. A registry-wide test asserts that every
feature is either defined past warmup or declares that it may not be, so an
undeclared gap fails the suite rather than reaching a model.

Normalization, scaling, and feature selection are deliberately absent: they must
be fit on training data only and belong to a later phase.

## Phase 3 regime classification

```text
FeatureSet
    ↓
rule evaluation (four trend rules, one volatility ratio)
    ↓
RegimeResult (trend axis, volatility axis, combined label, rule measurements)
```

Trend and volatility are independent properties, so they are classified on
separate axes and combined afterwards. A market trending up while volatility is
elevated is described by both facts instead of being forced into one of them.

`RegimeDetector` consumes a `FeatureSet` and computes no indicator of its own.
Every threshold, window, and agreement requirement lives in `RegimeSettings`, so
a classification can be reproduced exactly from its configuration.

### Rules

Four directional rules compare a measurement against a configured threshold and
vote bullish, bearish, or neutral:

| Rule | Measurement |
|---|---|
| `trend_ma_spread` | fast simple average relative to the slow one |
| `trend_price_distance` | close relative to its fast simple average |
| `trend_ema_slope` | slope of the fast exponential average |
| `trend_ema_spread` | fast exponential average relative to the slow one |

A direction is reported only when at least `trend_min_agreeing_rules` vote for it
and those votes outweigh the opposing ones; otherwise the market is `sideways`.
Requiring agreement rather than a single signal is what stops one noisy rule from
flipping the classification.

Volatility compares `volatility_atr_pct` with its own trailing mean over
`volatility_lookback` candles. The bands are relative to the instrument's recent
norm rather than an absolute level, which would not transfer across symbols or
market eras.

### Reasoning

`RegimeResult.explain(position)` returns one `RegimeEvidence` per rule plus an
aggregate entry, each carrying the rule name, the regime it supports, the
measured value, and the threshold it was compared against. The explanation is
built from the stored measurements rather than reconstructed, so it cannot
disagree with the label it accompanies.

### Warm-up

The two axes warm up independently: trend needs its slow average, and volatility
needs its ATR period plus the whole trailing norm window. An axis that is not yet
classifiable reports `unknown` instead of a guess, and the combined label reports
`unknown` until neither axis is.

Regime labels inherit the feature layer's causality, and the regime tests verify
it directly: replacing every candle after a cut point leaves every earlier label
and rule measurement unchanged.
