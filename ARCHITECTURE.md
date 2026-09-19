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

`RegimeResult.explain(position)` returns one `Evidence` per rule plus an
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

## Phase 3 setup detection

```text
candles + FeatureSet + RegimeResult
    ↓
setup rules (trend continuation, breakout, mean reversion)
    ↓
SetupScan of SetupSignal records
```

A setup is a hypothesis about market context. It is not a claim about
profitability: nothing in this layer estimates a return, a win rate, or a
probability of profit.

`SetupDetector` reads the feature and regime layers rather than recomputing
anything, so each family inherits the causality of the values beneath it. Every
family also requires its regime context, which means a trend continuation cannot
fire in a market that is not trending, and a mean reversion stays inside a
sideways market unless that requirement is switched off.

### Families

| Family | Requires | Direction |
|---|---|---|
| trend continuation | a trending regime, price pulled back to its fast average, slope still with the trend, momentum not broken down, participation | with the trend |
| breakout | a close beyond the prior window's extreme, band width narrow against its own trailing norm, volume expansion, price not already extended | with the break |
| mean reversion | a sideways regime, price stretched from its fast average, momentum at an extreme, close outside a Bollinger band | against the stretch |

Band width is compared against its own trailing norm rather than an absolute
width, for the same reason volatility bands are relative: a fixed width does not
transfer between instruments. The squeeze is measured on the candle *before* the
break, so the break itself cannot widen the band that qualifies it.

### Signal record

Each `SetupSignal` carries the symbol, the timestamp and the `close_time` it
became known at, the direction and family, the measured `evidence` for every
rule, an entry reference price, an invalidation level with its reasoning, and the
feature readings the rules consumed.

Invalidation is structural rather than a fixed distance: continuation and
reversion are invalidated by the trailing window extreme, and a breakout is
invalidated by a close back through the level it broke.

### Warm-up

Setups inherit the warm-up of the layers beneath them. A condition that compares
a missing value is false, so no signal can be produced before every input it
needs exists; the regime context alone rules out the earliest rows.

## Phase 4 backtesting

```text
candles + SetupScan
    ↓
fill simulation (one signal, one trade)
    ↓
portfolio assembly (sizing, equity curve, drawdown)
    ↓
metrics
```

Phase 4 exists to measure, not to tune. It changes no feature, no regime
threshold, and no setup threshold, because any of those changes would invalidate
the holdout.

### Execution model

A signal is decided at its candle's close, so the fill happens at the open of a
later bar. `execution_delay_bars` is constrained to be at least one, which makes
same-bar execution unrepresentable rather than merely discouraged.

The stop comes from the signal's structural invalidation level. The target is an
R multiple of the distance from the fill to that stop, so every trade risks the
same fraction of equity and the R multiple is comparable across setups and
symbols.

Each bar is examined in a fixed order: a gap through a level first, then an
ambiguous bar, then a single-sided touch, then the holding limit. A bar that
opens beyond the stop fills at that open, not at the stop, because a gap is a
real and unfavourable fact.

### The unknowable intrabar path

OHLC data does not reveal whether a bar's high or its low came first. When a bar
contains both the stop and the target, one of them must be chosen. The default is
the pessimistic reading: the stop came first. The case is counted, so a reader can
see how often it mattered rather than having to trust the choice.

The choice can be replaced by a measurement. `IntrabarWindows` holds finer candles
for the specific bars that were ambiguous, and the simulator consults it before
falling back to the assumption. `scripts/resolve_ambiguity.py` collects those bars
from a run, fetches the finer candles, and caches them, so a simulation itself never
touches the network. On the current dataset the finer candles decide about 98% of
the ambiguous cases, and they showed the pessimistic default to be right roughly
71% of the time.

### Portfolio layer

Trades are simulated independently, then assembled. Each trade risks
`risk_fraction` of equity at entry, subject to a notional cap, because risk-based
sizing alone is unbounded: a structural stop a few basis points from the entry
would otherwise imply enormous leverage and a round-trip fee that dwarfs the risk.
Equity is marked to market every bar, so the drawdown reflects open positions
rather than only closed ones. Entries stop once equity reaches zero, and the ruin
is reported rather than drawn as a negative balance.

This is a portfolio proxy, not an account. It assumes the modelled fills and
ignores margin, funding, borrow, and any limit on notional beyond the cap.

### Preventing look-ahead

| Failure | How it is prevented |
|---|---|
| Filling at the price that triggered the signal | `entry_index = signal_index + delay`, with `delay >= 1` enforced by the type |
| Using the signal bar's range to trigger an exit | The holding loop starts at the entry bar; earlier bars are unreachable |
| Stop or target derived from a later bar | Levels come from the signal record, already covered by the Phases 2 and 3 causality sweeps |
| Reading past the end of the window | The simulation is bounded by the last bar inside the window; a trade still open there is censored and excluded from metrics |
| A December trade exiting in January | Censoring rather than completion, so a development run never reads a holdout bar |

The engine also refuses to place a signal whose timestamp does not match a candle,
and a runtime check rejects any fill that is not strictly later than the moment
the signal became known.

### Stated limitations

Market impact beyond fixed slippage, funding and borrow, partial fills, exchange
outages that would prevent an exit, and queue position are all unmodelled. Fees
and slippage are stated assumptions, never fitted to the data.
