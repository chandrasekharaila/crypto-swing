# AGENTS.md

## 1. Project Overview

Project name: **Crypto Swing Trade Analyzer**

The goal of this project is to build a research-oriented cryptocurrency market analysis system that:

1. Collects free/public cryptocurrency market data.
2. Cleans and validates historical market data.
3. Generates quantitative features.
4. Detects market regimes and potential swing-trading setups.
5. Backtests those setups on historical data.
6. Trains machine-learning models to estimate future market outcomes.
7. Scans multiple cryptocurrency pairs.
8. Produces transparent, explainable analysis.
9. Eventually uses an LLM as a research assistant that can propose and evaluate feature-engineering hypotheses.

This project is for **research and experimentation**.

It must NOT place trades, manage exchange accounts, or execute financial transactions.

---

# 2. Core Philosophy

The project should be built incrementally.

Do NOT attempt to implement the entire system at once.

Development order:

```text
Phase 1
Data Infrastructure
        ↓
Phase 2
Feature Engineering
        ↓
Phase 3
Rule-Based Analysis
        ↓
Phase 4
Backtesting
        ↓
Phase 5
Machine Learning
        ↓
Phase 6
Multi-Timeframe Analysis
        ↓
Phase 7
Market Scanner
        ↓
Phase 8
LLM Research Agent
        ↓
Phase 9
API / Dashboard / Reporting
```

Each phase must be functional and tested before moving to the next phase.

Do not implement future phases prematurely unless explicitly requested.

---

# 3. Technology Stack

Primary language:

- Python 3.11+

Core libraries:

- pandas
- numpy
- scipy
- scikit-learn
- pydantic
- httpx or requests
- ccxt where appropriate
- pytest
- matplotlib

Optional later:

- XGBoost
- FastAPI
- Qdrant
- OpenAI API
- LangGraph

Prefer the standard library and existing dependencies where possible.

Do not add dependencies without a clear reason.

---

# 4. Data Source

The primary data source is **Binance public market data**.

The initial implementation must work without requiring:

- Binance API keys
- Binance account credentials
- trading permissions

Initially support:

```text
BTC/USDT
ETH/USDT
SOL/USDT
```

The architecture must allow additional symbols to be added through configuration.

Initial timeframes:

```text
15m
1h
4h
1d
```

Primary swing-analysis timeframe:

```text
4h
```

Higher timeframes provide market context.

Lower timeframes may be used for entry refinement.

---

# 5. Data Engineering Rules

The data pipeline must be deterministic and reproducible.

Expected flow:

```text
Exchange API
     ↓
Raw OHLCV
     ↓
Validation
     ↓
Cleaning
     ↓
Storage
     ↓
Feature Engineering
```

The data collector must handle:

- API errors
- HTTP errors
- network failures
- rate limits
- retries
- missing candles
- duplicate candles
- malformed responses
- timestamp normalization
- incomplete candles

Do not use the currently forming candle for historical backtesting unless explicitly required.

Prefer closed candles.

Downloaded data should be cached locally.

Do not repeatedly request historical data that already exists locally.

---

# 6. Data Storage

Use a local storage format suitable for analytical workloads.

Preferred:

```text
Parquet
```

The project should maintain separation between:

```text
data/raw/
data/processed/
data/features/
data/experiments/
```

Never overwrite raw downloaded data during feature engineering.

Raw data should remain reproducible.

---

# 7. Project Architecture

Use a modular architecture.

Suggested structure:

```text
crypto-swing-analyzer/
│
├── AGENTS.md
├── README.md
├── ARCHITECTURE.md
├── pyproject.toml
│
├── src/
│   └── crypto_analyzer/
│       │
│       ├── config/
│       │
│       ├── data/
│       │   ├── collectors/
│       │   ├── validators/
│       │   ├── processors/
│       │   └── storage/
│       │
│       ├── features/
│       │
│       ├── regimes/
│       │
│       ├── strategies/
│       │
│       ├── backtesting/
│       │
│       ├── models/
│       │
│       ├── analysis/
│       │
│       ├── reporting/
│       │
│       └── api/
│
├── tests/
│
├── scripts/
│
├── notebooks/
│
└── data/
    ├── raw/
    ├── processed/
    ├── features/
    └── experiments/
```

Maintain separation of concerns.

Do not place unrelated functionality into one large file.

---

# 8. Coding Standards

Use:

- type hints
- descriptive names
- small functions
- clear interfaces
- docstrings where useful
- meaningful exceptions
- structured logging

Prefer composition over unnecessarily complex inheritance.

Avoid:

- giant functions
- global mutable state
- duplicated logic
- magic numbers
- hard-coded symbols
- hard-coded timeframes
- hard-coded strategy parameters

Configuration belongs in configuration files or typed configuration objects.

---

# 9. Configuration

Centralize configurable parameters.

Examples:

```text
symbols
timeframes
historical period
primary timeframe
TP percentage
SL percentage
holding horizon
transaction fees
slippage
minimum signal score
model parameters
data directories
```

Never scatter these values throughout the codebase.

---

# 10. Feature Engineering

Features must only use information available at the feature timestamp.

Initial feature groups:

### Price

- returns
- log returns
- rolling returns
- candle body
- candle range
- wick size
- price changes

### Trend

- SMA
- EMA
- EMA slope
- distance from moving averages
- trend direction

### Momentum

- RSI
- MACD
- MACD histogram
- ROC
- stochastic oscillator

### Volatility

- ATR
- rolling standard deviation
- Bollinger Bands
- Bollinger Band width

### Volume

- volume change
- rolling volume
- relative volume
- volume moving averages
- price-volume relationships

### Market Structure

- recent highs
- recent lows
- higher highs
- higher lows
- lower highs
- lower lows
- consolidation ranges
- breakout candidates
- support/resistance proximity

Feature generation must be deterministic.

Every feature should have a clear definition.

---

# 11. Look-Ahead Bias Prevention

This is one of the most important rules in the entire project.

NEVER allow future information to leak into:

- features
- labels
- training
- signal generation
- backtesting

For a feature at timestamp `t`, only information available at or before `t` may be used.

Be especially careful with:

- rolling calculations
- normalization
- scaling
- feature selection
- support/resistance calculations
- market regime detection
- target construction
- train/test preprocessing

Fit preprocessing transformations only on training data.

Never use the test set to influence model development.

---

# 12. Machine Learning Rules

Machine learning should be introduced only after the rule-based analysis and backtesting infrastructure works.

The initial ML objective should NOT be exact price prediction.

Prefer classification problems such as:

> Given the information available at timestamp `t`, what is the probability that the price reaches a predefined target before a predefined stop within a future time horizon?

Possible labels:

```text
0 = stop reached first
1 = neither reached
2 = target reached first
```

Make target, stop, and horizon configurable.

Initial models:

- Logistic Regression
- Random Forest
- Gradient Boosting

Additional models may be added later.

Do not add sophisticated models simply for complexity.

---

# 13. Time-Series Validation

Never randomly shuffle market data for the primary evaluation.

Use chronological splits.

Example:

```text
Historical Data
│
├── Training
│
├── Validation
│
└── Final Test
```

The final test set should remain untouched until the experiment is finalized.

If walk-forward validation is appropriate, implement it explicitly.

Record:

- training period
- validation period
- test period
- feature set
- model
- hyperparameters
- metrics

Experiments must be reproducible.

---

# 14. Rule-Based Strategy Engine

Before ML, implement transparent rule-based setups.

Potential setup categories:

### Trend Continuation

Possible evidence:

- higher-timeframe trend
- price relative to moving averages
- moving-average slope
- pullback
- momentum recovery
- volume confirmation

### Breakout

Possible evidence:

- consolidation
- established resistance
- breakout
- volume expansion
- confirmation/retest

### Mean Reversion

Possible evidence:

- unusually extended price
- volatility conditions
- momentum extremes
- distance from moving averages

These are hypotheses, not assumptions of profitability.

Every strategy must be backtested.

---

# 15. Market Regime Detection

Create a regime-analysis component.

Initial regimes:

```text
Bullish Trend
Bearish Trend
Sideways / Consolidation
High Volatility
Low Volatility
```

Start with transparent rule-based classification.

ML-based regime detection can be explored later.

The regime classifier must not use future information.

---

# 16. Backtesting

The backtesting engine must be independent from the strategy implementation.

Expected flow:

```text
Historical Data
      ↓
Features
      ↓
Strategy
      ↓
Signal
      ↓
Entry
      ↓
Position Management
      ↓
Exit
      ↓
Trade Record
      ↓
Performance Metrics
```

Each trade record should contain:

```text
symbol
timestamp
direction
entry price
exit price
stop loss
take profit
setup
holding period
return
fees
slippage
```

Calculate:

- number of trades
- win rate
- average return
- median return
- expectancy
- profit factor
- maximum drawdown
- Sharpe ratio
- average holding period
- consecutive wins
- consecutive losses
- total return

Include configurable:

- transaction fees
- slippage

Do not hide losing trades.

---

# 17. Multi-Timeframe Analysis

The system should eventually combine:

```text
1D
 ↓
4H
 ↓
1H
 ↓
15M
```

Higher timeframes provide context.

Lower timeframes provide more granular information.

The system should identify:

- alignment
- disagreement
- trend direction
- momentum
- volatility
- potential setup

Do not force a signal when timeframes disagree.

---

# 18. Market Scanner

The scanner should eventually:

```text
Symbols
   ↓
Download/update data
   ↓
Feature generation
   ↓
Regime detection
   ↓
Strategy detection
   ↓
ML prediction
   ↓
Scoring
   ↓
Ranking
   ↓
Report
```

Initially scan only:

```text
BTC/USDT
ETH/USDT
SOL/USDT
```

The architecture must support expansion.

---

# 19. Signal Explainability

Every generated setup must be explainable.

Example structure:

```text
Symbol: BTC/USDT

Direction: LONG

Setup:
Trend Pullback

Evidence:
- 4H trend bullish
- Price above EMA 50
- EMA slope positive
- RSI recovering
- Relative volume increasing
- Price near previous support

Model probability:
67%

Risk factors:
- High volatility
- Resistance nearby
- Lower timeframe disagreement
```

Do not output unexplained scores.

---

# 20. Scoring

A future scoring system may combine:

```text
Trend
Momentum
Volume
Volatility
Market Structure
Multi-Timeframe Alignment
ML Probability
```

The scoring system must be transparent.

Every score should be decomposable into its underlying factors.

Avoid arbitrary scoring unless justified and documented.

A high score must NEVER be described as a guarantee of future profitability.

---

# 21. LLM Research Layer

The LLM layer is a later phase.

The LLM should initially act as a **research assistant**, not as the primary numerical analysis engine.

Preferred architecture:

```text
Market Data
     ↓
Python Analysis
     ↓
Structured JSON
     ↓
LLM
     ↓
Research Explanation
```

The LLM must not invent:

- prices
- indicators
- statistics
- backtest results
- model probabilities

Give the LLM structured results generated by deterministic Python components.

---

# 22. Future Autonomous Research Loop

A future version may allow the LLM to generate feature-engineering hypotheses.

Architecture:

```text
Existing Dataset
       ↓
LLM generates hypothesis
       ↓
Feature implementation
       ↓
Experiment
       ↓
Model training
       ↓
Backtest
       ↓
Evaluation
       ↓
Results
       ↓
LLM analyzes results
       ↓
New hypothesis
```

The agent must NOT modify the core research system without explicit safeguards.

Every experiment must be logged.

Every experiment must have:

```text
hypothesis
features
dataset period
model
parameters
metrics
result
```

The system must distinguish genuine improvements from overfitting.

---

# 23. Agent Operating Rules

When working on this repository, follow this process.

### Before implementation

1. Inspect the repository.
2. Read `AGENTS.md`.
3. Inspect existing architecture.
4. Identify the current development phase.
5. Identify relevant files.
6. Explain the implementation plan briefly.

### During implementation

1. Implement only the requested phase/task.
2. Preserve existing architecture.
3. Avoid unrelated refactoring.
4. Reuse existing components where appropriate.
5. Add tests for new functionality.
6. Maintain type safety and clean interfaces.

### After implementation

1. Run relevant tests.
2. Run lint/type checks if configured.
3. Fix errors caused by the implementation.
4. Review the diff.
5. Summarize:
   - files changed
   - functionality added
   - tests run
   - known limitations
   - recommended next step

Do not silently move to the next major phase.

---

# 24. Change Management

Before major architectural changes:

- explain why the change is necessary
- identify affected components
- consider backwards compatibility
- wait for explicit approval when the change is substantial

Do not rewrite the architecture merely because another design appears cleaner.

Prefer incremental improvement.

---

# 25. Testing Philosophy

Every major component should have tests.

Test:

- data ingestion
- data validation
- feature calculations
- label generation
- regime detection
- signal generation
- scoring
- backtesting
- model training
- API endpoints

Include deterministic synthetic datasets where possible.

Tests should verify both expected behavior and important edge cases.

---

# 26. Error Handling

Errors should be:

- explicit
- actionable
- logged
- traceable

Do not silently swallow exceptions.

Differentiate between:

- network errors
- API errors
- validation errors
- data errors
- configuration errors
- model errors
- programming errors

---

# 27. Logging

Use structured logging where practical.

Logs should help answer:

```text
What happened?
When did it happen?
Which symbol?
Which timeframe?
Which component?
What caused the failure?
```

Avoid excessive logging inside tight numerical loops.

---

# 28. Security

Never commit:

- API keys
- secrets
- passwords
- tokens

Use environment variables for future authenticated integrations.

The initial Binance data collector should not require credentials.

---

# 29. Performance

Do not prematurely optimize.

First prioritize:

1. correctness
2. reproducibility
3. testability
4. clarity

Then optimize bottlenecks based on evidence.

Cache expensive operations where appropriate.

Avoid unnecessary API requests.

---

# 30. Documentation

Maintain:

```text
README.md
ARCHITECTURE.md
EXPERIMENTS.md
```

README should explain:

- project purpose
- installation
- usage
- configuration
- examples

ARCHITECTURE should explain:

- components
- data flow
- dependencies
- design decisions

EXPERIMENTS should record:

- hypothesis
- methodology
- dataset
- model
- parameters
- results
- interpretation

---

# 31. Development Phases

The official development sequence is:

## Phase 1 — Data Foundation

Build:

- configuration
- Binance public-data collector
- pagination
- retries
- validation
- local storage
- caching
- tests

Do not build ML or LLM functionality.

---

## Phase 2 — Feature Engineering

Build:

- price features
- trend features
- momentum
- volatility
- volume
- market structure

Add tests.

---

## Phase 3 — Rule-Based Analysis

Build:

- market regime detection
- trend analysis
- swing setup detection
- transparent scoring

No ML yet.

---

## Phase 4 — Backtesting

Build:

- event-driven backtester
- position management
- fees
- slippage
- performance metrics
- equity curve
- drawdown

Validate against synthetic examples.

---

## Phase 5 — Machine Learning

Build:

- label generation
- chronological splits
- preprocessing
- baseline models
- evaluation
- experiment tracking

Prevent leakage.

---

## Phase 6 — Multi-Timeframe Analysis

Combine:

```text
1D
4H
1H
15M
```

Build timeframe alignment analysis.

---

## Phase 7 — Market Scanner

Scan multiple configured symbols.

Generate structured opportunities and explanations.

---

## Phase 8 — LLM Research Agent

Add:

- structured analysis input
- LLM interpretation
- feature hypothesis generation
- experiment generation
- experiment evaluation
- research loop

Only after the deterministic system is stable.

---

## Phase 9 — API / Interface

Add:

- FastAPI
- reporting
- visualization
- optional dashboard

---

# 32. Current Development Rule

At the beginning of every new task, determine which phase the project is currently in.

Do not implement functionality from a later phase unless explicitly requested.

When uncertain about an architectural decision that materially affects future phases, explain the tradeoff before proceeding.

The priority is:

```text
Correctness
>
Reproducibility
>
Testability
>
Explainability
>
Performance
>
Complexity
```

Build the simplest system that can reliably support the next phase.
