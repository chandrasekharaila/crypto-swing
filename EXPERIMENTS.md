# Experiments

Reproducible research records. Each entry states the hypothesis, the method, the
data, the parameters, the result, and what it means. Results are measurements, not
claims of profitability.

---

## E001 — Do the Phase 3 rule-based setups have edge?

**Date:** 2026-09-19
**Status:** complete, negative

### Hypothesis

The trend-continuation, breakout, and mean-reversion setups detect market contexts
that are followed by a favourable move more often than the cost of trading them.
The question is whether they pay for themselves, not whether they occasionally
predict direction.

### Method

Signals are detected once over the full history, then simulated by
`BacktestEngine` with no parameter tuning of any kind. Each signal is decided at
its candle's close and filled at the open of a later bar. The stop is the signal's
structural invalidation level; the target is `take_profit_r_multiple` times the
distance from the fill to that stop. A bar that opens beyond a level fills at the
open. When one bar contains both levels, the stop is assumed to have come first
and the case is counted.

### Data

48 liquid Binance USDT pairs, 4h candles, each from its listing (2017 onwards):
711,232 four-hour rows. Fees 10 bps per side and slippage 5 bps per side, both
stated assumptions rather than fitted values.

Two windows, fixed before the run:

| Window | Period | Purpose |
|---|---|---|
| Development | 2022-01-01 to 2025-12-31 | reported in full |
| Holdout | 2026-01-01 to present | reported once, never tuned against |

### Parameters

`execution_delay_bars=1`, `take_profit_r_multiple=2.0`, `max_holding_bars=42`,
`risk_fraction=0.01`, `max_notional_multiple=1.0`, `max_concurrent_positions=5`,
`one_position_per_symbol=True`, `ambiguous_exit_policy=STOP_FIRST`.

A 2R target needs a 33.3% win rate to break even before costs.

### Results

Development window, 6,492 trades:

| Setup | Trades | Win rate | Mean R | Profit factor | Mean return |
|---|---|---|---|---|---|
| trend continuation | 1,388 | 29.8% | −0.310 | 0.837 | −0.44% |
| breakout | 3,558 | 27.0% | −0.691 | 0.640 | −0.29% |
| mean reversion | 1,546 | 22.1% | −0.842 | 0.540 | −0.57% |
| **all** | **6,492** | **26.4%** | **−0.646** | **0.705** | **−0.39%** |

Holdout window, 1,265 trades:

| Setup | Trades | Win rate | Mean R | Profit factor | Mean return |
|---|---|---|---|---|---|
| trend continuation | 248 | 35.9% | −0.186 | 1.010 | +0.02% |
| breakout | 707 | 25.5% | −0.781 | 0.577 | −0.33% |
| mean reversion | 310 | 20.3% | −0.982 | 0.412 | −0.72% |
| **all** | **1,265** | **26.2%** | **−0.713** | **0.688** | **−0.36%** |

Equity reached zero in both windows. 1,163 of the 6,492 development exits were
ambiguous bars where the stop and the target both sat inside one candle.

### The dominant mechanical effect

6.9% of trades have a stop closer than 0.2% to the entry price. Those trades
average −2.78R and account for **30% of all the negative R** in the development
window. A 0.2% round-trip cost against a stop 0.05% away is four times the risk
being taken, so such a trade is unwinnable before the market moves at all.

Excluding them, mean R improves from −0.646 to −0.486 in development and from
−0.713 to −0.515 in the holdout. The artifact is large, but removing it does not
turn a loss into a profit: the win rate on wide stops is 28%, still short of the
33.3% required.

### Interpretation

The setups show **no evidence of edge at these parameters**. The holdout
corroborates the development window, so this is not an overfitted backtest
collapsing — it is a consistent absence of edge, which is the more useful finding.

### Follow-up — the intrabar path, measured rather than guessed

The result above rests on an assumption: 17.9% of trades hit both the stop and the
target inside one candle, and the backtest guessed that the stop came first.
Switching the guess to the target gives a mean R of −0.108 instead of −0.646, so
the assumption was carrying most of the answer.

`scripts/resolve_ambiguity.py` replaces the guess with a measurement. It collected
every ambiguous bar, fetched the 1-minute candles covering it, and cached them; the
simulator now consults that cache instead of assuming. 1,405 windows were fetched,
and the finer candles decided **98.0% of the development cases and 99.2% of the
holdout cases**. The rest remain undecided because a single minute candle contained
both levels, which is the same problem one scale down.

| | Policy (guessed) | Finer candles (measured) |
|---|---|---|
| Development win rate | 26.4% | 31.0% |
| Development mean R | −0.646 | **−0.494** |
| Development profit factor | 0.705 | 0.750 |
| Holdout win rate | 26.2% | 30.4% |
| Holdout mean R | −0.713 | **−0.573** |
| Holdout profit factor | 0.688 | 0.730 |

**The pessimistic default was mostly right.** Of the resolved cases, 813 were the
stop first and 327 the target, so the guess was correct about 71% of the time. That
also means the crude "flip the policy" bound above was misleading: assuming every
ambiguous bar resolved as a win put mean R at −0.108, but measuring put it at
−0.494. Guessing in either direction was wrong; only the finer data settled it.

The correction is real and worth about 0.15R, and the conclusion is unchanged. The
setups still lose in both windows once the ambiguity is measured rather than
assumed. Breakout is the family most affected — its development win rate rises from
27.0% to 33.1%, because a break near a range extreme is exactly the situation that
produces a bar touching both levels — and it is still clearly negative.

Two things stand out:

1. **Trend continuation is the only family near break-even.** In the holdout its
   win rate (36.3%) clears the 33.3% requirement, but mean R is still −0.173
   because costs and stop-outs consume the difference. It remains the one candidate
   worth revisiting.
2. **Breakout is both the most frequent and clearly negative.** It fires 3,558
   times in development and loses. Frequency is not evidence.

### Limitations

- The squeeze threshold was calibrated on 2022–2026 distributions before the
  holdout was fixed, so even the holdout is not perfectly pristine.
- The universe is today's survivor list, so results are optimistic; delisted pairs
  are absent.
- Portfolio construction, not signal quality, caused most of the skips: 10,760
  development signals were dropped by the five-position cap, so the setups were
  not given a fair sample of their own signals.
- Fees and slippage are assumptions. At these effect sizes, different assumptions
  change the magnitude but not the sign.
- Intrabar resolution depends on a cached 1-minute dataset that is not committed.
  Rebuild it with `scripts/resolve_ambiguity.py`; without it the run falls back to
  the stop-first assumption and reports the more pessimistic numbers.

### Next step

Phase 5 should not be expected to rescue a signal set with no edge. With the
intrabar path now measured rather than assumed, the remaining questions are, in
order: why only 31% of trades reach a 2R target when 33% is needed; whether the
tight-stop artifact belongs in the setup layer as a validity filter; and whether
the five-position cap is discarding the better signals. Only then is a learned
model worth adding.
