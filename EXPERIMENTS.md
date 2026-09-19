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

Two things stand out:

1. **Trend continuation is the only family near break-even.** In the holdout its
   win rate (35.9%) clears the 33.3% requirement, but mean R is still slightly
   negative because costs and stop-outs consume the difference. It is the one
   candidate worth revisiting.
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

### Next step

Phase 5 should not be expected to rescue a signal set with no edge. In order: why
only 26% of trades reach a 2R target rather than 33%; whether the tight-stop
artifact belongs in the setup layer as a validity filter; whether the position cap
is discarding the better signals. Only then is a learned model worth adding.
