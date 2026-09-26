# Concepts

Background for reading the results in the README. Nothing here is specific to
this codebase -- it is the set of ideas you need in order to judge whether the
numbers mean anything. If you already work in quantitative finance, skip to
[PROTOCOL.md](PROTOCOL.md).

---

## 1. The residual target

**Target** is the thing a model is trained to predict. **Residual** is what is
left of a stock's return once you remove the parts that are not about the
stock.

Any stock's daily move decomposes:

```
AAPL return  =  market move x its market beta
              + sector move x its sector beta
              + everything else            <- the residual
```

Concretely, on some day:

| | |
| --- | --- |
| AAPL actually returned | **+1.50%** |
| SPY returned +1.2%, AAPL's market beta is 1.2 | +1.44% |
| XLK returned +1.0%, sector beta 0.3 | +0.30% |
| so its factor exposures predicted | +1.74% |
| **residual** | **-0.24%** |

AAPL rose 1.5% and still *underperformed* by 0.24%. Owning it that day made
money, but nothing about that was AAPL-specific -- the market did the work. The
residual says the stock-specific news was mildly bad.

### Why forecast the residual instead of the return

**Most of a stock's return is not about the stock.** Roughly 70-80% of a large
cap's daily variance is the market moving. A model trained on raw returns
spends its capacity forecasting the index -- the least predictable series in
finance, and not what this strategy trades.

**Raw returns let a model fake skill.** A model that learns "markets drift up"
scores well on raw returns while knowing nothing about any company. Stripping
factor exposure closes that escape route: the only way to score well on
residuals is to say something about the stock.

**It is what the book collects.** The strategy is dollar-neutral and
factor-hedged -- long some names, short others, with offsetting SPY and sector
ETF positions. The market component cancels by construction. The residual is
the only part left to harvest.

### The intercept

The rolling regression fits `r = alpha + b1*mkt + b2*sector`. The intercept
must be *in the fit* -- omitting it biases the betas when a stock has drifted.
But it must not be *subtracted from the target*, because `r - alpha - b*f` is
not a quantity any position earns. You can hedge factor exposure; you cannot
hedge a stock's own trailing drift.

This distinction was worth two thirds of the apparent forecast skill in this
repo. See the README section "The finding that changed the result".

---

## 2. "Well enough to trade after costs"

A forecast can point the right way on average and still lose money, because
acting on it is not free. If the edge per trade is smaller than the cost per
trade, a *correct* model yields a losing strategy.

### What gets charged

- **7 bps per unit traded** -- 5 bps commission and impact, 2 bps crossing the
  spread
- **50 bps/yr borrow** on short notional
- charged on the **hedge legs too**. The SPY and sector ETF positions are real
  trades, not a free abstraction. Booking residual PnL directly instead would
  assume a costless hedge, which is the difference between a plausible Sharpe
  and a fictional one.

### Why turnover dominates

The strategy re-ranks 40 names daily, so the book turns over ~45% a day --
about **113 times a year**. The 7 bps is paid on all of it. The forecast does
not need to beat 7 bps once; it needs to beat it 113 times.

### The arithmetic, for ridge

Per unit of book turned over:

| | |
| --- | --- |
| what the forecast earns | **0.65 bps** |
| what it costs to collect | **7 bps** |

Annualised: **+0.73%** gross, **-8.30%** costs, **-7.30%** net. The
`cost_share_of_gross` column reports that ratio directly -- **10.7x**.

So the model is not wrong. It is right by about a tenth of what it costs to act
on being right.

### What would have been enough

Roughly any one of:

- an edge ~11x larger (IC near 0.10 rather than 0.011)
- ~11x less turnover -- a weekly or monthly horizon rather than daily
- materially cheaper execution, which mostly means being a different kind of
  institution
- far more names, spreading the same thin edge across more independent bets

The failure is specific, which is the useful part: the edge exists and is about
an order of magnitude too thin for this holding period. A longer horizon is the
obvious next experiment, since it divides costs directly.

---

## 3. Uncertainty, and why it is not direction

Every model here emits **9 quantiles** rather than a point forecast. One
prediction looks like:

```
AAPL, tomorrow's residual return:
  10th percentile   -1.4%     10% chance it is worse than this
  50th percentile   +0.1%     central guess
  90th percentile   +1.6%     10% chance it is better than this
```

Two independent claims are in there:

- **Direction** -- where the centre sits.
- **Uncertainty** -- how wide the spread is.

A model is good at uncertainty when that width is honest: when it draws an 80%
interval, outcomes land inside it about 80% of the time.

### How it is measured

Fraction of realised returns falling inside each model's own 10th-90th
percentile band, over 68,876 predictions:

| model | 80% coverage | reading |
| --- | --- | --- |
| zero | 0.771 | overconfident, intervals too narrow |
| gbm | 0.778 | overconfident |
| ridge | 0.794 | mildly overconfident |
| timesfm3 | **0.801** | honest |

`tail_mass` and the chi-square occupancy test say the same thing from other
angles. Pinball loss scores the whole quantile set at once.

### Calibration is not discrimination

These are independent properties, and conflating them is the trap this repo
was built to avoid.

A forecaster who says "70% chance of rain" every day, in a climate where it
rains 70% of days, is **perfectly calibrated and completely useless**. They
cannot tell you whether to take an umbrella *today* rather than tomorrow,
because they never distinguish one day from another.

That is approximately TimesFM-3's result here. Its stated uncertainty is
trustworthy; its ranking of which stock beats which is not (IC 0.0033,
t = 0.75). And this is a ranking strategy -- [signal.py](../src/ralpha/signal.py)
takes the mean of each predictive distribution, sorts the 40 names, and goes
long the top against the bottom. Honest error bars do not change that sort
order.

Pinball loss is dominated by distribution width rather than location, so a
model can win it decisively while adding nothing tradeable. That is exactly
what happened, and it is why the `zero` null and the PnL simulation both exist:
either one alone would have been fooled.

### Where uncertainty was supposed to pay

**The gate.** The idea in [gate.py](../src/ralpha/gate.py) was to trade only
names whose predicted interval is tight -- trade where the model is confident.
That is a direct use of uncertainty. It failed for an unrelated reason:
interval width turned out to be 0.85-correlated with trailing realised
volatility, making the gate a low-volatility filter rather than a confidence
filter.

**Risk management**, which this repo does not attempt. If you were sizing
positions by predicted variance, setting stop levels, or reporting
value-at-risk, calibrated intervals would be the entire point and TimesFM-3
would be the right tool. The finding is not that it is a bad model. It is that
it is good at something this particular strategy is not paid for.
