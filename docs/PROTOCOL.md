# Research protocol

This document exists because the hard part of this project is not building a
forecaster. It is avoiding self-deception while building one. Financial data is
short, non-stationary, and low signal-to-noise; a researcher with a flexible
model and an unrestricted number of looks at the data will find something
beautiful every time, and it will not survive contact with a real book.

The rules below are the ones this repo enforces in code. Read the enforcement,
not the prose -- prose does not stop anyone.

---

## 1. Forecast the tradeable quantity

A single stock's daily return is mostly beta times the market's. Forecasting it
means forecasting the index, which is both the hardest part of the problem and
not what the strategy trades. So the target is the **beta-residual return**:
regress each stock on the market (SPY) and its sector ETF over a trailing
252-day window, and take what is left.

The residual at date `t` uses betas fit on data through `t-1` together with the
*contemporaneous* factor returns at `t`. That is deliberate and not a leak: the
residual is a **label**, not a feature. At decision time we do not know
`r_mkt[t]`; we forecast the residual and hedge the factor exposure in portfolio
construction. `tests/test_targets.py` pins this down -- perturbing the tail of a
series must not move any earlier coefficient.

### The intercept is estimated but not subtracted

This is the subtlest decision in the repo, and it was originally wrong.

The rolling regression is `r = alpha + b1*mkt + b2*sector`. The intercept has to
be *in the fit* -- omitting it biases the betas whenever a stock has drifted
over the window. But an earlier version also *subtracted* it, making the target
`r - alpha - b*f`. That quantity is not tradeable. A market/sector-hedged book
earns `r - b*f`; there is no instrument that lets you short a stock's own
trailing drift.

The size of the error was not cosmetic. Holding the weights fixed and only
changing what they were scored against:

| Scored against | Sharpe |
| --- | --- |
| `r - alpha - b*f` (alpha subtracted) | 2.33 |
| simulated hedged book | 0.37 |

The entire 6x gap was the drift term. A target that flatters the model by 6x is
worse than no target at all, because it survives every sanity check you would
think to run -- the lookahead tests all passed.

Correcting it also cut the reported forecast skill, which is the more useful
number because it does not depend on any cost assumption:

| | IC | IC t-stat |
| --- | --- | --- |
| alpha subtracted | 0.030 | 7.2 |
| alpha retained | 0.011 | 2.5 |

About two thirds of the apparent skill was the model predicting each stock's
trailing drift from features that contained it. A t-stat of 7.2 reads as a
settled result; 2.5 reads as "maybe". Those are different projects.

`target.subtract_alpha` now defaults to `false`. Setting it `true` reproduces
the old behaviour and is kept only so the discrepancy can be re-derived.

**Generalisation worth carrying to the next project:** a forecasting target is
only honest if some position earns exactly it. Write down the trade that
collects the target before you fit anything.

---

## 2. Walk-forward, purged, embargoed

No single train/test split. The development window is cut into rolling folds:
4 years train, 6 months test, rolled forward. Each model is refit from scratch
on each fold -- no state crosses a fold boundary.

Two separate protections against leakage across the boundary:

- **Purge.** A label at `t` is the return over `(t+lag, t+lag+horizon]`, so it
  consumes data up to `t + execution_lag + horizon`. Training rows within that
  distance of the test window are dropped. The purge width is computed from
  `label_end_offset()`, not hardcoded, so it cannot drift out of sync with the
  target definition. `assert_no_leakage` re-checks every fold at construction.
- **Embargo.** A further 10 days after each test block are removed from the
  *next* training set, to blunt residual autocorrelation that purging alone
  does not handle.

`ralpha folds` prints the schedule.

---

## 3. Every model is scored against a null

`zero` is a real entry in the model list: it forecasts zero residual return with
an empirical spread. It is there because pinball loss on daily stock returns is
dominated by the width of the distribution, not the location of the point
forecast, so a model can post an impressive-looking absolute loss while adding
nothing.

`pinball_skill` is therefore reported relative to `zero`. A positive number is
the only version of the claim that means anything. `ridge` and `gbm` are also
in the bake-off so that "the fancy model won" has to be demonstrated, not
assumed.

Reported alongside:

- `ic_mean` / `ic_t` -- daily cross-sectional information coefficient and its
  t-stat. Below |t| = 2, it is noise.
- `cov80_empirical` -- fraction of outcomes inside the 80% predictive interval.
  Far below 0.80 means the intervals are too narrow and the gate is meaningless.
- `chi2_pvalue` -- calibration test on bin occupancy between adjacent predicted
  quantiles. This replaced a KS test on the PIT, which was flagging
  linear-interpolation artefacts rather than real miscalibration.

---

## 4. The confidence gate is assumed guilty

"Only trade when the model's q90-q10 interval is tight" sounds like selecting on
model confidence. It is much more likely to be a volatility forecast in
disguise, which means the strategy is a low-volatility tilt -- a known factor
that is short volatility, looks excellent for years, and returns the whole gain
in a week.

`ralpha ablate-gate` runs four books that are identical except for name
selection: no gate, the width gate, a trailing-realised-volatility gate, and a
random gate. The volatility and random gates **select exactly as many names per
day as the width gate**, because otherwise the books differ in breadth as well
as in selection rule and the Sharpe comparison is uninterpretable.

The bar: the width gate must beat the volatility gate by more than one Sharpe
standard error. Otherwise the honest description is "low-volatility filter", and
the Jaccard overlap between the two gates is reported so the reader can see how
close they are regardless of what the returns say.

**It failed.** On ridge, Jaccard overlap is 0.845 and Spearman correlation
between interval width and trailing realised vol is 0.859; on gbm, 0.809 and
0.808. Neither model's width gate clears its matched vol gate by even one
standard error. The gate is a low-volatility tilt and is documented as one.
The `gate.enabled` default is left `true` so the ablation stays reproducible
against the configuration the results were produced under, but nothing here
supports believing the gate is selecting on model confidence.

---

## 5. The holdout is touched once, and the code counts

Everything from 2023-01-01 is reserved. `ralpha backtest`, `ralpha bakeoff` and
`ralpha ablate-gate` call `assert_development_only()` and **raise** if a single
reserved date enters the computation. Reaching the holdout requires the separate
`ralpha holdout` command.

That command appends to `.holdout_ledger.json`: timestamp, look number, config
fingerprint, metrics. A second look prints the first one next to it, and states
whether the config changed. If it did, you tuned against holdout feedback and
the result is in-sample -- the banner says so.

The ledger is committed to git on purpose. Discipline that depends on
remembering is not discipline.

---

## 6. Costs are part of the result, not a footnote

Reported net of: 5bps per unit turnover, 2bps spread, 50bps annual borrow on
short notional, with daily turnover capped at 25% of gross. Positions are
dollar-neutral, capped at 5% per name, gross leverage 1.0.

A daily-rebalanced residual strategy turns over a large fraction of the book
every day, so cost assumptions dominate the conclusion. `cost_share_of_gross` is
reported precisely so that a strategy whose gross edge is smaller than its
frictions cannot be presented as a winner. On this universe, it currently is
smaller. See the README.

---

## Degrees of freedom already spent

Honesty requires listing these. Each was a choice made while looking at
development-window results, and each inflates the development numbers relative
to what a truly fresh sample would give:

1. Universe: 40 large-cap US names, chosen for liquidity and sector spread.
2. Target: residual rather than raw; 252-day beta window; 6-sigma winsorisation.
3. Intercept handling: switched from subtracted to retained after the
   diagnostic above. This is a *correction*, and it moved results down, not up.
4. Fold geometry: 4y train / 6m test / 10d embargo.
5. Gate: tightest 60% of names, floor of 10.
6. Cost model: the numbers in section 6.
7. Model hyperparameters: the ridge alpha grid and the GBM settings in
   `config/default.yaml`.
8. TimesFM-3 settings: 512-day context, per-fold affine recalibration on 2,000
   samples, `make_positive=False`. The recalibration is itself a fitted
   quantity and a source of instability -- its slope changes sign across folds
   (see the README), so it is a degree of freedom that does not pay for itself.

## A degree of freedom this harness cannot control

TimesFM-3 is pretrained on a large corpus of public time series. If that corpus
includes these tickers over the development window, its forecasts are
contaminated before this repo ever touches the data, and no amount of purging,
embargoing or holdout discipline can repair it. Purge and embargo protect
against leakage *within* the experiment; they are silent about leakage that
happened during someone else's pretraining run.

The recalibration slopes decay sharply after the first three folds, which is
consistent with contamination and equally consistent with the model simply
having no stable signal. Nothing here distinguishes the two. This is a
structural limitation of evaluating any zero-shot foundation model on
historical market data, and it should be stated whenever such a result is
reported -- including favourable ones.

None of these were selected against the holdout. All of them were selected with
some view of development results, which is exactly why the development Sharpe
is not an estimate of future Sharpe.

## Adding an experiment

1. Copy `config/default.yaml`, change what you are testing, run with `-c`.
2. Run `ralpha bakeoff` first. If `pinball_skill` and `ic_t` do not move, stop
   -- there is no point looking at PnL.
3. Add the change to the degrees-of-freedom list above.
4. Do not run `ralpha holdout`.
