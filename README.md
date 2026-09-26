# residual-alpha

An evaluation harness for time-series foundation models on cross-sectional
equity returns. It asks one question and tries hard to answer it honestly:

> Does a foundation model like TimesFM-3, given market context as covariates,
> forecast single-stock **beta-residual** returns well enough to trade after
> costs?

**That question is not yet answered.** TimesFM-3 has not been run. The adapter
in [`models/timesfm3.py`](src/ralpha/models/timesfm3.py) is written but
unexercised -- no checkpoint installed, no test covers it, `timesfm3` is
commented out of `models.enabled`.

What *has* been measured is the bake-off's control arm -- ridge and GBM on the
same features, against a zero-forecast null -- and it produced a result that
constrains the original question:

- There is real but marginal forecast skill: **IC 0.011, t = 2.5**.
- The strategy's cost structure imposes an **8.3%/yr drag** at 45% daily
  turnover. That number comes from the holding period and the cost assumptions,
  not from the model.
- So the gross edge needed to break even is roughly **10x** what ridge
  achieves. TimesFM-3 would face the same bar.

The honest summary: at a one-day horizon on 40 large caps, the economics are
the binding constraint, and no forecaster has cleared them here. Whether a
foundation model clears it is an open question this repo is set up to answer
but has not.

## Results

Development window 2012-2022, walk-forward out-of-sample, 13 folds, 1,762
trading days, 68,876 predictions. Holdout (2023+) untouched — the ledger is
empty. **Baselines only — TimesFM-3 is not in any table below.**

**Forecast quality** — skill is measured against the `zero` null, not in
absolute terms:

| model | pinball skill | IC | IC t-stat | 80% coverage | calibration p |
| --- | --- | --- | --- | --- | --- |
| zero  | 0.0%  | –      | –    | 0.771 | 0.000 |
| ridge | 2.09% | 0.0108 | 2.46 | 0.794 | 0.002 |
| gbm   | 1.94% | 0.0095 | 2.21 | 0.778 | 0.000 |

There is skill, and it is small. An IC of 0.011 at t = 2.5 is barely
distinguishable from noise on eleven years of daily data, and it is an order of
magnitude below what the pre-correction target reported (see below).

**Portfolio** — dollar-neutral, market/sector hedged, net of costs:

| model | gross | net | vol | Sharpe | Sharpe t | turnover | costs ÷ gross |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | +0.73% | **−7.30%** | 3.10% | −2.35 | −3.20 | 0.45 | 10.7× |
| gbm   | +2.72% | **−4.88%** | 2.14% | −2.28 | −3.18 | 0.42 | 2.8× |

The gross edge is real and positive. It is 3–11× too small to pay for the
turnover required to collect it. The negative Sharpe carries a large t-stat
not because the strategy is reliably skilful in reverse, but because costs are
deterministic: you pay them every day whether or not the forecast lands.

**The confidence gate is a volatility filter.** Matched trade counts, ridge:

| variant | Sharpe | costs ÷ gross |
| --- | --- | --- |
| no gate | −1.32 | 3.0× |
| width gate | −2.35 | 10.7× |
| vol gate (matched) | −2.32 | 9.7× |
| random gate (matched) | −1.64 | 3.4× |

Jaccard overlap between the width gate and the volatility gate is **0.845**;
Spearman correlation between interval width and trailing realised vol is
**0.859**. The width gate does not clear the vol gate by even one standard
error — on either model. It is a low-volatility tilt wearing a model's
clothes, and here it is actively harmful: it concentrates the book into names
whose gross edge is too thin to cover their share of the costs.



## What it does

```
ralpha build          download prices, build the panel
ralpha folds          print the purged/embargoed walk-forward schedule
ralpha bakeoff        forecast quality: every model vs. the null
ralpha backtest       forecast quality -> portfolio PnL, net of costs
ralpha ablate-gate    is the confidence gate just a volatility filter?
ralpha holdout        spend one look at the reserved window
```

The pipeline:

1. **Data** -- 40 liquid US large caps, SPY as the market factor, 9 sector ETFs,
   plus VIX / rates / breadth / credit series used only as covariates.
2. **Target** -- rolling 252-day regression of each stock on market and sector;
   forecast the residual. The intercept is estimated but *not* subtracted --
   see below, it matters more than it sounds.
3. **Features** -- per-stock momentum/reversal/volume/volatility, regime
   features, and genuinely known-in-advance calendar features (day of week,
   turn of month, scheduled FOMC dates).
4. **Models** -- a `zero` null, `ridge`, `gbm`, and a TimesFM-3 adapter. All
   emit 9 quantiles, so intervals are available, not just point forecasts.
5. **Backtest** -- purged and embargoed walk-forward, rank-based cross-sectional
   weights, dollar-neutral, 5% per-name cap, then market/sector hedged and
   charged realistic costs.
6. **Holdout** -- everything from 2023 is mechanically unreachable until you
   spend a look, and looks are counted in a committed ledger.

## Quick start

```bash
pip install -e '.[dev]'

python scripts/fetch_fomc_dates.py --start 2012 --end 2027   # scheduled meetings
ralpha build                                                  # ~1 min, caches to data/
ralpha bakeoff                                                # forecast quality
ralpha backtest                                               # PnL, net of costs
ralpha ablate-gate --model ridge                              # is the gate real?
pytest                                                        # 61 tests
```

TimesFM-3 is optional and off by default. To enable it, install the upstream
package (`pip install -e '/path/to/timesfm[torch]'`) and uncomment `timesfm3`
under `models.enabled`. Its weights are released for non-commercial use only.

## The finding that changed the result

An earlier version of the target subtracted the rolling regression's intercept,
making it `r - alpha - beta*f`. Every lookahead test passed. The model scored
Sharpe 2.33 against it.

But a hedged book earns `r - beta*f`. Nothing lets you short a stock's own
trailing drift. Scoring the *same weights* against what a book actually
collects gave Sharpe 0.37 -- the 6x gap was entirely the drift term.

Correcting it moved the headline forecast number too, not just the PnL:

| | IC | IC t-stat |
| --- | --- | --- |
| alpha subtracted (old) | 0.030 | 7.2 |
| alpha retained (now)   | 0.011 | 2.5 |

Roughly two thirds of the apparent forecast skill was the model successfully
predicting each stock's own trailing drift — a quantity it could see in the
features and that no hedged book can sell.

This is the failure mode worth remembering: the target was not leaking future
information, which is what the tests were built to catch. It was leaking
*unhedgeable* information. A forecasting target is only honest if some position
earns exactly it.

`target.subtract_alpha` defaults to `false`. Setting it `true` reproduces the
old behaviour, so the discrepancy can be re-derived rather than taken on faith.

## Design notes

**Why residual returns.** A stock's daily return is mostly beta times the
market's. Forecasting raw returns means forecasting the index -- the least
predictable part of the problem -- and lets a model look skilful by learning
market direction. Stripping factor exposure isolates the part a
market-neutral book can actually monetise.

**Why a null model is a first-class citizen.** Pinball loss on daily returns is
dominated by distribution width, not forecast location. A model can post a
strong absolute loss while adding nothing. `zero` makes skill a relative
number, which is the only version of the claim that survives scrutiny.

**Why the confidence gate is assumed guilty.** "Trade only when the predicted
interval is tight" is, to a first approximation, "trade only when volatility is
low" -- a known factor exposure that is short volatility and pays out steadily
right up until it does not. `ralpha ablate-gate` runs the identical strategy
with the gate driven by trailing realised vol, matching the number of names
traded per day, and reports the overlap. If the model gate does not clear the
vol gate by more than one Sharpe standard error, the honest description is
"low-volatility tilt".

**Why the holdout is enforced in code.** Every look at held-out data is a
degree of freedom, and nothing in the output distinguishes a tenth look from a
first. `assert_development_only()` raises if a reserved date reaches the
computation, and `.holdout_ledger.json` -- committed, not ignored -- records
every look with the config hash that produced it.

Full methodology, including the complete list of degrees of freedom already
spent, is in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Layout

```
src/ralpha/
  config.py            dot-access YAML config
  pipeline.py          raw prices -> Panel (features, target, covariates)
  targets.py           rolling-beta residuals; the subtract_alpha decision
  features.py          past-only and known-future features
  signal.py            quantiles -> expected return -> capped neutral weights
  gate.py              width gate + the matched-count volatility ablation
  holdout.py           touch-once ledger and the development-only assertion
  runner.py            walk-forward orchestration, scoring, ablation
  models/              base protocol, zero/ridge/gbm baselines, timesfm3 adapter
  backtest/            purged+embargoed splits, hedged PnL simulation
  evaluation/          pinball loss, coverage, calibration tests
tests/                 61 tests, weighted toward lookahead and alignment
docs/PROTOCOL.md       research protocol and degrees of freedom
```

## Limitations

- **The headline model has not been run.** TimesFM-3 is the reason this repo
  exists and it is entirely untested here. The adapter is written against the
  real v3 API and handles the traps documented at the top of that file, but
  code that has never executed should be assumed broken until it runs. Nothing
  in the results speaks to foundation-model performance either way.
- **40 names is a small cross-section.** Cross-sectional strategies get most of
  their Sharpe from breadth; a 40-name universe caps what is achievable and
  makes the daily IC noisy.
- **Survivorship.** The universe is today's large caps over a 2012-2024 window,
  so it is biased toward names that did well. The strategy is dollar-neutral
  and cross-sectional, which blunts this considerably, but does not remove it.
- **Costs are assumed, not measured.** No market impact model, no participation
  constraint. At this turnover, the cost assumption drives the conclusion, and
  a different assumption would change the sign.
- **Daily bars only.** Signals formed at the close and executed at the next
  close; nothing here speaks to intraday behaviour.
- **Development results are not forward estimates.** Seven categories of
  research choices were made while looking at this window. See the degrees of
  freedom list in the protocol doc.

## Licence

Code is unlicensed for redistribution; TimesFM-3 weights, if you enable them,
are non-commercial use only under Google's terms. Nothing here is investment
advice.
