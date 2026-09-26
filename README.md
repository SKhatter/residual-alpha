# residual-alpha

An evaluation harness for time-series foundation models on cross-sectional
equity returns. It asks one question and tries hard to answer it honestly:

> Does a foundation model like TimesFM-3, given market context as covariates,
> forecast single-stock **beta-residual** returns well enough to trade after
> costs?

**Answer: no -- and how it fails is the interesting part.** TimesFM-3 is the
best-calibrated forecaster in this repo and its worst trading signal.

- Best *distributions* of any model tested: highest pinball skill (2.59% over
  the null), near-exact interval coverage (0.801 against 0.800 nominal), lowest
  calibration chi-square.
- Worst *direction* of any fitted model: IC 0.0033 at t = 0.75, against ridge's
  0.0108 at t = 2.46. Ridge is the only model here with statistically
  significant directional skill.
- Last on the book: +0.21%/yr gross versus ridge's +0.73% and GBM's +2.45%,
  with costs at 32x its gross edge.

Knowing how uncertain you are is not the same as knowing which way to bet, and
a cross-sectional strategy is paid only for the second. Pinball loss is
dominated by distribution width, so a model can win it decisively while adding
nothing a book can trade -- which is exactly what happened, and is the reason
the `zero` null and the PnL simulation both exist.

Underneath that, the cost structure binds every model: **8.3%/yr of drag** at
45% daily turnover, which is arithmetic about the holding period rather than a
property of any forecaster. No model tested clears it.

## Results

Development window 2012-2022, walk-forward out-of-sample, 14 folds, 1,762
trading days, 68,876 predictions per model. Holdout (2023+) untouched — the
ledger is empty.

**Forecast quality** — skill is measured against the `zero` null, not in
absolute terms:

| model | pinball skill | IC | IC t-stat | 80% coverage | calibration p | tail mass (nom. 0.200) |
| --- | --- | --- | --- | --- | --- | --- |
| zero     | 0.0%      | –          | –        | 0.771     | 0.000     | 0.229     |
| ridge    | 2.09%     | **0.0108** | **2.46** | 0.794     | 0.002     | 0.207     |
| gbm      | 1.94%     | 0.0073     | 1.74     | 0.778     | 0.000     | 0.223     |
| timesfm3 | **2.59%** | 0.0033     | 0.75     | **0.801** | **0.007** | **0.199** |

The two columns disagree, and that disagreement is the main result.
`timesfm3` wins every distributional measure; `ridge` is the only model whose
directional skill clears t = 2. An IC of 0.011 is small in absolute terms and
an order of magnitude below what the pre-correction target reported (see below).

**Portfolio** — dollar-neutral, market/sector hedged, net of costs:

| model | gross | net | vol | Sharpe | Sharpe t | turnover | costs ÷ gross |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ridge    | +0.73% | **−7.30%** | 3.10% | −2.35 | −3.20 | 0.45 | 10.7× |
| gbm      | +2.45% | **−5.14%** | 2.09% | −2.46 | −3.24 | 0.42 | 3.2× |
| timesfm3 | +0.21% | **−7.47%** | 2.47% | −3.02 | −3.39 | 0.43 | 32.5× |

Every model has a positive gross edge and every one is 3–32x too small to pay
for the turnover required to collect it. The negative Sharpes carry large
t-stats not because any strategy is reliably skilful in reverse, but because
costs are deterministic: you pay them daily whether or not the forecast lands.

Note that GBM has the largest gross edge while ranking second-worst on
calibration — another instance of the two axes coming apart.

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

TimesFM-3 needs torch >= 2.4 (the v3 model uses `nn.RMSNorm`) and pandas < 3
(`stack(dropna=False)` is an error in pandas 3). A dedicated venv is the
path of least resistance:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]' 'timesfm[torch]==3.0.2' 'torch>=2.4' 'pandas<3'

python scripts/fetch_fomc_dates.py --start 2012 --end 2027   # scheduled meetings
.venv/bin/ralpha build                                       # ~1 min, caches to data/
.venv/bin/ralpha bakeoff                                     # forecast quality
.venv/bin/ralpha backtest --cached                           # PnL, reuses predictions
.venv/bin/ralpha ablate-gate --model ridge                   # is the gate real?
.venv/bin/python -m pytest                                   # 61 tests
```

Runtimes on an M3 Pro: `bakeoff` with all four models is **~2 hours** (TimesFM
is ~510s per fold and the checkpoint is reloaded every fold), `backtest
--cached` is ~4 min, and dropping `timesfm3` from `models.enabled` brings a
full run down to ~9 min. Start with `-c config/smoke_timesfm.yaml` for a
12-name, ~20-minute version. TimesFM-3 weights are non-commercial use only.

## What running TimesFM-3 actually taught us

**Its uncertainty estimates need no correction.** The adapter fits an affine
recalibration per fold on the assumption -- stated in its own docstring -- that
pretrained quantile heads are "near-certainly too narrow on financial returns."
They are not. The fitted spread rescale factors across 14 folds:

```
0.975  0.955  0.971  0.936  0.961  0.980  1.002
1.022  0.978  1.014  0.995  0.991  0.959  0.961
```

All within ±6% of 1.0, mean 0.977. TimesFM-3 is essentially calibrated
out of the box for daily equity residuals, which is why its coverage beats
every fitted baseline. That contradicts the assumption the adapter was written
under and is the most transferable thing in this repo.

**Its directional signal is unstable, not merely weak.** The median
recalibration slope -- realised return regressed on predicted median, fit per
fold -- should be consistently positive for a forecaster with real signal:

```
+0.82  +0.50  +0.52  +0.04  +0.07  −0.54  −0.02
−0.81  +0.10  +0.33  +0.32  +0.08  +0.00  −0.25
```

Mean +0.08, with 4 of 14 negative. The first three folds are strongly positive
and it collapses after. A strategy whose recalibration coefficient changes sign
between folds is trading against its own forecast a quarter of the time.

**A 12-name pilot reversed under scaling.** On a reduced 12-name universe
(`config/smoke_timesfm.yaml`) TimesFM looked like the winner: gross Sharpe
+0.51 against ridge's −0.26. At 40 names it came last. The pilot's IC t-stats
were all below 1, so nothing there was significant -- but it is a clean
reminder that an underpowered comparison can point the wrong way, not merely
be noisy.

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
spent, is in [docs/PROTOCOL.md](docs/PROTOCOL.md). If the terms above are
unfamiliar -- residual returns, why costs dominate, what "calibrated" means and
why it is not the same as being right -- start with
[docs/CONCEPTS.md](docs/CONCEPTS.md).

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
config/default.yaml    the 40-name research config; every knob is a decision
config/smoke_timesfm.yaml  12-name reduced config, ~20 min with TimesFM
docs/CONCEPTS.md       residual targets, cost arithmetic, calibration
docs/PROTOCOL.md       research protocol and degrees of freedom
```

## Limitations

- **Pretraining overlap cannot be ruled out.** TimesFM-3 is pretrained on a
  large corpus of public time series. If that corpus includes these tickers
  over 2012-2022, its forecasts are contaminated in a way no amount of purging
  or embargoing can fix, because the leak happened before this repo saw the
  data. The decay pattern in the recalibration slopes (strong in the earliest
  folds, gone later) is consistent with contamination but also with plain
  instability; this harness cannot distinguish them. Any zero-shot foundation
  model evaluated on historical market data has this problem.
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
