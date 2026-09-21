"""Signal construction, gating, and the gate ablation machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ralpha.gate import (
  gate_overlap,
  matched_count_vol_gate,
  trailing_vol,
  vol_gate,
  width_gate,
)
from ralpha.signal import (
  cross_sectional_score,
  expected_return,
  interval_width,
  to_weights,
)

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
DATES = pd.bdate_range("2021-01-01", periods=40)
TICKERS = [f"S{i:02d}" for i in range(30)]
IDX = pd.MultiIndex.from_product([DATES, TICKERS], names=["date", "ticker"])


def _quantile_frame(loc, scale):
  z = stats.norm.ppf(QUANTILES)
  values = np.asarray(loc)[:, None] + np.asarray(scale)[:, None] * z[None, :]
  return pd.DataFrame(values, index=IDX, columns=[f"q{int(q*100)}" for q in QUANTILES])


def test_expected_return_recovers_a_symmetric_mean():
  loc = np.full(len(IDX), 0.004)
  preds = _quantile_frame(loc, np.full(len(IDX), 0.02))
  mu = expected_return(preds, QUANTILES)
  # Trapezoidal integration over 5 quantiles is coarse but unbiased in sign
  # and close in magnitude for a symmetric distribution.
  assert np.allclose(mu.to_numpy(), 0.004, atol=5e-3)
  assert mu.std() == pytest.approx(0.0, abs=1e-12)


def test_interval_width_is_the_q90_q10_spread():
  scale = np.linspace(0.005, 0.05, len(IDX))
  preds = _quantile_frame(np.zeros(len(IDX)), scale)
  width = interval_width(preds, QUANTILES, 0.1, 0.9)
  expected = scale * (stats.norm.ppf(0.9) - stats.norm.ppf(0.1))
  assert np.allclose(width.to_numpy(), expected)


def test_cross_sectional_rank_is_centred_each_day():
  rng = np.random.default_rng(0)
  s = pd.Series(rng.normal(size=len(IDX)), index=IDX)
  ranked = cross_sectional_score(s, "rank")
  daily_mean = ranked.groupby(level="date").mean()
  assert daily_mean.abs().max() < 1e-12
  assert ranked.min() >= -0.5 and ranked.max() <= 0.5


def test_weights_are_dollar_neutral_and_respect_the_cap():
  rng = np.random.default_rng(1)
  s = pd.Series(rng.normal(size=len(IDX)), index=IDX)
  score = cross_sectional_score(s, "rank")
  w = to_weights(score, gross_leverage=1.0, max_weight=0.05)

  by_day = w.groupby(level="date")
  assert by_day.sum().abs().max() < 1e-9
  assert w.abs().max() <= 0.05 + 1e-9
  gross = by_day.apply(lambda x: x.abs().sum())
  assert (gross <= 1.0 + 1e-9).all()
  assert gross.min() > 0.5  # the cap must not collapse the book


def test_cap_binds_when_the_universe_is_small():
  small_idx = pd.MultiIndex.from_product(
    [DATES[:2], ["A", "B", "C"]], names=["date", "ticker"]
  )
  score = pd.Series([1.0, 0.0, -1.0] * 2, index=small_idx)
  w = to_weights(score, gross_leverage=1.0, max_weight=0.1)
  assert w.abs().max() <= 0.1 + 1e-9


def test_width_gate_keeps_the_requested_fraction():
  rng = np.random.default_rng(2)
  width = pd.Series(rng.random(len(IDX)), index=IDX)
  result = width_gate(width, percentile=0.5, min_names=1)
  assert result.n_selected.between(14, 16).all()
  # Selected names must all be tighter than the rejected ones.
  for date in DATES[:5]:
    day = width.loc[date]
    chosen = result.mask.loc[date]
    assert day[chosen].max() <= day[~chosen].min() + 1e-12


def test_gate_respects_min_names_floor():
  rng = np.random.default_rng(3)
  width = pd.Series(rng.random(len(IDX)), index=IDX)
  result = width_gate(width, percentile=0.01, min_names=8)
  assert (result.n_selected >= 8).all()


def test_matched_count_vol_gate_matches_trade_count_exactly():
  """The ablation is only fair if breadth is held constant."""
  rng = np.random.default_rng(4)
  width = pd.Series(rng.random(len(IDX)), index=IDX)
  rv = pd.Series(rng.random(len(IDX)), index=IDX)

  reference = width_gate(width, percentile=0.4, min_names=5)
  matched = matched_count_vol_gate(rv, reference, min_names=5)

  pd.testing.assert_series_equal(
    reference.n_selected, matched.n_selected, check_names=False
  )


def test_gate_overlap_detects_an_identical_gate():
  rng = np.random.default_rng(5)
  width = pd.Series(rng.random(len(IDX)), index=IDX)
  a = width_gate(width, percentile=0.5, min_names=1)
  b = vol_gate(width, percentile=0.5, min_names=1)  # same criterion
  stats_ = gate_overlap(a, b)
  assert stats_["jaccard"] == pytest.approx(1.0)
  assert stats_["corr_criterion"] == pytest.approx(1.0)


def test_gate_overlap_detects_an_unrelated_gate():
  rng = np.random.default_rng(6)
  a = width_gate(pd.Series(rng.random(len(IDX)), index=IDX), 0.5, 1)
  b = vol_gate(pd.Series(rng.random(len(IDX)), index=IDX), 0.5, 1)
  assert gate_overlap(a, b)["jaccard"] < 0.6


def test_trailing_vol_is_causal():
  rng = np.random.default_rng(7)
  resid = pd.DataFrame(
    rng.normal(0, 0.01, (len(DATES), 3)), index=DATES, columns=["A", "B", "C"]
  )
  base = trailing_vol(resid, window=10)

  shocked = resid.copy()
  shocked.iloc[30:] *= 50
  after = trailing_vol(shocked, window=10)

  early = [(d, t) for d in DATES[:30] for t in ["A", "B", "C"]]
  pd.testing.assert_series_equal(base.loc[early], after.loc[early])
