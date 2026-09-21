"""Lookahead tests for target construction.

These are the tests that matter. A lookahead bug here does not crash, does not
warn, and produces a beautiful equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ralpha.targets import (
  _rolling_ols_2f,
  forward_target,
  label_end_offset,
  residual_returns,
)


@pytest.fixture
def toy():
  rng = np.random.default_rng(0)
  n = 400
  idx = pd.bdate_range("2020-01-01", periods=n)
  mkt = pd.Series(rng.normal(0, 0.01, n), index=idx)
  sec = pd.Series(rng.normal(0, 0.01, n), index=idx)
  y = 1.2 * mkt + 0.5 * sec + pd.Series(rng.normal(0, 0.005, n), index=idx)
  return idx, y, mkt, sec


def test_rolling_beta_uses_only_past_data(toy):
  """Coefficients at row t must equal an OLS fit on rows strictly before t."""
  idx, y, mkt, sec = toy
  window, min_periods = 100, 100
  coefs = _rolling_ols_2f(y, mkt, sec, window, min_periods)

  t = 250
  past = slice(t - window, t)  # excludes t
  X = np.column_stack(
    [np.ones(window), mkt.iloc[past].to_numpy(), sec.iloc[past].to_numpy()]
  )
  expected, *_ = np.linalg.lstsq(X, y.iloc[past].to_numpy(), rcond=None)

  row = coefs.iloc[t]
  assert row["alpha"] == pytest.approx(expected[0], abs=1e-9)
  assert row["beta_mkt"] == pytest.approx(expected[1], abs=1e-9)
  assert row["beta_sector"] == pytest.approx(expected[2], abs=1e-9)


def test_future_data_cannot_change_past_betas(toy):
  """Mutating the tail of the series must not alter earlier coefficients."""
  idx, y, mkt, sec = toy
  base = _rolling_ols_2f(y, mkt, sec, 100, 100)

  perturbed = y.copy()
  perturbed.iloc[300:] += 5.0  # an absurd future shock
  after = _rolling_ols_2f(perturbed, mkt, sec, 100, 100)

  pd.testing.assert_frame_equal(base.iloc[:300], after.iloc[:300])


def test_residual_returns_recover_idiosyncratic_component(toy):
  """With a stable true beta, residuals should track the injected noise."""
  idx, y, mkt, sec = toy
  noise = y - 1.2 * mkt - 0.5 * sec

  resid, betas = residual_returns(
    stock_returns=pd.DataFrame({"AAA": y}),
    market_returns=mkt,
    sector_returns=pd.DataFrame({"XLK": sec}),
    sector_map={"AAA": "XLK"},
    window=150,
    min_periods=150,
    winsorize_sigma=None,
  )

  valid = resid["AAA"].dropna()
  assert len(valid) > 100
  assert valid.corr(noise.reindex(valid.index)) > 0.95
  assert betas.loc[(valid.index[-1], "AAA"), "beta_mkt"] == pytest.approx(1.2, abs=0.1)


def test_residuals_have_no_market_exposure(toy):
  """The point of the exercise: residuals should be ~uncorrelated with the market."""
  idx, y, mkt, sec = toy
  resid, _ = residual_returns(
    pd.DataFrame({"AAA": y}), mkt, pd.DataFrame({"XLK": sec}),
    {"AAA": "XLK"}, window=150, min_periods=150, winsorize_sigma=None,
  )
  valid = resid["AAA"].dropna()
  assert abs(valid.corr(mkt.reindex(valid.index))) < 0.15
  assert abs(y.corr(mkt)) > 0.7  # the raw series very much does


def test_default_target_is_the_hedgeable_quantity(toy):
  """Default residual must be exactly r - beta*f, with no intercept removed."""
  idx, y, mkt, sec = toy
  resid, betas = residual_returns(
    pd.DataFrame({"AAA": y}), mkt, pd.DataFrame({"XLK": sec}),
    {"AAA": "XLK"}, window=150, min_periods=150, winsorize_sigma=None,
  )
  b = betas.xs("AAA", level="ticker")
  hedged = y - b["beta_mkt"] * mkt - b["beta_sector"] * sec

  valid = resid["AAA"].dropna().index
  pd.testing.assert_series_equal(
    resid["AAA"].loc[valid], hedged.loc[valid], check_names=False
  )


def test_subtract_alpha_flag_removes_exactly_the_intercept(toy):
  """The two targets must differ by the trailing alpha and nothing else."""
  idx, y, mkt, sec = toy
  args = dict(
    market_returns=mkt,
    sector_returns=pd.DataFrame({"XLK": sec}),
    sector_map={"AAA": "XLK"},
    window=150,
    min_periods=150,
    winsorize_sigma=None,
  )
  kept, betas = residual_returns(pd.DataFrame({"AAA": y}), **args)
  removed, _ = residual_returns(
    pd.DataFrame({"AAA": y}), **args, subtract_alpha=True
  )

  alpha = betas.xs("AAA", level="ticker")["alpha"]
  valid = kept["AAA"].dropna().index
  pd.testing.assert_series_equal(
    (kept["AAA"] - removed["AAA"]).loc[valid], alpha.loc[valid], check_names=False
  )


def test_drift_survives_in_the_default_target():
  """A stock that only drifts has no factor exposure to hedge away.

  The old target subtracted that drift and so scored against a quantity no
  book can earn. The default target must keep it.
  """
  n = 500
  idx = pd.bdate_range("2020-01-01", periods=n)
  rng = np.random.default_rng(1)
  # Factors have real variance so the regression is well posed, but the stock
  # is independent of them: its true beta is zero and all it does is drift.
  mkt = pd.Series(rng.normal(0, 0.01, n), index=idx)
  sec = pd.Series(rng.normal(0, 0.01, n), index=idx)
  drift = 0.0008
  y = pd.Series(drift + rng.normal(0, 0.002, n), index=idx)

  args = dict(
    market_returns=mkt,
    sector_returns=pd.DataFrame({"XLK": sec}),
    sector_map={"AAA": "XLK"},
    window=200,
    min_periods=200,
    winsorize_sigma=None,
  )
  kept, _ = residual_returns(pd.DataFrame({"AAA": y}), **args)
  removed, _ = residual_returns(pd.DataFrame({"AAA": y}), **args, subtract_alpha=True)

  assert kept["AAA"].dropna().mean() == pytest.approx(drift, abs=2e-4)
  assert abs(removed["AAA"].dropna().mean()) < drift / 4


@pytest.mark.parametrize("lag,horizon", [(0, 1), (1, 1), (2, 1), (1, 3)])
def test_forward_target_alignment(lag, horizon):
  """target[t] must be the return earned after executing at t+lag."""
  n = 60
  idx = pd.bdate_range("2021-01-01", periods=n)
  resid = pd.DataFrame({"AAA": np.arange(n, dtype=float) / 1000.0}, index=idx)

  target = forward_target(resid, horizon=horizon, execution_lag=lag)

  t = 20
  start = t + lag + 1
  window = resid["AAA"].iloc[start : start + horizon]
  expected = float((1.0 + window).prod() - 1.0)
  assert target["AAA"].iloc[t] == pytest.approx(expected, rel=1e-12)


def test_forward_target_tail_is_unresolvable():
  """The last few rows have no future left and must be NaN, not zero."""
  n = 30
  idx = pd.bdate_range("2021-01-01", periods=n)
  resid = pd.DataFrame({"AAA": np.zeros(n)}, index=idx)
  target = forward_target(resid, horizon=1, execution_lag=1)
  assert target["AAA"].iloc[-2:].isna().all()


def test_label_end_offset_matches_forward_target():
  """The purge width must cover exactly how far a label reaches."""
  for lag in (0, 1, 3):
    for h in (1, 5):
      n = 80
      idx = pd.bdate_range("2021-01-01", periods=n)
      resid = pd.DataFrame({"A": np.zeros(n)}, index=idx)
      target = forward_target(resid, horizon=h, execution_lag=lag)
      last_valid = int(np.flatnonzero(target["A"].notna().to_numpy())[-1])
      # Row `last_valid` consumes data through last_valid + lag + h.
      assert last_valid + label_end_offset(h, lag) == n - 1
