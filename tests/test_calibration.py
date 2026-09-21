"""Metric correctness, verified against closed forms where one exists."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ralpha.evaluation.calibration import (
  coverage,
  directional_accuracy,
  information_coefficient,
  interval_coverage,
  pinball_loss,
  pit_uniformity,
  pit_values,
  skill_vs_null,
)

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def test_pinball_loss_matches_hand_calculation():
  y = np.array([1.0])
  pred = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
  out = pinball_loss(y, pred, QUANTILES)
  # Under-prediction: loss is q * (y - pred) = q * 1.
  for q in QUANTILES:
    assert out[f"pinball_q{int(q * 100):02d}"] == pytest.approx(q)


def test_pinball_loss_penalises_over_prediction_by_one_minus_q():
  y = np.array([0.0])
  pred = np.array([[1.0] * 5])
  out = pinball_loss(y, pred, QUANTILES)
  for q in QUANTILES:
    assert out[f"pinball_q{int(q * 100):02d}"] == pytest.approx(1.0 - q)


def test_pinball_minimised_at_true_quantiles():
  """The scoring rule must actually be proper: truth beats any shift."""
  rng = np.random.default_rng(1)
  y = rng.normal(0, 1, 20000)
  truth = np.tile(stats.norm.ppf(QUANTILES), (len(y), 1))
  best = pinball_loss(y, truth, QUANTILES)["pinball_mean"]

  for shift in (-0.3, -0.1, 0.1, 0.3):
    worse = pinball_loss(y, truth + shift, QUANTILES)["pinball_mean"]
    assert worse > best

  narrow = pinball_loss(y, truth * 0.5, QUANTILES)["pinball_mean"]
  wide = pinball_loss(y, truth * 2.0, QUANTILES)["pinball_mean"]
  assert narrow > best and wide > best


def test_coverage_recovers_nominal_for_a_correct_model():
  rng = np.random.default_rng(2)
  y = rng.normal(0, 1, 50000)
  pred = np.tile(stats.norm.ppf(QUANTILES), (len(y), 1))
  cov = coverage(y, pred, QUANTILES)
  assert np.allclose(cov["empirical"], cov["nominal"], atol=0.01)


def test_coverage_detects_intervals_that_are_too_narrow():
  """The expected failure mode on financial returns."""
  rng = np.random.default_rng(3)
  y = stats.t.rvs(df=3, size=50000, random_state=rng) / np.sqrt(3.0)
  # Gaussian intervals fitted to a fat-tailed truth.
  pred = np.tile(stats.norm.ppf(QUANTILES), (len(y), 1))
  iv = interval_coverage(y, pred, QUANTILES, 0.1, 0.9)
  assert iv["empirical"] > iv["nominal"]  # too wide in the body...

  tails = np.mean(np.abs(y) > 3.0)
  assert tails > 0.002  # ...yet the tails are much heavier than Gaussian


def test_pit_uniform_for_calibrated_model():
  """A correct model must pass both the shape test and the tail-mass test."""
  rng = np.random.default_rng(4)
  y = rng.normal(0, 1, 20000)
  pred = np.tile(stats.norm.ppf(QUANTILES), (len(y), 1))
  out = pit_uniformity(pit_values(y, pred, QUANTILES), QUANTILES)

  assert out["chi2_pvalue"] > 0.01
  # With a 10/90 grid, 20% of observations fall outside by construction.
  assert out["tail_mass_nominal"] == pytest.approx(0.2)
  assert abs(out["tail_mass_excess"]) < 0.02


def test_pit_detects_intervals_that_are_too_narrow():
  """The failure mode we actually expect from a pretrained model on returns."""
  rng = np.random.default_rng(5)
  y = rng.normal(0, 3, 20000)  # truth is 3x wider than claimed
  pred = np.tile(stats.norm.ppf(QUANTILES), (len(y), 1))
  out = pit_uniformity(pit_values(y, pred, QUANTILES), QUANTILES)

  # Far more mass outside the stated interval than the grid allows for.
  assert out["tail_mass_excess"] > 0.3


def test_pit_detects_intervals_that_are_too_wide():
  rng = np.random.default_rng(6)
  y = rng.normal(0, 0.3, 20000)
  pred = np.tile(stats.norm.ppf(QUANTILES), (len(y), 1))
  out = pit_uniformity(pit_values(y, pred, QUANTILES), QUANTILES)
  assert out["tail_mass_excess"] < -0.15


def _panel(pred_vals, actual_vals, n_days=100, n_names=20):
  idx = pd.MultiIndex.from_product(
    [pd.bdate_range("2021-01-01", periods=n_days), [f"S{i}" for i in range(n_names)]],
    names=["date", "ticker"],
  )
  return pd.Series(pred_vals, index=idx), pd.Series(actual_vals, index=idx)


def test_information_coefficient_is_zero_for_random_predictions():
  rng = np.random.default_rng(6)
  n = 100 * 20
  pred, actual = _panel(rng.normal(size=n), rng.normal(size=n))
  out = information_coefficient(pred, actual)
  assert abs(out["ic_mean"]) < 0.05
  assert abs(out["ic_t"]) < 3.0


def test_information_coefficient_detects_a_real_signal():
  rng = np.random.default_rng(7)
  n = 100 * 20
  pred_vals = rng.normal(size=n)
  actual_vals = 0.3 * pred_vals + rng.normal(size=n)
  pred, actual = _panel(pred_vals, actual_vals)
  out = information_coefficient(pred, actual)
  assert out["ic_mean"] > 0.15
  assert out["ic_t"] > 5.0


def test_directional_accuracy_ignores_zero_predictions():
  idx = pd.MultiIndex.from_tuples(
    [(pd.Timestamp("2021-01-04"), t) for t in "ABCD"], names=["date", "ticker"]
  )
  pred = pd.Series([1.0, -1.0, 0.0, 1.0], index=idx)
  actual = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
  # Of the three non-zero predictions, two matched.
  assert directional_accuracy(pred, actual) == pytest.approx(2 / 3)


def test_skill_vs_null_signs_correctly():
  assert skill_vs_null({"pinball_mean": 0.9}, {"pinball_mean": 1.0})[
    "pinball_skill"
  ] == pytest.approx(0.1)
  assert skill_vs_null({"pinball_mean": 1.1}, {"pinball_mean": 1.0})[
    "pinball_skill"
  ] < 0
