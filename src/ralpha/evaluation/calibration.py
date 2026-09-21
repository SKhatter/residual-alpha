"""Forecast quality metrics, with calibration treated as a first-class check.

Two distinct questions, often conflated:

  1. Is the *location* informative?  -> information coefficient, directional
     accuracy, pinball loss versus the null.
  2. Is the *spread* honest?         -> coverage, PIT uniformity.

The second matters here specifically because the confidence gate keys off
interval width. If the intervals are systematically too narrow -- the expected
failure mode for a model pretrained mostly on non-financial series with thin
tails -- then the gate is selecting on a miscalibrated quantity and every
decision downstream of it inherits the error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def pinball_loss(
  y: np.ndarray, q_pred: np.ndarray, quantiles: list[float]
) -> pd.Series:
  """Per-quantile pinball (quantile) loss, plus the mean across quantiles.

  This is the proper scoring rule for quantile forecasts: lower is better, and
  it rewards sharpness only when calibration holds.
  """
  y = np.asarray(y, dtype=float)
  q_pred = np.asarray(q_pred, dtype=float)
  mask = np.isfinite(y) & np.isfinite(q_pred).all(axis=1)
  y, q_pred = y[mask], q_pred[mask]

  losses = {}
  for i, q in enumerate(quantiles):
    diff = y - q_pred[:, i]
    losses[f"pinball_q{int(round(q * 100)):02d}"] = float(
      np.mean(np.maximum(q * diff, (q - 1.0) * diff))
    )
  out = pd.Series(losses)
  out["pinball_mean"] = float(out.mean())
  out["n"] = int(len(y))
  return out


def coverage(
  y: np.ndarray, q_pred: np.ndarray, quantiles: list[float]
) -> pd.DataFrame:
  """Empirical vs nominal coverage at each quantile level."""
  y = np.asarray(y, dtype=float)
  q_pred = np.asarray(q_pred, dtype=float)
  mask = np.isfinite(y) & np.isfinite(q_pred).all(axis=1)
  y, q_pred = y[mask], q_pred[mask]

  rows = []
  for i, q in enumerate(quantiles):
    empirical = float(np.mean(y <= q_pred[:, i]))
    rows.append(
      {
        "quantile": q,
        "nominal": q,
        "empirical": empirical,
        "error": empirical - q,
      }
    )
  return pd.DataFrame(rows)


def interval_coverage(
  y: np.ndarray, q_pred: np.ndarray, quantiles: list[float], lo: float, hi: float
) -> dict[str, float]:
  """Coverage and mean width of one central interval."""
  q_arr = np.asarray(quantiles)
  i_lo = int(np.argmin(np.abs(q_arr - lo)))
  i_hi = int(np.argmin(np.abs(q_arr - hi)))
  y = np.asarray(y, dtype=float)
  q_pred = np.asarray(q_pred, dtype=float)
  mask = np.isfinite(y) & np.isfinite(q_pred).all(axis=1)
  y, q_pred = y[mask], q_pred[mask]

  inside = (y >= q_pred[:, i_lo]) & (y <= q_pred[:, i_hi])
  return {
    "nominal": float(q_arr[i_hi] - q_arr[i_lo]),
    "empirical": float(np.mean(inside)),
    "mean_width": float(np.mean(q_pred[:, i_hi] - q_pred[:, i_lo])),
  }


def pit_values(
  y: np.ndarray, q_pred: np.ndarray, quantiles: list[float]
) -> np.ndarray:
  """Probability integral transform via linear interpolation of the quantiles.

  If the predictive distribution is correct, these are uniform on [0, 1].
  A U-shaped histogram means intervals are too narrow (the usual result on
  returns); a hump in the middle means too wide.
  """
  y = np.asarray(y, dtype=float)
  q_pred = np.asarray(q_pred, dtype=float)
  mask = np.isfinite(y) & np.isfinite(q_pred).all(axis=1)
  y, q_pred = y[mask], q_pred[mask]

  out = np.empty(len(y))
  for i in range(len(y)):
    out[i] = np.interp(y[i], q_pred[i], quantiles, left=0.0, right=1.0)
  return out


def pit_uniformity(pit: np.ndarray, quantiles: list[float]) -> dict[str, float]:
  """Test calibration by how observations distribute across quantile bins.

  Why not a KS test on the PIT values
  -----------------------------------
  `pit_values` interpolates linearly between quantile levels, and the true
  quantile function is not linear between them. On a coarse grid that
  interpolation error is large enough that a KS test rejects a *perfectly
  calibrated* model given enough samples -- it detects the interpolation, not
  the forecast.

  The bin test avoids this entirely. A quantile forecast makes exactly one
  claim: P(y <= q_k) = k for each level k. So the fraction of observations
  landing between adjacent predicted quantiles should match the gap between
  those levels. Chi-square on those counts is exact under the null and needs
  no interpolation.

  Returns:
    chi2_pvalue      -- small means the predictive distribution is wrong.
    tail_mass_excess -- observed mass outside the outermost quantiles minus
                        the nominal amount. Positive means intervals are too
                        narrow, the expected failure mode on returns.
  """
  pit = pit[np.isfinite(pit)]
  lo, hi = float(quantiles[0]), float(quantiles[-1])
  nominal_tail = lo + (1.0 - hi)

  nan_result = {
    "chi2_stat": float("nan"),
    "chi2_pvalue": float("nan"),
    "tail_mass": float("nan"),
    "tail_mass_nominal": float(nominal_tail),
    "tail_mass_excess": float("nan"),
    "n": int(len(pit)),
  }
  if len(pit) < 20:
    return nan_result

  # Bin edges in probability space: [0, q0, q1, ..., qK, 1]. Interpolation is
  # monotone within a bin, so bin membership is exact even though the PIT
  # value inside the bin is approximate.
  edges = np.concatenate([[0.0], np.asarray(quantiles, dtype=float), [1.0]])
  expected_probs = np.diff(edges)
  keep = expected_probs > 1e-12
  observed = np.histogram(np.clip(pit, 0.0, 1.0), bins=edges)[0].astype(float)

  observed, expected_probs = observed[keep], expected_probs[keep]
  expected = expected_probs / expected_probs.sum() * observed.sum()

  if (expected < 5).any() or len(observed) < 2:
    chi2_stat = chi2_p = float("nan")
  else:
    chi2_stat = float(((observed - expected) ** 2 / expected).sum())
    chi2_p = float(stats.chi2.sf(chi2_stat, df=len(observed) - 1))

  observed_tail = float(np.mean((pit <= lo) | (pit >= hi)))
  return {
    "chi2_stat": chi2_stat,
    "chi2_pvalue": chi2_p,
    "tail_mass": observed_tail,
    "tail_mass_nominal": float(nominal_tail),
    "tail_mass_excess": float(observed_tail - nominal_tail),
    "n": int(len(pit)),
  }


def information_coefficient(
  predictions: pd.Series, actuals: pd.Series
) -> dict[str, float]:
  """Daily cross-sectional rank correlation, and whether its mean is real.

  Reported with a Newey-West style t-stat on the daily IC series. A mean IC of
  0.02 with a t-stat of 1.1 is noise; the same IC with a t-stat of 4 is a
  finding. Report both or neither.
  """
  df = pd.DataFrame({"pred": predictions, "actual": actuals}).dropna()
  if df.empty:
    return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_t": float("nan")}

  def _spearman(g: pd.DataFrame) -> float:
    if len(g) < 5 or g["pred"].nunique() < 2 or g["actual"].nunique() < 2:
      return np.nan
    return float(stats.spearmanr(g["pred"], g["actual"]).statistic)

  daily = df.groupby(level="date").apply(_spearman).dropna()
  if len(daily) < 2:
    return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_t": float("nan")}

  mean, sd = float(daily.mean()), float(daily.std(ddof=1))
  t_stat = mean / (sd / np.sqrt(len(daily))) if sd > 0 else float("nan")
  return {
    "ic_mean": mean,
    "ic_std": sd,
    "ic_t": float(t_stat),
    "ic_hit_rate": float((daily > 0).mean()),
    "n_days": int(len(daily)),
  }


def directional_accuracy(predictions: pd.Series, actuals: pd.Series) -> float:
  """Fraction of non-zero predictions whose sign matched.

  Compare against 0.5, and remember that on a panel with a positive drift even
  a constant long prediction beats 0.5.
  """
  df = pd.DataFrame({"pred": predictions, "actual": actuals}).dropna()
  df = df[df["pred"] != 0.0]
  if df.empty:
    return float("nan")
  return float((np.sign(df["pred"]) == np.sign(df["actual"])).mean())


def evaluate_forecasts(
  q_pred: pd.DataFrame,
  actuals: pd.Series,
  quantiles: list[float],
  median_idx: int | None = None,
) -> dict[str, float]:
  """Full metric bundle for one model on one evaluation set."""
  aligned = q_pred.reindex(actuals.index)
  y = actuals.to_numpy(dtype=float)
  q = aligned.to_numpy(dtype=float)

  if median_idx is None:
    median_idx = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
  median = pd.Series(q[:, median_idx], index=actuals.index)

  out: dict[str, float] = {}
  out.update(pinball_loss(y, q, quantiles).to_dict())
  out.update(information_coefficient(median, actuals))
  out["directional_accuracy"] = directional_accuracy(median, actuals)

  iv = interval_coverage(y, q, quantiles, 0.1, 0.9)
  out["cov80_nominal"] = iv["nominal"]
  out["cov80_empirical"] = iv["empirical"]
  out["cov80_mean_width"] = iv["mean_width"]
  out.update(pit_uniformity(pit_values(y, q, quantiles), quantiles))
  return out


def skill_vs_null(model_metrics: dict, null_metrics: dict) -> dict[str, float]:
  """Fractional pinball improvement over the null model.

  Positive means better than predicting zero. This is the number to lead with;
  a raw pinball loss is unreadable without it.
  """
  m, n = model_metrics.get("pinball_mean"), null_metrics.get("pinball_mean")
  if not m or not n or not np.isfinite(m) or not np.isfinite(n) or n == 0:
    return {"pinball_skill": float("nan")}
  return {"pinball_skill": float(1.0 - m / n)}
