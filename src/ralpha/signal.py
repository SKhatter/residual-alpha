"""Turning quantile forecasts into target weights.

The signal is cross-sectional by construction: we rank names against each
other on the same day rather than predicting each one's return in isolation.
That is the form the edge actually takes. It also removes the need to forecast
the market's direction, which we have no ability to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_return(q_pred: pd.DataFrame, quantiles: list[float]) -> pd.Series:
  """Mean of the predictive distribution, from its quantiles.

  Uses the trapezoidal average across quantile levels rather than just the
  median. For a skewed predictive distribution -- common after a large move --
  the mean and the median disagree, and the mean is what a linear utility
  cares about.
  """
  q = np.asarray(quantiles, dtype=float)
  values = q_pred.to_numpy(dtype=float)
  # Treat the quantile function as piecewise linear in probability space.
  widths = np.diff(q)
  mids = (values[:, 1:] + values[:, :-1]) / 2.0
  interior = (mids * widths).sum(axis=1)
  # Attribute the unmodelled tails to the extreme quantiles.
  tail_lo = values[:, 0] * q[0]
  tail_hi = values[:, -1] * (1.0 - q[-1])
  return pd.Series(interior + tail_lo + tail_hi, index=q_pred.index)


def interval_width(
  q_pred: pd.DataFrame, quantiles: list[float], lo: float = 0.1, hi: float = 0.9
) -> pd.Series:
  """Width of a central predictive interval -- the model's own uncertainty."""
  q = np.asarray(quantiles, dtype=float)
  i_lo = int(np.argmin(np.abs(q - lo)))
  i_hi = int(np.argmin(np.abs(q - hi)))
  values = q_pred.to_numpy(dtype=float)
  return pd.Series(values[:, i_hi] - values[:, i_lo], index=q_pred.index)


def cross_sectional_score(scores: pd.Series, method: str = "rank") -> pd.Series:
  """Normalise scores within each date.

  'rank' is the default because it is robust to the outliers that a quantile
  model will occasionally emit after a gap. 'zscore' keeps magnitude
  information at the cost of letting one name dominate.
  """
  grouped = scores.groupby(level="date")
  if method == "rank":
    # (r - (n+1)/2) / n, which is exactly mean-zero and bounded by +/-0.5.
    # Plain `rank(pct=True) - 0.5` is off-centre by 1/(2n) and leaves a small
    # net long tilt in what is supposed to be a dollar-neutral book.
    return grouped.transform(
      lambda s: (s.rank() - (len(s) + 1) / 2.0) / len(s) if len(s) else s
    )
  if method == "zscore":
    return grouped.transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))
  if method == "demean":
    return grouped.transform(lambda s: s - s.mean())
  raise ValueError(f"unknown method {method!r}")


def to_weights(
  scores: pd.Series,
  gross_leverage: float = 1.0,
  max_weight: float = 0.05,
  dollar_neutral: bool = True,
) -> pd.Series:
  """Scale normalised scores into position weights.

  Applies the per-name cap and the gross target iteratively, because capping
  changes the gross and rescaling can push names back over the cap.
  """

  def _one_day(s: pd.Series) -> pd.Series:
    w = s.astype(float)
    if w.empty:
      return w
    if dollar_neutral:
      w = w - w.mean()
    if w.abs().sum() < 1e-12:
      return w * 0.0

    # A small universe under a tight cap cannot reach the gross target at all;
    # asking for it anyway is how a backtest ends up silently over the cap.
    target_gross = min(gross_leverage, max_weight * len(w))

    # Alternating projection onto {gross == target} and {|w| <= cap}, and onto
    # {sum w == 0} when neutral. Ending on the clip guarantees the cap holds
    # exactly; the capital freed by clipping is redistributed to uncapped
    # names on the next rescale, which is the water-filling behaviour we want.
    for _ in range(64):
      gross = w.abs().sum()
      if gross < 1e-12:
        break
      previous = w
      w = w * (target_gross / gross)
      w = w.clip(lower=-max_weight, upper=max_weight)
      if dollar_neutral:
        w = (w - w.mean()).clip(lower=-max_weight, upper=max_weight)
      if np.max(np.abs(w.to_numpy() - previous.to_numpy())) < 1e-15:
        break
    return w

  return scores.groupby(level="date", group_keys=False).apply(_one_day)


def build_signal(
  q_pred: pd.DataFrame,
  quantiles: list[float],
  gross_leverage: float = 1.0,
  max_weight: float = 0.05,
  method: str = "rank",
  mask: pd.Series | None = None,
) -> pd.DataFrame:
  """Full path from quantiles to weights.

  Returns a frame with the intermediate columns kept, so a disappointing
  backtest can be diagnosed without re-running inference.
  """
  mu = expected_return(q_pred, quantiles)
  width = interval_width(q_pred, quantiles)

  if mask is not None:
    mu = mu.where(mask.reindex(mu.index).fillna(False), np.nan)

  score = cross_sectional_score(mu.dropna(), method=method)
  weights = to_weights(score, gross_leverage, max_weight)

  out = pd.DataFrame({"mu": mu, "width": width})
  out["score"] = score.reindex(out.index)
  out["weight"] = weights.reindex(out.index).fillna(0.0)
  return out
