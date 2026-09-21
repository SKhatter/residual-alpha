"""Confidence gate, and the ablation that tells you whether it is real.

The suspicion this module is built around
-----------------------------------------
"Trade only when the model's q90-q10 interval is tight" sounds like it is
selecting on model confidence. But interval width is, to a first
approximation, a volatility forecast. So the gate may be doing nothing more
than "trade only when volatility is low" -- which is a well-known factor
exposure (betting-against-beta, short volatility), not a property of your
model. It will look excellent for years and then lose all of it in a week.

`ablate` answers the question directly: run the identical strategy with the
gate driven by trailing realised volatility instead of model interval width,
holding the number of traded names fixed. If the model gate does not beat the
volatility gate, you have rediscovered low-volatility investing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GateResult:
  mask: pd.Series  # (date, ticker) -> bool, True = tradable
  criterion: pd.Series  # the value that was thresholded
  n_selected: pd.Series  # per-date count


def _select_tightest(
  criterion: pd.Series, percentile: float, min_names: int
) -> pd.Series:
  """Keep the lowest-criterion fraction of names on each date."""

  def _one_day(s: pd.Series) -> pd.Series:
    valid = s.dropna()
    if len(valid) == 0:
      return pd.Series(False, index=s.index)
    keep_n = max(int(np.floor(len(valid) * percentile)), min(min_names, len(valid)))
    threshold = valid.nsmallest(keep_n).max()
    return (s <= threshold) & s.notna()

  return criterion.groupby(level="date", group_keys=False).apply(_one_day)


def width_gate(
  width: pd.Series, percentile: float = 0.6, min_names: int = 10
) -> GateResult:
  """Gate on the model's own predictive interval width."""
  mask = _select_tightest(width, percentile, min_names)
  return GateResult(
    mask=mask,
    criterion=width,
    n_selected=mask.groupby(level="date").sum(),
  )


def vol_gate(
  realised_vol: pd.Series, percentile: float = 0.6, min_names: int = 10
) -> GateResult:
  """Gate on trailing realised volatility. The null hypothesis for the gate."""
  mask = _select_tightest(realised_vol, percentile, min_names)
  return GateResult(
    mask=mask,
    criterion=realised_vol,
    n_selected=mask.groupby(level="date").sum(),
  )


def trailing_vol(residuals: pd.DataFrame, window: int = 20) -> pd.Series:
  """Causal realised volatility of residual returns, as a long series."""
  rv = residuals.rolling(window).std()
  out = rv.stack(dropna=False)
  out.index.names = ["date", "ticker"]
  return out


def gate_overlap(a: GateResult, b: GateResult) -> dict[str, float]:
  """How similar two gates are.

  A Jaccard index near 1 means the model gate and the volatility gate are
  selecting the same names, which settles the question before you even look
  at returns.
  """
  ma, mb = a.mask.align(b.mask, join="inner", fill_value=False)
  both = (ma & mb).sum()
  either = (ma | mb).sum()
  return {
    "jaccard": float(both / either) if either else float("nan"),
    "agreement": float((ma == mb).mean()),
    "corr_criterion": float(
      a.criterion.corr(b.criterion.reindex(a.criterion.index), method="spearman")
    ),
    "mean_selected_a": float(a.n_selected.mean()),
    "mean_selected_b": float(b.n_selected.mean()),
  }


def matched_count_vol_gate(
  realised_vol: pd.Series, reference: GateResult, min_names: int = 10
) -> GateResult:
  """A volatility gate that selects exactly as many names per day as `reference`.

  Matching the trade count is what makes the ablation fair: otherwise the two
  strategies differ in breadth as well as in selection rule, and any
  difference in Sharpe is uninterpretable.
  """
  target = reference.n_selected

  def _one_day(s: pd.Series) -> pd.Series:
    date = s.index.get_level_values("date")[0]
    want = int(target.get(date, 0))
    valid = s.dropna()
    if want <= 0 or len(valid) == 0:
      return pd.Series(False, index=s.index)
    want = min(max(want, min(min_names, len(valid))), len(valid))
    keep = valid.nsmallest(want).index
    return pd.Series(s.index.isin(keep), index=s.index)

  mask = realised_vol.groupby(level="date", group_keys=False).apply(_one_day)
  return GateResult(
    mask=mask,
    criterion=realised_vol,
    n_selected=mask.groupby(level="date").sum(),
  )
