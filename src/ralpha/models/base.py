"""Common interface for every forecaster in the bake-off.

All models emit the same thing: a full set of quantiles over the forward
residual return, for every (date, ticker) in the requested index. That is what
makes the comparison meaningful -- a foundation model and a ridge regression
are scored with the same pinball loss on the same rows.

Models receive the whole `Panel` plus an index, rather than a pre-sliced X/y
matrix, because sequence models need raw history that a flat design matrix has
already destroyed.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Panel:
  """Everything a model might need, with an explicit time axis."""

  features: pd.DataFrame  # (date, ticker) x feature
  target: pd.Series  # (date, ticker) -> forward residual return
  residuals: pd.DataFrame  # date x ticker, the series being forecast
  covariates: dict[str, pd.DataFrame] = field(default_factory=dict)
  calendar: pd.DataFrame = field(default_factory=pd.DataFrame)

  @property
  def dates(self) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(self.features.index.get_level_values("date").unique())

  def slice(self, index: pd.MultiIndex) -> tuple[pd.DataFrame, pd.Series]:
    return self.features.loc[index], self.target.loc[index]


class QuantileForecaster(abc.ABC):
  """A model that maps (panel, index) to quantiles of the forward target."""

  name: str = "base"

  def __init__(self, quantiles: list[float]):
    self.quantiles = list(quantiles)
    self.n_quantiles = len(self.quantiles)
    if sorted(self.quantiles) != self.quantiles:
      raise ValueError("quantiles must be given in ascending order")

  @abc.abstractmethod
  def fit(self, panel: Panel, train_index: pd.MultiIndex) -> "QuantileForecaster":
    ...

  @abc.abstractmethod
  def predict_quantiles(
    self, panel: Panel, index: pd.MultiIndex
  ) -> np.ndarray:  # (len(index), n_quantiles)
    ...

  # -- shared helpers -------------------------------------------------------
  @property
  def median_idx(self) -> int:
    return int(np.argmin(np.abs(np.asarray(self.quantiles) - 0.5)))

  @staticmethod
  def enforce_monotone(q: np.ndarray) -> np.ndarray:
    """Sort each row so quantiles never cross.

    Independently fitted quantile regressions routinely produce q60 < q40 in
    sparse regions. Sorting is the standard cheap fix and cannot make the
    pinball loss worse.
    """
    return np.sort(q, axis=1)

  def predict_frame(self, panel: Panel, index: pd.MultiIndex) -> pd.DataFrame:
    q = self.predict_quantiles(panel, index)
    cols = [f"q{int(round(x * 100)):02d}" for x in self.quantiles]
    return pd.DataFrame(q, index=index, columns=cols)
