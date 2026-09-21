"""Baseline forecasters.

These exist to answer the only question that matters before investing in a
foundation model: does the foundation model beat a ridge regression on the
same features? If it does not, the transformer is decoration.

`ZeroForecaster` is the null model and should be run every time. On residual
returns it is a genuinely hard baseline to beat, which is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .base import Panel, QuantileForecaster


def _clean(X: pd.DataFrame) -> np.ndarray:
  return X.to_numpy(dtype=float, copy=True)


class ZeroForecaster(QuantileForecaster):
  """Predicts zero conditional mean with a constant, unconditional spread.

  The honest null for residual returns: after hedging out market and sector,
  the best guess for tomorrow really is roughly zero. Any model that cannot
  beat this on pinball loss has learned nothing.
  """

  name = "zero"

  def fit(self, panel: Panel, train_index: pd.MultiIndex) -> "ZeroForecaster":
    y = panel.target.loc[train_index].dropna()
    self._q = np.quantile(y.to_numpy(), self.quantiles)
    # Centre so the median is exactly zero; the spread is what we keep.
    self._q = self._q - self._q[self.median_idx]
    return self

  def predict_quantiles(self, panel: Panel, index: pd.MultiIndex) -> np.ndarray:
    return np.tile(self._q, (len(index), 1))


class RidgeForecaster(QuantileForecaster):
  """Linear location model plus a linear scale model.

  Two stages: ridge for the conditional mean, then a second ridge on log
  absolute residuals for the conditional scale. The scale stage matters -- a
  constant-width interval would make the confidence gate meaningless, since
  the gate keys off interval width.
  """

  name = "ridge"

  def __init__(self, quantiles: list[float], alphas: list[float] | None = None):
    super().__init__(quantiles)
    self.alphas = alphas or [1.0, 10.0, 100.0, 1000.0]

  def _select_alpha(self, X: np.ndarray, y: np.ndarray) -> float:
    """Pick alpha on a trailing slice of the training window.

    Deliberately a single chronological split rather than K-fold: shuffled
    folds leak future information into hyperparameter choice.
    """
    cut = int(len(y) * 0.8)
    if cut < 50 or len(y) - cut < 20:
      return float(self.alphas[len(self.alphas) // 2])
    best, best_mse = self.alphas[0], np.inf
    for a in self.alphas:
      m = Ridge(alpha=a).fit(X[:cut], y[:cut])
      mse = float(np.mean((m.predict(X[cut:]) - y[cut:]) ** 2))
      if mse < best_mse:
        best, best_mse = a, mse
    return float(best)

  def fit(self, panel: Panel, train_index: pd.MultiIndex) -> "RidgeForecaster":
    X_df, y_s = panel.slice(train_index)
    mask = y_s.notna().to_numpy()
    X, y = _clean(X_df)[mask], y_s.to_numpy(dtype=float)[mask]

    self._medians = np.nanmedian(X, axis=0)
    self._medians = np.where(np.isfinite(self._medians), self._medians, 0.0)
    X = np.where(np.isfinite(X), X, self._medians)

    self._scaler = StandardScaler().fit(X)
    Xs = self._scaler.transform(X)

    alpha = self._select_alpha(Xs, y)
    self._loc = Ridge(alpha=alpha).fit(Xs, y)

    resid = y - self._loc.predict(Xs)
    # Model log-scale so the fitted scale is positive by construction.
    log_abs = np.log(np.abs(resid) + 1e-8)
    self._scale = Ridge(alpha=alpha * 10).fit(Xs, log_abs)

    # Empirical quantiles of the standardised residual, so the shape of the
    # predictive distribution comes from the data rather than a normal
    # assumption that would understate the tails.
    scale_hat = np.exp(self._scale.predict(Xs))
    scale_hat = np.clip(scale_hat, 1e-8, None)
    z = resid / scale_hat
    self._z_q = np.quantile(z, self.quantiles)
    return self

  def _prepare(self, X_df: pd.DataFrame) -> np.ndarray:
    X = _clean(X_df)
    X = np.where(np.isfinite(X), X, self._medians)
    return self._scaler.transform(X)

  def predict_quantiles(self, panel: Panel, index: pd.MultiIndex) -> np.ndarray:
    Xs = self._prepare(panel.features.loc[index])
    loc = self._loc.predict(Xs)[:, None]
    scale = np.clip(np.exp(self._scale.predict(Xs)), 1e-8, None)[:, None]
    return self.enforce_monotone(loc + scale * self._z_q[None, :])


class GBMForecaster(QuantileForecaster):
  """Gradient boosting with pinball loss, one model per quantile.

  Uses HistGradientBoostingRegressor, which handles NaNs natively -- useful
  here because early rows of any rolling feature are legitimately missing and
  imputing them would invent data.
  """

  name = "gbm"

  def __init__(
    self,
    quantiles: list[float],
    max_iter: int = 200,
    learning_rate: float = 0.05,
    max_depth: int | None = 3,
    min_samples_leaf: int = 200,
    l2_regularization: float = 1.0,
    random_state: int = 7,
  ):
    super().__init__(quantiles)
    self.params = dict(
      max_iter=max_iter,
      learning_rate=learning_rate,
      max_depth=max_depth,
      min_samples_leaf=min_samples_leaf,
      l2_regularization=l2_regularization,
      random_state=random_state,
      early_stopping=False,
    )

  def fit(self, panel: Panel, train_index: pd.MultiIndex) -> "GBMForecaster":
    from sklearn.ensemble import HistGradientBoostingRegressor

    X_df, y_s = panel.slice(train_index)
    mask = y_s.notna().to_numpy()
    X, y = _clean(X_df)[mask], y_s.to_numpy(dtype=float)[mask]

    self._models = []
    for q in self.quantiles:
      m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.params)
      m.fit(X, y)
      self._models.append(m)
    return self

  def predict_quantiles(self, panel: Panel, index: pd.MultiIndex) -> np.ndarray:
    X = _clean(panel.features.loc[index])
    preds = np.column_stack([m.predict(X) for m in self._models])
    return self.enforce_monotone(preds)


def build_model(name: str, cfg) -> QuantileForecaster:
  """Instantiate a forecaster by config name."""
  quantiles = list(cfg.models.quantiles)
  if name == "zero":
    return ZeroForecaster(quantiles)
  if name == "ridge":
    return RidgeForecaster(quantiles, alphas=list(cfg.get("models.ridge.alphas", [])))
  if name == "gbm":
    return GBMForecaster(quantiles, **(cfg.get("models.gbm", {}) or {}))
  if name == "timesfm3":
    from .timesfm3 import TimesFM3Forecaster as Adapter

    return Adapter(quantiles, cfg)
  raise ValueError(f"unknown model {name!r}")
