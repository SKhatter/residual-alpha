"""TimesFM-3 adapter.

Built against the real v3 API in google-research/timesfm (src/timesfm3), not
against the blog post. Things that bite you:

* `TimesFM3Evaluator` defaults to `make_positive=True`, which clamps negative
  forecasts to zero. On return series that silently destroys half the signal.
  We use the base `TimesFM3Forecaster` and pin `make_positive=False`.
* The implementation caps covariates at 31 slots and *subsamples* beyond that
  rather than erroring. We assert instead, so a silent drop can't happen.
* `past_future_covariates` must span `context_len + horizon`. Only genuinely
  known-in-advance series qualify -- calendar effects, FOMC dates, earnings
  dates. VIX and volume are past-only no matter how tempting.
* Quantiles are per-step. They are not additive across steps, so multi-day
  cumulative intervals are an approximation -- see `_collapse_horizon`.

The model is zero-shot; `fit()` does no gradient work. It fits an affine
recalibration on the training window only, because the pretrained quantile
heads are calibrated to general time series and are near-certainly too narrow
on financial returns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import Panel, QuantileForecaster

log = logging.getLogger(__name__)

MAX_COVARIATE_SLOTS = 31


def _resolve_device(requested: str | None) -> str:
  if requested:
    return requested
  try:
    import torch
  except ImportError:
    return "cpu"
  if torch.cuda.is_available():
    return "cuda"
  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    return "mps"
  return "cpu"


class TimesFM3Forecaster(QuantileForecaster):
  """Zero-shot multivariate forecaster over residual return series."""

  name = "timesfm3"

  def __init__(self, quantiles: list[float], cfg):
    super().__init__(quantiles)
    mcfg = cfg.models.timesfm3
    self.checkpoint = mcfg.get("checkpoint", "google/timesfm-3.0-pytorch")
    self.context_length = int(mcfg.get("context_length", 512))
    self.batch_size = int(mcfg.get("batch_size", 4))
    self.device = _resolve_device(mcfg.get("device"))
    self.max_covariates = int(mcfg.get("max_covariates", MAX_COVARIATE_SLOTS))
    self.recalibrate = bool(mcfg.get("recalibrate", True))
    self.calibration_samples = int(mcfg.get("calibration_samples", 2000))

    self.horizon = int(cfg.target.horizon_days)
    self.execution_lag = int(cfg.backtest.execution_lag)
    self.steps = self.execution_lag + self.horizon

    self._model = None
    self._loc_a, self._loc_b = 0.0, 1.0
    self._spread_scale = 1.0

  # -- model plumbing -------------------------------------------------------
  def _ensure_model(self):
    if self._model is not None:
      return self._model
    try:
      from timesfm3 import TimesFM3Forecaster as _TFM
    except ImportError as exc:
      raise ImportError(
        "timesfm3 is not installed. From a clone of google-research/timesfm:\n"
        "    pip install -e '.[torch]'\n"
        "The TimesFM-3 weights are released for non-commercial, "
        "non-production use only -- check the licence before relying on this."
      ) from exc

    log.info("loading %s on %s", self.checkpoint, self.device)
    self._model = _TFM.from_pretrained(self.checkpoint, device=self.device)
    return self._model

  # -- input assembly -------------------------------------------------------
  def _covariate_names(self, panel: Panel) -> list[str]:
    names = [n for n in panel.covariates if n != "target"]
    n_future = len(panel.calendar.columns) if not panel.calendar.empty else 0
    budget = self.max_covariates - n_future
    if budget < 0:
      raise ValueError(
        f"{n_future} known-future covariates exceeds the {self.max_covariates} "
        "slot cap on their own"
      )
    if len(names) > budget:
      raise ValueError(
        f"{len(names)} past-only covariates + {n_future} known-future exceeds "
        f"the {self.max_covariates} slot cap. TimesFM-3 would silently "
        "subsample these; drop some explicitly instead."
      )
    return names

  def _build_inputs(
    self, panel: Panel, date: pd.Timestamp, tickers: list[str]
  ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str]]:
    """Assemble contexts and covariates for every ticker on one date."""
    resid = panel.residuals
    pos = resid.index.searchsorted(date)
    if pos < self.context_length:
      return [], [], [], []
    window = slice(pos - self.context_length + 1, pos + 1)
    hist_dates = resid.index[window]

    cov_names = self._covariate_names(panel)
    contexts, past_only, past_future, kept = [], [], [], []

    for ticker in tickers:
      ctx = resid[ticker].iloc[window].to_numpy(dtype=float)
      if not np.isfinite(ctx).all():
        continue  # incomplete history; skip rather than impute a return series

      po = []
      ok = True
      for name in cov_names:
        frame = panel.covariates[name]
        col = ticker if ticker in frame.columns else frame.columns[0]
        series = frame[col].reindex(hist_dates).to_numpy(dtype=float)
        if not np.isfinite(series).all():
          ok = False
          break
        po.append(series)
      if not ok:
        continue

      pf = []
      if not panel.calendar.empty:
        # Known-future covariates must cover context AND horizon.
        future_pos = slice(pos + 1, pos + 1 + self.steps)
        future_dates = resid.index[future_pos]
        if len(future_dates) < self.steps:
          continue
        span = hist_dates.append(future_dates)
        block = panel.calendar.reindex(span)
        if block.isna().any().any():
          block = block.ffill().fillna(0.0)
        pf = [block[c].to_numpy(dtype=float) for c in block.columns]

      contexts.append(ctx)
      past_only.append(np.stack(po) if po else None)
      past_future.append(np.stack(pf) if pf else None)
      kept.append(ticker)

    return contexts, past_only, past_future, kept

  # -- inference ------------------------------------------------------------
  def _collapse_horizon(self, q_steps: np.ndarray) -> np.ndarray:
    """Reduce per-step quantiles to one cumulative forecast.

    q_steps is (steps, n_quantiles) for the residual return path. We want the
    return over (execution_lag, execution_lag + horizon].

    For horizon == 1 this is an exact single step. For horizon > 1 quantiles
    are not additive, so we sum the medians and widen the spread by sqrt(h) --
    a random-walk approximation that understates tail dependence. Prefer
    horizon == 1 unless you have a reason not to.
    """
    window = q_steps[self.execution_lag : self.execution_lag + self.horizon]
    if self.horizon == 1:
      return window[0]
    med = window[:, self.median_idx].sum()
    spread = window[0] - window[0, self.median_idx]
    return med + spread * np.sqrt(self.horizon)

  def _raw_predict(
    self, panel: Panel, index: pd.MultiIndex
  ) -> pd.DataFrame:
    model = self._ensure_model()
    frame = pd.DataFrame(
      np.nan, index=index, columns=[f"q{i}" for i in range(self.n_quantiles)]
    )

    by_date = pd.Series(index=index, data=0).groupby(level="date")
    for date, group in by_date:
      tickers = list(group.index.get_level_values("ticker"))
      contexts, po, pf, kept = self._build_inputs(panel, date, tickers)
      if not contexts:
        continue

      for start in range(0, len(contexts), self.batch_size):
        stop = start + self.batch_size
        outputs = list(
          model.predict_batch(
            contexts=contexts[start:stop],
            horizon=self.steps,
            past_only_covariates=po[start:stop],
            past_future_covariates=pf[start:stop],
            return_quantiles=True,
            make_positive=False,  # returns are signed
            use_znorm=False,
            padding_mode="none",
          )
        )
        for offset, out in enumerate(outputs):
          if out.quantiles is None:
            continue
          q = np.asarray(out.quantiles)
          if q.ndim == 3:  # multivariate output; target is variate 0
            q = q[0]
          frame.loc[(date, kept[start + offset])] = self._collapse_horizon(q)

    return frame

  # -- interface ------------------------------------------------------------
  def fit(self, panel: Panel, train_index: pd.MultiIndex) -> "TimesFM3Forecaster":
    """Zero-shot: no weights change. Fits affine recalibration only."""
    if not self.recalibrate:
      return self

    # Subsample the training window -- running the full panel through a 330M
    # model to fit two scalars is not a good trade.
    rng = np.random.default_rng(0)
    n = min(self.calibration_samples, len(train_index))
    take = rng.choice(len(train_index), size=n, replace=False)
    sample = train_index[np.sort(take)]

    raw = self._raw_predict(panel, sample)
    y = panel.target.loc[sample]
    med = raw.iloc[:, self.median_idx]
    ok = med.notna() & y.notna()
    if ok.sum() < 100:
      log.warning("too few calibration points (%d); skipping", int(ok.sum()))
      return self

    # Shrink the median toward zero by regressing realised on predicted. A
    # slope below 1 is the normal outcome and is information, not a failure.
    x, yy = med[ok].to_numpy(), y[ok].to_numpy()
    var = float(np.var(x))
    if var > 1e-18:
      self._loc_b = float(np.cov(x, yy, ddof=1)[0, 1] / var)
      self._loc_a = float(np.mean(yy) - self._loc_b * np.mean(x))
    log.info("median recalibration: y = %.4g + %.4g * pred", self._loc_a, self._loc_b)

    # Scale the spread so realised central coverage matches nominal.
    lo_i, hi_i = 0, self.n_quantiles - 1
    nominal = self.quantiles[hi_i] - self.quantiles[lo_i]
    width = (raw.iloc[:, hi_i] - raw.iloc[:, lo_i])[ok].to_numpy()
    centred = np.abs(yy - (self._loc_a + self._loc_b * x))
    half = np.maximum(width / 2.0, 1e-12)
    # Fraction of nominal width needed to reach the nominal coverage level.
    need = float(np.quantile(centred / half, nominal))
    self._spread_scale = float(np.clip(need, 0.25, 10.0))
    log.info("spread rescaled by %.3f", self._spread_scale)
    return self

  def predict_quantiles(self, panel: Panel, index: pd.MultiIndex) -> np.ndarray:
    raw = self._raw_predict(panel, index)
    med = raw.iloc[:, self.median_idx].to_numpy()[:, None]
    q = raw.to_numpy(dtype=float)
    centred = q - med
    out = (self._loc_a + self._loc_b * med) + centred * self._spread_scale
    # Rows we could not forecast fall back to a flat zero-median prediction so
    # downstream code gets a full-length array; the gate will drop them.
    missing = ~np.isfinite(out).all(axis=1)
    if missing.any():
      log.debug("timesfm3 produced no forecast for %d rows", int(missing.sum()))
      out[missing] = 0.0
    return self.enforce_monotone(out)
