"""Walk-forward orchestration: fit, predict, score, trade, ablate."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import gate as gate_mod
from . import signal as signal_mod
from .backtest.engine import BacktestResult, simulate
from .backtest.splits import Fold, assert_no_leakage, index_for_dates, walk_forward
from .config import Config
from .evaluation.calibration import evaluate_forecasts, skill_vs_null
from .models.base import Panel
from .models.baselines import build_model
from .pipeline import usable_index
from .targets import label_end_offset

log = logging.getLogger(__name__)


def build_folds(cfg: Config, dates: pd.DatetimeIndex) -> list[Fold]:
  purge = label_end_offset(
    int(cfg.target.horizon_days), int(cfg.backtest.execution_lag)
  )
  folds = walk_forward(
    dates=dates,
    train_years=float(cfg.backtest.train_years),
    test_months=int(cfg.backtest.test_months),
    purge_days=purge,
    embargo_days=int(cfg.backtest.embargo_days),
  )
  for f in folds:
    assert_no_leakage(f, purge)
  return folds


def run_walk_forward(
  cfg: Config,
  panel: Panel,
  model_names: list[str],
  dates: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
  """Fit and predict every model across every fold.

  Returns {model_name: (date, ticker) x quantile predictions}. Models are
  refit from scratch on each fold's training window -- no state survives a
  fold boundary, which is the whole point.
  """
  folds = build_folds(cfg, dates)
  if not folds:
    raise RuntimeError(
      "no folds produced; shorten backtest.train_years or widen the date range"
    )
  log.info("built %d walk-forward folds", len(folds))
  for f in folds:
    log.info("  %s", f.describe())

  eligible = usable_index(panel)
  predictions: dict[str, list[pd.DataFrame]] = {n: [] for n in model_names}

  for fold in folds:
    train_idx = index_for_dates(eligible, fold.train_dates)
    test_idx = index_for_dates(eligible, fold.test_dates)
    if len(train_idx) == 0 or len(test_idx) == 0:
      log.warning("fold %d empty after filtering; skipped", fold.index)
      continue

    for name in model_names:
      started = time.time()
      model = build_model(name, cfg)
      model.fit(panel, train_idx)
      preds = model.predict_frame(panel, test_idx)
      predictions[name].append(preds)
      log.info(
        "fold %d %-9s train=%d test=%d %.1fs",
        fold.index,
        name,
        len(train_idx),
        len(test_idx),
        time.time() - started,
      )

  return {
    name: pd.concat(parts).sort_index()
    for name, parts in predictions.items()
    if parts
  }


def score_models(
  cfg: Config, panel: Panel, predictions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
  """Forecast-quality table, with skill measured against the null model."""
  quantiles = list(cfg.models.quantiles)
  rows = {}
  for name, preds in predictions.items():
    actual = panel.target.reindex(preds.index)
    rows[name] = evaluate_forecasts(preds, actual, quantiles)

  if "zero" in rows:
    for name in rows:
      rows[name].update(skill_vs_null(rows[name], rows["zero"]))

  table = pd.DataFrame(rows).T
  ordered = [
    c
    for c in [
      "pinball_mean",
      "pinball_skill",
      "ic_mean",
      "ic_t",
      "directional_accuracy",
      "cov80_empirical",
      "cov80_mean_width",
      "chi2_pvalue",
      "tail_mass_excess",
      "n",
    ]
    if c in table.columns
  ]
  return table[ordered + [c for c in table.columns if c not in ordered]]


def _trailing_vol_for(panel: Panel, index: pd.MultiIndex) -> pd.Series:
  return gate_mod.trailing_vol(panel.residuals).reindex(index)


def backtest_predictions(
  cfg: Config,
  panel: Panel,
  preds: pd.DataFrame,
  use_gate: bool = True,
  gate_override: pd.Series | None = None,
) -> tuple[BacktestResult, pd.DataFrame]:
  """Quantiles -> weights -> PnL."""
  quantiles = list(cfg.models.quantiles)
  meta = panel.meta  # type: ignore[attr-defined]

  mask = None
  if gate_override is not None:
    mask = gate_override
  elif use_gate and cfg.get("gate.enabled", True):
    width = signal_mod.interval_width(preds, quantiles)
    mask = gate_mod.width_gate(
      width,
      percentile=float(cfg.gate.width_percentile),
      min_names=int(cfg.gate.min_names),
    ).mask

  sig = signal_mod.build_signal(
    preds,
    quantiles,
    gross_leverage=float(cfg.backtest.gross_leverage),
    max_weight=float(cfg.backtest.max_weight),
    mask=mask,
  )
  weights = sig["weight"].unstack("ticker").fillna(0.0)

  result = simulate(
    weights=weights,
    stock_returns=meta["stock_returns"],
    betas=meta["betas"],
    etf_returns=meta["etf_returns"],
    sector_map=meta["sector_map"],
    market=meta["market"],
    execution_lag=int(cfg.backtest.execution_lag),
    cost_bps=float(cfg.backtest.cost_bps),
    spread_bps=float(cfg.backtest.spread_bps),
    borrow_bps_annual=float(cfg.backtest.borrow_bps_annual),
    max_daily_turnover=cfg.get("backtest.max_daily_turnover"),
  )
  # Restrict reported PnL to dates the predictions actually cover.
  span = preds.index.get_level_values("date")
  window = (result.returns.index >= span.min()) & (result.returns.index <= span.max())
  result.returns = result.returns[window]
  result.gross_returns = result.gross_returns[window]
  result.costs = result.costs[window]
  result.turnover = result.turnover[window]
  from .backtest.engine import performance_metrics

  result.metrics = performance_metrics(result)
  return result, sig


def ablate_gate(
  cfg: Config, panel: Panel, preds: pd.DataFrame
) -> pd.DataFrame:
  """Does the confidence gate beat a plain volatility filter?

  Four books, identical except for how names are selected:
    no_gate      -- trade everything
    width_gate   -- the model's predictive interval width
    vol_gate     -- trailing realised volatility, matched trade count
    random_gate  -- random selection, matched trade count (sanity floor)

  If width_gate does not clear vol_gate by more than the Sharpe standard
  error, the gate is a volatility filter and should be described as one.
  """
  quantiles = list(cfg.models.quantiles)
  width = signal_mod.interval_width(preds, quantiles)
  rv = _trailing_vol_for(panel, preds.index)

  wg = gate_mod.width_gate(
    width,
    percentile=float(cfg.gate.width_percentile),
    min_names=int(cfg.gate.min_names),
  )
  vg = gate_mod.matched_count_vol_gate(rv, wg, min_names=int(cfg.gate.min_names))

  rng = np.random.default_rng(int(cfg.get("seed", 7)))
  noise = pd.Series(rng.random(len(preds.index)), index=preds.index)
  rg = gate_mod.matched_count_vol_gate(noise, wg, min_names=int(cfg.gate.min_names))

  variants = {
    "no_gate": None,
    "width_gate": wg.mask,
    "vol_gate": vg.mask,
    "random_gate": rg.mask,
  }

  rows = {}
  for label, mask in variants.items():
    result, _ = backtest_predictions(
      cfg, panel, preds, use_gate=False, gate_override=mask
    )
    rows[label] = result.metrics

  table = pd.DataFrame(rows).T
  overlap = gate_mod.gate_overlap(wg, vg)
  table["jaccard_vs_vol"] = np.nan
  table.loc["width_gate", "jaccard_vs_vol"] = overlap["jaccard"]
  table.attrs["overlap"] = overlap
  return table


def save_predictions(preds: dict[str, pd.DataFrame], out_dir: Path) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  for name, df in preds.items():
    df.to_parquet(out_dir / f"predictions_{name}.parquet")


def load_predictions(names: list[str], out_dir: Path) -> dict[str, pd.DataFrame]:
  out: dict[str, pd.DataFrame] = {}
  for name in names:
    path = out_dir / f"predictions_{name}.parquet"
    if path.exists():
      out[name] = pd.read_parquet(path)
  return out
