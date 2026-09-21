"""Feature construction.

Three families, kept separate because they have different information-timing
properties:

  1. Per-stock features      -- computed from that name's own history.
  2. Market/regime features  -- identical across names on a given date.
  3. Calendar features       -- KNOWN IN ADVANCE. These are the only ones that
                                can legitimately be handed to TimesFM-3 as
                                `past_future_covariates`.

Everything uses causal pandas rolling windows, so a value stamped at date t
depends only on data up to and including t. The gap between t and the traded
return is handled in targets.forward_target, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EPS = 1e-12


# --------------------------------------------------------------------------
# per-stock
# --------------------------------------------------------------------------
def stock_features(
  close: pd.DataFrame,
  volume: pd.DataFrame,
  residuals: pd.DataFrame,
  betas: pd.DataFrame,
) -> pd.DataFrame:
  """Per-name features as a long frame indexed by (date, ticker)."""
  ret = close.pct_change()
  frames: dict[str, pd.DataFrame] = {}

  # Short-horizon reversal. Negated because the documented effect is that
  # recent winners underperform over the next few days.
  frames["reversal_1d"] = -residuals
  frames["reversal_5d"] = -residuals.rolling(5).sum()

  # Residual momentum, skipping the most recent week to separate it from
  # reversal.
  for w in (21, 63, 126):
    frames[f"resid_mom_{w}d"] = residuals.rolling(w).sum().shift(5)

  # Volatility and its term structure.
  rv20 = ret.rolling(20).std()
  rv60 = ret.rolling(60).std()
  frames["rv_20d"] = rv20
  frames["rv_ratio"] = rv20 / (rv60 + EPS)
  idio20 = residuals.rolling(20).std()
  frames["idio_vol_20d"] = idio20
  frames["idio_share"] = idio20 / (rv20 + EPS)

  # Volume: level relative to its own recent norm, and its trend.
  log_vol = np.log1p(volume)
  vol_mean = log_vol.rolling(20).mean()
  vol_std = log_vol.rolling(60).std()
  frames["volume_z"] = (log_vol - vol_mean) / (vol_std + EPS)
  frames["volume_trend"] = vol_mean - log_vol.rolling(60).mean()

  # Amihud illiquidity: price impact per dollar traded.
  dollar_vol = (close * volume).replace(0.0, np.nan)
  frames["amihud"] = (ret.abs() / dollar_vol).rolling(20).mean() * 1e9

  # Distance from moving averages, scaled by vol so it is comparable across
  # names with different volatility.
  for w in (50, 200):
    ma = close.rolling(w).mean()
    frames[f"dist_ma{w}"] = (close / (ma + EPS) - 1.0) / (rv20 * np.sqrt(w) + EPS)

  # High-low range as an intraday vol proxy the close-to-close series misses.
  frames["overnight_gap"] = (close / close.shift(1) - 1.0) - ret

  long = pd.concat(
    {name: df.stack(dropna=False) for name, df in frames.items()}, axis=1
  )
  long.index.names = ["date", "ticker"]

  # Betas are already a (date, ticker) long frame from targets.residual_returns.
  beta_cols = betas[["beta_mkt", "beta_sector"]].reindex(long.index)
  return long.join(beta_cols).sort_index()


# --------------------------------------------------------------------------
# market / regime
# --------------------------------------------------------------------------
def market_features(
  aux_close: pd.DataFrame,
  market_ret: pd.Series,
  universe_close: pd.DataFrame,
) -> pd.DataFrame:
  """Date-indexed features shared by every name."""
  out = pd.DataFrame(index=market_ret.index)

  out["mkt_ret_1d"] = market_ret
  out["mkt_ret_5d"] = market_ret.rolling(5).sum()
  out["mkt_ret_21d"] = market_ret.rolling(21).sum()
  mkt_rv = market_ret.rolling(20).std()
  out["mkt_rv_20d"] = mkt_rv

  def series(name: str) -> pd.Series | None:
    if name in aux_close.columns:
      return aux_close[name].reindex(market_ret.index)
    log.warning("auxiliary series %s missing; dependent features skipped", name)
    return None

  vix = series("^VIX")
  if vix is not None:
    out["vix"] = vix
    out["vix_chg_1d"] = vix.pct_change()
    out["vix_chg_5d"] = vix.pct_change(5)
    out["vix_z_60d"] = (vix - vix.rolling(60).mean()) / (
      vix.rolling(60).std() + EPS
    )
    # Variance risk premium proxy: implied vol vs realised vol, annualised.
    out["vrp"] = vix / 100.0 - mkt_rv * np.sqrt(252)

  tnx, irx = series("^TNX"), series("^IRX")
  if tnx is not None:
    out["rate_10y"] = tnx
    out["rate_10y_chg_5d"] = tnx.diff(5)
  if tnx is not None and irx is not None:
    out["term_spread"] = tnx - irx

  # Breadth: equal-weight vs cap-weight is the cleanest single proxy, and the
  # share of the universe above its own 50d average is the direct measure.
  rsp = series("RSP")
  if rsp is not None:
    out["breadth_ew_cw_5d"] = rsp.pct_change(5) - market_ret.rolling(5).sum()
  above_ma = (universe_close > universe_close.rolling(50).mean()).mean(axis=1)
  out["breadth_pct_above_ma50"] = above_ma.reindex(market_ret.index)
  out["breadth_chg_5d"] = out["breadth_pct_above_ma50"].diff(5)

  hyg, tlt = series("HYG"), series("TLT")
  if hyg is not None:
    out["credit_ret_5d"] = hyg.pct_change(5)
  if tlt is not None:
    out["duration_ret_5d"] = tlt.pct_change(5)
  if hyg is not None and tlt is not None:
    out["credit_excess_5d"] = hyg.pct_change(5) - tlt.pct_change(5)

  # Cross-sectional dispersion: how much idiosyncratic movement is on offer.
  out["xs_dispersion_20d"] = (
    universe_close.pct_change().rolling(20).std().mean(axis=1)
  )
  return out


# --------------------------------------------------------------------------
# calendar (known in advance)
# --------------------------------------------------------------------------
def _load_event_dates(path: Path | None) -> pd.DatetimeIndex:
  if path is None or not Path(path).exists():
    return pd.DatetimeIndex([])
  df = pd.read_csv(path)
  col = "date" if "date" in df.columns else df.columns[0]
  return pd.DatetimeIndex(pd.to_datetime(df[col]).dt.normalize().unique())


def _days_to_next(dates: pd.DatetimeIndex, events: pd.DatetimeIndex) -> pd.Series:
  if len(events) == 0:
    return pd.Series(np.nan, index=dates)
  ev = np.sort(events.values.astype("datetime64[D]"))
  d = dates.values.astype("datetime64[D]")
  idx = np.searchsorted(ev, d, side="left")
  out = np.full(len(d), np.nan)
  valid = idx < len(ev)
  out[valid] = (ev[idx[valid]] - d[valid]).astype(int)
  return pd.Series(out, index=dates)


def calendar_features(
  dates: pd.DatetimeIndex,
  fomc_path: Path | None = None,
  earnings_path: Path | None = None,
  day_of_week: bool = True,
  turn_of_month: bool = True,
) -> pd.DataFrame:
  """Features knowable arbitrarily far in advance.

  These are the genuine `past_future_covariates` for TimesFM-3. Published
  benchmark gains from covariate support come overwhelmingly from known-future
  inputs, and none of the market channels (VIX, volume, rates) qualify.
  """
  dates = pd.DatetimeIndex(dates)
  out = pd.DataFrame(index=dates)

  if day_of_week:
    dow = dates.dayofweek
    # Cyclical encoding so Monday and Friday are not treated as far apart by
    # linear models.
    out["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)

  if turn_of_month:
    month_id = dates.year * 12 + dates.month
    pos = pd.Series(dates, index=dates).groupby(month_id).cumcount()
    size = pd.Series(month_id, index=dates).map(
      pd.Series(month_id).value_counts()
    )
    from_end = size.values - 1 - pos.values
    out["tom_start"] = (pos.values < 3).astype(float)
    out["tom_end"] = (from_end < 3).astype(float)

  fomc = _load_event_dates(fomc_path)
  if len(fomc):
    days = _days_to_next(dates, fomc)
    out["days_to_fomc"] = days.clip(upper=30)
    out["fomc_window"] = (days <= 1).astype(float)
  else:
    log.warning("no FOMC dates loaded; skipping FOMC calendar features")

  earnings = _load_event_dates(earnings_path)
  if len(earnings):
    days = _days_to_next(dates, earnings)
    out["days_to_earnings"] = days.clip(upper=60)
  return out


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def cross_sectional_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
  """Rank-normalise each column to [-0.5, 0.5] within each date.

  Cross-sectional ranking is what makes the panel comparable across regimes:
  a 2% move means something different in 2017 than in March 2020, but "third
  from the top today" means the same thing in both.
  """
  out = df.copy()
  grouped = out.groupby(level="date")[cols]
  ranks = grouped.rank()
  counts = out.groupby(level="date")[cols].transform("count")
  # Exactly mean-zero within each date; `rank(pct=True) - 0.5` is not.
  out[cols] = (ranks - (counts + 1) / 2.0).div(counts.replace(0, np.nan))
  return out


def build_features(
  close: pd.DataFrame,
  volume: pd.DataFrame,
  residuals: pd.DataFrame,
  betas: pd.DataFrame,
  aux_close: pd.DataFrame,
  market_ret: pd.Series,
  calendar: pd.DataFrame | None = None,
  rank_normalise: bool = True,
) -> pd.DataFrame:
  """Assemble the full (date, ticker) design matrix."""
  per_stock = stock_features(close, volume, residuals, betas)
  if rank_normalise:
    per_stock = cross_sectional_rank(per_stock, list(per_stock.columns))

  regime = market_features(aux_close, market_ret, close)
  if calendar is not None and not calendar.empty:
    regime = regime.join(calendar, how="left")

  # Broadcast date-level features across tickers.
  merged = per_stock.join(regime, on="date")
  merged = merged.replace([np.inf, -np.inf], np.nan)
  return merged.sort_index()


def feature_columns(df: pd.DataFrame) -> list[str]:
  return [c for c in df.columns if not c.startswith("_")]
