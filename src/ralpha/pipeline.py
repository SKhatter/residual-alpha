"""Assembles raw prices into the Panel every model consumes."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import features as feat
from . import targets as tgt
from .config import Config
from .data.loader import load_panel, to_wide, trading_calendar
from .models.base import Panel

log = logging.getLogger(__name__)


def _sector_returns_per_ticker(
  etf_returns: pd.DataFrame, tickers: list[str], sector_map: dict[str, str]
) -> pd.DataFrame:
  """Broadcast each sector ETF's return onto the tickers that load on it."""
  cols = {}
  for t in tickers:
    etf = sector_map.get(t)
    if etf in etf_returns.columns:
      cols[t] = etf_returns[etf]
    else:
      cols[t] = pd.Series(0.0, index=etf_returns.index)
  return pd.DataFrame(cols)


def _peer_returns(
  stock_returns: pd.DataFrame, sector_map: dict[str, str]
) -> pd.DataFrame:
  """Equal-weight return of same-sector peers, excluding the name itself.

  Self-exclusion matters: including the name would leak its own contemporaneous
  return into its own covariate channel.
  """
  cols = {}
  for t in stock_returns.columns:
    etf = sector_map.get(t)
    peers = [
      o for o in stock_returns.columns if o != t and sector_map.get(o) == etf
    ]
    cols[t] = (
      stock_returns[peers].mean(axis=1)
      if peers
      else pd.Series(0.0, index=stock_returns.index)
    )
  return pd.DataFrame(cols)


def build_panel(cfg: Config, refresh: bool = False) -> Panel:
  """Download, align, and derive everything downstream code needs."""
  data = cfg.data
  universe = list(data.universe)
  market = data.market
  sector_map = dict(data.sector_map.as_dict())
  sector_etfs = sorted(set(sector_map.values()))
  aux = list(data.auxiliary)
  cache = cfg.path(data.cache_dir)

  all_tickers = sorted(set(universe + [market] + sector_etfs + aux))
  log.info("loading %d tickers from %s to %s", len(all_tickers), data.start, data.end)
  raw = load_panel(all_tickers, data.start, data.end, cache, refresh=refresh)

  close_all = to_wide(raw, "close")
  volume_all = to_wide(raw, "volume")

  present = [t for t in universe if t in close_all.columns]
  if len(present) < len(universe):
    log.warning("universe reduced to %d/%d names", len(present), len(universe))
  if market not in close_all.columns:
    raise RuntimeError(f"market factor {market} failed to load; cannot continue")

  calendar_dates = trading_calendar(close_all[present])
  close = close_all.loc[calendar_dates, present]
  volume = volume_all.loc[calendar_dates, present]
  etf_close = close_all.loc[calendar_dates, [c for c in ([market] + sector_etfs) if c in close_all.columns]]
  aux_close = close_all.loc[calendar_dates, [c for c in aux if c in close_all.columns]]

  stock_returns = tgt.simple_returns(close)
  etf_returns = tgt.simple_returns(etf_close)
  market_ret = etf_returns[market]

  log.info("computing rolling-beta residuals")
  residuals, betas = tgt.residual_returns(
    stock_returns=stock_returns,
    market_returns=market_ret,
    sector_returns=etf_returns,
    sector_map=sector_map,
    window=int(cfg.target.beta_window),
    min_periods=int(cfg.target.beta_min_periods),
    winsorize_sigma=cfg.get("target.winsorize_sigma"),
    subtract_alpha=bool(cfg.get("target.subtract_alpha", False)),
  )

  cal_cfg = cfg.features.calendar
  calendar = (
    feat.calendar_features(
      dates=close.index,
      fomc_path=cfg.path(cal_cfg.get("fomc_dates", "")) if cal_cfg.get("fomc_dates") else None,
      earnings_path=(
        cfg.path(cal_cfg.get("earnings_dates", ""))
        if cal_cfg.get("earnings_dates")
        else None
      ),
      day_of_week=bool(cal_cfg.get("day_of_week", True)),
      turn_of_month=bool(cal_cfg.get("turn_of_month", True)),
    )
    if cal_cfg.get("enabled", True)
    else pd.DataFrame(index=close.index)
  )

  log.info("building features")
  design = feat.build_features(
    close=close,
    volume=volume,
    residuals=residuals,
    betas=betas,
    aux_close=aux_close,
    market_ret=market_ret,
    calendar=calendar,
  )

  target = tgt.forward_target(
    residuals,
    horizon=int(cfg.target.horizon_days),
    execution_lag=int(cfg.backtest.execution_lag),
  ).stack(dropna=False)
  target.index.names = ["date", "ticker"]
  target = target.reindex(design.index)

  # Sequence covariates for TimesFM-3, in the order of the original diagram.
  rv20 = stock_returns.rolling(20).std()
  log_vol = np.log1p(volume)
  volume_z = (log_vol - log_vol.rolling(20).mean()) / (
    log_vol.rolling(60).std() + 1e-12
  )
  breadth = (close > close.rolling(50).mean()).mean(axis=1)

  def shared(series: pd.Series, name: str) -> pd.DataFrame:
    return pd.DataFrame({name: series.reindex(close.index).ffill()})

  covariates = {
    "mkt_ret": shared(market_ret, "mkt_ret"),
    "sector_ret": _sector_returns_per_ticker(etf_returns, present, sector_map),
    "peer_ret": _peer_returns(stock_returns, sector_map),
    "volume_z": volume_z,
    "realized_vol": rv20,
  }
  if "^VIX" in aux_close.columns:
    covariates["vix"] = shared(np.log(aux_close["^VIX"]), "vix")
  if "^TNX" in aux_close.columns:
    covariates["rate_10y"] = shared(aux_close["^TNX"], "rate_10y")
  covariates["breadth"] = shared(breadth, "breadth")

  # TimesFM skips any ticker whose context window contains a NaN, so fill the
  # interior and let the leading burn-in be trimmed by the beta window anyway.
  covariates = {k: v.ffill().fillna(0.0) for k, v in covariates.items()}

  panel = Panel(
    features=design,
    target=target,
    residuals=residuals.fillna(0.0),
    covariates=covariates,
    calendar=calendar.fillna(0.0),
  )
  panel.meta = {  # type: ignore[attr-defined]
    "close": close,
    "volume": volume,
    "stock_returns": stock_returns,
    "etf_returns": etf_returns,
    "market_ret": market_ret,
    "betas": betas,
    "sector_map": sector_map,
    "market": market,
    "universe": present,
  }
  log.info(
    "panel ready: %d dates x %d tickers, %d features",
    len(close.index),
    len(present),
    design.shape[1],
  )
  return panel


def usable_index(panel: Panel, min_features: float = 0.8) -> pd.MultiIndex:
  """Rows with a resolved label and enough non-missing features.

  Dropping sparse rows here rather than inside each model keeps every model
  scored on exactly the same population.
  """
  has_target = panel.target.notna()
  completeness = panel.features.notna().mean(axis=1)
  keep = has_target & (completeness >= min_features)
  return panel.features.index[keep.to_numpy()]
