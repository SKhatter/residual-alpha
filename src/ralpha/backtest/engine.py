"""Portfolio simulation.

Design choices that change the answer
-------------------------------------
* PnL is computed on RAW stock returns with explicit hedge positions in the
  market and sector ETFs, not on residual returns directly. Booking residual
  PnL assumes a free, frictionless hedge. Here the hedge is a real position
  that pays real costs, which is the difference between a plausible Sharpe and
  a fictional one.
* Positions are rebalanced to target daily and intra-period drift is ignored.
  At daily frequency the error is second order; at monthly it would not be.
* Costs are charged on every leg including the hedge, and the short book pays
  borrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
  returns: pd.Series  # net daily portfolio return
  gross_returns: pd.Series
  costs: pd.Series
  turnover: pd.Series
  weights: pd.DataFrame
  hedges: pd.DataFrame
  metrics: dict[str, float] = field(default_factory=dict)

  @property
  def equity_curve(self) -> pd.Series:
    return (1.0 + self.returns.fillna(0.0)).cumprod()

  def summary(self) -> pd.Series:
    return pd.Series(self.metrics)


def apply_turnover_cap(target: pd.DataFrame, cap: float) -> pd.DataFrame:
  """Limit how far the book can move in one day.

  Sequential by necessity: today's achievable position depends on yesterday's
  actual position, not yesterday's target. Partial rebalancing toward the
  target is what a real execution desk does when the signal moves faster than
  liquidity allows.
  """
  if cap is None or cap <= 0 or not np.isfinite(cap):
    return target

  out = np.zeros_like(target.to_numpy(dtype=float))
  values = target.to_numpy(dtype=float)
  values = np.where(np.isfinite(values), values, 0.0)
  prev = np.zeros(values.shape[1])

  for i in range(values.shape[0]):
    desired = values[i]
    move = desired - prev
    required = np.abs(move).sum()
    if required > cap and required > 0:
      move = move * (cap / required)
    prev = prev + move
    out[i] = prev

  return pd.DataFrame(out, index=target.index, columns=target.columns)


def _hedge_positions(
  weights: pd.DataFrame,
  betas: pd.DataFrame,
  sector_map: dict[str, str],
  market: str,
) -> pd.DataFrame:
  """Offsetting ETF positions that neutralise factor exposure.

  Market hedge is the negative of the book's beta-weighted net exposure;
  sector hedges are the same per sector ETF. Anything the rolling betas get
  wrong shows up as residual factor PnL, which is the honest outcome.
  """
  beta_mkt = betas["beta_mkt"].unstack("ticker").reindex(
    index=weights.index, columns=weights.columns
  )
  beta_sec = betas["beta_sector"].unstack("ticker").reindex(
    index=weights.index, columns=weights.columns
  )

  w = weights.fillna(0.0)
  hedges = pd.DataFrame(0.0, index=weights.index, columns=[market])
  hedges[market] = -(w * beta_mkt.fillna(0.0)).sum(axis=1)

  for etf in sorted(set(sector_map.values())):
    members = [t for t in weights.columns if sector_map.get(t) == etf]
    if not members:
      continue
    exposure = (w[members] * beta_sec[members].fillna(0.0)).sum(axis=1)
    hedges[etf] = -exposure

  return hedges


def simulate(
  weights: pd.DataFrame,
  stock_returns: pd.DataFrame,
  betas: pd.DataFrame,
  etf_returns: pd.DataFrame,
  sector_map: dict[str, str],
  market: str = "SPY",
  execution_lag: int = 1,
  cost_bps: float = 5.0,
  spread_bps: float = 2.0,
  borrow_bps_annual: float = 50.0,
  max_daily_turnover: float | None = 0.25,
  hedge: bool = True,
) -> BacktestResult:
  """Run the book.

  Args:
    weights: date x ticker target weights, indexed by DECISION date.
    stock_returns: date x ticker raw simple returns.
    betas: long (date, ticker) frame with beta_mkt / beta_sector.
    etf_returns: date x symbol returns for the market and sector ETFs.
    sector_map: ticker -> sector ETF.
    execution_lag: trading days between decision and execution.
    cost_bps: commission + impact per unit of turnover.
    spread_bps: half-spread paid per unit of turnover.
    borrow_bps_annual: financing on short notional.
    max_daily_turnover: cap on daily gross position change.
    hedge: if False, run unhedged (useful to show how much of the PnL is beta).
  """
  weights = weights.sort_index().fillna(0.0)
  weights = apply_turnover_cap(weights, max_daily_turnover)

  # Position earning the return on date d was decided execution_lag+1 days ago.
  held = weights.shift(execution_lag + 1).fillna(0.0)
  held = held.reindex(index=stock_returns.index, columns=stock_returns.columns)
  held = held.fillna(0.0)

  rets = stock_returns.reindex_like(held).fillna(0.0)
  stock_pnl = (held * rets).sum(axis=1)

  if hedge:
    hedges = _hedge_positions(held, betas, sector_map, market)
    hedge_rets = etf_returns.reindex(
      index=hedges.index, columns=hedges.columns
    ).fillna(0.0)
    hedge_pnl = (hedges * hedge_rets).sum(axis=1)
  else:
    hedges = pd.DataFrame(0.0, index=held.index, columns=[market])
    hedge_pnl = pd.Series(0.0, index=held.index)

  gross = stock_pnl + hedge_pnl

  # Turnover on every leg we actually trade.
  stock_turnover = held.diff().abs().sum(axis=1)
  hedge_turnover = hedges.diff().abs().sum(axis=1)
  turnover = (stock_turnover + hedge_turnover).fillna(0.0)

  trade_cost = turnover * (cost_bps + spread_bps) / 10_000.0

  short_notional = held.clip(upper=0.0).abs().sum(axis=1) + hedges.clip(
    upper=0.0
  ).abs().sum(axis=1)
  borrow_cost = short_notional * (borrow_bps_annual / 10_000.0) / TRADING_DAYS

  costs = trade_cost + borrow_cost
  net = gross - costs

  result = BacktestResult(
    returns=net,
    gross_returns=gross,
    costs=costs,
    turnover=turnover,
    weights=held,
    hedges=hedges,
  )
  result.metrics = performance_metrics(result)
  return result


def performance_metrics(result: BacktestResult) -> dict[str, float]:
  r = result.returns.dropna()
  if len(r) < 2:
    return {"n_days": int(len(r))}

  ann_return = float((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0)
  ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
  sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")

  curve = (1.0 + r).cumprod()
  drawdown = curve / curve.cummax() - 1.0
  max_dd = float(drawdown.min())

  downside = r[r < 0]
  sortino = (
    float(ann_return / (downside.std(ddof=1) * np.sqrt(TRADING_DAYS)))
    if len(downside) > 1 and downside.std(ddof=1) > 0
    else float("nan")
  )

  gross = result.gross_returns.dropna()
  gross_ann = float((1.0 + gross).prod() ** (TRADING_DAYS / len(gross)) - 1.0)

  # Standard error of the Sharpe, so nobody reads 0.6 as meaningfully
  # different from 0.2 on three years of data.
  sharpe_se = float(np.sqrt((1.0 + 0.5 * sharpe**2) / len(r)) * np.sqrt(TRADING_DAYS))

  return {
    "n_days": int(len(r)),
    "ann_return_net": ann_return,
    "ann_return_gross": gross_ann,
    "ann_vol": ann_vol,
    "sharpe_net": sharpe,
    "sharpe_se": sharpe_se,
    "sharpe_t": float(sharpe / sharpe_se) if sharpe_se > 0 else float("nan"),
    "sortino": sortino,
    "max_drawdown": max_dd,
    "hit_rate": float((r > 0).mean()),
    "avg_daily_turnover": float(result.turnover.mean()),
    "ann_cost_drag": float(result.costs.mean() * TRADING_DAYS),
    "cost_share_of_gross": (
      float(result.costs.sum() / abs(gross.sum())) if gross.sum() != 0 else float("nan")
    ),
  }
