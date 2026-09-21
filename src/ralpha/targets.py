"""Target construction: beta-residual returns.

Why residual and not raw returns
--------------------------------
A single stock's return is mostly beta times the market's return. Forecasting
the raw return therefore means forecasting the index, which is the least
predictable part of the problem, and it lets a model score well by learning
market direction rather than anything stock-specific. We strip out market and
sector exposure with rolling betas and forecast what is left.

Lookahead discipline
--------------------
Betas at date t are estimated on a window ending at t-1. The residual at t then
uses those stale betas together with the *contemporaneous* factor returns at t.
That is correct: the residual is a target, not a feature. At decision time we
do not know r_mkt[t]; we predict the residual and hedge the factor exposure in
portfolio construction.

Why the intercept is estimated but not subtracted
-------------------------------------------------
The rolling regression fits `r = alpha + b1*mkt + b2*sector`. Including the
intercept is necessary -- omitting it biases the betas whenever a stock has
drifted over the window. But *subtracting* it makes the target `r - alpha - bf`,
and that is not a tradeable quantity. A market/sector-hedged book earns
`r - bf`; there is no instrument that shorts a stock's own trailing drift.

This is not a cosmetic distinction. Measured on the same weights, the
alpha-subtracted target scored Sharpe 2.33 while the simulated book earned
0.37 -- the entire 6x gap was the drift term. So `subtract_alpha` defaults to
False: estimate the intercept, keep the betas honest, forecast what a book can
actually collect. Setting it True reproduces the old (non-tradeable) target and
is kept only so the discrepancy can be re-derived.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_COLS = ["mkt", "sector"]


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
  """Simple (not log) returns, so portfolio arithmetic stays exact."""
  return prices.sort_index().pct_change()


def _rolling_ols_2f(
  y: pd.Series,
  f1: pd.Series,
  f2: pd.Series,
  window: int,
  min_periods: int,
) -> pd.DataFrame:
  """Rolling OLS of y on [f1, f2] with intercept, using strictly past data.

  Solved from rolling second moments rather than a Python loop over windows.
  Returns alpha/beta1/beta2 already shifted so row t holds coefficients fit on
  data through t-1.
  """
  df = pd.concat({"y": y, "f1": f1, "f2": f2}, axis=1).astype(float)
  roll = df.rolling(window, min_periods=min_periods)

  m_y, m_1, m_2 = roll["y"].mean(), roll["f1"].mean(), roll["f2"].mean()
  # Central second moments.
  c11 = roll["f1"].var(ddof=1)
  c22 = roll["f2"].var(ddof=1)
  c12 = df["f1"].rolling(window, min_periods=min_periods).cov(df["f2"])
  c1y = df["f1"].rolling(window, min_periods=min_periods).cov(df["y"])
  c2y = df["f2"].rolling(window, min_periods=min_periods).cov(df["y"])

  det = c11 * c22 - c12 * c12
  # Where the factors are collinear or the window is degenerate, fall back to a
  # single-factor (market only) fit instead of emitting exploding betas.
  degenerate = det.abs() < 1e-16
  det_safe = det.where(~degenerate, np.nan)

  b1 = (c22 * c1y - c12 * c2y) / det_safe
  b2 = (c11 * c2y - c12 * c1y) / det_safe

  fallback_b1 = c1y / c11.replace(0.0, np.nan)
  b1 = b1.where(~degenerate, fallback_b1)
  b2 = b2.where(~degenerate, 0.0)

  alpha = m_y - b1 * m_1 - b2 * m_2

  out = pd.DataFrame({"alpha": alpha, "beta_mkt": b1, "beta_sector": b2})
  # Shift so that row t only reflects information through t-1.
  return out.shift(1)


def residual_returns(
  stock_returns: pd.DataFrame,
  market_returns: pd.Series,
  sector_returns: pd.DataFrame,
  sector_map: dict[str, str],
  window: int = 252,
  min_periods: int = 120,
  winsorize_sigma: float | None = 6.0,
  subtract_alpha: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Compute residual returns and the betas used to produce them.

  Args:
    stock_returns: date x ticker simple returns.
    market_returns: date-indexed market factor returns.
    sector_returns: date x sector-ETF simple returns.
    sector_map: ticker -> sector ETF symbol.
    window: rolling beta window in trading days.
    min_periods: minimum observations before a beta is emitted.
    winsorize_sigma: clip residuals at +/- N rolling sigma. None disables.
    subtract_alpha: also remove the trailing intercept. Off by default -- see
      the module docstring; the resulting target is not tradeable.

  Returns:
    (residuals, betas) where residuals is date x ticker and betas is a long
    frame indexed by (date, ticker) with alpha/beta_mkt/beta_sector.
  """
  residuals: dict[str, pd.Series] = {}
  beta_rows: dict[str, pd.DataFrame] = {}

  for ticker in stock_returns.columns:
    sector_etf = sector_map.get(ticker)
    y = stock_returns[ticker]
    f1 = market_returns.reindex(y.index)
    if sector_etf is not None and sector_etf in sector_returns.columns:
      f2 = sector_returns[sector_etf].reindex(y.index)
    else:
      # No sector mapping -> market-only model. Keeps the ticker in the
      # universe rather than silently dropping it.
      f2 = pd.Series(0.0, index=y.index)

    coefs = _rolling_ols_2f(y, f1, f2, window, min_periods)
    # The intercept is always estimated -- dropping it from the fit would bias
    # the betas -- but only removed from the target when explicitly asked for.
    fitted = coefs["beta_mkt"] * f1 + coefs["beta_sector"] * f2
    if subtract_alpha:
      fitted = fitted + coefs["alpha"]
    resid = y - fitted

    if winsorize_sigma:
      # Rolling sigma is causal, so the clip level at t uses only past residuals.
      sigma = resid.rolling(window, min_periods=min_periods).std().shift(1)
      limit = winsorize_sigma * sigma
      resid = resid.clip(lower=-limit, upper=limit)

    residuals[ticker] = resid
    beta_rows[ticker] = coefs

  resid_df = pd.DataFrame(residuals).sort_index()
  betas = (
    pd.concat(beta_rows, names=["ticker", "date"])
    .reorder_levels(["date", "ticker"])
    .sort_index()
  )
  return resid_df, betas


def forward_target(
  residuals: pd.DataFrame,
  horizon: int = 1,
  execution_lag: int = 1,
) -> pd.DataFrame:
  """Cumulative residual return actually earned by a decision made at t.

  Timeline for execution_lag=1, horizon=1:
      close t     -> features observed, signal formed
      close t+1   -> position established
      close t+2   -> position closed; this is the return we book

  So target[t] = residual return over (t+1, t+2].
  """
  if horizon < 1:
    raise ValueError("horizon must be >= 1")
  if execution_lag < 0:
    raise ValueError("execution_lag must be >= 0")

  # Cumulative forward return over `horizon` days, stamped at the window start.
  fwd = (1.0 + residuals).rolling(horizon).apply(np.prod, raw=True) - 1.0
  fwd = fwd.shift(-(horizon - 1))  # stamp at first day of the window
  # Shift again so the window begins after execution.
  return fwd.shift(-(execution_lag + 1))


def label_end_offset(horizon: int, execution_lag: int) -> int:
  """Trading days between a decision at t and the last date its label uses.

  The purge width in walk-forward splitting must be at least this, or training
  labels will overlap the test window.
  """
  return execution_lag + horizon
