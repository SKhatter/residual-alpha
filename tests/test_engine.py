"""Portfolio accounting: execution lag, turnover, costs, hedging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ralpha.backtest.engine import apply_turnover_cap, performance_metrics, simulate

DATES = pd.bdate_range("2021-01-01", periods=120)
TICKERS = ["AAA", "BBB", "CCC", "DDD"]
SECTOR_MAP = {"AAA": "XLK", "BBB": "XLK", "CCC": "XLF", "DDD": "XLF"}


def _inputs(seed=0, beta=1.0):
  rng = np.random.default_rng(seed)
  stock_returns = pd.DataFrame(
    rng.normal(0, 0.01, (len(DATES), len(TICKERS))), index=DATES, columns=TICKERS
  )
  etf_returns = pd.DataFrame(
    rng.normal(0, 0.008, (len(DATES), 3)), index=DATES, columns=["SPY", "XLK", "XLF"]
  )
  idx = pd.MultiIndex.from_product([DATES, TICKERS], names=["date", "ticker"])
  betas = pd.DataFrame(
    {"beta_mkt": beta, "beta_sector": 0.0}, index=idx
  )
  return stock_returns, etf_returns, betas


def test_turnover_cap_limits_daily_movement():
  target = pd.DataFrame(
    [[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]],
    index=DATES[:3],
    columns=["AAA", "BBB"],
  )
  capped = apply_turnover_cap(target, cap=0.5)
  moves = capped.diff().abs().sum(axis=1)
  assert capped.iloc[0].abs().sum() == pytest.approx(0.5)
  assert (moves.iloc[1:] <= 0.5 + 1e-9).all()


def test_turnover_cap_is_a_noop_when_generous():
  target = pd.DataFrame(
    [[0.1, -0.1], [0.1, -0.1]], index=DATES[:2], columns=["AAA", "BBB"]
  )
  capped = apply_turnover_cap(target, cap=100.0)
  pd.testing.assert_frame_equal(capped, target, check_dtype=False)


def test_execution_lag_shifts_the_earning_position():
  """A weight decided at t must not earn the return at t."""
  stock_returns, etf_returns, betas = _inputs()
  weights = pd.DataFrame(0.0, index=DATES, columns=TICKERS)
  weights.loc[DATES[10], "AAA"] = 1.0

  result = simulate(
    weights, stock_returns, betas, etf_returns, SECTOR_MAP,
    execution_lag=1, cost_bps=0, spread_bps=0, borrow_bps_annual=0,
    max_daily_turnover=None, hedge=False,
  )
  # Decided at index 10, executed at 11, earns the return at 12.
  assert result.gross_returns.iloc[10] == pytest.approx(0.0, abs=1e-12)
  assert result.gross_returns.iloc[11] == pytest.approx(0.0, abs=1e-12)
  assert result.gross_returns.iloc[12] == pytest.approx(
    stock_returns["AAA"].iloc[12]
  )


def test_costs_reduce_returns_and_scale_with_turnover():
  stock_returns, etf_returns, betas = _inputs()
  rng = np.random.default_rng(1)
  weights = pd.DataFrame(
    rng.normal(0, 0.02, (len(DATES), len(TICKERS))), index=DATES, columns=TICKERS
  )

  free = simulate(
    weights, stock_returns, betas, etf_returns, SECTOR_MAP,
    cost_bps=0, spread_bps=0, borrow_bps_annual=0, max_daily_turnover=None,
  )
  costly = simulate(
    weights, stock_returns, betas, etf_returns, SECTOR_MAP,
    cost_bps=10, spread_bps=5, borrow_bps_annual=0, max_daily_turnover=None,
  )

  assert (costly.costs >= 0).all()
  assert costly.returns.sum() < free.returns.sum()
  expected = free.turnover * 15 / 10_000.0
  pd.testing.assert_series_equal(costly.costs, expected, check_names=False)


def test_borrow_is_charged_on_the_short_book_only():
  stock_returns, etf_returns, betas = _inputs()
  long_only = pd.DataFrame(0.25, index=DATES, columns=TICKERS)

  result = simulate(
    long_only, stock_returns, betas, etf_returns, SECTOR_MAP,
    cost_bps=0, spread_bps=0, borrow_bps_annual=100, max_daily_turnover=None,
    hedge=False,
  )
  assert result.costs.abs().max() == pytest.approx(0.0, abs=1e-15)

  short_only = -long_only
  shorted = simulate(
    short_only, stock_returns, betas, etf_returns, SECTOR_MAP,
    cost_bps=0, spread_bps=0, borrow_bps_annual=100, max_daily_turnover=None,
    hedge=False,
  )
  assert shorted.costs.iloc[20] > 0


def test_hedge_removes_market_exposure():
  """With beta 1 and an all-long book, the hedge should neutralise the market."""
  rng = np.random.default_rng(2)
  market = pd.Series(rng.normal(0, 0.02, len(DATES)), index=DATES)
  # Each stock is pure market with no idiosyncratic component.
  stock_returns = pd.DataFrame(
    {t: market for t in TICKERS}, index=DATES
  )
  etf_returns = pd.DataFrame(
    {"SPY": market, "XLK": market * 0, "XLF": market * 0}, index=DATES
  )
  idx = pd.MultiIndex.from_product([DATES, TICKERS], names=["date", "ticker"])
  betas = pd.DataFrame({"beta_mkt": 1.0, "beta_sector": 0.0}, index=idx)

  weights = pd.DataFrame(0.25, index=DATES, columns=TICKERS)

  hedged = simulate(
    weights, stock_returns, betas, etf_returns, SECTOR_MAP,
    cost_bps=0, spread_bps=0, borrow_bps_annual=0, max_daily_turnover=None,
    hedge=True,
  )
  unhedged = simulate(
    weights, stock_returns, betas, etf_returns, SECTOR_MAP,
    cost_bps=0, spread_bps=0, borrow_bps_annual=0, max_daily_turnover=None,
    hedge=False,
  )
  assert hedged.gross_returns.abs().max() < 1e-12
  assert unhedged.gross_returns.abs().max() > 1e-3


def test_metrics_are_internally_consistent():
  stock_returns, etf_returns, betas = _inputs(seed=3)
  rng = np.random.default_rng(4)
  weights = pd.DataFrame(
    rng.normal(0, 0.01, (len(DATES), len(TICKERS))), index=DATES, columns=TICKERS
  )
  result = simulate(
    weights, stock_returns, betas, etf_returns, SECTOR_MAP,
    max_daily_turnover=None,
  )
  m = performance_metrics(result)
  assert m["max_drawdown"] <= 0
  assert m["ann_vol"] > 0
  assert 0 <= m["hit_rate"] <= 1
  assert m["sharpe_se"] > 0
  # Net must never exceed gross once costs are positive.
  assert result.returns.sum() <= result.gross_returns.sum() + 1e-12
