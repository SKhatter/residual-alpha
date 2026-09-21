"""Purged, embargoed walk-forward splitting.

The failure this prevents
-------------------------
A sample stamped at date t carries a label that resolves at t + execution_lag
+ horizon. If t sits just before the test window, its label is drawn from
inside the test window, and the model has been trained on the answer. On daily
data with a one-day horizon the leak is small; on a weekly horizon it is
enormous, and it is invisible in the metrics -- it just makes everything look
good.

Purge removes training dates whose labels reach into the test block. Embargo
removes an additional buffer, because a sample immediately before the test
block is strongly serially correlated with the start of the test block even
when its label does not literally overlap.

Because this is walk-forward -- training data always precedes test data -- both
purge and embargo apply at the boundary before the test block. Nothing after a
test block ever enters that fold's training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
  """One train/test division along the time axis."""

  index: int
  train_dates: pd.DatetimeIndex
  test_dates: pd.DatetimeIndex
  purge_days: int
  embargo_days: int

  @property
  def train_span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
    return self.train_dates[0], self.train_dates[-1]

  @property
  def test_span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
    return self.test_dates[0], self.test_dates[-1]

  def describe(self) -> str:
    tr0, tr1 = self.train_span
    te0, te1 = self.test_span
    return (
      f"fold {self.index}: train {tr0.date()}..{tr1.date()} "
      f"({len(self.train_dates)}d) -> test {te0.date()}..{te1.date()} "
      f"({len(self.test_dates)}d), gap {self.purge_days}+{self.embargo_days}d"
    )


def walk_forward(
  dates: pd.DatetimeIndex,
  train_years: float,
  test_months: int,
  purge_days: int,
  embargo_days: int = 0,
  min_train_days: int = 250,
) -> list[Fold]:
  """Build rolling folds over a trading calendar.

  Args:
    dates: sorted trading dates covering the research period.
    train_years: rolling training window length in calendar years.
    test_months: length of each out-of-sample block in calendar months.
    purge_days: trading days a label reaches past its stamp date. Must be
      >= targets.label_end_offset(horizon, execution_lag).
    embargo_days: extra trading-day buffer before each test block.
    min_train_days: folds with less training data than this are skipped.

  Returns:
    Folds in chronological order.
  """
  dates = pd.DatetimeIndex(dates).sort_values()
  if purge_days < 0 or embargo_days < 0:
    raise ValueError("purge_days and embargo_days must be non-negative")
  if len(dates) == 0:
    return []

  gap = purge_days + embargo_days
  train_delta = pd.DateOffset(days=int(round(train_years * 365.25)))
  test_delta = pd.DateOffset(months=test_months)

  folds: list[Fold] = []
  test_start = dates[0] + train_delta

  while test_start < dates[-1]:
    test_end = test_start + test_delta
    test_mask = (dates >= test_start) & (dates < test_end)
    test_dates = dates[test_mask]
    if len(test_dates) == 0:
      test_start = test_end
      continue

    # Positional cut: drop the last `gap` trading days before the test block.
    first_test_pos = int(dates.searchsorted(test_dates[0]))
    train_stop_pos = max(first_test_pos - gap, 0)

    train_floor = test_start - train_delta
    candidate = dates[:train_stop_pos]
    train_dates = candidate[candidate >= train_floor]

    if len(train_dates) >= min_train_days:
      folds.append(
        Fold(
          index=len(folds),
          train_dates=train_dates,
          test_dates=test_dates,
          purge_days=purge_days,
          embargo_days=embargo_days,
        )
      )
    test_start = test_end

  return folds


def index_for_dates(
  panel_index: pd.MultiIndex, dates: pd.DatetimeIndex
) -> pd.MultiIndex:
  """Restrict a (date, ticker) index to a set of dates."""
  level = panel_index.get_level_values("date")
  return panel_index[level.isin(dates)]


def assert_no_leakage(fold: Fold, label_end_offset: int) -> None:
  """Fail loudly if any training label could reach the test window.

  Called by the backtest engine on every fold. This check is cheap and the bug
  it catches is both catastrophic and silent, so it is an assertion rather
  than a warning.
  """
  if len(fold.train_dates) == 0:
    return
  last_train = fold.train_dates[-1]
  first_test = fold.test_dates[0]
  # Business-day approximation is intentionally conservative: it can only
  # over-estimate how far a label reaches.
  reach = last_train + pd.tseries.offsets.BDay(label_end_offset)
  if reach >= first_test:
    raise AssertionError(
      f"fold {fold.index}: training label from {last_train.date()} reaches "
      f"{reach.date()}, at or past test start {first_test.date()}. "
      f"Increase purge_days (currently {fold.purge_days})."
    )
