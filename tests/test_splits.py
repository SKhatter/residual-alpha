"""Walk-forward splitting: purge, embargo, and strict chronology."""

from __future__ import annotations

import pandas as pd
import pytest

from ralpha.backtest.splits import (
  Fold,
  assert_no_leakage,
  index_for_dates,
  walk_forward,
)

DATES = pd.bdate_range("2015-01-01", "2024-12-31")


def test_train_always_precedes_test():
  folds = walk_forward(DATES, train_years=3, test_months=6, purge_days=2)
  assert folds
  for f in folds:
    assert f.train_dates[-1] < f.test_dates[0]


def test_purge_removes_exactly_the_overlapping_days():
  purge = 5
  folds = walk_forward(DATES, train_years=3, test_months=6, purge_days=purge,
                       embargo_days=0)
  for f in folds:
    first_test_pos = DATES.searchsorted(f.test_dates[0])
    last_train_pos = DATES.searchsorted(f.train_dates[-1])
    assert first_test_pos - last_train_pos > purge


def test_embargo_widens_the_gap():
  narrow = walk_forward(DATES, 3, 6, purge_days=2, embargo_days=0)
  wide = walk_forward(DATES, 3, 6, purge_days=2, embargo_days=20)

  for a, b in zip(narrow, wide):
    assert a.test_dates[0] == b.test_dates[0]
    gap_a = DATES.searchsorted(a.test_dates[0]) - DATES.searchsorted(a.train_dates[-1])
    gap_b = DATES.searchsorted(b.test_dates[0]) - DATES.searchsorted(b.train_dates[-1])
    assert gap_b == gap_a + 20


def test_test_blocks_tile_without_overlap():
  folds = walk_forward(DATES, 3, 6, purge_days=2)
  seen = pd.DatetimeIndex([])
  for f in folds:
    assert len(seen.intersection(f.test_dates)) == 0
    seen = seen.append(f.test_dates)
    assert f.test_dates.is_monotonic_increasing


def test_rolling_window_length_is_bounded():
  """Training windows roll rather than expand, so they stay comparable."""
  folds = walk_forward(DATES, train_years=3, test_months=6, purge_days=2)
  lengths = [len(f.train_dates) for f in folds]
  assert max(lengths) - min(lengths) < 60  # a couple of months of calendar drift


def test_leakage_assertion_fires_when_purge_too_small():
  """A hand-built fold with an inadequate gap must be rejected."""
  train = DATES[:100]
  test = DATES[100:150]
  bad = Fold(0, train, test, purge_days=0, embargo_days=0)
  with pytest.raises(AssertionError, match="reaches"):
    assert_no_leakage(bad, label_end_offset=5)


def test_leakage_assertion_passes_with_adequate_purge():
  train = DATES[:100]
  test = DATES[110:160]
  ok = Fold(0, train, test, purge_days=5, embargo_days=5)
  assert_no_leakage(ok, label_end_offset=5)  # must not raise


def test_short_history_yields_no_folds():
  short = pd.bdate_range("2023-01-01", "2023-06-30")
  assert walk_forward(short, train_years=3, test_months=6, purge_days=2) == []


def test_index_for_dates_filters_by_date_level():
  idx = pd.MultiIndex.from_product(
    [DATES[:5], ["AAA", "BBB"]], names=["date", "ticker"]
  )
  subset = index_for_dates(idx, DATES[:2])
  assert len(subset) == 4
  assert set(subset.get_level_values("date")) == set(DATES[:2])
