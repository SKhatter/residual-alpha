"""The holdout lock: it has to actually stop you."""

from __future__ import annotations

import pandas as pd
import pytest

from ralpha.holdout import (
  HoldoutLedger,
  assert_development_only,
  config_fingerprint,
  split_development,
)

DATES = pd.bdate_range("2020-01-01", "2024-12-31")


def test_split_is_exhaustive_and_disjoint():
  dev, hold = split_development(DATES, "2023-01-01")
  assert len(dev) + len(hold) == len(DATES)
  assert dev.max() < pd.Timestamp("2023-01-01") <= hold.min()


def test_development_guard_rejects_holdout_dates():
  with pytest.raises(RuntimeError, match="holdout dates"):
    assert_development_only(DATES, "2023-01-01", context="backtest")


def test_development_guard_passes_on_clean_dates():
  dev, _ = split_development(DATES, "2023-01-01")
  assert_development_only(dev, "2023-01-01")  # must not raise


def test_fingerprint_is_stable_and_order_independent():
  a = {"x": 1, "y": {"z": 2}}
  b = {"y": {"z": 2}, "x": 1}
  assert config_fingerprint(a) == config_fingerprint(b)
  assert config_fingerprint(a) != config_fingerprint({"x": 2, "y": {"z": 2}})


def test_first_look_has_no_banner(tmp_path):
  ledger = HoldoutLedger(tmp_path / "ledger.json")
  assert ledger.warning_banner("abc") is None


def test_second_look_warns_and_names_the_count(tmp_path):
  path = tmp_path / "ledger.json"
  ledger = HoldoutLedger(path)
  ledger.record("abc", {"sharpe_net": 0.7}, note="first")

  reloaded = HoldoutLedger(path)
  banner = reloaded.warning_banner("abc")
  assert banner is not None
  assert "look #2" in banner
  assert "re-run, not a new experiment" in banner


def test_changed_config_is_flagged_as_tuned(tmp_path):
  path = tmp_path / "ledger.json"
  ledger = HoldoutLedger(path)
  ledger.record("abc", {"sharpe_net": 0.7})

  banner = HoldoutLedger(path).warning_banner("def")
  assert "tuned" in banner
  assert "in-sample" in banner


def test_ledger_is_append_only_across_reloads(tmp_path):
  path = tmp_path / "ledger.json"
  HoldoutLedger(path).record("abc", {"sharpe_net": 0.1})
  HoldoutLedger(path).record("abc", {"sharpe_net": 0.2})
  final = HoldoutLedger(path)
  assert final.n_looks == 2
  assert [e["look_number"] for e in final.entries] == [1, 2]
  assert final.entries[0]["metrics"]["sharpe_net"] == 0.1


def test_previous_for_filters_by_fingerprint(tmp_path):
  path = tmp_path / "ledger.json"
  led = HoldoutLedger(path)
  led.record("aaa", {"sharpe_net": 0.1})
  led.record("bbb", {"sharpe_net": 0.2})
  led.record("aaa", {"sharpe_net": 0.3})
  assert len(led.previous_for("aaa")) == 2
  assert len(led.previous_for("bbb")) == 1
