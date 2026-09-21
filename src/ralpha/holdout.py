"""The touch-once holdout.

Why this is code and not a note in the README
----------------------------------------------
Every look at a held-out result is a degree of freedom. Look ten times, tweak
something between each look, and the final number is an in-sample number
wearing a disguise -- but nothing in the output distinguishes it from a real
one. Discipline that depends on remembering is not discipline.

So the holdout window is enforced mechanically: the development backtest
refuses to read dates inside it, and every evaluation on it is appended to a
ledger with the config hash that produced it. The ledger cannot be silently
overwritten, and a second look prints the first one alongside it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def config_fingerprint(cfg_dict: dict) -> str:
  """Stable hash of a config, so re-runs are distinguishable from re-tunes."""
  blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode()
  return hashlib.sha256(blob).hexdigest()[:16]


def split_development(
  dates: pd.DatetimeIndex, holdout_start: str
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
  """Split a calendar into (development, holdout)."""
  cut = pd.Timestamp(holdout_start)
  dates = pd.DatetimeIndex(dates).sort_values()
  return dates[dates < cut], dates[dates >= cut]


class HoldoutLedger:
  """Append-only record of holdout evaluations."""

  def __init__(self, path: Path):
    self.path = Path(path)
    self.entries: list[dict] = []
    if self.path.exists():
      with open(self.path) as fh:
        self.entries = json.load(fh).get("entries", [])

  @property
  def n_looks(self) -> int:
    return len(self.entries)

  def previous_for(self, fingerprint: str) -> list[dict]:
    return [e for e in self.entries if e.get("config_fingerprint") == fingerprint]

  def record(
    self, fingerprint: str, metrics: dict, note: str = ""
  ) -> dict:
    entry = {
      "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
      "look_number": self.n_looks + 1,
      "config_fingerprint": fingerprint,
      "note": note,
      "metrics": {
        k: (float(v) if isinstance(v, (int, float)) else v)
        for k, v in metrics.items()
      },
    }
    self.entries.append(entry)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with open(self.path, "w") as fh:
      json.dump({"entries": self.entries}, fh, indent=2)
    return entry

  def warning_banner(self, fingerprint: str) -> str | None:
    """Text to print before a holdout run, if this is not the first look."""
    if self.n_looks == 0:
      return None

    same = self.previous_for(fingerprint)
    lines = [
      "",
      "=" * 72,
      f"  HOLDOUT WARNING: this is look #{self.n_looks + 1} at the holdout.",
      "=" * 72,
      "  Each additional look is a degree of freedom. The reported numbers",
      "  are no longer a clean out-of-sample estimate, and no correction",
      "  applied afterwards will fully restore one.",
      "",
    ]
    if same:
      lines.append(f"  This exact config has been evaluated {len(same)} time(s):")
      for e in same[-3:]:
        sharpe = e["metrics"].get("sharpe_net")
        shown = f"{sharpe:.3f}" if isinstance(sharpe, float) else "n/a"
        lines.append(f"    {e['timestamp']}  look #{e['look_number']}  sharpe={shown}")
      lines.append("  Identical config: this is a re-run, not a new experiment.")
    else:
      lines.append("  Config differs from all previous looks -- you have tuned")
      lines.append("  something against holdout feedback. Treat results as")
      lines.append("  in-sample.")
    lines.extend(["", "=" * 72, ""])
    return "\n".join(lines)


def assert_development_only(
  dates: pd.DatetimeIndex, holdout_start: str, context: str = "backtest"
) -> None:
  """Raise if any date falls inside the reserved window."""
  cut = pd.Timestamp(holdout_start)
  intruders = pd.DatetimeIndex(dates)[pd.DatetimeIndex(dates) >= cut]
  if len(intruders):
    raise RuntimeError(
      f"{context} touched {len(intruders)} holdout dates "
      f"(first {intruders[0].date()}, holdout starts {cut.date()}). "
      "Use `ralpha holdout` if you intend to spend a look."
    )
