"""Configuration loading.

Config is a plain nested dict behind dot-access. Deliberately not a typed
schema: research configs change shape constantly and a rigid dataclass tree
turns every experiment into a refactor.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


class Config:
  """Dot-accessible view over a nested dict."""

  def __init__(self, data: dict[str, Any], root: Path | None = None):
    self._data = data
    self.root = root or REPO_ROOT

  def __getattr__(self, name: str) -> Any:
    try:
      value = self._data[name]
    except KeyError as exc:
      raise AttributeError(
        f"no config key {name!r} (available: {sorted(self._data)})"
      ) from exc
    if isinstance(value, dict):
      return Config(value, self.root)
    return value

  def __getitem__(self, name: str) -> Any:
    return getattr(self, name)

  def __contains__(self, name: str) -> bool:
    return name in self._data

  def get(self, path: str, default: Any = None) -> Any:
    """Fetch a dotted path, e.g. cfg.get('models.gbm.max_iter', 100)."""
    node: Any = self._data
    for part in path.split("."):
      if not isinstance(node, dict) or part not in node:
        return default
      node = node[part]
    return node

  def as_dict(self) -> dict[str, Any]:
    return copy.deepcopy(self._data)

  def path(self, relative: str) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(relative)
    return p if p.is_absolute() else self.root / p

  def __repr__(self) -> str:
    return f"Config({sorted(self._data)})"


def load_config(path: str | Path | None = None) -> Config:
  cfg_path = Path(path) if path else DEFAULT_CONFIG
  if not cfg_path.exists():
    raise FileNotFoundError(f"config not found: {cfg_path}")
  with open(cfg_path) as fh:
    data = yaml.safe_load(fh)
  return Config(data, REPO_ROOT)
