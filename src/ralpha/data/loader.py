"""Price data loading with an on-disk cache.

yfinance is the default source because it is free and adequate for deciding
whether a research direction is worth pursuing. It is NOT adequate for a
production backtest: it is survivorship-biased (delisted tickers are gone),
split/dividend adjustments are restated over time, and there is no
point-in-time universe. Those biases flatter a backtest. Read docs/PROTOCOL.md
before believing any number that comes out of this.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Fields we keep from the vendor.
_FIELDS = ["open", "high", "low", "close", "volume"]


def _cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
  safe = ticker.replace("^", "_idx_").replace("/", "_")
  return cache_dir / f"{safe}__{start}__{end}.parquet"


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
  import yfinance as yf

  raw = yf.download(
    ticker,
    start=start,
    end=end,
    auto_adjust=True,  # split & dividend adjusted
    progress=False,
    threads=False,
  )
  if raw is None or raw.empty:
    raise ValueError(f"no data returned for {ticker}")

  # yfinance returns a MultiIndex column frame when given a list, and a flat
  # one for a single ticker. Normalise to flat lowercase names.
  if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
  raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
  raw = raw[[c for c in _FIELDS if c in raw.columns]].copy()
  raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
  raw.index.name = "date"
  return raw[~raw.index.duplicated(keep="last")].sort_index()


def load_ticker(
  ticker: str, start: str, end: str, cache_dir: Path, refresh: bool = False
) -> pd.DataFrame:
  """Load one ticker's OHLCV, using the parquet cache when possible."""
  cache_dir = Path(cache_dir)
  cache_dir.mkdir(parents=True, exist_ok=True)
  path = _cache_path(cache_dir, ticker, start, end)

  if path.exists() and not refresh:
    return pd.read_parquet(path)

  df = _download(ticker, start, end)
  df.to_parquet(path)
  log.info("downloaded %s: %d rows", ticker, len(df))
  return df


def load_panel(
  tickers: list[str],
  start: str,
  end: str,
  cache_dir: Path,
  refresh: bool = False,
) -> dict[str, pd.DataFrame]:
  """Load many tickers. Failures are logged and dropped, not raised.

  A missing ticker is a data problem, not a code problem; the caller gets a
  smaller universe and a warning rather than a dead run.
  """
  out: dict[str, pd.DataFrame] = {}
  failed: list[str] = []
  for t in tickers:
    try:
      out[t] = load_ticker(t, start, end, cache_dir, refresh=refresh)
    except Exception as exc:  # noqa: BLE001 - vendor errors are varied
      log.warning("failed to load %s: %s", t, exc)
      failed.append(t)
  if failed:
    log.warning("dropped %d/%d tickers: %s", len(failed), len(tickers), failed)
  if not out:
    raise RuntimeError("no tickers loaded; check network or cache")
  return out


def to_wide(panel: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
  """Stack a {ticker: ohlcv} dict into a wide date x ticker frame."""
  cols = {t: df[field] for t, df in panel.items() if field in df.columns}
  wide = pd.DataFrame(cols).sort_index()
  wide.columns.name = "ticker"
  return wide


def trading_calendar(wide: pd.DataFrame, min_coverage: float = 0.6) -> pd.DatetimeIndex:
  """Dates where enough of the universe traded.

  Guards against holidays and half-populated rows introduced by aligning
  tickers with different listing histories.
  """
  coverage = wide.notna().mean(axis=1)
  return wide.index[coverage >= min_coverage]
