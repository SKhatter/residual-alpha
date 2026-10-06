#!/usr/bin/env python3
"""Event study: do insider purchases predict forward returns?

Why an event study and not a feature
------------------------------------
Insider purchases touch ~0.5% of rows in a 40-name daily panel, so adding them
as a model feature answers nothing -- there is no density to learn from. The
question "is there signal here" is better asked directly: take the rows that
*do* have a purchase, and look at what happened next.

Design
------
* Signal date is the FILING date, never the trade date. See
  fetch_insider_data.py -- 36% of filings land more than 2 days after the
  trade and the 99th percentile lag is 675 days.
* Entry is the close *after* the filing date, matching the repo's
  execution_lag=1. A reader of that day's filings could have traded there.
* Returns are benchmarked against the **same universe on the same date**, not
  against SPY. This matters enormously and was got wrong first: insider
  purchases concentrate in small caps, and these 800 tickers underperformed SPY
  by 2.9% over an average 126-day window in 2022-2024. Benchmarking them
  against SPY therefore measures the size factor and reports a spurious
  negative insider effect of the same magnitude.
  Subtracting the cross-sectional median of the universe on the event date
  removes both the size tilt and the event-timing effect -- insider buying
  spikes after selloffs, so a control pooled across all dates would still be
  confounded.
* The null is not zero and it is not SPY. It is "a stock from this universe,
  on this date, with no insider purchase".

Usage
-----
    python scripts/insider_event_study.py --max-tickers 800 --min-price 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 21, 63, 126)
BENCHMARK = "SPY"


def _load_events(trans_path: Path, min_price: float) -> pd.DataFrame:
  df = pd.read_parquet(trans_path)
  df = df[df.price >= min_price]
  # Drop pre-arranged 10b5-1 purchases: scheduled before the insider knew
  # anything, so they carry no information by construction.
  df = df[~df.is_planned.astype(bool)]
  # Keep plain US listings. 5-letter tickers ending in F or Y are typically
  # foreign/ADR OTC lines with unreliable price history.
  t = df.ticker.astype(str).str.upper().str.strip()
  df = df[t.str.fullmatch(r"[A-Z]{1,4}")]
  df["ticker"] = t
  return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
  """One row per (filing_date, ticker) -- the unit a reader could act on."""
  g = df.groupby(["filing_date", "ticker"])
  out = g.agg(
    n_insiders=("owner_name", "nunique"),
    total_value=("value_usd", "sum"),
    any_officer=("is_officer", "any"),
    any_ceo_cfo=("is_ceo_cfo", "any"),
    median_lag=("filing_lag_days", "median"),
  ).reset_index()
  out["is_cluster"] = out.n_insiders >= 2
  return out


def _download(tickers: list[str], start: str, end: str, cache: Path) -> pd.DataFrame:
  """Batch-download adjusted closes, caching the result."""
  if cache.exists():
    px = pd.read_parquet(cache)
    have = set(px.columns)
    if set(tickers) - have == set() or len(have) >= len(tickers) * 0.9:
      return px

  import yfinance as yf

  frames = []
  batch = 100
  symbols = sorted(set(tickers) | {BENCHMARK})
  for i in range(0, len(symbols), batch):
    chunk = symbols[i : i + batch]
    print(f"  downloading {i+1}-{i+len(chunk)} of {len(symbols)}", file=sys.stderr)
    raw = yf.download(
      chunk, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )
    if raw is None or raw.empty:
      continue
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    frames.append(close)

  px = pd.concat(frames, axis=1)
  px = px.loc[:, ~px.columns.duplicated()].sort_index()
  cache.parent.mkdir(parents=True, exist_ok=True)
  px.to_parquet(cache)
  return px


def _abnormal_panel(px: pd.DataFrame) -> dict[int, pd.DataFrame]:
  """Universe-demeaned forward returns, one frame per horizon.

  abnormal[h].loc[t, ticker] is the ticker's h-day forward return from t,
  minus the cross-sectional median forward return of the universe at t.
  Demeaning against the universe on the *same date* is what makes this a
  matched control.

  The benchmark is the **median**, not the mean. Small-cap forward returns are
  heavily right-skewed (at 63 days the cross-section runs to +182% with a skew
  of 1.9), so the mean sits above the typical stock and subtracting it makes
  the median name look reliably negative. Using the mean reported a -1.7%
  "abnormal" return at a one-day horizon, which is how the error announced
  itself.
  """
  cols = [c for c in px.columns if c != BENCHMARK]
  out = {}
  for h in HORIZONS:
    fwd = px[cols].shift(-h) / px[cols] - 1.0
    out[h] = fwd.sub(fwd.median(axis=1), axis=0)
  return out


def _forward_returns(px: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
  """Abnormal forward return for each event, at several horizons.

  Entry is the close one trading day after the filing, so a filing seen on day
  t is traded at t+1 and the h-day return runs from t+1 to t+1+h.
  """
  dates = px.index
  abnormal = _abnormal_panel(px)
  # Also keep the naive SPY-relative figure, so the size artefact stays visible
  # rather than being quietly corrected away.
  spy_fwd = {h: px[BENCHMARK].shift(-h) / px[BENCHMARK] - 1.0 for h in HORIZONS}
  raw_fwd = {
    h: px[[c for c in px.columns if c != BENCHMARK]].shift(-h)
    / px[[c for c in px.columns if c != BENCHMARK]]
    - 1.0
    for h in HORIZONS
  }

  rows = []
  for ev in events.itertuples(index=False):
    pos = dates.searchsorted(pd.Timestamp(ev.filing_date), side="right")
    if pos >= len(dates):
      continue
    entry_date = dates[pos]

    rec = {
      "filing_date": ev.filing_date,
      "entry_date": entry_date,
      "ticker": ev.ticker,
      "n_insiders": ev.n_insiders,
      "total_value": ev.total_value,
      "is_cluster": ev.is_cluster,
      "any_officer": ev.any_officer,
      "any_ceo_cfo": ev.any_ceo_cfo,
    }
    ok = False
    for h in HORIZONS:
      frame = abnormal[h]
      if ev.ticker not in frame.columns:
        rec[f"ar_{h}d"] = np.nan
        rec[f"spy_{h}d"] = np.nan
        continue
      v = frame.at[entry_date, ev.ticker]
      rec[f"ar_{h}d"] = v if np.isfinite(v) else np.nan
      r = raw_fwd[h].at[entry_date, ev.ticker]
      b = spy_fwd[h].at[entry_date]
      rec[f"spy_{h}d"] = (r - b) if (np.isfinite(r) and np.isfinite(b)) else np.nan
      if np.isfinite(v):
        ok = True
    if ok:
      rows.append(rec)

  return pd.DataFrame(rows)


def _report(df: pd.DataFrame, label: str) -> None:
  """Median abnormal return plus a sign test.

  The sign test -- what fraction of events beat the universe median, against a
  50% null -- is the primary statistic. It is robust to the return skew that
  breaks mean-based tests, and it does not assume events are independent in
  magnitude. The mean is shown alongside for reference only.
  """
  from scipy import stats

  print(f"\n--- {label}  (n = {len(df):,}) ---")
  if len(df) < 30:
    print("  too few events to say anything")
    return
  print(f"  {'horizon':>8} {'median AR':>10} {'% beating':>10} {'sign p':>9} {'mean':>9}")
  for h in HORIZONS:
    s = df[f"ar_{h}d"].dropna()
    if len(s) < 30:
      continue
    wins, n = int((s > 0).sum()), len(s)
    p = stats.binomtest(wins, n, 0.5).pvalue
    flag = " *" if p < 0.01 else ""
    print(f"  {h:>7}d {s.median()*100:+9.3f}% {wins/n*100:9.1f}% {p:9.4f} "
          f"{s.mean()*100:+8.3f}%{flag}")


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--transactions", default="data/insider_purchases.parquet")
  p.add_argument("--min-price", type=float, default=5.0,
                 help="drop purchases below this price; penny names are untradeable")
  p.add_argument("--max-tickers", type=int, default=800,
                 help="keep the N tickers with the most events, to bound the download")
  p.add_argument("--price-cache", default="data/cache/insider_prices.parquet")
  p.add_argument("--out", default="data/insider_event_study.parquet")
  args = p.parse_args(argv)

  trans = Path(args.transactions)
  if not trans.exists():
    print(f"missing {trans}; run scripts/fetch_insider_data.py first", file=sys.stderr)
    return 1

  df = _load_events(trans, args.min_price)
  events = _aggregate(df)
  print(f"events after filtering: {len(events):,} on {events.ticker.nunique():,} tickers")

  top = events.ticker.value_counts().head(args.max_tickers).index.tolist()
  events = events[events.ticker.isin(top)]
  print(f"restricted to top {len(top)} tickers: {len(events):,} events")

  lo = (pd.Timestamp(events.filing_date.min()) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
  hi = (pd.Timestamp(events.filing_date.max()) + pd.Timedelta(days=400)).strftime("%Y-%m-%d")
  px = _download(top, lo, hi, Path(args.price_cache))
  print(f"price panel: {px.shape[0]} dates x {px.shape[1]} symbols")

  res = _forward_returns(px, events)
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  res.to_parquet(args.out, index=False)
  print(f"matched {len(res):,} events to prices -> {args.out}")

  print(f"\n{'='*62}\nTHE SIZE ARTEFACT, for the record\n{'='*62}")
  print("Benchmarking this universe against SPY instead of against itself")
  print("reports the opposite sign, because these names underperformed SPY")
  print("regardless of insider activity:")
  print(f"  {'horizon':>8} {'vs SPY':>10} {'vs universe':>12}")
  for h in HORIZONS:
    a, s = res[f"ar_{h}d"].dropna(), res[f"spy_{h}d"].dropna()
    if len(a) < 30:
      continue
    print(f"  {h:>7}d {s.mean()*100:9.3f}% {a.mean()*100:11.3f}%")

  print(f"\n{'='*62}\nABNORMAL RETURNS vs MATCHED UNIVERSE (same date)\n{'='*62}")
  _report(res, "all insider purchases")
  _report(res[res.is_cluster], "cluster buys (2+ insiders, same day)")
  _report(res[~res.is_cluster], "single-insider buys")
  _report(res[res.any_officer], "officer purchases")
  _report(res[res.any_ceo_cfo], "CEO/CFO purchases")
  big = res[res.total_value >= 100_000]
  _report(big, "purchases >= $100k")
  huge = res[res.total_value >= 1_000_000]
  _report(huge, "purchases >= $1M")

  print("\nNote: events cluster in calendar time, so even the sign test is")
  print("optimistic. The 2022-2024 window is short, and yfinance history")
  print("excludes delisted names -- a real upward bias at 6-month horizons.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
