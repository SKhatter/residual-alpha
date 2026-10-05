#!/usr/bin/env python3
"""Fetch SEC Form 4 insider transactions from EDGAR's quarterly data sets.

Why the bulk data sets and not scraping
---------------------------------------
There are ~250k Form 4 filings a year. Fetching each filing's XML at SEC's
rate limit would take days. The DERA "Insider Transactions Data Sets" publish
the same content as quarterly TSV bundles -- one HTTP request per quarter.

Why FILING_DATE and not TRANS_DATE
----------------------------------
This is the entire reason to be careful with insider data. Form 4 is due within
two business days of the trade, but compliance is poor: measured on 2024Q1
open-market purchases, 44% were filed more than 2 days late, 15% more than 30
days late, and the 99th percentile lag is 794 days. A signal keyed to the trade
date is therefore trading on information that was not public for months.

Every row this script emits is stamped with FILING_DATE. The trade date is kept
alongside only so the lag can be audited, never to align a signal.

What gets kept
--------------
Open-market purchases (TRANS_CODE 'P') by default. The other codes dominate the
raw data and carry little information: 'F' is tax withholding, 'A' is a grant,
'M' is an option exercise, 'S' is a sale (mostly diversification and scheduled
plans). On 2024Q1, 5,954 of 111,404 non-derivative transactions were code 'P'.

`is_planned` comes from the AFF10B5ONE flag -- trades under a pre-arranged
10b5-1 plan, scheduled before the insider knew anything. These are the "routine"
half of the routine/opportunistic distinction and should normally be excluded.

Usage
-----
    python scripts/fetch_insider_data.py --start 2014 --end 2024
    python scripts/fetch_insider_data.py --start 2024 --end 2024 --diagnose
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

# SEC requires a descriptive User-Agent with contact information. Requests
# without one are refused.
USER_AGENT = "residual-alpha research (sumedhakhatter482@gmail.com)"

# The hosting path moved partway through 2026; try the newer one first.
URL_TEMPLATES = [
  "https://www.sec.gov/files/datastandardsinnovation/data/"
  "insider-transactions-data-sets/{q}_form345.zip",
  "https://www.sec.gov/files/structureddata/data/"
  "insider-transactions-data-sets/{q}_form345.zip",
]

PURCHASE_CODES = ("P",)
SALE_CODES = ("S",)


def _quarters(start_year: int, end_year: int) -> list[str]:
  return [f"{y}q{q}" for y in range(start_year, end_year + 1) for q in (1, 2, 3, 4)]


def _download(quarter: str, cache: Path) -> bytes | None:
  """Fetch one quarterly bundle, caching the raw zip."""
  import urllib.error
  import urllib.request

  dest = cache / f"{quarter}_form345.zip"
  if dest.exists():
    return dest.read_bytes()

  for template in URL_TEMPLATES:
    url = template.format(q=quarter)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
      with urllib.request.urlopen(req, timeout=120) as fh:
        blob = fh.read()
    except urllib.error.HTTPError as exc:
      if exc.code == 404:
        continue
      print(f"  {quarter}: HTTP {exc.code}", file=sys.stderr)
      return None
    except Exception as exc:  # noqa: BLE001 - network errors vary
      print(f"  {quarter}: {type(exc).__name__}: {exc}", file=sys.stderr)
      return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    # Be a polite client; SEC's published ceiling is 10 requests/second and we
    # need nowhere near that.
    time.sleep(0.5)
    return blob

  print(f"  {quarter}: not published", file=sys.stderr)
  return None


def _read(zf: zipfile.ZipFile, name: str, usecols: list[str]) -> pd.DataFrame:
  """Read one TSV, tolerating columns that do not exist in that vintage.

  The schema drifts across quarters -- AFF10B5ONE, the 10b5-1 plan checkbox,
  only appears from 2022Q4 once the SEC added it to the form. Requesting it
  from an earlier bundle is a hard error in pandas, so intersect against the
  real header first and let the caller fill the gap.
  """
  with zf.open(name) as fh:
    header = pd.read_csv(
      io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
      sep="\t",
      nrows=0,
    ).columns
  present = [c for c in usecols if c in header]
  missing = [c for c in usecols if c not in header]
  with zf.open(name) as fh:
    df = pd.read_csv(
      io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
      sep="\t",
      usecols=present,
      low_memory=False,
    )
  for col in missing:
    df[col] = pd.NA
  return df


def _normalise_flag(series: pd.Series) -> pd.Series:
  """AFF10B5ONE arrives as a mix of 0/1 and 'false'/'true' across quarters."""
  return (
    series.astype(str).str.strip().str.lower().isin({"1", "true", "y", "yes"})
  )


def _parse_quarter(blob: bytes, codes: tuple[str, ...]) -> pd.DataFrame:
  """Join submission / transaction / owner tables into one tidy frame."""
  with zipfile.ZipFile(io.BytesIO(blob)) as zf:
    sub = _read(zf, "SUBMISSION.tsv", [
      "ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
      "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL", "AFF10B5ONE",
    ])
    tr = _read(zf, "NONDERIV_TRANS.tsv", [
      "ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
      "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
    ])
    own = _read(zf, "REPORTINGOWNER.tsv", [
      "ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
      "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE",
    ])

  # Amendments (4/A) restate an earlier filing. Keeping them would double count
  # the same economic trade, so only original Form 4s are used.
  sub = sub[sub.DOCUMENT_TYPE == "4"]
  tr = tr[tr.TRANS_CODE.isin(codes)]

  df = tr.merge(sub, on="ACCESSION_NUMBER", how="inner")
  # One filing can carry several reporting owners; collapse to the first so a
  # joint filing counts once, and keep the count separately.
  owners = (
    own.groupby("ACCESSION_NUMBER")
    .agg(
      n_owners=("RPTOWNERCIK", "nunique"),
      owner_name=("RPTOWNERNAME", "first"),
      relationship=("RPTOWNER_RELATIONSHIP", "first"),
      title=("RPTOWNER_TITLE", "first"),
    )
    .reset_index()
  )
  df = df.merge(owners, on="ACCESSION_NUMBER", how="left")

  for col in ("FILING_DATE", "TRANS_DATE"):
    df[col] = pd.to_datetime(df[col], format="%d-%b-%Y", errors="coerce")

  df["shares"] = pd.to_numeric(df.TRANS_SHARES, errors="coerce")
  df["price"] = pd.to_numeric(df.TRANS_PRICEPERSHARE, errors="coerce")
  df["value_usd"] = df.shares * df.price
  df["filing_lag_days"] = (df.FILING_DATE - df.TRANS_DATE).dt.days
  df["is_planned"] = _normalise_flag(df.AFF10B5ONE)

  rel = df.relationship.fillna("")
  df["is_officer"] = rel.str.contains("Officer", case=False)
  df["is_director"] = rel.str.contains("Director", case=False)
  title = df.title.fillna("").str.upper()
  # CEO/CFO purchases are the subset with the strongest documented effect.
  df["is_ceo_cfo"] = title.str.contains(
    r"\bCEO\b|CHIEF EXECUTIVE|\bCFO\b|CHIEF FINANCIAL", regex=True
  )

  out = df.rename(columns={
    "FILING_DATE": "filing_date",
    "TRANS_DATE": "trade_date",
    "TRANS_CODE": "trans_code",
    "ISSUERTRADINGSYMBOL": "ticker",
    "ISSUERNAME": "issuer",
    "ISSUERCIK": "issuer_cik",
  })
  cols = [
    "filing_date", "trade_date", "filing_lag_days", "ticker", "issuer",
    "issuer_cik", "trans_code", "shares", "price", "value_usd",
    "owner_name", "relationship", "title", "n_owners",
    "is_officer", "is_director", "is_ceo_cfo", "is_planned",
    "ACCESSION_NUMBER",
  ]
  out = out[cols].rename(columns={"ACCESSION_NUMBER": "accession"})
  out = out[out.ticker.notna() & (out.ticker.astype(str).str.strip() != "")]
  return out.sort_values(["filing_date", "ticker"]).reset_index(drop=True)


def daily_panel(df: pd.DataFrame) -> pd.DataFrame:
  """Collapse filings to one row per (filing_date, ticker).

  This is the shape a feature pipeline wants: stamped at the date the
  information became public, with the cluster size and the dollar amount that a
  reader of that day's filings could have seen.
  """
  excl = df[~df.is_planned]
  g = excl.groupby(["filing_date", "ticker"])
  out = g.agg(
    n_filings=("accession", "nunique"),
    n_insiders=("owner_name", "nunique"),
    total_value_usd=("value_usd", "sum"),
    max_value_usd=("value_usd", "max"),
    any_officer=("is_officer", "any"),
    any_ceo_cfo=("is_ceo_cfo", "any"),
    median_lag_days=("filing_lag_days", "median"),
  ).reset_index()
  # Cluster buys -- several distinct insiders buying the same name -- are the
  # variant with the strongest effect in the literature.
  out["is_cluster"] = out.n_insiders >= 2
  return out


def diagnose(df: pd.DataFrame, universe: list[str] | None) -> None:
  print(f"\n{'='*66}\nDIAGNOSTIC\n{'='*66}")
  print(f"rows (code {sorted(df.trans_code.unique())}): {len(df):,}")
  print(f"date span: {df.filing_date.min():%Y-%m-%d} .. {df.filing_date.max():%Y-%m-%d}")
  print(f"distinct tickers: {df.ticker.nunique():,}")

  lag = df.filing_lag_days.dropna()
  print("\nfiling lag (calendar days) -- why trade_date must not be used:")
  for p in (50, 75, 90, 95, 99):
    print(f"   p{p:<3d} {lag.quantile(p/100):8.0f}")
  print(f"   filed >2 days late : {(lag > 2).mean():.1%}")
  print(f"   filed >30 days late: {(lag > 30).mean():.1%}")

  print(f"\npre-arranged (10b5-1) share: {df.is_planned.mean():.1%}")
  print(f"officer share: {df.is_officer.mean():.1%}   "
        f"CEO/CFO share: {df.is_ceo_cfo.mean():.1%}")

  panel = daily_panel(df)
  print(f"\ndaily panel rows: {len(panel):,}  "
        f"({panel.is_cluster.mean():.1%} are cluster buys)")

  if universe:
    hit = df[df.ticker.isin(universe)]
    years = max((df.filing_date.max() - df.filing_date.min()).days / 365.25, 1e-9)
    print(f"\n--- coverage of the supplied {len(universe)}-name universe ---")
    print(f"purchases: {len(hit):,} over {years:.1f}y "
          f"= {len(hit)/years/max(len(universe),1):.2f} per name per year")
    if len(hit):
      print(hit.ticker.value_counts().head(12).to_string())
    else:
      print("  none")


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--start", type=int, required=True, help="first year")
  p.add_argument("--end", type=int, required=True, help="last year")
  p.add_argument("--out", default="data/insider_purchases.parquet")
  p.add_argument("--panel-out", default="data/insider_daily.parquet")
  p.add_argument("--cache", default="data/cache/insider")
  p.add_argument("--include-sales", action="store_true",
                 help="also keep code S; off because sales are mostly noise")
  p.add_argument("--diagnose", action="store_true")
  p.add_argument("--universe", default=None,
                 help="comma-separated tickers to report coverage for")
  args = p.parse_args(argv)

  codes = PURCHASE_CODES + (SALE_CODES if args.include_sales else ())
  cache = Path(args.cache)
  frames = []
  for q in _quarters(args.start, args.end):
    blob = _download(q, cache)
    if blob is None:
      continue
    part = _parse_quarter(blob, codes)
    frames.append(part)
    print(f"  {q}: {len(part):,} transactions")

  if not frames:
    print("no data retrieved", file=sys.stderr)
    return 1

  df = pd.concat(frames, ignore_index=True)
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  df.to_parquet(args.out, index=False)
  panel = daily_panel(df)
  panel.to_parquet(args.panel_out, index=False)
  print(f"\nwrote {len(df):,} transactions -> {args.out}")
  print(f"wrote {len(panel):,} (filing_date, ticker) rows -> {args.panel_out}")

  if args.diagnose:
    universe = (
      [t.strip().upper() for t in args.universe.split(",") if t.strip()]
      if args.universe else None
    )
    diagnose(df, universe)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
