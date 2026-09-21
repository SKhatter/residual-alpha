#!/usr/bin/env python3
"""Scrape FOMC decision dates from federalreserve.gov into config/fomc_dates.csv.

Shipped as a script rather than a checked-in CSV on purpose: a hardcoded list
of policy dates rots silently, and a wrong date in a known-future covariate is
a lookahead bug wearing a calendar's clothes.

    python scripts/fetch_fomc_dates.py --start 2012 --end 2027

Two sources, in order of trust:

  1. Statement release links (`/monetary20260128a.htm`). The statement is
     published on the final day of the meeting, so the URL *is* the decision
     date. Exact, no parsing of prose.
  2. The month/date cells on the calendar page, for scheduled meetings that
     have not happened yet and so have no statement. Needed for live use --
     a known-future covariate is worthless if it stops at today.

Years that fail the plausibility check are dropped, not written. Writing a
half-parsed policy calendar is worse than writing none, because the pipeline
will happily consume it.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HISTORICAL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"

# The FOMC holds 8 scheduled meetings a year. Unscheduled actions do happen
# (March 2020 had two), so allow headroom -- but not enough to hide a parser
# that has started matching arbitrary dates in the page text.
MIN_PER_YEAR, MAX_PER_YEAR = 6, 12

STATEMENT_RE = re.compile(r"/monetary(\d{8})a")
PANEL_RE = re.compile(r'<a id="\d+">(\d{4}) FOMC Meetings</a>')
MONTH_RE = re.compile(r'fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([A-Za-z/]+)')
DATE_RE = re.compile(r'fomc-meeting__date[^>]*>\s*([0-9–—\-]+)')
MONTHS = {
  m: i + 1
  for i, m in enumerate(
    "January February March April May June July August September "
    "October November December".split()
  )
}


def fetch(url: str) -> str:
  req = urllib.request.Request(url, headers={"User-Agent": "residual-alpha/0.1"})
  with urllib.request.urlopen(req, timeout=30) as resp:
    return resp.read().decode("utf-8", errors="replace")


def statement_dates(html: str) -> set[pd.Timestamp]:
  """Decision dates taken directly from statement release URLs."""
  out = set()
  for stamp in STATEMENT_RE.findall(html):
    try:
      out.add(pd.Timestamp(stamp))
    except ValueError:
      continue
  return out


def scheduled_dates(html: str) -> set[pd.Timestamp]:
  """Final day of each scheduled meeting, from the calendar's month/date cells.

  Handles meetings that straddle a month boundary ("April/May", "30-1"): the
  decision lands on the last day, which belongs to the second month.
  """
  out: set[pd.Timestamp] = set()
  # Split into per-year panels so a date cell is attributed to the right year.
  chunks = re.split(r'<div class="panel panel-default">', html)
  for chunk in chunks:
    year_match = PANEL_RE.search(chunk)
    if not year_match:
      continue
    year = int(year_match.group(1))

    months = MONTH_RE.findall(chunk)
    days = DATE_RE.findall(chunk)
    if len(months) != len(days):
      print(
        f"  {year}: {len(months)} month cells vs {len(days)} date cells; "
        "skipping structured parse for this year",
        file=sys.stderr,
      )
      continue

    for month_text, day_text in zip(months, days):
      parts = [p for p in month_text.split("/") if p in MONTHS]
      if not parts:
        continue
      day_parts = re.split(r"[–—\-]", day_text)
      day_parts = [d for d in day_parts if d.strip().isdigit()]
      if not day_parts:
        continue

      month_name = parts[-1]
      day = int(day_parts[-1])
      # A December/January meeting rolls the year forward.
      stamp_year = year + 1 if month_name == "January" and parts[0] == "December" else year
      try:
        out.add(pd.Timestamp(year=stamp_year, month=MONTHS[month_name], day=day))
      except ValueError:
        continue
  return out


def collect(start: int, end: int) -> dict[int, set[pd.Timestamp]]:
  by_year: dict[int, set[pd.Timestamp]] = defaultdict(set)

  calendar = fetch(CALENDAR)
  for source in (statement_dates(calendar), scheduled_dates(calendar)):
    for d in source:
      if start <= d.year <= end:
        by_year[d.year].add(d)

  # Anything the calendar page does not reach lives on a historical page.
  for year in range(start, end + 1):
    if len(by_year.get(year, ())) >= MIN_PER_YEAR:
      continue
    try:
      html = fetch(HISTORICAL.format(year=year))
    except Exception as exc:  # noqa: BLE001 - 404 for years without a page
      print(f"  {year}: no historical page ({exc})", file=sys.stderr)
      continue
    for d in statement_dates(html):
      if d.year == year:
        by_year[year].add(d)

  return by_year


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--start", type=int, default=2012)
  ap.add_argument("--end", type=int, default=2027)
  ap.add_argument("--out", default="config/fomc_dates.csv")
  args = ap.parse_args()

  by_year = collect(args.start, args.end)

  accepted: list[pd.Timestamp] = []
  rejected: list[int] = []
  for year in range(args.start, args.end + 1):
    dates = sorted(by_year.get(year, ()))
    if not dates:
      print(f"  {year}: no dates found")
      rejected.append(year)
      continue
    if not MIN_PER_YEAR <= len(dates) <= MAX_PER_YEAR:
      print(
        f"  {year}: {len(dates)} dates, outside {MIN_PER_YEAR}-{MAX_PER_YEAR}; "
        "REJECTED",
        file=sys.stderr,
      )
      rejected.append(year)
      continue
    print(f"  {year}: {len(dates)} meetings  ({dates[0].date()} .. {dates[-1].date()})")
    accepted.extend(dates)

  if not accepted:
    print("\nnothing parsed cleanly; page layout has probably changed", file=sys.stderr)
    return 1

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  pd.DataFrame({"date": sorted(set(accepted))}).to_csv(out, index=False)
  print(f"\nwrote {len(set(accepted))} dates to {out}")
  if rejected:
    print(f"years omitted: {rejected} -- FOMC features are absent for these.")
    print("Fix the parser or supply them by hand before trusting those periods.")
  return 1 if rejected else 0


if __name__ == "__main__":
  raise SystemExit(main())
