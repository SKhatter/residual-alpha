"""Command line interface.

    ralpha build          download data and cache the panel
    ralpha folds          print the walk-forward schedule
    ralpha bakeoff        forecast quality: every model vs the null
    ralpha backtest       forecast quality -> portfolio PnL
    ralpha ablate-gate    is the confidence gate just a vol filter?
    ralpha holdout        spend one look at the reserved window
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .holdout import (
  HoldoutLedger,
  assert_development_only,
  config_fingerprint,
  split_development,
)
from .pipeline import build_panel
from .runner import (
  ablate_gate,
  backtest_predictions,
  build_folds,
  load_predictions,
  run_walk_forward,
  save_predictions,
  score_models,
)

ARTIFACTS = Path("artifacts")


def _setup_logging(verbose: bool) -> None:
  logging.basicConfig(
    level=logging.DEBUG if verbose else logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
  )
  logging.getLogger("yfinance").setLevel(logging.ERROR)
  logging.getLogger("peewee").setLevel(logging.ERROR)


def _show(title: str, table: pd.DataFrame | pd.Series) -> None:
  print(f"\n{title}")
  print("-" * len(title))
  with pd.option_context(
    "display.width", 200, "display.max_columns", 50, "display.float_format", "{:.4f}".format
  ):
    print(table)
  print()


def _models(cfg, override: str | None) -> list[str]:
  if override:
    return [m.strip() for m in override.split(",") if m.strip()]
  return list(cfg.models.enabled)


def _development_dates(cfg, panel) -> pd.DatetimeIndex:
  dev, _ = split_development(panel.dates, cfg.holdout.start)
  if len(dev) == 0:
    raise RuntimeError("no development dates; check holdout.start vs data.start")
  return dev


# --------------------------------------------------------------------------
def cmd_build(args) -> int:
  cfg = load_config(args.config)
  panel = build_panel(cfg, refresh=args.refresh)
  dev, hold = split_development(panel.dates, cfg.holdout.start)
  print(f"\npanel: {len(panel.dates)} dates, {panel.features.shape[1]} features")
  print(f"  development: {dev[0].date()} .. {dev[-1].date()}  ({len(dev)} days)")
  if len(hold):
    print(f"  holdout:     {hold[0].date()} .. {hold[-1].date()}  ({len(hold)} days) [reserved]")
  print(f"  labelled rows: {int(panel.target.notna().sum())}")
  print(f"  timesfm covariate channels: {len(panel.covariates)}")
  print(f"  known-future calendar channels: {panel.calendar.shape[1]}\n")
  return 0


def cmd_folds(args) -> int:
  cfg = load_config(args.config)
  panel = build_panel(cfg)
  folds = build_folds(cfg, _development_dates(cfg, panel))
  print()
  for f in folds:
    print("  " + f.describe())
  print(f"\n{len(folds)} folds. Purge and embargo verified on each.\n")
  return 0


def cmd_bakeoff(args) -> int:
  cfg = load_config(args.config)
  panel = build_panel(cfg)
  dates = _development_dates(cfg, panel)
  assert_development_only(dates, cfg.holdout.start, "bakeoff")

  names = _models(cfg, args.models)
  preds = run_walk_forward(cfg, panel, names, dates)
  save_predictions(preds, ARTIFACTS)

  table = score_models(cfg, panel, preds)
  _show("Forecast quality (development, walk-forward out-of-sample)", table)
  print("  pinball_skill > 0 means the model beat the zero-forecast null.")
  print("  ic_t is the t-stat of the daily cross-sectional IC; |t| < 2 is noise.")
  print("  cov80_empirical far below 0.80 means intervals are too narrow.\n")
  return 0


def cmd_backtest(args) -> int:
  cfg = load_config(args.config)
  panel = build_panel(cfg)
  dates = _development_dates(cfg, panel)
  assert_development_only(dates, cfg.holdout.start, "backtest")

  names = _models(cfg, args.models)
  preds = load_predictions(names, ARTIFACTS) if args.cached else {}
  missing = [n for n in names if n not in preds]
  if missing:
    preds.update(run_walk_forward(cfg, panel, missing, dates))
    save_predictions(preds, ARTIFACTS)

  _show("Forecast quality", score_models(cfg, panel, preds))

  rows = {}
  for name, p in preds.items():
    result, _ = backtest_predictions(cfg, panel, p, use_gate=not args.no_gate)
    rows[name] = result.metrics
  _show("Portfolio performance (net of costs, market/sector hedged)", pd.DataFrame(rows).T)
  print("  sharpe_t is the t-stat of the Sharpe ratio. Below ~2, the strategy")
  print("  is not distinguishable from zero on this sample length.\n")
  return 0


def cmd_ablate_gate(args) -> int:
  cfg = load_config(args.config)
  panel = build_panel(cfg)
  dates = _development_dates(cfg, panel)
  assert_development_only(dates, cfg.holdout.start, "ablate-gate")

  name = args.model
  preds = load_predictions([name], ARTIFACTS)
  if name not in preds:
    preds = run_walk_forward(cfg, panel, [name], dates)
    save_predictions(preds, ARTIFACTS)

  table = ablate_gate(cfg, panel, preds[name])
  _show(f"Gate ablation for '{name}' (matched trade counts)", table)

  overlap = table.attrs.get("overlap", {})
  print(f"  Jaccard(width gate, vol gate) = {overlap.get('jaccard', float('nan')):.3f}")
  print(f"  Spearman(width, trailing vol) = {overlap.get('corr_criterion', float('nan')):.3f}")
  print()
  w = table.loc["width_gate", "sharpe_net"]
  v = table.loc["vol_gate", "sharpe_net"]
  se = table.loc["width_gate", "sharpe_se"]
  if pd.notna(w) and pd.notna(v) and pd.notna(se):
    if w - v > se:
      print("  The confidence gate beats the volatility filter by more than one")
      print("  standard error. It is plausibly doing something of its own.")
    else:
      print("  The confidence gate does NOT clear the volatility filter by even")
      print("  one standard error. Treat it as a low-volatility tilt, and note")
      print("  that it carries a short-volatility exposure you are not paid for.")
  print()
  return 0


def cmd_holdout(args) -> int:
  cfg = load_config(args.config)
  fingerprint = config_fingerprint(cfg.as_dict())
  ledger = HoldoutLedger(cfg.path(cfg.holdout.lock_file))

  banner = ledger.warning_banner(fingerprint)
  if banner:
    print(banner)
    if not args.yes:
      reply = input("  Spend another look? [y/N] ").strip().lower()
      if reply != "y":
        print("  Aborted. Nothing recorded.\n")
        return 1

  panel = build_panel(cfg)
  dev, hold = split_development(panel.dates, cfg.holdout.start)
  if len(hold) == 0:
    print("no holdout dates in range")
    return 1

  # Train through development, predict the holdout in one pass -- no refitting
  # inside the holdout, so there is exactly one decision boundary.
  all_dates = panel.dates
  names = _models(cfg, args.models)
  preds = run_walk_forward(cfg, panel, names, all_dates)
  preds = {
    n: p[p.index.get_level_values("date") >= pd.Timestamp(cfg.holdout.start)]
    for n, p in preds.items()
  }
  preds = {n: p for n, p in preds.items() if len(p)}

  _show("HOLDOUT forecast quality", score_models(cfg, panel, preds))

  rows = {}
  for name, p in preds.items():
    result, _ = backtest_predictions(cfg, panel, p, use_gate=not args.no_gate)
    rows[name] = result.metrics
  table = pd.DataFrame(rows).T
  _show("HOLDOUT portfolio performance", table)

  primary = args.primary if args.primary in rows else next(iter(rows), None)
  if primary:
    entry = ledger.record(fingerprint, rows[primary], note=args.note or primary)
    print(f"  recorded look #{entry['look_number']} for model '{primary}'")
    print(f"  ledger: {ledger.path}\n")
  return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(prog="ralpha", description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("-c", "--config", default=None, help="path to a YAML config")
  p.add_argument("-v", "--verbose", action="store_true")
  sub = p.add_subparsers(dest="command", required=True)

  b = sub.add_parser("build", help="download data and build the panel")
  b.add_argument("--refresh", action="store_true", help="bypass the price cache")
  b.set_defaults(func=cmd_build)

  f = sub.add_parser("folds", help="print the walk-forward schedule")
  f.set_defaults(func=cmd_folds)

  k = sub.add_parser("bakeoff", help="forecast quality vs the null model")
  k.add_argument("--models", default=None, help="comma-separated override")
  k.set_defaults(func=cmd_bakeoff)

  t = sub.add_parser("backtest", help="run the full pipeline to PnL")
  t.add_argument("--models", default=None)
  t.add_argument("--cached", action="store_true", help="reuse saved predictions")
  t.add_argument("--no-gate", action="store_true")
  t.set_defaults(func=cmd_backtest)

  a = sub.add_parser("ablate-gate", help="confidence gate vs volatility filter")
  a.add_argument("--model", default="gbm")
  a.set_defaults(func=cmd_ablate_gate)

  h = sub.add_parser("holdout", help="spend one look at the reserved window")
  h.add_argument("--models", default=None)
  h.add_argument("--primary", default="gbm", help="model recorded in the ledger")
  h.add_argument("--note", default=None)
  h.add_argument("--no-gate", action="store_true")
  h.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
  h.set_defaults(func=cmd_holdout)

  return p


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  _setup_logging(args.verbose)
  return args.func(args)


if __name__ == "__main__":
  raise SystemExit(main())
