"""Walk-forward backtest of all models; prints metrics and stores predictions.

Example:
    python scripts/run_backtest.py --seasons 2023 2024 2025 --run-id baseline
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from soccer import storage
from soccer.backtest import common_matches, walk_forward
from soccer.data import load_matches
from soccer.metrics import summary_table
from soccer.tuned import DC_PARAMS, ELO_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--leagues", nargs="+", default=None)
    ap.add_argument("--freq", default="W", choices=["W", "D"])
    ap.add_argument("--dc", default="{}", help="JSON overrides for DixonColes params")
    ap.add_argument("--elo", default="{}", help="JSON overrides for Elo params")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    dc_kwargs = {**DC_PARAMS, **json.loads(args.dc)}
    elo_kwargs = {**ELO_PARAMS, **json.loads(args.elo)}
    matches = load_matches(leagues=args.leagues)
    t0 = time.time()
    preds = walk_forward(matches, args.seasons, leagues=args.leagues, dc_kwargs=dc_kwargs,
                         elo_kwargs=elo_kwargs, freq=args.freq)
    print(f"backtest: {time.time() - t0:.0f}s, {len(preds)} prediction rows")

    models = sorted(preds["model"].unique())
    fair = common_matches(preds, models)
    pd.set_option("display.width", 200)
    print("\n== All leagues (matches every model predicted) ==")
    print(summary_table(fair).round(4).to_string(index=False))
    print("\n== By league ==")
    print(summary_table(fair, by=("league", "model")).round(4).to_string(index=False))
    print("\n== By season ==")
    print(summary_table(fair, by=("season", "model")).round(4).to_string(index=False))

    if not args.no_save:
        run_id = args.run_id or f"bt_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
        storage.save_results(matches)
        n = storage.save_run(run_id, "backtest", preds,
                             description=f"seasons={args.seasons} freq={args.freq}",
                             params={"dc": dc_kwargs, "elo": elo_kwargs, "seasons": args.seasons})
        print(f"\nsaved {n} predictions as run '{run_id}'")


if __name__ == "__main__":
    main()
