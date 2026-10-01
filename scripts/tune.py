"""Grid search of model hyperparameters on VALIDATION seasons (never the test seasons).

    python scripts/tune.py dc   --seasons 2019 2020 2021 2022
    python scripts/tune.py elo  --seasons 2019 2020 2021 2022
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from soccer.backtest import walk_forward
from soccer.config import DATA_DIR
from soccer.data import load_matches
from soccer.metrics import summarize
from soccer.tuned import DC_PARAMS, ELO_PARAMS

GRIDS = {
    # round 1 (no newcomer prior): best xi=0.0015, ridge=6, window=5y (logloss 0.99346)
    "dc": {"xi": [0.001, 0.0015, 0.002],
           "ridge": [4.0, 6.0, 9.0],
           "window_days": [365 * 3, 365 * 5],
           "newcomer_shift": [0.0, 0.1, 0.2],
           "newcomer_ridge_mult": [1.0, 3.0]},
    # first pass (k 15-35, home 50-95, regress 0-0.33) hit the k=15 / regress=0 edge;
    # home_adv barely matters because the ordered logit re-learns the home intercept.
    "elo": {"k": [8.0, 10.0, 12.0, 14.0, 16.0],
            "home_adv": [65.0],
            "season_regress": [0.0, 0.05, 0.1]},
}

_MATCHES = None


def _evaluate(model: str, params: dict, seasons) -> dict:
    global _MATCHES
    if _MATCHES is None:
        _MATCHES = load_matches()
    name = "dixon_coles" if model == "dc" else "elo"
    kw = {"dc_kwargs": params} if model == "dc" else {"elo_kwargs": params}
    preds = walk_forward(_MATCHES, seasons, models=(name,), verbose=False, n_jobs=1, **kw)
    return summarize(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(GRIDS))
    ap.add_argument("--seasons", type=int, nargs="+", default=[2019, 2020, 2021, 2022])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--grid", default=None, help='JSON grid override, e.g. {"xi": [0.002]}')
    args = ap.parse_args()

    load_matches()  # warm the download cache before forking workers
    grid = json.loads(args.grid) if args.grid else GRIDS[args.model]
    base = DC_PARAMS if args.model == "dc" else ELO_PARAMS
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_evaluate, args.model, {**base, **c}, args.seasons): c for c in combos}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            res = {**c, **fut.result()}
            rows.append(res)
            print(f"[{i}/{len(combos)}] {c} -> logloss {res['log_loss']:.5f} "
                  f"rps {res['rps']:.5f}", flush=True)
    df = pd.DataFrame(rows).sort_values("log_loss")
    out = DATA_DIR / "tuning" / f"{args.model}_{'_'.join(map(str, args.seasons))}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print("\nTop 10:\n", df.head(10).round(5).to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
