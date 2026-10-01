"""Daily job: refresh data, store finished results, record forecasts for the next days.

    .venv\\Scripts\\python.exe scripts\\update.py            (today + 3 days)
    .venv\\Scripts\\python.exe scripts\\update.py --days 7
"""
import argparse
from datetime import timedelta

import pandas as pd

from soccer.engine import Engine

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    eng = Engine(refresh=not args.no_refresh, verbose=True)
    n_res = eng.save_results()
    today = pd.Timestamp.now().normalize()
    n_pred = eng.record_days([today + timedelta(days=i) for i in range(args.days)])
    print(f"results stored: {n_res:,} | forecast rows stored: {n_pred:,} | "
          f"fresh league results: {eng.n_fresh}")
