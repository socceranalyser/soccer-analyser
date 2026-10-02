r"""Daily job: refresh data, store finished results, record forecasts for the next days.

    .venv\Scripts\python.exe scripts\update.py            (today + 3 days)
    .venv\Scripts\python.exe scripts\update.py --days 7
    .venv\Scripts\pythonw.exe scripts\update.py --log data\update.log   (no window)
"""
import argparse
import sys
import traceback
from datetime import datetime, timedelta

import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--log", default=None, help="append output to this file (for pythonw)")
    ap.add_argument("--notify", action="store_true", help="send the daily Telegram digest")
    args = ap.parse_args()
    if args.log:
        sys.stdout = sys.stderr = open(args.log, "a", encoding="utf-8", buffering=1)
    print(f"==== {datetime.now():%Y-%m-%d %H:%M:%S} ====")
    try:
        from soccer.engine import Engine
        eng = Engine(refresh=not args.no_refresh, verbose=True)
        n_res = eng.save_results()
        today = pd.Timestamp.now().normalize()
        n_pred = eng.record_days([today + timedelta(days=i) for i in range(args.days)])
        print(f"results stored: {n_res:,} | forecast rows stored: {n_pred:,} | "
              f"fresh league results: {eng.n_fresh}")
        if args.notify:
            from soccer.notify import send_daily_digest
            print(send_daily_digest(eng))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
