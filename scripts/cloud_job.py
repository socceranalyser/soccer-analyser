r"""Daily job for GitHub Actions: restore state -> update -> forecasts -> Telegram -> save state.

    python scripts/cloud_job.py
"""
import sys
import traceback
from datetime import timedelta

import pandas as pd

from soccer import storage

if __name__ == "__main__":
    print("state imported:", storage.import_state())
    try:
        from soccer.engine import Engine
        eng = Engine(refresh=True, verbose=True)
        print("results stored:", eng.save_results())
        today = pd.Timestamp.now().normalize()
        print("forecast rows:", eng.record_days([today + timedelta(days=i) for i in range(4)]))
        from soccer.notify import send_daily_digest
        print(send_daily_digest(eng))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("state exported:", storage.export_state())
