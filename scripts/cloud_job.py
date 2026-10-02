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
        try:
            from soccer.misli import save_market_snapshot
            print("bookmaker odds stored:", save_market_snapshot(eng))
        except Exception:
            traceback.print_exc()
        from soccer.notify import send_daily_digest
        print(send_daily_digest(eng))
        from soccer import analysis
        from soccer.notify import send
        age = (pd.Timestamp.now() - pd.Timestamp(analysis.REPORT.stat().st_mtime, unit="s")).days             if analysis.REPORT.exists() else 99
        if pd.Timestamp.now().weekday() == 0 or age >= 7:
            report = analysis.weekly_report()
            print("weekly analysis written")
            send("🧪 <b>Еженедельный автоанализ</b>\n" + report.replace("# ", "").replace("## ", "▪️ ")
                 .replace("**", "")[:3800])
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("state exported:", storage.export_state())
