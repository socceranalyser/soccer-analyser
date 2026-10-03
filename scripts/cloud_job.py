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
        from soccer import analysis
        print("daily learning:", analysis.daily_learning())  # before today's forecasts
        today = pd.Timestamp.now().normalize()
        print("forecast rows:", eng.record_days([today + timedelta(days=i) for i in range(4)]))
        try:
            from soccer.misli import events_with_model, save_market_snapshot
            ev = events_with_model(eng)
            print("bookmaker odds stored:", save_market_snapshot(eng, ev))
            from soccer import news
            if news.available():  # Claude reads today's news; the maths weighs it
                print("news analysed:", len(news.analyse(ev)), "W, G, n =", news.weights())
            else:
                print("news: ANTHROPIC_API_KEY not set - skipped")
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
