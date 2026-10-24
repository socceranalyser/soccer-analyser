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
        try:  # starting XIs of the last days -> each team's regulars (soccer/lineups.py)
            from soccer.lineups import update_history
            print("line-ups added:", update_history(eng.matches))
        except Exception:
            traceback.print_exc()
        from soccer import analysis
        print("daily learning:", analysis.daily_learning())  # before today's forecasts
        today = pd.Timestamp.now().normalize()
        print("forecast rows:", eng.record_days([today + timedelta(days=i) for i in range(4)]))
        try:
            from soccer.misli import events_with_model, save_market_snapshot
            ev = events_with_model(eng)
            print("bookmaker odds stored:", save_market_snapshot(eng, ev))
            from soccer.coupons import log_offers
            baku = pd.Timestamp.now(tz="Asia/Baku").tz_localize(None).normalize()
            print("coupons offered today:", log_offers(ev, baku, "Asia/Baku"))
            from soccer.health import link_check
            from soccer.notify import send as _send
            sus = link_check(ev)
            print("suspicious links:", sus or "none")
            # each match once (it used to repeat every day), in calm words: such matches are
            # already kept out of the coupons; the owner only needs to know when it is new
            import json
            from soccer.config import DATA_DIR
            seen_f = DATA_DIR / "state" / "link_alerts.json"
            seen = set(json.loads(seen_f.read_text(encoding="utf-8"))) if seen_f.exists() else set()
            key = lambda s: s.split("«")[-1].split("»")[0]  # the match as misli names it
            new = [s for s in sus if key(s) not in seen]
            if new:
                _send("ℹ️ <b>Модель и букмекер сильно расходятся</b> — такие матчи не берутся в "
                      "купоны; я проверю, не перепутаны ли команды:"
                      + "".join(f"\n• {s.split('(misli) = ')[-1]}" for s in new[:10]))
                seen_f.write_text(json.dumps(sorted(seen | {key(s) for s in new})[-300:]),
                                  encoding="utf-8")
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
        try:
            from soccer import health
            problems = health.run(eng)
            print("health:", problems or "ok")
            if problems:
                send("⚠️ <b>Проверка данных нашла проблемы</b>\n" + "\n".join("• " + p for p in problems))
        except Exception:
            traceback.print_exc()
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
