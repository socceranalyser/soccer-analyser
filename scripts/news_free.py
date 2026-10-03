r"""Free news analysis done by Claude Code instead of the paid API (see soccer/news.py).

    python scripts/news_free.py todo      # pick today's matches -> data/news_todo.json
    python scripts/news_free.py submit    # data/news_answer.json -> maths, site, Telegram
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from soccer import news  # noqa: E402

if __name__ == "__main__":
    if sys.argv[1:] == ["todo"]:
        from soccer.engine import Engine
        from soccer.misli import events_with_model
        eng = Engine()
        for t in news.write_todo(events_with_model(eng)):
            print(t["text"])
        print("\nRULES:\n" + news.SYSTEM)
    elif sys.argv[1:] == ["submit"]:
        recs = news.submit_answer()
        W, G, n = news.weights()
        print(f"saved {len(recs)} analyses; W={W:.3f} G={G:.3f} (finished news matches: {n})")
        from soccer.notify import _e, configured, send
        notable = [r for r in recs if abs(r["s"]) >= 1 or r["goals_shift"] or r["avoid"]]
        if configured() and notable:
            lines = ["📰 <b>Новости дня — что я учёл</b>"]
            for r in notable:
                (ph, px, pa), _, _ = news.adjust((r["p_home"], r["p_draw"], r["p_away"]), r["s"],
                                                 w=(W, G, n))
                lines.append(f"\n<b>{_e(r['home'])} — {_e(r['away'])}</b>"
                             + (" ⛔ <i>исключён из купонов</i>" if r["avoid"] else "")
                             + f"\nП1 {r['p_home']:.0%}→{ph:.0%} · Х {r['p_draw']:.0%}→{px:.0%}"
                             f" · П2 {r['p_away']:.0%}→{pa:.0%}\n{_e(r['summary'])}")
            print("telegram:", send("\n".join(lines)))
    else:
        print(__doc__)
