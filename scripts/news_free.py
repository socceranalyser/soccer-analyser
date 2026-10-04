r"""Free news analysis done by Claude Code instead of the paid API (see soccer/news.py).

    python scripts/news_free.py todo      # pick today's matches -> data/news_todo.json
    python scripts/news_free.py submit    # data/news_answer.json -> maths, site, Telegram
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from soccer import news  # noqa: E402

def publish():
    """Save today's analysis to the site (GitHub) as part of 'submit', so the scheduled task
    needs no separate git command (each new command waits for the user's approval)."""
    import datetime
    import subprocess
    from soccer import storage
    storage.import_state()
    storage.export_state()
    files = ["data/state/news.json", "data/state/news_log.csv", "data/state/live_predictions.csv",
             "data/state/runs.csv"]
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True)  # noqa: E731
    run("add", *files)
    if run("diff", "--cached", "--quiet").returncode == 0:
        print("publish: nothing new")
        return
    run("commit", "-q", "-m", f"Daily news analysis {datetime.date.today()}",
        "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    run("pull", "-q", "--rebase", "--autostash")
    r = run("push", "-q")
    print("publish:", "ok" if r.returncode == 0 else f"FAILED {r.stderr.strip()[:200]}")


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
        publish()
    else:
        print(__doc__)
