r"""Snapshot of every football match in play: minute, score, red cards, misli.az live odds and
livescore.com statistics (shots, shots on target, possession, corners, attacks...).

Run every 15 minutes by .github/workflows/live_collect.yml; rows are appended to
<out>/live_snapshots_YYYY-MM.jsonl (branch 'livedata', so the dashboard is not redeployed).
One JSON object per row: livescore and misli rows carry different columns, and appending
them to one CSV under the first header made the 2026-10 CSV unreadable.
Purpose: after a few hundred finished matches, test whether in-match statistics (and the
live market) improve our in-play model (soccer/inplay.py) - only then use them.

    python scripts/collect_live.py --out data/live
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

from soccer.livescore import API as LS_API, HEADERS as LS_HEADERS, _events, _get
from soccer.misli import fetch_live_odds

STAT_KEYS = ["Shon", "Shof", "Shbl", "Shwd", "Pss", "Cos", "Att", "Dat", "Fls", "Ofs",
             "Crs", "Ycs", "Rcs", "Gks", "Ths"]


def livescore_snapshot(stats: bool = True) -> pd.DataFrame:
    rows = []
    data = _get("live/soccer/0?MD=1")
    for r in _events(data):
        if not r["live"]:
            continue
        eid = r["id"][2:]
        row = {k: r[k] for k in ("home_raw", "away_raw", "kickoff", "minute", "hg", "ag",
                                 "red_h", "red_a", "competition")}
        row["ls_id"] = eid
        if not stats:  # quick score check for goal alerts
            rows.append(row)
            continue
        try:
            st = requests.get(f"{LS_API}/statistics/soccer/{eid}", headers=LS_HEADERS,
                              timeout=20).json().get("Stat") or []
            for side, s in zip(("h", "a"), sorted(st, key=lambda x: x.get("Tnb", 0))[:2]):
                for k in STAT_KEYS:
                    if k in s:
                        row[f"{side}_{k}"] = s[k]
        except (requests.RequestException, ValueError):
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def goal_alerts(ls: pd.DataFrame, out: Path, now: pd.Timestamp) -> int:
    """Telegram message when the score changes in a match the owner starred (⭐ on the site).
    Stars come from the private Telegram backup; last seen scores live in <out>/goal_state.json.
    Checked every minute (--goals-only), so an alert comes within about a minute of the goal."""
    import json
    from soccer.backup import fetch
    from soccer.names import CLUB_ALIASES, similarity
    from soccer.notify import configured, send
    if ls.empty or not configured():
        return 0
    favs = [f for f in (fetch().get("favorites") or [])
            if abs((pd.Timestamp(f[1]) - now.tz_localize(None).normalize()).days) <= 1]
    if not favs:
        return 0
    state_f = out / "goal_state.json"
    state = json.loads(state_f.read_text(encoding="utf-8")) if state_f.exists() else {}
    alias = lambda n: CLUB_ALIASES.get(n, n)
    sent = 0
    from soccer.engine import Engine
    for r in ls.itertuples():
        if any(Engine.LS_NOT_MEN_TEAM.search(str(x)) for x in (r.home_raw, r.away_raw))                 or Engine.LS_NOT_MEN.search(str(getattr(r, "competition", "") or "")):
            continue  # youth / women's game of the same clubs
        fav = next((f for f in favs if min(similarity(alias(r.home_raw), f[2]),
                                           similarity(alias(r.away_raw), f[3])) >= 0.75), None)
        if fav is None or pd.isna(r.hg):
            continue
        score, old = f"{int(r.hg)}:{int(r.ag)}", state.get(str(r.ls_id))
        state[str(r.ls_id)] = score
        if old is not None and old != score:
            minute = f", {int(r.minute)}'" if pd.notna(r.minute) else ""
            who = fav[2] if int(r.hg) > int(old.split(":")[0]) else fav[3]
            sent += send(f"⚽ <b>Гол!</b> {who}\n⭐ {fav[2]} <b>{score}</b> {fav[3]}{minute}")
    state_f.write_text(json.dumps(state), encoding="utf-8")
    return sent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/live")
    ap.add_argument("--goals-only", action="store_true",
                    help="only check scores of starred matches (run every minute)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    if args.goals_only:
        print("goal alerts sent:", goal_alerts(livescore_snapshot(stats=False), out, now))
        sys.exit(0)
    parts = []
    try:
        ls = livescore_snapshot()
        if len(ls):
            parts.append(ls.assign(source="livescore"))
            try:
                print("goal alerts sent:", goal_alerts(ls, out, now))
            except Exception as exc:
                print("goal alerts failed:", exc, file=sys.stderr)
    except Exception as exc:
        print("livescore failed:", exc, file=sys.stderr)
    try:
        mo = fetch_live_odds(max_age_s=0)
        if len(mo):
            parts.append(mo.assign(source="misli"))
    except Exception as exc:
        print("misli failed:", exc, file=sys.stderr)
    if not parts:
        print("nothing live")
        sys.exit(0)
    snap = pd.concat(parts, ignore_index=True).assign(taken_at=now)
    f = out / f"live_snapshots_{now:%Y-%m}.jsonl"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(snap.to_json(orient="records", lines=True, force_ascii=False, date_format="iso"))
    print(f"{len(snap)} rows -> {f}")
