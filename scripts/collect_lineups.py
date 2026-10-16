r"""Collect starting line-ups of past league games from livescore.com (available from the
2025/26 season) for the line-up experiment (scripts/test_lineups.py).

    python scripts/collect_lineups.py [--start 2025-08-01]

Cache: data/raw/livescore_days/<date>.json (day lists), data/raw/ls_lineups/<eid>.json.
Output: data/raw/lineups.csv - one row per player: eid, date, competition, home_raw,
away_raw, hg, ag, side (1 home / 2 away), pid, name, pos (1-4 starter by line, 5 bench).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from soccer.config import DATA_DIR
from soccer.engine import Engine
from soccer.livescore import _events, _get

DAYS = DATA_DIR / "raw" / "livescore_days"
LU = DATA_DIR / "raw" / "ls_lineups"
OUT = DATA_DIR / "raw" / "lineups.csv"


def day_events(day: pd.Timestamp) -> list:
    f = DAYS / f"{day:%Y%m%d}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    ev = _events(_get(f"date/soccer/{day:%Y%m%d}/0?MD=1"))
    for e in ev:
        e["kickoff"] = None if pd.isna(e["kickoff"]) else e["kickoff"].isoformat()
    if day < pd.Timestamp.now().normalize() - pd.Timedelta(days=1):
        DAYS.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(ev), encoding="utf-8")
    return ev


def lineup(eid: str):
    f = LU / f"{eid}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    for attempt in range(3):
        try:
            d = _get(f"lineups/soccer/{eid}") or {}
            break
        except Exception:
            time.sleep(2 + 3 * attempt)
    else:
        return None
    LU.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d), encoding="utf-8")
    return d


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-08-01")
    a = ap.parse_args()
    countries = set(Engine.LS_COUNTRY.values())
    days = pd.date_range(a.start, pd.Timestamp.now().normalize() - pd.Timedelta(days=1))
    games = []
    for i, d in enumerate(days):
        try:
            ev = day_events(d)
        except Exception as exc:
            print(d.date(), "day list failed:", exc)
            continue
        for e in ev:
            country, _, stage = e["competition"].partition(" · ")
            if (country in countries and e.get("ended") and not Engine.LS_NOT_MEN.search(stage)
                    and not Engine.LS_CUP.search(stage)
                    and not any(Engine.LS_NOT_MEN_TEAM.search(t) for t in (e["home_raw"], e["away_raw"]))):
                games.append({"eid": str(e["id"]).removeprefix("ls"), "date": d.date(),
                              "competition": e["competition"], "home_raw": e["home_raw"],
                              "away_raw": e["away_raw"], "hg": e.get("hg"), "ag": e.get("ag")})
        if i % 30 == 0:
            print(d.date(), "games so far", len(games), flush=True)
    print("games:", len(games), flush=True)
    rows = []
    with ThreadPoolExecutor(4) as pool:
        for g, d in zip(games, pool.map(lambda g: lineup(g["eid"]), games)):
            for t in (d or {}).get("Lu", []):
                for p in t.get("Ps", []):
                    if p.get("Pos") in (1, 2, 3, 4, 5):
                        rows.append({**g, "side": t.get("Tnb"), "pid": p.get("Pid"),
                                     "name": f"{p.get('Fn', '')} {p.get('Ln', '')}".strip(),
                                     "pos": p.get("Pos")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("players rows:", len(df), "| games with line-ups:", df["eid"].nunique() if len(df) else 0)
