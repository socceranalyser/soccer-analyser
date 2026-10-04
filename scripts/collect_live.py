r"""Snapshot of every football match in play: minute, score, red cards, misli.az live odds and
livescore.com statistics (shots, shots on target, possession, corners, attacks...).

Run every 15 minutes by .github/workflows/live_collect.yml; rows are appended to
<out>/live_snapshots_YYYY-MM.csv (branch 'livedata', so the dashboard is not redeployed).
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


def livescore_snapshot() -> pd.DataFrame:
    rows = []
    data = _get("live/soccer/0?MD=1")
    for r in _events(data):
        if not r["live"]:
            continue
        eid = r["id"][2:]
        row = {k: r[k] for k in ("home_raw", "away_raw", "kickoff", "minute", "hg", "ag",
                                 "red_h", "red_a", "competition")}
        row["ls_id"] = eid
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/live")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    parts = []
    try:
        ls = livescore_snapshot()
        if len(ls):
            parts.append(ls.assign(source="livescore"))
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
    f = out / f"live_snapshots_{now:%Y-%m}.csv"
    snap.to_csv(f, mode="a", header=not f.exists(), index=False)
    print(f"{len(snap)} rows -> {f}")
