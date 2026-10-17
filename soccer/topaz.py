"""Topaz (etopaz.az) pre-match odds - the second legal bookmaker in Azerbaijan.

Public JSON of its site (odds from Sportradar): /feed/events?sportTypeId=1:sr:sport:1&
eventType=PREMATCH&startDate=<ms>&endDate=<ms>&offset=&limit=50. Markets (Sportradar ids):
1 = 1X2 (outcomes 1/2/3), 10 = double chance (9 = 1X, 10 = 12, 11 = X2), 18 = total with
specifier total (12 = over, 13 = under), 29 = both score (74 = yes, 76 = no).
Used to compare with misli.az: bet where the price is higher, and spot the rare cases where
the two books together give a profit whatever happens.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

URL = "https://ops.etopaz.az/feed/events"
HEADERS = {"User-Agent": "Mozilla/5.0 (soccer-analyser)"}
OUT = {("1:1", "1:1"): "o1", ("1:1", "1:2"): "ox", ("1:1", "1:3"): "o2",
       ("1:10", "1:9"): "o1x", ("1:10", "1:10"): "o12", ("1:10", "1:11"): "ox2",
       ("1:29", "1:74"): "o_btts_yes", ("1:29", "1:76"): "o_btts_no"}


def fetch_events(days: int = 3) -> pd.DataFrame:
    """One row per football match: kickoff (UTC), home, away, odds columns like misli's."""
    now = int(time.time() * 1000)
    rows, off = [], 0
    while True:
        r = requests.get(URL, params={"sportTypeId": "1:sr:sport:1", "eventType": "PREMATCH",
                                      "startDate": now, "endDate": now + days * 86400 * 1000,
                                      "offset": off, "limit": 50}, headers=HEADERS, timeout=30)
        r.raise_for_status()
        batch = [(s, e) for s in r.json()["item"]["seasons"] for e in s["events"]]
        for s, e in batch:
            row = {"tz_id": e["id"], "kickoff": pd.Timestamp(e["startedAt"], unit="s", tz="UTC"),
                   "home": e["teams"]["home"]["name"], "away": e["teams"]["away"]["name"],
                   "tournament": f"{s.get('categoryName', '')} · {s.get('seasonName', '')}"}
            for m in e.get("markets", []):
                if m.get("status") != 0:
                    continue
                if m["marketRefId"] == "1:18":
                    if str((m.get("specifiers") or {}).get("total")) == "2.5":
                        for o in m["outcomes"]:
                            k = {"1:12": "o_over", "1:13": "o_under"}.get(o["id"])
                            if k and o.get("status") == 0:
                                row[k] = o["odd"]
                    continue
                for o in m["outcomes"]:
                    k = OUT.get((m["marketRefId"], o["id"]))
                    if k and o.get("status") == 0:
                        row[k] = o["odd"]
            rows.append(row)
        if len(batch) < 50:
            break
        off += 50
    return pd.DataFrame(rows)


def link(misli: pd.DataFrame, topaz: pd.DataFrame) -> pd.DataFrame:
    """misli events + the same match's Topaz odds (columns tz_*): same kick-off (+-10 min)
    and both team names alike (misli's Azerbaijani spelling is transliterated first)."""
    from .misli import az_to_latin
    from .names import similarity
    out = misli.copy()
    if topaz.empty or misli.empty:
        return out
    cols = [c for c in topaz.columns if c.startswith("o")]
    found = []
    for _, r in misli.iterrows():
        cand = topaz[(topaz["kickoff"] - r["kickoff"]).abs() <= pd.Timedelta(minutes=10)]
        best, score = None, 0.6
        for i, t in cand.iterrows():
            sh = max(similarity(az_to_latin(r["home_raw"]), t["home"]),
                     similarity(str(r.get("home") or ""), t["home"]))
            sa = max(similarity(az_to_latin(r["away_raw"]), t["away"]),
                     similarity(str(r.get("away") or ""), t["away"]))
            s = min(sh, sa)
            if s >= score:
                best, score = i, s
        found.append(topaz.loc[best, cols].add_prefix("tz_") if best is not None else None)
    tz = pd.DataFrame([f.to_dict() if f is not None else {} for f in found], index=misli.index)
    return pd.concat([out, tz], axis=1)
