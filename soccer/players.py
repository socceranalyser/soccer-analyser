"""How important is a missing player? Minutes and attacking output from understat.com.

importance = share of the season's minutes the player played (a regular starter ~1, a
squad player ~0.2) + k * (xG + xA per 90) * share  (k fitted in scripts/test_player_importance.py).
Last season's numbers are used for a match (no peeking into the season being predicted),
players without last-season data in the top-5 leagues get UNKNOWN_SHARE.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from functools import lru_cache

import pandas as pd
import requests

from .xg import CACHE, HEADERS, UNDERSTAT

UNKNOWN_SHARE = 0.3


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z ]+", " ", s).split())


def season_players(code: str, year: int) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{code}_{year}_players.json"
    if f.exists():
        rows = json.loads(f.read_text(encoding="utf-8"))
    else:
        lg = UNDERSTAT[code]
        r = requests.get(f"https://understat.com/getLeagueData/{lg}/{year}", timeout=60,
                         headers={**HEADERS, "Referer": f"https://understat.com/league/{lg}/{year}"})
        r.raise_for_status()
        rows = [{"name": p["player_name"], "team": p["team_title"], "time": int(p["time"]),
                 "xg": float(p["xG"]), "xa": float(p["xA"])} for p in r.json()["players"]]
        f.write_text(json.dumps(rows), encoding="utf-8")
        time.sleep(1.0)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["share"] = (df["time"] / df["time"].max()).clip(0, 1)
    df["att90"] = (df["xg"] + df["xa"]) / (df["time"].clip(lower=90) / 90)
    df["key"] = df["name"].map(_norm)
    return df


@lru_cache(maxsize=8)
def season_table(year: int) -> pd.DataFrame:
    """All top-5-league players of a season (a player who moved leagues appears twice)."""
    parts = []
    for code in UNDERSTAT:
        try:
            parts.append(season_players(code, year).assign(league=code))
        except (requests.RequestException, ValueError, KeyError):
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def lookup(name: str, year: int) -> dict | None:
    """Last-season record for an API-Football name like 'J. Butland' / 'K. De Bruyne' / 'Rodri'."""
    tab = season_table(year)
    if tab.empty:
        return None
    n = _norm(name)
    m = re.match(r"^([a-z]) (.+)$", n)
    if m:
        ini, sur = m.groups()
        hit = tab[tab["key"].str.endswith(" " + sur) & tab["key"].str.startswith(ini)]
    else:
        hit = tab[(tab["key"] == n) | tab["key"].str.endswith(" " + n)
                  | tab["key"].str.startswith(n + " ")]
    if hit.empty:
        return None
    best = hit.sort_values("time", ascending=False).iloc[0]   # the regular, if ambiguous
    agg = hit[hit["key"] == best["key"]]                       # same player in two leagues
    return {"share": float(min(1.0, agg["share"].sum())),
            "att90": float((agg["xg"].sum() + agg["xa"].sum()) / max(agg["time"].sum() / 90, 1))}


def importance(name: str, year: int, k: float) -> float:
    rec = lookup(name, year - 1)
    if rec is None:
        return UNKNOWN_SHARE
    return rec["share"] * (1 + k * rec["att90"])
