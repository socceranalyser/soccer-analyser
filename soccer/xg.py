"""Expected goals (xG) per match from understat.com (top-5 leagues since 2014/15).

xG measures the quality of the chances a team created, not just whether they went in, so
it is a less noisy signal of strength than goals. Used by Dixon-Coles as a blend:
    effective goals = (1 - w) * goals + w * xG       (w = DixonColes.xg_weight)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR
from .names import best_match

UNDERSTAT = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "I1": "Serie_A", "F1": "Ligue_1"}
FIRST_SEASON = 2014
CACHE = DATA_DIR / "raw" / "understat"
ALIASES = {"FC Cologne": "FC Koln"}  # understat name -> football-data name
HEADERS = {"User-Agent": "Mozilla/5.0 (soccer-analyser personal use)",
           "X-Requested-With": "XMLHttpRequest"}


def _season(code: str, year: int, current: int) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{code}_{year}.json"
    if f.exists() and (year < current or time.time() - f.stat().st_mtime < 6 * 3600):
        return json.loads(f.read_text(encoding="utf-8"))
    lg = UNDERSTAT[code]
    r = requests.get(f"https://understat.com/getLeagueData/{lg}/{year}", timeout=60,
                     headers={**HEADERS, "Referer": f"https://understat.com/league/{lg}/{year}"})
    r.raise_for_status()
    rows = [{"date": d["datetime"][:10], "home": d["h"]["title"], "away": d["a"]["title"],
             "xg_h": float(d["xG"]["h"]), "xg_a": float(d["xG"]["a"])}
            for d in r.json()["dates"] if d.get("isResult")]
    f.write_text(json.dumps(rows), encoding="utf-8")
    time.sleep(1.0)
    return rows


def attach_xg(matches: pd.DataFrame) -> pd.DataFrame:
    """Add xg_h / xg_a (NaN where understat has no data)."""
    out = matches.copy()
    out["xg_h"], out["xg_a"] = np.nan, np.nan
    current = int(matches["season"].max())
    for code in UNDERSTAT:
        lm = out[out["league"] == code]
        for season, g in lm.groupby("season"):
            if season < FIRST_SEASON:
                continue
            try:
                us = pd.DataFrame(_season(code, int(season), current))
            except (requests.RequestException, ValueError, KeyError):
                continue
            if us.empty:
                continue
            teams = sorted(set(g["home"]) | set(g["away"]))
            names = {t: best_match(ALIASES.get(t, t), teams, threshold=0.6)[0] for t in set(us["home"]) | set(us["away"])}
            us["home"], us["away"] = us["home"].map(names), us["away"].map(names)
            us = us.dropna(subset=["home", "away"])
            us["date"] = pd.to_datetime(us["date"])
            key = g.reset_index()[["index", "date", "home", "away"]]
            m = key.merge(us, on=["home", "away"], how="inner")
            m = m[(m["date_x"] - m["date_y"]).abs() <= pd.Timedelta(days=1)]
            out.loc[m["index"], "xg_h"] = m["xg_h"].to_numpy()
            out.loc[m["index"], "xg_a"] = m["xg_a"].to_numpy()
    return out
