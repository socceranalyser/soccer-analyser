"""Scores and live matches from livescore.com (the public JSON its website uses).

Second, independent source next to misli.az: English team names, very wide coverage.
Returns frames in the same format as soccer.misli.fetch_results / fetch_live.
"""
from __future__ import annotations

import re

import pandas as pd
import requests

from .config import DATA_DIR

API = "https://prod-public-api.livescore.com/v1/api/app"
HEADERS = {"User-Agent": "Mozilla/5.0 (soccer-analyser personal use)", "Accept": "application/json",
           "Origin": "https://www.livescore.com", "Referer": "https://www.livescore.com/"}
ENDED = {"FT", "AET", "AP", "Pen.", "FT Pen", "AP."}
NOT_LIVE = {"NS", "Postp.", "Canc.", "Abd.", "Susp.", "TBA", "Int.", "Aband.", "Del."}
ARCHIVE = DATA_DIR / "livescore_results.csv"
COLS = ["home_raw", "away_raw", "kickoff", "status", "minute", "ended", "live", "hg", "ag",
        "competition"]


def _get(path: str):
    r = requests.get(f"{API}/{path}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _events(data) -> list[dict]:
    rows = []
    for stage in (data or {}).get("Stages", []):
        comp = f"{stage.get('Cnm', '')} · {stage.get('Snm', '')}".strip(" ·")
        for e in stage.get("Events", []):
            try:
                home, away = e["T1"][0]["Nm"], e["T2"][0]["Nm"]
            except (KeyError, IndexError):
                continue
            eps = str(e.get("Eps", ""))
            ended = eps in ENDED
            minute = None
            m = re.match(r"(\d+)", eps)
            if m and not ended:
                minute = int(m.group(1))
            live = not ended and (eps == "HT" or minute is not None)
            hg, ag = e.get("Tr1"), e.get("Tr2")
            hg = int(hg) if str(hg).isdigit() else None
            ag = int(ag) if str(ag).isdigit() else None
            kick = pd.to_datetime(str(e.get("Esd")), format="%Y%m%d%H%M%S", errors="coerce")
            rows.append({"home_raw": home, "away_raw": away,
                         "kickoff": kick.tz_localize("UTC") if pd.notna(kick) else pd.NaT,
                         "status": "перерыв" if eps == "HT" else ("2-й тайм" if (minute or 0) > 45
                                                                  else "1-й тайм" if live else eps),
                         "minute": minute, "ended": ended, "live": live, "hg": hg, "ag": ag,
                         "competition": comp, "red_h": int(e.get("Tr1RC") or 0) if str(
                             e.get("Tr1RC") or 0).isdigit() else 0,
                         "red_a": int(e.get("Tr2RC") or 0) if str(e.get("Tr2RC") or 0).isdigit() else 0,
                         "id": f"ls{e.get('Eid')}", "top": False})
    return rows


def fetch_results(dates) -> pd.DataFrame:
    """Finished and live scores for calendar dates (UTC), archived locally."""
    rows = []
    for d in dates:
        try:
            rows += _events(_get(f"date/soccer/{pd.Timestamp(d):%Y%m%d}/0?MD=1"))
        except (requests.RequestException, ValueError):
            continue
    df = pd.DataFrame(rows, columns=COLS + ["red_h", "red_a", "id", "top"])
    done = df[df["ended"] & df["hg"].notna()] if len(df) else df
    try:
        old = pd.read_csv(ARCHIVE) if ARCHIVE.exists() else pd.DataFrame(columns=COLS)
        allr = pd.concat([old, done[COLS]], ignore_index=True)
        allr["kickoff"] = pd.to_datetime(allr["kickoff"], utc=True, format="ISO8601")
        allr = allr.drop_duplicates(["home_raw", "away_raw", "kickoff"], keep="last")
        if len(done):
            allr.to_csv(ARCHIVE, index=False)
        lo = pd.Timestamp(min(dates)).tz_localize("UTC") - pd.Timedelta(days=1)
        hi = pd.Timestamp(max(dates)).tz_localize("UTC") + pd.Timedelta(days=2)
        old = allr[(allr["kickoff"] >= lo) & (allr["kickoff"] < hi)].assign(live=False)
        df = pd.concat([df, old], ignore_index=True) if len(df) else old
    except (OSError, ValueError):
        pass
    if df.empty:
        return df
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    df["ended"] = df["ended"].astype(bool)
    df["live"] = df["live"].fillna(False).astype(bool)
    return df.drop_duplicates(["home_raw", "away_raw", "kickoff"]).reset_index(drop=True)


def fetch_live() -> pd.DataFrame:
    rows = [r for r in _events(_get("live/soccer/0?MD=1")) if r["live"]]
    return pd.DataFrame(rows)
