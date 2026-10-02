"""API-Football (api-sports.io): injuries / suspensions and line-ups.

Free plan: 100 requests per day; past seasons 2022-2024 by league, the current season by date.
Every response is cached on disk so the daily quota is never spent twice on the same query.
The key is read from .env / environment (API_FOOTBALL_KEY) and never written by the code.
"""
from __future__ import annotations

import hashlib
import json
import time

import pandas as pd
import requests

from .config import RAW_DIR

BASE = "https://v3.football.api-sports.io"
CACHE = RAW_DIR / "APIF"

# API-Football league id -> our league code
LEAGUE_IDS = {39: "E0", 140: "SP1", 135: "I1", 78: "D1", 61: "F1", 40: "E1", 88: "N1",
              94: "P1", 203: "T1", 179: "SC0"}
HISTORY_SEASONS = (2022, 2023, 2024)
MIN_INTERVAL_S = 6.5
_last_call = 0.0


def _key() -> str | None:
    from .notify import load_config
    k = load_config().get("API_FOOTBALL_KEY") or ""
    return k if k and "ВСТАВЬТЕ" not in k else None


def available() -> bool:
    return _key() is not None


def get(endpoint: str, params: dict, max_age_h: float | None = None):
    """Cached GET. max_age_h=None -> cache forever (finished seasons)."""
    tag = hashlib.md5(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"{endpoint.strip('/').replace('/', '_')}_{tag}.json"
    if path.exists() and (max_age_h is None or
                          time.time() - path.stat().st_mtime < max_age_h * 3600):
        return json.loads(path.read_text(encoding="utf-8"))
    key = _key()
    if key is None:
        return None
    global _last_call
    for attempt in range(5):
        wait = MIN_INTERVAL_S - (time.time() - _last_call)  # free plan: 10 requests/minute
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        r = requests.get(f"{BASE}/{endpoint.strip('/')}", params=params,
                         headers={"x-apisports-key": key}, timeout=60)
        data = r.json() if r.status_code in (200, 429) else None
        limited = r.status_code == 429 or (isinstance(data, dict) and isinstance(
            data.get("errors"), dict) and "rateLimit" in data["errors"])
        if not limited:
            break
        time.sleep(61)  # per-minute limit hit: wait for the window to reset
    r.raise_for_status()
    if data.get("errors"):
        raise RuntimeError(f"API-Football: {data['errors']}")
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def requests_left() -> int | None:
    key = _key()
    if key is None:
        return None
    r = requests.get(f"{BASE}/status", headers={"x-apisports-key": key}, timeout=30).json()
    req = (r.get("response") or {}).get("requests") or {}
    return int(req.get("limit_day", 0)) - int(req.get("current", 0))


def _injury_rows(data) -> list[dict]:
    rows = []
    for x in (data or {}).get("response", []):
        fx, team, pl, lg = x.get("fixture") or {}, x.get("team") or {}, x.get("player") or {}, \
            x.get("league") or {}
        rows.append({"fixture_id": fx.get("id"), "date": fx.get("date"),
                     "league_id": lg.get("id"), "season": lg.get("season"),
                     "team_id": team.get("id"), "team": team.get("name"),
                     "player_id": pl.get("id"), "player": pl.get("name"),
                     "type": pl.get("type"), "reason": pl.get("reason")})
    return rows


def injuries_history(leagues=None, seasons=HISTORY_SEASONS) -> pd.DataFrame:
    """Injuries/suspensions per fixture for past seasons (1 request per league-season)."""
    rows = []
    for lid in (leagues or LEAGUE_IDS):
        for s in seasons:
            data = get("injuries", {"league": lid, "season": s})
            rows += _injury_rows(data)
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["league"] = df["league_id"].map(LEAGUE_IDS)
    return df


def injuries_on(date) -> pd.DataFrame:
    """Current-season injuries/suspensions for all fixtures on a date (refreshed every 3 h)."""
    data = get("injuries", {"date": pd.Timestamp(date).strftime("%Y-%m-%d")}, max_age_h=3)
    df = pd.DataFrame(_injury_rows(data))
    if len(df):
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df
