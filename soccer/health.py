"""Daily data sanity check: catch broken data BEFORE it turns into wrong forecasts.

Born from real incidents (2026-10): women's results stored as Premier League games, old cup
matches dated years ahead, a source silently stopping. Each check returns plain-Russian
warnings; scripts/cloud_job.py sends them to Telegram and writes data/state/health.json.
"""
from __future__ import annotations

import json

import pandas as pd

from .config import DATA_DIR, LEAGUES

HEALTH_FILE = DATA_DIR / "state" / "health.json"


def check(engine) -> list[str]:
    today = pd.Timestamp.now().normalize()
    m = engine.matches
    out = []

    fut = m[m["date"] > today + pd.Timedelta(days=1)]
    if len(fut):
        out.append(f"{len(fut)} результатов с датой в будущем (напр. {fut.iloc[0]['home']} — "
                   f"{fut.iloc[0]['away']}, {fut.iloc[0]['date']:%d.%m.%Y})")

    recent = m[m["date"] >= today - pd.Timedelta(days=60)]
    games = pd.concat([recent[["league", "date", "home"]].rename(columns={"home": "team"}),
                       recent[["league", "date", "away"]].rename(columns={"away": "team"})])
    games = games.sort_values("date")
    gap = games.groupby(["league", "team"])["date"].diff().dt.days
    twice = games[gap < 2]
    if len(twice):
        ex = twice.iloc[0]
        out.append(f"{len(twice)} случаев «команда сыграла дважды за 2 дня» — похоже на чужие "
                   f"результаты (напр. {ex['team']}, {LEAGUES.get(ex['league'], {}).get('name', ex['league'])}, "
                   f"{ex['date']:%d.%m})")

    s = engine.schedule
    past = s[(s["kind"] == "league") & (s["date"] < today - pd.Timedelta(days=3))
             & (s["date"] >= today - pd.Timedelta(days=21)) & ~s["played"].astype(bool)]
    if len(past):
        have = set(zip(m["home"], m["away"]))
        missing = past[[(h, a) not in have for h, a in zip(past["home"], past["away"])]]
        by = missing.groupby("competition").size().sort_values(ascending=False)
        big = by[by >= 3]
        if len(big):
            names = ", ".join(f"{LEAGUES.get(k, {}).get('name', k)} ({v})" for k, v in big.head(5).items())
            out.append(f"нет результатов прошедших матчей больше 3 дней: {names}")
    return out


def run(engine) -> list[str]:
    warnings = check(engine)
    HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_FILE.write_text(json.dumps({"checked": pd.Timestamp.now().isoformat(timespec="minutes"),
                                       "warnings": warnings}, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    return warnings
