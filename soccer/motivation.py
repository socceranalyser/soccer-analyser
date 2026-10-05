"""End-of-season motivation: a side with nothing left to play for is weaker than its rating.

From the current table (results so far) and games left L, a team can gain at most 3L points:
    chasing - can still reach the top quarter (title / European places)
    danger  - can still drop into the bottom 3
    nothing - neither ("dead rubber")
Only within the last LAST_ROUNDS rounds. scripts/test_motivation.py (bt_all, last 10 rounds):
fit 2023-24, test 2025 (4 000 matches): 1.0232 -> 1.0227 with +0.071 log-odds for the side whose
opponent has nothing to play for; the relegation fight showed no effect; the bookmaker already
prices it (weight ~0).
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

BETA_NOTHING = 0.071
LAST_ROUNDS = 10


def season_flags(results: pd.DataFrame, teams) -> dict:
    """{team: (games_left, nothing_to_play_for)} from one league-season's results so far."""
    teams = sorted(set(teams))
    n = len(teams)
    if n < 10:
        return {}
    pts, played = defaultdict(int), defaultdict(int)
    for r in results.itertuples():
        played[r.home] += 1
        played[r.away] += 1
        if r.hg > r.ag:
            pts[r.home] += 3
        elif r.hg < r.ag:
            pts[r.away] += 3
        else:
            pts[r.home] += 1
            pts[r.away] += 1
    p_sorted = sorted((pts[t] for t in teams), reverse=True)
    top_line, rel_line, mid = p_sorted[max(n // 4 - 1, 0)], p_sorted[n - 3], p_sorted[n // 2]
    out = {}
    for t in teams:
        left = 2 * (n - 1) - played[t]
        chasing = pts[t] + 3 * left >= top_line
        danger = pts[t] - rel_line <= 3 * left and pts[t] <= mid
        out[t] = (left, not chasing and not danger)
    return out


def shift(flags: dict, home: str, away: str) -> float:
    """Log-odds shift for the home win (and minus it for the away win)."""
    if home not in flags or away not in flags:
        return 0.0
    (lh, nh), (la, na) = flags[home], flags[away]
    if min(lh, la) > LAST_ROUNDS:
        return 0.0
    return BETA_NOTHING * (float(na) - float(nh))
