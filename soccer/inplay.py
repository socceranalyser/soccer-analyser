"""In-play (live) probabilities: the rest of the match from the current minute and score.

Pre-match expected goals (lam, mu) come from our model. For the time left the goal rates are
scaled by the share of a match's goals still to come (more goals fall in the second half),
by red cards, and by the game state (the trailing side pushes, the leader sits back).
Remaining goals ~ independent Poisson; final score = current score + remaining goals.
Parameters fitted on half-time scores of 2023-24 league matches, checked on 2025
(scripts/test_inplay.py).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

SECOND_HALF_SHARE = 0.57   # share of goals scored after half-time (fitted 2023-24)
GAME_STATE = 0.10          # leader sits back, trailing side pushes (fitted 2023-24)
RED_OWN = 0.30             # a team down to 10 men scores 30% less ...
RED_OPP = 0.25             # ... and concedes 25% more
MAX_REMAINING = 10


def remaining_share(minute: float, first_half_share: float | None = None) -> float:
    """Share of the match's expected goals still to come after `minute` (0..90+)."""
    s2 = SECOND_HALF_SHARE if first_half_share is None else 1 - first_half_share
    s1 = 1 - s2
    m = float(np.clip(minute, 0, 95))
    if m <= 45:
        return s1 * (45 - m) / 45 + s2
    return max(s2 * (90 - m) / 45, 0.01)


def score_matrix_live(lam: float, mu: float, minute: float, hg: int, ag: int,
                      red_h: int = 0, red_a: int = 0, game_state: float | None = None,
                      second_half_share: float | None = None) -> np.ndarray:
    """Probability matrix of the FINAL score (rows home goals, cols away goals)."""
    gs = GAME_STATE if game_state is None else game_state
    share = remaining_share(minute, None if second_half_share is None else 1 - second_half_share)
    lh, la = lam * share, mu * share
    lh *= (1 - RED_OWN) ** red_h * (1 + RED_OPP) ** red_a
    la *= (1 - RED_OWN) ** red_a * (1 + RED_OPP) ** red_h
    if hg > ag:
        lh, la = lh * (1 - gs), la * (1 + gs)
    elif ag > hg:
        lh, la = lh * (1 + gs), la * (1 - gs)
    k = np.arange(MAX_REMAINING + 1)
    ph, pa = poisson.pmf(k, lh), poisson.pmf(k, la)
    n = hg + ag + MAX_REMAINING * 2 + 1
    m = np.zeros((hg + MAX_REMAINING + 1, ag + MAX_REMAINING + 1))
    m[hg:, ag:] = np.outer(ph, pa)
    return m / m.sum()


def live_markets(m: np.ndarray, line: float = 2.5) -> dict:
    i, j = np.indices(m.shape)
    return {"o1": float(m[i > j].sum()), "ox": float(m[i == j].sum()), "o2": float(m[i < j].sum()),
            "o1x": float(m[i >= j].sum()), "o12": float(m[i != j].sum()), "ox2": float(m[i <= j].sum()),
            "o_over": float(m[i + j > line].sum()), "o_under": float(m[i + j < line].sum()),
            "o_btts_yes": float(m[1:, 1:].sum()), "o_btts_no": float(1 - m[1:, 1:].sum())}
