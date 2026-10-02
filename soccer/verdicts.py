"""Plain-language calls from probabilities: who wins, over/under 2.5, both teams to score.

Each call names the outcome(s) it covers, so it can be scored against the final result.
When nobody is a clear favourite the outcome call falls back to a double chance
("X will not lose"), which is what a careful tipster would say.
"""
from __future__ import annotations

import numpy as np

CLEAR_FAVOURITE = 0.50  # below this a single outcome is a coin-flip -> double chance


def outcome_call(p, home: str, away: str) -> tuple[str, frozenset, float]:
    """(label, covered results {0=home,1=draw,2=away}, probability of the call)."""
    p = np.asarray(p, float)
    k = int(np.argmax(p))
    if p[k] >= CLEAR_FAVOURITE and k != 1:
        return f"Победа {home if k == 0 else away}", frozenset({k}), float(p[k])
    hx, xa = p[0] + p[1], p[1] + p[2]
    if hx >= xa:
        return f"{home} не проиграет", frozenset({0, 1}), float(hx)
    return f"{away} не проиграет", frozenset({1, 2}), float(xa)


def total_call(p_over: float, line: float = 2.5) -> tuple[str, bool, float]:
    """(label, True if the call is 'over', probability)."""
    return (f"Больше {line:g}", True, p_over) if p_over >= 0.5 else \
        (f"Меньше {line:g}", False, 1 - p_over)


def btts_call(p_btts: float) -> tuple[str, bool, float]:
    return ("Да", True, p_btts) if p_btts >= 0.5 else ("Нет", False, 1 - p_btts)


def result_index(hg: int, ag: int) -> int:
    return 0 if hg > ag else 1 if hg == ag else 2


def score_calls(p, p_over, p_btts, hg, ag, line: float = 2.5) -> dict:
    """Which of the three calls came true for a final score."""
    _, covered, _ = outcome_call(p, "", "")
    out = {"outcome": result_index(hg, ag) in covered}
    if p_over is not None and not np.isnan(p_over):
        out["total"] = (hg + ag > line) == total_call(p_over, line)[1]
    if p_btts is not None and not np.isnan(p_btts):
        out["btts"] = (hg > 0 and ag > 0) == btts_call(p_btts)[1]
    return out
