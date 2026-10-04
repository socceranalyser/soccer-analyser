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


LIVE_TIP_MIN = 0.72    # without odds: show the outcomes the model gives at least this
LIVE_VALUE_MIN = 0.05  # with misli live odds: "выгодно" when p * odds - 1 >= 5%
LABELS = {"o1": "П1", "ox": "Х", "o2": "П2", "o1x": "1X", "o12": "12", "ox2": "X2",
          "o_over": "ТБ 2.5", "o_under": "ТМ 2.5", "o_btts_yes": "Обе забьют — да",
          "o_btts_no": "Обе забьют — нет"}


def live_suggestion(lam, mu, minute, hg, ag, red_h=0, red_a=0, odds: dict | None = None) -> str:
    """One line for the live panel. With live odds: the best bet whose chance is >= 55% and
    worth at least 5% more than its fair price; otherwise the most likely outcomes with their
    fair odds (1/p), so they can be compared with the bookmaker by eye."""
    m = score_matrix_live(lam, mu, minute, hg, ag, red_h, red_a)
    mk = live_markets(m)
    if odds:
        i, j = np.indices(m.shape)
        cands = [(LABELS[k], mk[k], float(odds[k])) for k in
                 ("o1", "ox", "o2", "o1x", "o12", "ox2", "o_btts_yes", "o_btts_no")
                 if odds.get(k) is not None and np.isfinite(odds[k]) and odds[k] > 1.01]
        for col in [c for c in odds if str(c).startswith("o_over_")]:
            line = float(col.split("_")[-1])
            over = float(m[i + j > line].sum())
            for name, p, o in ((f"ТБ {line:g}", over, odds.get(col)),
                               (f"ТМ {line:g}", 1 - over, odds.get(f"o_under_{line:g}"))):
                if o is not None and np.isfinite(o) and o > 1.01:
                    cands.append((name, p, float(o)))
        good = [c for c in cands if 0.55 <= c[1] < 0.98 and c[1] * c[2] - 1 >= LIVE_VALUE_MIN]
        if good:
            name, p, o = max(good, key=lambda c: c[1] * c[2])
            return f"⚡ live: {name} @ {o:.2f} — мой шанс {p:.0%}, выгодно (+{p * o - 1:.0%})"
    picks = [(k, p) for k, p in mk.items() if LIVE_TIP_MIN <= p < 0.97]
    if not picks:
        return "⚡ live: сейчас выгодных ставок не вижу" if odds else ""
    fam = {"o1": "x", "ox": "x", "o2": "x", "o1x": "x", "o12": "x", "ox2": "x",
           "o_over": "t", "o_under": "t", "o_btts_yes": "b", "o_btts_no": "b"}
    best = {}
    for k, p in sorted(picks, key=lambda kp: kp[1]):
        best[fam[k]] = (k, p)
    parts = [f"{LABELS[k]} {p:.0%} (честный кф {1 / p:.2f})"
             for k, p in sorted(best.values(), key=lambda kp: -kp[1])[:2]]
    return ("⚡ live (выгодных по misli нет): " if odds else "⚡ live сейчас: ") + " · ".join(parts)


# ---------------------------------------------------------------- reading the match itself
# Chance quality from in-match statistics: an average shot on target is worth ~0.30 goals, a
# shot off target / blocked ~0.07 (typical xG values). The pre-match expectation is updated
# with what the teams actually created, weighted by minutes played (STATS_PRIOR_MIN = how
# many minutes of evidence count as much as the pre-match forecast).
# START VALUES: being checked on the snapshots collected by scripts/collect_live.py.
XG_ON, XG_OFF = 0.30, 0.07
STATS_PRIOR_MIN = 90.0


def created_xg(s: dict) -> float:
    return XG_ON * (s.get("Shon") or 0) + XG_OFF * ((s.get("Shof") or 0) + (s.get("Shbl") or 0))


def stats_adjust(lam: float, mu: float, minute: float, stats: dict | None) -> tuple[float, float, dict]:
    """Pre-match goal rates updated with the chances created so far (per 90 minutes)."""
    if not stats or minute < 10:
        return lam, mu, {}
    w = minute / (minute + STATS_PRIOR_MIN)
    xh, xa = created_xg(stats["h"]), created_xg(stats["a"])
    lam2 = (1 - w) * lam + w * xh * 90 / minute
    mu2 = (1 - w) * mu + w * xa * 90 / minute
    return lam2, mu2, {"xh": xh, "xa": xa, "w": w}


def describe(minute, hg, ag, stats, red_h, red_a, home, away, info) -> str:
    """A plain-Russian reading of the match so far."""
    bits = [f"{int(minute)}', {hg}:{ag}"]
    if red_h or red_a:
        bits.append(("у " + home if red_h else "у " + away) + " удаление — в меньшинстве")
    if stats:
        h, a = stats["h"], stats["a"]
        on_h, on_a = h.get("Shon") or 0, a.get("Shon") or 0
        pss = h.get("Pss")
        line = f"удары в створ {on_h}:{on_a}"
        if pss:
            line += f", владение {pss}% — {100 - pss}%"
        bits.append(line)
        xh, xa = info.get("xh", 0), info.get("xa", 0)
        if xh - xa > 0.6 and hg <= ag:
            bits.append(f"{home} создаёт больше, чем говорит счёт — их гол вероятнее")
        elif xa - xh > 0.6 and ag <= hg:
            bits.append(f"{away} создаёт больше, чем говорит счёт — их гол вероятнее")
        elif xh + xa < 0.4 * minute / 45:
            bits.append("моментов мало — игра закрытая")
        elif xh + xa > 1.2 * minute / 45:
            bits.append("много моментов — игра открытая")
    else:
        bits.append("статистики по матчу нет — учитываю только счёт, время и карточки")
    return "; ".join(bits)


def live_analysis(lam, mu, minute, hg, ag, red_h=0, red_a=0, odds: dict | None = None,
                  stats: dict | None = None, home: str = "хозяева", away: str = "гости") -> tuple[str, str]:
    """(reading of the match, suggestion) for the live panel."""
    lam2, mu2, info = stats_adjust(lam, mu, minute, stats)
    text = describe(minute, hg, ag, stats, red_h, red_a, home, away, info)
    return text, live_suggestion(lam2, mu2, minute, hg, ag, red_h, red_a, odds)
