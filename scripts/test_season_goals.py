r"""Does following the current season's goal level improve totals?  (2026-10-04)

The 2026/27 season started with more goals (Jul-Oct 2.89 per match vs 2.70-2.75 before) and
the model, fitted on several seasons, lags. Walk-forward test on bt_all league forecasts:
before each match day, offset = how far this season's finished matches went over/under
our (calibrated) over-2.5 forecasts, as a MAP logit shift:
    pooled over all leagues (prior n0) + league-specific on top (prior n1)
then the over-2.5 chance of the day's matches is shifted by it. Tune n0/n1 on 2023-2024,
test on 2025 and the started 2026 season.
    python scripts/test_season_goals.py
"""
import json
import sys

import numpy as np
import pandas as pd

from soccer import storage
from soccer.config import DATA_DIR
from soccer.probability import over_prob, rescale_to_outcomes, score_matrix


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def load():
    p = storage.load_predictions("bt_all")
    p = p[(p.model == "final") & p.result.notna() & p.xg_home.notna()].copy()
    m = rescale_to_outcomes(score_matrix(p.xg_home.to_numpy(), p.xg_away.to_numpy(), -0.05),
                            p[["p_home", "p_draw", "p_away"]].to_numpy())
    cal = json.loads((DATA_DIR / "goals_calibration.json").read_text())
    z = cal["a"] + cal["b"] * logit(over_prob(m)) + p.league.map(cal["league"]).fillna(0).to_numpy()
    p["p"] = 1 / (1 + np.exp(-z))           # what production shows today (before this fix)
    p["y"] = ((p.hg + p.ag) > 2.5).astype(float)
    return p.sort_values("date").reset_index(drop=True)


def season_offsets(p: pd.DataFrame, n0: float, n1: float) -> np.ndarray:
    """Offset for each match from earlier days of the same season only."""
    off = np.zeros(len(p))
    for s, g in p.groupby("season"):
        pool_r = pool_v = 0.0
        lg_r, lg_v = {}, {}
        for d, day in g.groupby("date", sort=True):
            o0 = pool_r / (pool_v + n0 * 0.25)
            for i, lg in zip(day.index, day.league):
                q = 1 / (1 + np.exp(-(logit(p.at[i, "p"]) + o0)))
                o1 = lg_r.get(lg, 0.0) / (lg_v.get(lg, 0.0) + n1 * 0.25)
                off[i] = o0 + o1
            r = day.y - day.p
            v = day.p * (1 - day.p)
            pool_r += r.sum()
            pool_v += v.sum()
            for lg, gg in day.groupby("league"):
                q = 1 / (1 + np.exp(-(logit(gg.p) + o0)))
                lg_r[lg] = lg_r.get(lg, 0.0) + (gg.y - q).sum()
                lg_v[lg] = lg_v.get(lg, 0.0) + (q * (1 - q)).sum()
    return off


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p = load()
    fit = (p.season <= 2024).to_numpy()
    best = None
    for n0 in (100, 300, 1000, 3000):
        for n1 in (100, 300, 1000, 1e9):
            off = season_offsets(p, n0, n1)
            q = 1 / (1 + np.exp(-(logit(p.p) + off)))
            v = ll(q[fit], p.y[fit])
            if best is None or v < best[0]:
                best = (v, n0, n1)
    _, n0, n1 = best
    off = season_offsets(p, n0, n1)
    q = 1 / (1 + np.exp(-(logit(p.p) + off)))
    print(f"подобрано на 2023-24: n0={n0}, n1={n1:g}")
    for s in (2023, 2024, 2025, 2026):
        sel = (p.season == s).to_numpy()
        print(f"  сезон {s}: n={sel.sum():5d} | ТБ: прогноз {p.p[sel].mean():.3f} -> {q[sel].mean():.3f}, "
              f"факт {p.y[sel].mean():.3f} | log loss {ll(p.p[sel], p.y[sel]):.4f} -> {ll(q[sel], p.y[sel]):.4f}")
