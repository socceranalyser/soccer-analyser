r"""Motivation at the end of the season: does "nothing left to play for" change results?

For every league match the table is rebuilt from results before that day (same league and
season). With L games left a team can still gain at most 3L points:
    chasing  - can still reach the top quarter of the table (title / European places)
    danger   - can still end up in the bottom 3 (its lead over 3rd-from-bottom <= 3L)
    nothing  - neither: mid-table, safe, cannot reach the top quarter ("dead rubber")
Feature for a match: nothing_away - nothing_home (the side with nothing to play for is
expected to be weaker), and danger_home - danger_away (a relegation fight lifts a side).
Log-odds shift on home/away win, only in the last 10 rounds. Fit on 2023-2024, test on 2025,
for our model and for the bookmaker.
    python scripts/test_motivation.py
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from soccer import storage
from soccer.data import load_matches
from soccer.metrics import log_loss

C = ["p_home", "p_draw", "p_away"]
LAST_ROUNDS = 10


def motivation_table(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lg, s), g in m.groupby(["league", "season"]):
        teams = sorted(set(g.home) | set(g.away))
        n = len(teams)
        if n < 10:
            continue
        total = 2 * (n - 1)
        pts, played = defaultdict(int), defaultdict(int)
        for day, dg in g.sort_values("date").groupby("date"):
            table = sorted(teams, key=lambda t: -pts[t])
            p_sorted = [pts[t] for t in table]
            top_line = p_sorted[max(n // 4 - 1, 0)]
            rel_line = p_sorted[n - 3]          # 3rd from bottom
            for r in dg.itertuples():
                feat = {}
                for side, t in (("h", r.home), ("a", r.away)):
                    left = total - played[t]
                    chasing = pts[t] + 3 * left >= top_line
                    danger = pts[t] - rel_line <= 3 * left and pts[t] <= p_sorted[n // 2]
                    feat[f"left_{side}"] = left
                    feat[f"nothing_{side}"] = float(not chasing and not danger)
                    feat[f"danger_{side}"] = float(danger)
                rows.append({"league": lg, "date": day, "home": r.home, "away": r.away, **feat})
            for r in dg.itertuples():  # update after the whole day (no peeking within a day)
                played[r.home] += 1
                played[r.away] += 1
                if r.hg > r.ag:
                    pts[r.home] += 3
                elif r.hg < r.ag:
                    pts[r.away] += 3
                else:
                    pts[r.home] += 1
                    pts[r.away] += 1
    return pd.DataFrame(rows)


def adjust(p, X, b):
    z = np.log(np.clip(p, 1e-9, 1))
    z[:, 0] += X @ b
    z[:, 2] -= X @ b
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    m = load_matches(seasons=[2023, 2024, 2025])
    mt = motivation_table(m)
    preds = storage.load_predictions("bt_all").dropna(subset=["result"])
    for model in ("final", "market"):
        p = preds[preds.model == model].merge(mt, on=["league", "date", "home", "away"])
        late = (np.minimum(p.left_h, p.left_a) <= LAST_ROUNDS).to_numpy()
        p = p[late]
        y = p["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        fit, test = (p.season <= 2024).to_numpy(), (p.season == 2025).to_numpy()
        P = p[C].to_numpy()
        X = np.c_[p.nothing_a - p.nothing_h, p.danger_h - p.danger_a].astype(float)
        b = minimize(lambda v: log_loss(adjust(P[fit], X[fit], v), y[fit]), np.zeros(2),
                     method="Nelder-Mead").x
        print(f"\n== {model}: последние {LAST_ROUNDS} туров, подбор 2023-24 n={fit.sum()}, тест 2025 n={test.sum()} ==")
        print(f"  веса: «нечего терять» {b[0]:+.3f}, «борьба за выживание» {b[1]:+.3f}")
        print(f"  тест: {log_loss(P[test], y[test]):.4f} -> {log_loss(adjust(P[test], X[test], b), y[test]):.4f}")
        if model == "final":
            q = p.assign(hw=(p.result == "H").astype(float))
            for name, sel in (("хозяевам нечего терять, гостям есть", (q.nothing_h == 1) & (q.nothing_a == 0)),
                              ("гостям нечего терять, хозяевам есть", (q.nothing_a == 1) & (q.nothing_h == 0))):
                g = q[sel]
                print(f"  {name}: матчей {len(g)}, модель П1 {g.p_home.mean():.1%} / факт {g.hw.mean():.1%}; "
                      f"П2 модель {g.p_away.mean():.1%} / факт {(g.result == 'A').mean():.1%}")
