r"""Fatigue: does rest before a match (league + European cups) improve the forecast?

Features per match (each team's previous competitive match, league or UCL/UEL/UECL):
    short_h / short_a = 1 if the team played within the last 3 days (<= 3 full days rest)
    rest diff         = min(rest_a, 7) - min(rest_h, 7)
Log-odds shift on home/away win. Fit on 2023-2024, test on 2025, all 38 leagues; also for
the bookmaker (if the market already prices fatigue, the gain there is ~0).
    python scripts/test_rest_days.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from soccer import storage
from soccer.data import load_matches
from soccer.euro import load_euro
from soccer.metrics import log_loss

C = ["p_home", "p_draw", "p_away"]


def rest_table() -> pd.DataFrame:
    m = load_matches()
    games = [m[["league", "date", "home"]].rename(columns={"home": "team"}).assign(cup=False),
             m[["league", "date", "away"]].rename(columns={"away": "team"}).assign(cup=False)]
    try:
        e = load_euro(m)
        if "home_key" in e:  # map cup teams back to league names (country|team keys)
            for side in ("home", "away"):
                g = e[["date", f"{side}_key"]].dropna()
                games.append(pd.DataFrame({"league": None, "date": g["date"],
                                           "team": g[f"{side}_key"].str.split("|").str[-1],
                                           "cup": True}))
    except Exception as exc:  # cups are optional for this test
        print("no cups:", exc)
    g = pd.concat(games, ignore_index=True).sort_values("date")
    g = g.drop_duplicates(["team", "date"])
    g["prev"] = g.groupby("team")["date"].shift()
    g["rest"] = (g["date"] - g["prev"]).dt.days
    g["prev_cup"] = g.groupby("team")["cup"].shift().fillna(False)
    return g[~g["cup"]][["league", "date", "team", "rest", "prev_cup"]]


def adjust(p, d, b):
    z = np.log(np.clip(p, 1e-9, 1))
    z[:, 0] += d @ b
    z[:, 2] -= d @ b
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rt = rest_table()
    preds = storage.load_predictions("bt_all").dropna(subset=["result"])
    for model in ("final", "market"):
        p = preds[preds.model == model]
        for side in ("home", "away"):
            p = p.merge(rt.rename(columns={"team": side, "rest": f"rest_{side[0]}",
                                           "prev_cup": f"cup_{side[0]}"}),
                        on=["league", "date", side], how="left")
        p = p.dropna(subset=["rest_h", "rest_a"])
        p = p[(p["rest_h"] < 60) & (p["rest_a"] < 60)]  # season starts: rest is meaningless
        y = p["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        fit, test = (p.season <= 2024).to_numpy(), (p.season == 2025).to_numpy()
        P = p[C].to_numpy()
        feats = {
            "разница отдыха (до 7 дней)": np.c_[p.rest_a.clip(upper=7) - p.rest_h.clip(upper=7)],
            "играл ≤3 дней назад": np.c_[(p.rest_a <= 3).astype(float) - (p.rest_h <= 3).astype(float)],
            "еврокубок перед матчем": np.c_[p.cup_a.astype(float) - p.cup_h.astype(float)],
            "всё вместе": np.c_[p.rest_a.clip(upper=7) - p.rest_h.clip(upper=7),
                                (p.rest_a <= 3).astype(float) - (p.rest_h <= 3).astype(float),
                                p.cup_a.astype(float) - p.cup_h.astype(float)],
        }
        print(f"\n== {model}: тест 2025, матчей {test.sum()} (без поправки {log_loss(P[test], y[test]):.4f}) ==")
        print(f"  матчей, где кто-то играл ≤3 дней назад: {((p.rest_h <= 3) | (p.rest_a <= 3)).mean():.1%}; "
              f"после еврокубка: {(p.cup_h | p.cup_a).mean():.1%}")
        for name, X in feats.items():
            X = X.astype(float)
            b = minimize(lambda v: log_loss(adjust(P[fit], X[fit], v), y[fit]),
                         np.zeros(X.shape[1]), method="Nelder-Mead").x
            print(f"  {name:28s} b={np.round(b, 3)}: тест -> "
                  f"{log_loss(adjust(P[test], X[test], b), y[test]):.4f}")
