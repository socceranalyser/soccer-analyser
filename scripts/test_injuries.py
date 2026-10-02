r"""Do injuries/suspensions (API-Football) improve the forecast?  python scripts/test_injuries.py

Feature per match: players missing (type 'Missing Fixture') for home and away.
Adjustment on log-odds: home win += b*d, away win -= b*d, d = away_missing - home_missing.
Fitted on season 2023 (2023/24), checked on 2024 (2024/25); for our model and for the bookmaker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from soccer import apifootball as af
from soccer import storage
from soccer.data import load_matches
from soccer.metrics import log_loss
from soccer.names import best_match

C = ["p_home", "p_draw", "p_away"]


def match_features() -> pd.DataFrame:
    inj = af.injuries_history()
    inj = inj[inj["league"].notna()]
    m = load_matches(leagues=sorted(inj["league"].unique()), seasons=[2022, 2023, 2024])
    out = []
    for (lg, s), g in inj.groupby(["league", "season"]):
        teams = sorted(set(m.loc[(m.league == lg) & (m.season == s), "home"]))
        names = {t: best_match(t, teams)[0] for t in g["team"].unique()}
        g = g.assign(our=g["team"].map(names), day=g["date"].dt.tz_convert("Europe/London")
                     .dt.tz_localize(None).dt.normalize())
        miss = g[g["type"] == "Missing Fixture"].groupby(["fixture_id", "our"]).size()
        quest = g[g["type"] == "Questionable"].groupby(["fixture_id", "our"]).size()
        days = g.groupby("fixture_id")["day"].first()
        mm = m[(m.league == lg) & (m.season == s)]
        for fid, day in days.items():
            teams_in = set(g.loc[g.fixture_id == fid, "our"].dropna())
            cand = mm[(mm["date"] - day).abs() <= pd.Timedelta(days=1)]
            cand = cand[cand["home"].isin(teams_in) | cand["away"].isin(teams_in)]
            for r in cand.itertuples():
                out.append({"league": lg, "date": r.date, "home": r.home, "away": r.away,
                            "miss_h": miss.get((fid, r.home), 0), "miss_a": miss.get((fid, r.away), 0),
                            "q_h": quest.get((fid, r.home), 0), "q_a": quest.get((fid, r.away), 0)})
    f = pd.DataFrame(out)
    # one row per match (a fixture appears once per team listed)
    return f.groupby(["league", "date", "home", "away"], as_index=False).max()


def adjust(p, d, b):
    z = np.log(np.clip(p, 1e-9, 1))
    z[:, 0] += b * d
    z[:, 2] -= b * d
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


if __name__ == "__main__":
    feats = match_features()
    print("matches with injury info:", len(feats))
    preds = storage.load_predictions("bt_all").dropna(subset=["result"])
    for model in ("final", "market"):
        p = preds[preds.model == model].merge(feats, on=["league", "date", "home", "away"])
        p["d"] = (p.miss_a - p.miss_h).astype(float)
        y = p["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        fit, test = (p.season == 2023).to_numpy(), (p.season == 2024).to_numpy()
        P, D = p[C].to_numpy(), p["d"].to_numpy()
        b = minimize(lambda x: log_loss(adjust(P[fit], D[fit], x[0]), y[fit]), [0.0],
                     method="Nelder-Mead").x[0]
        base, new = log_loss(P[test], y[test]), log_loss(adjust(P[test], D[test], b), y[test])
        print(f"{model:7s}: n_fit={fit.sum()} n_test={test.sum()} | b={b:+.3f} per player | "
              f"test 2024/25 logloss {base:.4f} -> {new:.4f} ({new - base:+.4f})")
    p = preds[preds.model == "final"].merge(feats, on=["league", "date", "home", "away"])
    p["d"] = p.miss_a - p.miss_h
    p["home_won"] = (p.result == "H").astype(float)
    print("\nкогда у гостей не хватает игроков больше, чем у хозяев (модель vs факт, победа хозяев):")
    print(p.groupby(pd.cut(p.d, [-30, -4, -2, 0, 1, 3, 30]), observed=True).agg(
        матчей=("d", "size"), модель=("p_home", "mean"), факт=("home_won", "mean")).round(3))
