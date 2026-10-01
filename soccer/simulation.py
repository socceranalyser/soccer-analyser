"""Monte Carlo simulation of the rest of a season with a fitted Dixon-Coles model.

Each simulation first draws team strengths from the Laplace posterior (parameter
uncertainty: we do not know exactly how good a team is), then draws scores for every
remaining fixture. Scores are independent Poisson (the DC low-score correction has a
negligible effect on final tables). Ties are broken by goal difference, then goals
scored (a simplification — Spain/Italy use head-to-head first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models.dixon_coles import DixonColes


def league_table(played: pd.DataFrame, teams=None) -> pd.DataFrame:
    teams = sorted(teams if teams is not None else set(played["home"]) | set(played["away"]))
    t = pd.DataFrame(0, index=teams, columns=["P", "W", "D", "L", "GF", "GA", "Pts"])
    for r in played.itertuples():
        for team, gf, ga in ((r.home, r.hg, r.ag), (r.away, r.ag, r.hg)):
            t.loc[team, ["P", "GF", "GA"]] += [1, gf, ga]
            res = "W" if gf > ga else "D" if gf == ga else "L"
            t.loc[team, res] += 1
            t.loc[team, "Pts"] += {"W": 3, "D": 1, "L": 0}[res]
    t["GD"] = t["GF"] - t["GA"]
    t = t.sort_values(["Pts", "GD", "GF"], ascending=False)
    t.insert(0, "#", range(1, len(t) + 1))
    return t.rename_axis("team").reset_index()


def remaining_fixtures(played: pd.DataFrame, teams) -> pd.DataFrame:
    done = set(zip(played["home"], played["away"]))
    rows = [(h, a) for h in teams for a in teams if h != a and (h, a) not in done]
    return pd.DataFrame(rows, columns=["home", "away"])


def simulate_season(model: DixonColes, played: pd.DataFrame, teams=None, n_sims: int = 10000,
                    param_uncertainty: bool = True, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    teams = sorted(teams if teams is not None else set(played["home"]) | set(played["away"]))
    tidx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    base = league_table(played, teams).set_index("team").loc[teams]
    rem = remaining_fixtures(played, teams)

    pts = np.tile(base["Pts"].to_numpy(float), (n_sims, 1))
    gd = np.tile(base["GD"].to_numpy(float), (n_sims, 1))
    gf = np.tile(base["GF"].to_numpy(float), (n_sims, 1))
    if len(rem):
        theta = model.sample_theta(n_sims, rng) if param_uncertainty else model.theta
        lam, mu = model.rates(rem["home"], rem["away"], theta)
        lam = np.broadcast_to(lam, (n_sims, len(rem)))
        mu = np.broadcast_to(mu, (n_sims, len(rem)))
        hg, ag = rng.poisson(lam), rng.poisson(mu)
        H = np.zeros((len(rem), T))
        A = np.zeros((len(rem), T))
        H[np.arange(len(rem)), rem["home"].map(tidx)] = 1
        A[np.arange(len(rem)), rem["away"].map(tidx)] = 1
        ph = 3.0 * (hg > ag) + (hg == ag)
        pa = 3.0 * (ag > hg) + (hg == ag)
        pts += ph @ H + pa @ A
        gd += (hg - ag) @ H + (ag - hg) @ A
        gf += hg @ H + ag @ A

    key = pts * 1e6 + gd * 1e3 + gf + rng.random(pts.shape) * 1e-3
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    pos[np.arange(n_sims)[:, None], order] = np.arange(T)
    pos_probs = np.stack([(pos == k).mean(axis=0) for k in range(T)], axis=1)  # team x position

    summary = pd.DataFrame({
        "team": teams,
        "pts_now": base["Pts"].to_numpy(),
        "exp_pts": pts.mean(axis=0),
        "pts_p10": np.percentile(pts, 10, axis=0),
        "pts_p90": np.percentile(pts, 90, axis=0),
        "exp_pos": (pos + 1).mean(axis=0),
    })
    return {"summary": summary, "position_probs": pd.DataFrame(pos_probs, index=teams,
                                                               columns=range(1, T + 1)),
            "n_remaining": len(rem), "n_sims": n_sims}


def add_probabilities(result: dict, top: int, relegated: int) -> pd.DataFrame:
    pp = result["position_probs"]
    s = result["summary"].set_index("team")
    T = pp.shape[1]
    s["p_title"] = pp[1]
    s[f"p_top{top}"] = pp.loc[:, 1:top].sum(axis=1)
    s["p_relegation"] = pp.loc[:, T - relegated + 1:T].sum(axis=1)
    return s.sort_values("exp_pts", ascending=False).reset_index()
