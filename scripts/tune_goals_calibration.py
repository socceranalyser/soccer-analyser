r"""Goals calibration for league forecasts (totals and both-teams-to-score).

Diagnosis 2026-10-04 (Albacete 1:3 Eibar): in the backtest the model is too sure of
low-scoring games - predicted over-2.5 31% -> happened 37%, 38% -> 43%. Fix tested here:

    logit(p_over') = a + b * logit(p_over) + c_league      (Platt scaling, league offsets
                                                             shrunk towards 0)
then the score matrix is tilted  m' ∝ m * exp(t * (i + j))  with t solved so that the
matrix gives p_over', and rescaled back to the same 1X2. BTTS and exact scores follow from
the tilted matrix, so BTTS is an honest out-of-sample check (it is never fitted).

Fit on seasons 2023-2024, test on 2025-2026 of run bt_all.

    python scripts/tune_goals_calibration.py           # report
    python scripts/tune_goals_calibration.py --save    # write data/goals_calibration.json
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from soccer import storage
from soccer.config import DATA_DIR
from soccer.probability import goal_tilt, over_prob, btts_prob, rescale_to_outcomes, score_matrix

OUT = DATA_DIR / "goals_calibration.json"
SHRINK = 200.0  # league offset prior: as if 200 matches with offset 0


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def load():
    p = storage.load_predictions("bt_all")
    p = p[(p["model"] == "final") & p["result"].notna() & p["xg_home"].notna()].copy()
    p["over"] = (p["hg"] + p["ag"] > 2.5).astype(float)
    p["btts"] = ((p["hg"] > 0) & (p["ag"] > 0)).astype(float)
    mats = score_matrix(p["xg_home"].to_numpy(), p["xg_away"].to_numpy(), -0.05)
    mats = rescale_to_outcomes(mats, p[["p_home", "p_draw", "p_away"]].to_numpy())
    p["m_over"] = over_prob(mats)
    p["m_btts"] = btts_prob(mats)
    return p, mats


def fit(tr: pd.DataFrame) -> dict:
    x, y = logit(tr["m_over"].to_numpy()), tr["over"].to_numpy()

    def nll(v):
        q = 1 / (1 + np.exp(-(v[0] + v[1] * x)))
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
    a, b = minimize(nll, [0.0, 1.0]).x
    base = a + b * x
    off = {}
    for lg, g in tr.assign(base=base).groupby("league"):
        res = g["over"] - 1 / (1 + np.exp(-g["base"]))
        q = 1 / (1 + np.exp(-g["base"]))
        # one Newton step of a Gaussian-prior offset (MAP), enough at these sizes
        off[lg] = float(res.sum() / ((q * (1 - q)).sum() + SHRINK * 0.25))
    return {"a": float(a), "b": float(b), "league": off}


def apply(df: pd.DataFrame, mats: np.ndarray, cal: dict, leagues: bool = True):
    z = cal["a"] + cal["b"] * logit(df["m_over"].to_numpy())
    if leagues:
        z = z + df["league"].map(cal["league"]).fillna(0.0).to_numpy()
    target = 1 / (1 + np.exp(-z))
    m2 = goal_tilt(mats, target, df[["p_home", "p_draw", "p_away"]].to_numpy())
    return over_prob(m2), btts_prob(m2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    df, mats = load()
    tr = (df["season"] <= 2024).to_numpy()
    te = ~tr
    cal = fit(df[tr])
    print(f"fit 2023-24 (n={tr.sum()}): a={cal['a']:.3f} b={cal['b']:.3f}")
    t = df[te]
    for name, kw in (("без лиговых поправок", {"leagues": False}), ("с лиговыми", {})):
        o, bt = apply(t, mats[te], cal, **kw)
        print(f"TEST 2025-26 (n={te.sum()}) {name}: over LL {ll(t['m_over'], t['over']):.4f} -> "
              f"{ll(o, t['over']):.4f};  BTTS LL {ll(t['m_btts'], t['btts']):.4f} -> "
              f"{ll(bt, t['btts']):.4f}")
    o, bt = apply(t, mats[te], cal)
    t = t.assign(o=o, bt=bt)
    print("\nкалибровка тотала на тесте (прогноз было / стало / факт):")
    for lo, hi in ((0, .35), (.35, .4), (.4, .45), (.45, .5), (.5, .55), (.55, .6), (.6, 1)):
        g = t[(t["m_over"] > lo) & (t["m_over"] <= hi)]
        print(f"  {lo:.2f}-{hi:.2f} n={len(g):5d}: {g['m_over'].mean():.3f} / {g['o'].mean():.3f}"
              f" / {g['over'].mean():.3f}")
    print(f"BTTS в среднем: было {t['m_btts'].mean():.3f} / стало {t['bt'].mean():.3f} / "
          f"факт {t['btts'].mean():.3f}")
    sp = t[t["league"] == "SP2"]
    print(f"SP2: тотал было {sp['m_over'].mean():.3f} / стало {sp['o'].mean():.3f} / факт "
          f"{sp['over'].mean():.3f}")
    if "--save" in sys.argv:
        full = fit(df)  # production: fitted on all seasons
        OUT.write_text(json.dumps(full, indent=1), encoding="utf-8")
        print("saved", OUT, f"a={full['a']:.3f} b={full['b']:.3f}")
