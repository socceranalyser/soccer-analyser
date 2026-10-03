r"""Does xG help? Dixon-Coles fitted on (1-w)*goals + w*xG, top-5 leagues (understat).

Pick w on validation seasons 2019-2022, confirm on 2023-2025; also checks the production
ensemble (50/50 DC + Elo) and totals.
    python scripts/tune_xg.py
"""
import sys

import numpy as np
import pandas as pd

from soccer.backtest import walk_forward
from soccer.data import load_matches
from soccer.metrics import log_loss
from soccer.tuned import DC_PARAMS, ELO_PARAMS
from soccer.xg import UNDERSTAT, attach_xg

LEAGUES = list(UNDERSTAT)
Y = {"H": 0, "D": 1, "A": 2}


def evaluate(matches, seasons, w, elo=None):
    p = walk_forward(matches, seasons, leagues=LEAGUES, dc_kwargs={**DC_PARAMS, "xg_weight": w},
                     elo_kwargs=ELO_PARAMS, models=("dixon_coles",), verbose=False)
    y = p["result"].map(Y).to_numpy()
    probs = p[["p_home", "p_draw", "p_away"]].to_numpy(float)
    over = (p["hg"] + p["ag"] > 2.5).to_numpy(float)
    po = p["p_over25"].clip(1e-6, 1 - 1e-6).to_numpy(float)
    ou = -np.mean(over * np.log(po) + (1 - over) * np.log(1 - po))
    res = {"dc": log_loss(probs, y), "ou": ou, "n": len(p)}
    if elo is not None:
        m = p.merge(elo, on=["league", "date", "home", "away"], suffixes=("", "_e"))
        ens = 0.5 * m[["p_home", "p_draw", "p_away"]].to_numpy(float) + \
            0.5 * m[["p_home_e", "p_draw_e", "p_away_e"]].to_numpy(float)
        res["ens"] = log_loss(ens, m["result"].map(Y).to_numpy())
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    matches = attach_xg(load_matches(leagues=LEAGUES))
    for name, seasons, ws in (("VALIDATION 2019-22", [2019, 2020, 2021, 2022], [0.0, 0.3, 0.5, 0.7, 0.9]),
                              ("TEST 2023-25", [2023, 2024, 2025], None)):
        if ws is None:
            ws = [0.0, best]
        e = walk_forward(matches, seasons, leagues=LEAGUES, elo_kwargs=ELO_PARAMS,
                         models=("elo",), verbose=False)[["league", "date", "home", "away",
                                                          "p_home", "p_draw", "p_away"]]
        rows = {w: evaluate(matches, seasons, w, e) for w in ws}
        print(f"\n== {name} ==")
        for w, r in rows.items():
            print(f"  xG weight {w:.1f}: DC {r['dc']:.4f} | ensemble DC+Elo {r['ens']:.4f} | "
                  f"total 2.5 {r['ou']:.4f}  (n={r['n']})")
        if ws[0] == 0.0 and name.startswith("VAL"):
            best = min(rows, key=lambda w: rows[w]["ens"])
            print(f"  -> best weight on validation: {best}")
