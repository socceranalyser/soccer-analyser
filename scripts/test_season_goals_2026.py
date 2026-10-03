r"""Season goal-level offset on the 2026/27 season (walk-forward forecasts made here).
    python scripts/test_season_goals_2026.py"""
import importlib.util
import sys

import numpy as np
import pandas as pd

from soccer.backtest import walk_forward
from soccer.data import load_matches
from soccer.engine import dc_weight
from soccer.tuned import DC_PARAMS, ELO_PARAMS, XG_WEIGHT
from soccer.xg import attach_xg

spec = importlib.util.spec_from_file_location("sg", "scripts/test_season_goals.py")
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    m = attach_xg(load_matches())
    wf = walk_forward(m, [2026], dc_kwargs={**DC_PARAMS, "xg_weight": XG_WEIGHT},
                      elo_kwargs=ELO_PARAMS, models=("dixon_coles", "elo"), verbose=False)
    K = ["league", "date", "home", "away"]
    dc = wf[wf.model == "dixon_coles"].set_index(K)
    el = wf[wf.model == "elo"].set_index(K)
    idx = dc.index.intersection(el.index)
    dc, el = dc.loc[idx].copy(), el.loc[idx]
    w = idx.get_level_values("league").map(dc_weight).to_numpy(float)[:, None]
    C = ["p_home", "p_draw", "p_away"]
    dc[C] = w * dc[C].to_numpy() + (1 - w) * el[C].to_numpy()
    p = dc.reset_index().assign(model="final")
    p = p[p.result.notna()]
    sg_load = sg.load
    sg.storage.load_predictions = lambda run: p          # feed the 2026 forecasts to load()
    q0 = sg_load()
    for n0, n1 in ((3000, 1000), (1000, 1000), (300, 300)):
        off = sg.season_offsets(q0, n0, n1)
        q = 1 / (1 + np.exp(-(sg.logit(q0.p) + off)))
        print(f"2026 (n={len(q0)}), n0={n0} n1={n1}: ТБ прогноз {q0.p.mean():.3f} -> {q.mean():.3f}, "
              f"факт {q0.y.mean():.3f} | log loss {sg.ll(q0.p, q0.y):.4f} -> {sg.ll(q, q0.y):.4f}")
