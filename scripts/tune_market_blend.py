"""How much to trust the model vs the bookmaker: fit on 2019-22, check on 2023-25.

Logarithmic pooling: p ∝ model^a · market^b  (a = b = 0.5 is the geometric mean).
    python scripts/tune_market_blend.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from soccer import storage
from soccer.backtest import walk_forward
from soccer.data import load_matches
from soccer.engine import ENSEMBLE_DC_WEIGHT
from soccer.euro import load_euro
from soccer.metrics import log_loss
from soccer.tuned import DC_PARAMS, ELO_PARAMS

C = ["p_home", "p_draw", "p_away"]
K = ["league", "date", "home", "away"]


def wide(p: pd.DataFrame):
    dc = p[p.model == "dixon_coles"].set_index(K)
    el = p[p.model == "elo"].set_index(K)
    mk = p[p.model == "market"].set_index(K)
    idx = dc.index.intersection(el.index).intersection(mk.index)
    model = ENSEMBLE_DC_WEIGHT * dc.loc[idx, C].to_numpy() + (1 - ENSEMBLE_DC_WEIGHT) * el.loc[idx, C].to_numpy()
    y = dc.loc[idx, "result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    return model, mk.loc[idx, C].to_numpy(), y


def pool(m, q, a, b):
    z = a * np.log(np.clip(m, 1e-9, 1)) + b * np.log(np.clip(q, 1e-9, 1))
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


if __name__ == "__main__":
    matches = load_matches()
    val = walk_forward(matches, [2019, 2020, 2021, 2022], dc_kwargs=DC_PARAMS,
                       elo_kwargs=ELO_PARAMS, cups=load_euro(matches), verbose=False)
    mv, qv, yv = wide(val)
    test = storage.load_predictions("bt_all").dropna(subset=["result"])
    mt, qt, yt = wide(test)
    res = minimize(lambda x: log_loss(pool(mv, qv, x[0], x[1]), yv), [0.5, 0.5], method="Nelder-Mead")
    a, b = res.x
    print(f"fitted on 2019-22: a(model)={a:.3f} b(market)={b:.3f}  (n={len(yv)})")
    for name, p in (("model", mt), ("market", qt), ("pooled", pool(mt, qt, a, b))):
        print(f"  test 2023-25 {name:7s} logloss {log_loss(p, yt):.4f}  accuracy {(p.argmax(1) == yt).mean():.3f}")
    share = a / (a + b)
    print(f"model share of the pooled opinion: {share:.0%}")
