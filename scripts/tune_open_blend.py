r"""Model vs EARLY bookmaker odds (like misli.az before kick-off): how to combine them.

Log pooling p ∝ model^a · market^b, fitted on 2019-22, checked on 2023-25,
separately for 1X2 and for over/under 2.5.   python scripts/tune_open_blend.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from soccer import storage
from soccer.backtest import walk_forward
from soccer.config import DATA_DIR, TOP5
from soccer.data import load_matches
from soccer.engine import ENSEMBLE_DC_WEIGHT
from soccer.euro import load_euro
from soccer.metrics import binary_log_loss, log_loss
from soccer.tuned import DC_PARAMS, ELO_PARAMS

K = ["league", "date", "home", "away"]
C = ["p_home", "p_draw", "p_away"]


def pool3(m, q, a, b):
    z = a * np.log(np.clip(m, 1e-9, 1)) + b * np.log(np.clip(q, 1e-9, 1))
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def pool2(m, q, a, b):
    lo = a * np.log(np.clip(m, 1e-9, 1)) + b * np.log(np.clip(q, 1e-9, 1))
    lu = a * np.log(np.clip(1 - m, 1e-9, 1)) + b * np.log(np.clip(1 - q, 1e-9, 1))
    return 1 / (1 + np.exp(lu - lo))


def frame(preds: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    dc = preds[preds.model == "dixon_coles"].set_index(K)
    el = preds[preds.model == "elo"].set_index(K)
    idx = dc.index.intersection(el.index)
    f = pd.DataFrame(index=idx)
    f[C] = ENSEMBLE_DC_WEIGHT * dc.loc[idx, C].to_numpy() + (1 - ENSEMBLE_DC_WEIGHT) * el.loc[idx, C].to_numpy()
    f["m_over"] = dc.loc[idx, "p_over25"].to_numpy()
    f = f.reset_index().merge(matches[K + ["hg", "ag", "result", "open_h", "open_d", "open_a",
                                           "open_o25", "open_u25"]], on=K)
    f = f.dropna(subset=["open_h", "open_d", "open_a"])
    inv = 1 / f[["open_h", "open_d", "open_a"]].to_numpy()
    f[["q_home", "q_draw", "q_away"]] = inv / inv.sum(1, keepdims=True)
    io, iu = 1 / f["open_o25"], 1 / f["open_u25"]
    f["q_over"] = io / (io + iu)
    f["y"] = f["result"].map({"H": 0, "D": 1, "A": 2})
    f["y_over"] = ((f.hg + f.ag) > 2.5).astype(float)
    return f


def report(name, f, a3, b3, a2, b2):
    m, q, y = f[C].to_numpy(), f[["q_home", "q_draw", "q_away"]].to_numpy(), f["y"].to_numpy()
    p = pool3(m, q, a3, b3)
    o = f.dropna(subset=["q_over", "m_over"])
    print(f"{name} (n={len(f)}): 1X2 model {log_loss(m, y):.4f} | bookmaker {log_loss(q, y):.4f} | "
          f"combined {log_loss(p, y):.4f}  ;  O/U2.5 model {binary_log_loss(o.m_over.to_numpy(), o.y_over.to_numpy()):.4f}"
          f" | bookmaker {binary_log_loss(o.q_over.to_numpy(), o.y_over.to_numpy()):.4f}"
          f" | combined {binary_log_loss(pool2(o.m_over.to_numpy(), o.q_over.to_numpy(), a2, b2), o.y_over.to_numpy()):.4f}")


if __name__ == "__main__":
    matches = load_matches()
    val = walk_forward(matches, [2019, 2020, 2021, 2022], dc_kwargs=DC_PARAMS, elo_kwargs=ELO_PARAMS,
                       cups=load_euro(matches), models=("dixon_coles", "elo"), verbose=False)
    fv = frame(val, matches)
    ft = frame(storage.load_predictions("bt_all").drop(columns=["hg", "ag", "result"]), matches)
    m, q, y = fv[C].to_numpy(), fv[["q_home", "q_draw", "q_away"]].to_numpy(), fv["y"].to_numpy()
    a3, b3 = minimize(lambda x: log_loss(pool3(m, q, *x), y), [0.5, 0.5], method="Nelder-Mead").x
    o = fv.dropna(subset=["q_over", "m_over"])
    a2, b2 = minimize(lambda x: binary_log_loss(pool2(o.m_over.to_numpy(), o.q_over.to_numpy(), *x),
                                                o.y_over.to_numpy()), [0.5, 0.5], method="Nelder-Mead").x
    print(f"fitted on 2019-22: 1X2 model a={a3:.3f} bookmaker b={b3:.3f} (model share {a3/(a3+b3):.0%}); "
          f"O/U model a={a2:.3f} bookmaker b={b2:.3f} (model share {a2/(a2+b2):.0%})")
    report("TEST 2023-25 all", ft, a3, b3, a2, b2)
    report("  top-5", ft[ft.league.isin(TOP5)], a3, b3, a2, b2)
    report("  other", ft[~ft.league.isin(TOP5)], a3, b3, a2, b2)
    (DATA_DIR / "blend_weights.json").write_text(json.dumps(
        {"x12": [a3, b3], "ou": [a2, b2], "fitted_on": "2019-2022 opening odds"}, indent=1))
