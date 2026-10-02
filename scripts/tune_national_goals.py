"""Tune the national-team GOALS model (totals / BTTS), not 1X2 (1X2 comes from Elo).

Walk-forward by month; metrics: log loss of Over 2.5 and of BTTS (lower is better).
    python scripts/tune_national_goals.py
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from soccer.models.dixon_coles import DixonColes
from soccer.national import NAT_DC, NAT_WEIGHTS, load_international, with_weights
from soccer.probability import btts_prob, over_prob

START, END = "2019-01-01", "2023-01-01"   # validation; pass TEST=1 for 2023-2026


def bll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run(cfg):
    ridge, xi, fw, shrink = cfg
    start, end = (("2023-01-01", "2026-12-31") if os.environ.get("TEST") else (START, END))
    df = load_international(since="1960-01-01")
    w = {**NAT_WEIGHTS, "friendly": fw}
    test = df[(df.date >= start) & (df.date < end)]
    po, pb, yo, yb, xg, tot = [], [], [], [], [], []
    for key, block in test.groupby(test.date.dt.to_period("M")):
        dc = DixonColes(**{**NAT_DC, "ridge": ridge, "xi": xi, "goal_shrink": shrink}).fit(
            with_weights(df, w), as_of=key.start_time, season=0, season_teams=())
        m = dc.score_matrices(block.home, block.away, block.neutral.to_numpy())
        po += list(over_prob(m, 2.5)); pb += list(btts_prob(m))
        yo += list((block.hg + block.ag > 2.5).astype(float)); yb += list(((block.hg > 0) & (block.ag > 0)).astype(float))
        g = np.arange(m.shape[-1]); xg += list((m.sum(-1) * g).sum(-1) + (m.sum(-2) * g).sum(-1)); tot += list(block.hg + block.ag)
    xg, tot = np.array(xg), np.array(tot)
    hi = xg >= 3.5
    return cfg, bll(np.array(po), np.array(yo)), bll(np.array(pb), np.array(yb)), xg[hi].mean() if hi.any() else np.nan, tot[hi].mean() if hi.any() else np.nan


if __name__ == "__main__":
    grid = list(itertools.product([1.0, 3.0, 10.0], [0.0005, 0.001], [0.7, 1.0], [0.0, 0.2, 0.35, 0.5]))
    if os.environ.get("GRID"):
        grid = eval(os.environ["GRID"])
    with ProcessPoolExecutor(min(len(grid), (os.cpu_count() or 4) - 2)) as ex:
        res = sorted(ex.map(run, grid), key=lambda r: r[1] + r[2])
    print("ridge xi friendly shrink | OU2.5 ll | BTTS ll | xg>=3.5: модель / факт")
    for cfg, o, b, x, t in res:
        print(cfg, f"{o:.4f} {b:.4f} | {x:.2f} / {t:.2f}")
