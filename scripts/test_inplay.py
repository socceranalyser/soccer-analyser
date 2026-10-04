r"""Check the in-play model from half-time: given the pre-match forecast (bt_all) and the
half-time score, how well does it predict the full-time result, total and BTTS?
Fit second-half share and game-state effect on 2023-24, test on 2025.
    python scripts/test_inplay.py
"""
import itertools
import sys

import numpy as np
import pandas as pd

from soccer import storage
from soccer.data import load_matches
from soccer.inplay import live_markets, score_matrix_live

KEY = ["league", "date", "home", "away"]


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def data():
    p = storage.load_predictions("bt_all")
    p = p[(p.model == "final") & p.result.notna() & p.xg_home.notna()]
    m = load_matches(seasons=[2023, 2024, 2025])[KEY + ["hthg", "htag", "hr", "ar"]]
    d = p.merge(m, on=KEY).dropna(subset=["hthg", "htag"])
    # only matches without red cards (their timing is unknown in the data)
    return d[(d.hr.fillna(0) == 0) & (d.ar.fillna(0) == 0)].reset_index(drop=True)


def evaluate(d, gs, s2):
    rows = []
    for r in d.itertuples():
        mk = live_markets(score_matrix_live(r.xg_home, r.xg_away, 45, int(r.hthg), int(r.htag),
                                            game_state=gs, second_half_share=s2))
        rows.append(mk)
    mk = pd.DataFrame(rows)
    y1 = (d.hg > d.ag).to_numpy(float)
    yx = (d.hg == d.ag).to_numpy(float)
    y2 = (d.hg < d.ag).to_numpy(float)
    P = mk[["o1", "ox", "o2"]].to_numpy()
    res_ll = -np.mean(np.log(np.clip((P * np.c_[y1, yx, y2]).sum(1), 1e-9, 1)))
    over = ((d.hg + d.ag) > 2.5).to_numpy(float)
    btts = ((d.hg > 0) & (d.ag > 0)).to_numpy(float)
    return {"1x2": res_ll, "ou": ll(mk.o_over, over), "btts": ll(mk.o_btts_yes, btts), "mk": mk}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    d = data()
    fit = d[d.season <= 2024].sample(6000, random_state=0)
    test = d[d.season == 2025]
    best = None
    for gs, s2 in itertools.product([0.0, 0.05, 0.10, 0.15, 0.20], [0.50, 0.53, 0.55, 0.57, 0.60]):
        e = evaluate(fit, gs, s2)
        score = e["1x2"] + e["ou"]
        if best is None or score < best[0]:
            best = (score, gs, s2)
    _, gs, s2 = best
    print(f"подобрано на 2023-24: эффект счёта {gs}, доля голов во 2-м тайме {s2}")
    e = evaluate(test, gs, s2)
    e0 = evaluate(test, 0.0, 0.5)
    print(f"тест 2025 (n={len(test)}), с перерыва: исход {e0['1x2']:.4f} -> {e['1x2']:.4f}, "
          f"тотал {e0['ou']:.4f} -> {e['ou']:.4f}, обе {e0['btts']:.4f} -> {e['btts']:.4f}  (простая -> подобранная)")
    mk = e["mk"]
    print("\nкалибровка (обещано / сбылось) на тесте:")
    for name, col, y in (("победа хозяев", "o1", test.hg > test.ag), ("ТБ 2.5", "o_over", test.hg + test.ag > 2.5),
                         ("обе забьют", "o_btts_yes", (test.hg > 0) & (test.ag > 0))):
        b = pd.cut(mk[col], [0, .1, .3, .5, .7, .9, 1])
        t = pd.DataFrame({"p": mk[col].to_numpy(), "y": y.to_numpy(float), "b": b.to_numpy()})
        print(f"  {name}: " + " | ".join(f"{g.p.mean():.0%}/{g.y.mean():.0%} (n={len(g)})"
                                         for _, g in t.groupby("b", observed=True)))
