r"""Several coupons a day ("portfolio") vs one: daily money result on 2023-2025 league matches.

Legs: 1X2 / over-under 2.5 outcomes where BOTH the margin-free market and our model give at
least `agree`, ranked by that agreement, one leg per match. Stake 5 per coupon.
    python scripts/backtest_portfolio.py
"""
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from backtest_coupons import candidates, simulate  # noqa: E402
from soccer.coupons import STYLES  # noqa: E402

STAKE = 5.0


def legs(c, lo, hi, agree):
    c = c[(c.odds >= lo) & (c.odds <= hi) & (np.minimum(c.p, c.p_model) >= agree)]
    return c.assign(a=np.minimum(c.p, c.p_model)).sort_values("a", ascending=False)


def system(c, k, n, lo, hi, agree, name):
    """n legs, every k-fold combination as its own coupon (k=1: n singles)."""
    rows = []
    for day, g in legs(c, lo, hi, agree).groupby("date"):
        g = g.drop_duplicates("match").head(n)
        if len(g) < n:
            continue
        o, w = g.odds.to_numpy(), g.won.to_numpy()
        ret = sum(STAKE * np.prod(o[list(ix)]) for ix in combinations(range(n), k) if w[list(ix)].all())
        cnt = len(list(combinations(range(n), k)))
        rows.append({"date": day, "staked": STAKE * cnt, "ret": ret})
    return name, pd.DataFrame(rows)


def report(name, r):
    if r.empty:
        return
    pl = r.ret - r.staked
    lose = (pl < 0).astype(int)
    streak = lose.groupby((lose == 0).cumsum()).sum().max()
    print(f"{name:44} дней {len(r):4} | ставка/день {r.staked.mean():5.1f} ₼ | день в плюсе "
          f"{(pl > 0).mean():5.1%} | в ноль/плюс {(pl >= 0).mean():5.1%} | итог {pl.sum():+8.0f} ₼ "
          f"({pl.sum() / r.staked.sum():+.1%}) | худшая серия {streak} дн.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    c = candidates()
    # today's ladder: one coupon of each style
    parts = []
    for s in STYLES:
        r = simulate(c, s)
        parts.append(pd.DataFrame({"date": r.date, "staked": STAKE, "ret": np.where(r.won, STAKE * r.odds, 0)}))
        report("сейчас: " + STYLES[s]["title"], parts[-1])
    report("сейчас: все 4 купона вместе", pd.concat(parts).groupby("date", as_index=False).sum())
    print()
    for args in [
        (1, 3, 1.25, 1.60, 0.70, "3 одиночных (кф 1.25–1.60, согласие ≥70%)"),
        (1, 3, 1.50, 2.20, 0.50, "3 одиночных (кф 1.50–2.20)"),
        (2, 3, 1.25, 1.60, 0.65, "система 2 из 3 (кф 1.25–1.60) = 3 купона"),
        (2, 3, 1.50, 2.20, 0.50, "система 2 из 3 (кф 1.50–2.20) = 3 купона"),
        (2, 3, 1.80, 2.40, 0.42, "система 2 из 3 (кф 1.80–2.40) = 3 купона"),
        (3, 4, 1.25, 1.60, 0.65, "система 3 из 4 (кф 1.25–1.60) = 4 купона"),
        (2, 4, 1.40, 2.00, 0.55, "система 2 из 4 (кф 1.40–2.00) = 6 купонов"),
        (1, 6, 1.25, 1.60, 0.65, "6 одиночных (кф 1.25–1.60)"),
    ]:
        report(*system(c, *args))
