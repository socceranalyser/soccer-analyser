r"""Do our coupons come true as often as promised? Daily coupons on 2023-2025 league matches.

Every match day: candidates are 1X2 and over/under 2.5 outcomes with real bookmaker odds
(opening odds where known - closest to what misli.az shows in the morning). Same rules as
soccer/coupons.py: chance = margin-free market ^1.10, our model must agree (agree_min), one
pick per match, odds floors. Reports, per style: promised chance vs how often the coupon
actually won, and the money result for 1 unit per coupon.
    python scripts/backtest_coupons.py
"""
import sys

import numpy as np
import pandas as pd

from soccer import storage
from soccer.coupons import MARKET_POWER, STYLES
from soccer.data import load_matches

KEY = ["league", "date", "home", "away"]


def candidates() -> pd.DataFrame:
    p = storage.load_predictions("bt_all")
    fin = p[(p.model == "final") & p.result.notna() & p.p_over25.notna()]
    m = load_matches(seasons=[2023, 2024, 2025])
    odds = {}
    for side, op, cl in (("1", "open_h", "odds_h"), ("x", "open_d", "odds_d"), ("2", "open_a", "odds_a"),
                         ("over", "open_o25", "odds_o25"), ("under", "open_u25", "odds_u25")):
        odds[side] = m[op].where(m[op] > 1, m[cl]) if op in m else m[cl]
    m = m.assign(**{f"o_{k}": v for k, v in odds.items()})[KEY + [f"o_{k}" for k in odds]]
    d = fin.merge(m, on=KEY).dropna(subset=["o_1", "o_x", "o_2", "o_over", "o_under"])
    rows = []
    for g, outs, pm in ((("o_1", "o_x", "o_2"), ("1", "x", "2"), ("p_home", "p_draw", "p_away")),
                        (("o_over", "o_under"), ("over", "under"), (None, None))):
        z = (1 / d[list(g)].to_numpy(float)) ** MARKET_POWER
        z = z / z.sum(1, keepdims=True)
        for j, o in enumerate(outs):
            model = d[pm[j]].to_numpy(float) if pm[0] else (
                d["p_over25"].to_numpy(float) if o == "over" else 1 - d["p_over25"].to_numpy(float))
            goals = d["hg"] + d["ag"]
            won = {"1": d["hg"] > d["ag"], "x": d["hg"] == d["ag"], "2": d["hg"] < d["ag"],
                   "over": goals > 2.5, "under": goals < 2.5}[o]
            rows.append(pd.DataFrame({"date": d["date"].to_numpy(), "match": (d["home"] + "-" + d["away"]).to_numpy(),
                                      "odds": d[g[j]].to_numpy(float), "p": z[:, j], "p_model": model,
                                      "won": won.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def simulate(c: pd.DataFrame, style: str) -> pd.DataFrame:
    st = STYLES[style]
    c = c[(c.odds >= st["min_odds"]) & (c.odds <= st["max_odds"]) & (c.p >= st["min_p"])
          & (np.minimum(c.p, c.p_model) >= st["agree_min"])]
    out = []
    for day, g in c.groupby("date"):
        g = g.assign(score=[st["score"](p, o, m) for p, o, m in zip(g.p, g.odds, g.p_model)])
        picks = g.sort_values("score", ascending=False).drop_duplicates("match").head(st["size"])
        if len(picks) < st["size"]:
            continue
        out.append({"date": day, "prob": picks.p.prod(), "odds": picks.odds.prod(),
                    "won": bool(picks.won.all()), "legs_won": int(picks.won.sum())})
    return pd.DataFrame(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    c = candidates()
    print(f"кандидатов: {len(c):,}")
    for style in STYLES:
        r = simulate(c, style)
        if r.empty:
            continue
        profit = np.where(r.won, r.odds - 1, -1.0)
        streak = (~r.won).astype(int).groupby(r.won.cumsum()).sum().max()
        print(f"\n{STYLES[style]['title']}: дней {len(r)}")
        print(f"  обещанный шанс в среднем {r.prob.mean():.1%} | сбылось на деле {r.won.mean():.1%}")
        print(f"  средний коэф. {r.odds.mean():.2f} | итог по 1 ₼ на купон: {profit.sum():+.1f} ₼ "
              f"({profit.mean():+.1%} за купон) | самая длинная серия проигрышей: {streak}")
        b = pd.cut(r.prob, [0, .1, .2, .35, .5, .65, 1])
        print(r.groupby(b, observed=True).agg(купонов=("won", "size"), обещано=("prob", "mean"),
                                               сбылось=("won", "mean")).round(3).to_string())
