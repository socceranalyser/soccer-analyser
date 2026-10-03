r"""Missing players weighted by importance vs simply counted (top-5 leagues).

    d = sum(importance | away missing) - sum(importance | home missing)
    importance = minutes share last season * (1 + k * (xG + xA) per 90)  (unknown: 0.3)
Log-odds shift b*d on home/away win. Fit b (and k) on seasons 2022-2023, test on 2024/25,
against the current "count" feature fitted the same way.
    python scripts/test_player_importance.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from soccer import apifootball as af
from soccer import storage
from soccer.data import load_matches
from soccer.metrics import log_loss
from soccer.names import best_match
from soccer.players import UNKNOWN_SHARE, lookup
from soccer.xg import UNDERSTAT

C = ["p_home", "p_draw", "p_away"]


def missing_players() -> pd.DataFrame:
    """One row per (match, side, missing player) with last-season share / att90."""
    inj = af.injuries_history()
    inj = inj[inj["league"].isin(list(UNDERSTAT)) & (inj["type"] == "Missing Fixture")]
    m = load_matches(leagues=list(UNDERSTAT), seasons=[2022, 2023, 2024])
    out = []
    cache = {}
    for (lg, s), g in inj.groupby(["league", "season"]):
        teams = sorted(set(m.loc[(m.league == lg) & (m.season == s), "home"]))
        names = {t: best_match(t, teams)[0] for t in g["team"].unique()}
        g = g.assign(our=g["team"].map(names), day=g["date"].dt.tz_convert("Europe/London")
                     .dt.tz_localize(None).dt.normalize()).drop_duplicates(["fixture_id", "player_id"])
        mm = m[(m.league == lg) & (m.season == s)]
        for fid, gf in g.groupby("fixture_id"):
            day = gf["day"].iloc[0]
            teams_in = set(gf["our"].dropna())
            cand = mm[((mm["date"] - day).abs() <= pd.Timedelta(days=1))
                      & (mm["home"].isin(teams_in) | mm["away"].isin(teams_in))]
            for r in cand.itertuples():
                for side, team in (("h", r.home), ("a", r.away)):
                    for p in gf[gf["our"] == team].itertuples():
                        key = (p.player, s - 1)
                        if key not in cache:
                            cache[key] = lookup(p.player, s - 1)
                        rec = cache[key]
                        out.append({"league": lg, "season": s, "date": r.date, "home": r.home,
                                    "away": r.away, "side": side, "player": p.player,
                                    "known": rec is not None,
                                    "share": rec["share"] if rec else np.nan,
                                    "att90": rec["att90"] if rec else np.nan})
    return pd.DataFrame(out)


def features(mp: pd.DataFrame, k: float, unknown: float = UNKNOWN_SHARE) -> pd.DataFrame:
    imp = np.where(mp["known"], mp["share"] * (1 + k * mp["att90"].fillna(0)), unknown)
    f = mp.assign(imp=imp, n=1.0)
    g = f.pivot_table(index=["league", "date", "home", "away"], columns="side",
                      values=["imp", "n"], aggfunc="sum", fill_value=0.0)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    for c in ("imp_h", "imp_a", "n_h", "n_a"):
        if c not in g:
            g[c] = 0.0
    return g.reset_index()


def adjust(p, d, b):
    z = np.log(np.clip(p, 1e-9, 1))
    z[:, 0] += b * d
    z[:, 2] -= b * d
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    mp = missing_players()
    print(f"missing-player rows: {len(mp)}, found in last-season stats: {mp['known'].mean():.0%}")
    preds = storage.load_predictions("bt_all").dropna(subset=["result"])
    for model in ("final", "market"):
        base = preds[preds.model == model]

        def data(k, unknown=UNKNOWN_SHARE):
            p = base.merge(features(mp, k, unknown), on=["league", "date", "home", "away"])
            return (p, p["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy(),
                    (p.season <= 2023).to_numpy(), (p.season == 2024).to_numpy())

        p, y, fit, test = data(0.0)
        P = p[C].to_numpy()
        res = {"без травм": log_loss(P[test], y[test])}
        dn = (p.n_a - p.n_h).to_numpy(float)
        b = minimize(lambda x: log_loss(adjust(P[fit], dn[fit], x[0]), y[fit]), [0.0],
                     method="Nelder-Mead").x[0]
        res[f"по количеству (b={b:.3f})"] = log_loss(adjust(P[test], dn[test], b), y[test])

        def loss(v, idx):
            k, b, u = v
            pk, yk, fk, tk = data(k, u)
            d = (pk.imp_a - pk.imp_h).to_numpy(float)
            sel = fk if idx == "fit" else tk
            return log_loss(adjust(pk[C].to_numpy()[sel], d[sel], b), yk[sel])
        best = None
        for k in (0.0, 0.5, 1.0, 2.0):
            for u in (0.1, 0.3, 0.5):
                r = minimize(lambda x: loss((k, x[0], u), "fit"), [0.05], method="Nelder-Mead")
                if best is None or r.fun < best[0]:
                    best = (r.fun, k, r.x[0], u)
        _, k, b2, u = best
        res[f"по важности (k={k}, b={b2:.3f}, неизв.={u})"] = loss((k, b2, u), "test")
        print(f"\n== {model}: тест 2024/25, матчей {test.sum()} ==")
        for name, v in res.items():
            print(f"  {name:40s} log loss {v:.4f}")


def mark_new(mp: pd.DataFrame) -> pd.DataFrame:
    """new = missing now but not missing in the team's previous league match."""
    m = load_matches(leagues=list(UNDERSTAT), seasons=[2022, 2023, 2024])
    games = pd.concat([m[["league", "date", "home"]].rename(columns={"home": "team"}),
                       m[["league", "date", "away"]].rename(columns={"away": "team"})])
    games = games.sort_values("date")
    prev = {}
    for (lg, t), g in games.groupby(["league", "team"]):
        d = list(g["date"])
        prev.update({(lg, t, d[i]): d[i - 1] for i in range(1, len(d))})
    mp = mp.assign(team=np.where(mp["side"] == "h", mp["home"], mp["away"]))
    out_sets = mp.groupby(["league", "team", "date"])["player"].apply(set).to_dict()
    new = [p not in out_sets.get((lg, t, prev.get((lg, t, d))), set())
           for lg, t, d, p in zip(mp["league"], mp["team"], mp["date"], mp["player"])]
    return mp.assign(new=new)


def run_new():
    mp = mark_new(missing_players())
    print(f"\nновые потери (не было в прошлом матче): {mp['new'].mean():.0%} строк")
    preds = storage.load_predictions("bt_all").dropna(subset=["result"])
    base = preds[preds.model == "final"]
    for label, sub, k, u in (("все отсутствующие, по количеству", mp, None, None),
                             ("только новые, по количеству", mp[mp["new"]], None, None),
                             ("только новые, по важности", mp[mp["new"]], 1.0, 0.3),
                             ("только новые, по важности (k=0)", mp[mp["new"]], 0.0, 0.3)):
        f = features(sub, k or 0.0, u or UNKNOWN_SHARE)
        p = base.merge(f, on=["league", "date", "home", "away"], how="left").fillna(
            {"imp_h": 0, "imp_a": 0, "n_h": 0, "n_a": 0})
        p = p[p["league"].isin(list(UNDERSTAT)) & p["season"].isin([2022, 2023, 2024])]
        y = p["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        fit, test = (p.season <= 2023).to_numpy(), (p.season == 2024).to_numpy()
        d = ((p.imp_a - p.imp_h) if k is not None else (p.n_a - p.n_h)).to_numpy(float)
        P = p[C].to_numpy()
        b = minimize(lambda x: log_loss(adjust(P[fit], d[fit], x[0]), y[fit]), [0.0],
                     method="Nelder-Mead").x[0]
        print(f"  {label:38s} b={b:+.3f}: тест {log_loss(P[test], y[test]):.4f} -> "
              f"{log_loss(adjust(P[test], d[test], b), y[test]):.4f} (n={test.sum()})")
