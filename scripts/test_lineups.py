r"""Do starting line-ups make the forecast better? (data: scripts/collect_lineups.py)

For each team its "regular XI" is learnt from its previous K league games with a known line-up:
w_p = share of those games player p started. Before a game: missing = sum of w_p of regulars
(w_p >= W_MIN) who are NOT in today's starting XI. The forecast is tilted towards the side
missing less: p_home * e^s, p_away * e^-s, s = beta * (missing_away - missing_home).
beta is fitted on the first part of the period (by date) and checked on the rest; the same
is done on the bookmaker's opening odds to see whether the market already knew.

    python scripts/test_lineups.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from soccer import storage
from soccer.config import LEAGUES
from soccer.data import load_matches
from soccer.engine import Engine
from soccer.names import CLUB_ALIASES, similarity

K, W_MIN, MIN_GAMES = 10, 0.5, 5
COUNTRY = {v: k for k, v in Engine.LS_COUNTRY.items()}  # English -> Russian


def link(lu: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """livescore game -> our (league, date, home, away) by country, date +-1 and both names."""
    games = lu.drop_duplicates("eid")[["eid", "date", "competition", "home_raw", "away_raw"]]
    m = m.assign(country=m["league"].map(lambda l: LEAGUES[l]["country"] if l in LEAGUES else None))
    by_c = {c: g for c, g in m.groupby("country")}
    alias = lambda n: CLUB_ALIASES.get(n, n)
    out = []
    for g in games.itertuples():
        c = COUNTRY.get(g.competition.split(" · ")[0])
        cand = by_c.get(c)
        if cand is None:
            continue
        d = pd.Timestamp(g.date)
        cand = cand[(cand["date"] - d).abs() <= pd.Timedelta(days=1)]
        if cand.empty:
            continue
        sh = cand["home"].map(lambda t: similarity(alias(g.home_raw), t))
        sa = cand["away"].map(lambda t: similarity(alias(g.away_raw), t))
        sc = np.minimum(sh, sa)
        if sc.max() >= 0.6:
            r = cand.loc[sc.idxmax()]
            out.append({"eid": g.eid, "league": r["league"], "date": r["date"], "home": r["home"],
                        "away": r["away"], "result": r["result"]})
    return pd.DataFrame(out)


def missing(lu: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Per linked game: weighted number of regulars missing from each side's starting XI."""
    st = lu[lu["pos"].between(1, 4)].merge(links, on="eid")
    st["team"] = np.where(st["side"] == 1, st["home"], st["away"])
    st["team"] = st["league"] + "|" + st["team"]
    games = st.drop_duplicates(["eid", "team"])[["eid", "team", "date"]].sort_values("date")
    xi = st.groupby(["eid", "team"])["pid"].apply(set).to_dict()
    hist: dict[str, list] = {}
    rows = []
    for g in games.itertuples():
        past = hist.setdefault(g.team, [])
        today = xi[(g.eid, g.team)]
        if len(past) >= MIN_GAMES:
            last = past[-K:]
            cnt = pd.Series([p for s in last for p in s]).value_counts() / len(last)
            reg = cnt[cnt >= W_MIN]
            rows.append({"eid": g.eid, "team": g.team, "miss": float(reg[~reg.index.isin(today)].sum()),
                         "n_reg": len(reg)})
        if len(today) >= 9:  # a real line-up (some feeds list only a few players)
            past.append(today)
    return pd.DataFrame(rows)


def tilt(P: np.ndarray, s: np.ndarray) -> np.ndarray:
    Q = P * np.stack([np.exp(s), np.ones_like(s), np.exp(-s)], 1)
    return Q / Q.sum(1, keepdims=True)


def ll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    lu = pd.read_csv(storage.DB_PATH.parent / "raw" / "lineups.csv", dtype={"eid": str, "pid": str})
    lu["date"] = pd.to_datetime(lu["date"])
    m = load_matches(seasons=[2025, 2026])
    links = link(lu, m)
    print(f"livescore games: {lu['eid'].nunique():,} | linked to our matches: {len(links):,}")
    ms = missing(lu, links)
    side = links.assign(hk=links["league"] + "|" + links["home"], ak=links["league"] + "|" + links["away"])
    side = side.merge(ms.rename(columns={"team": "hk", "miss": "miss_h", "n_reg": "reg_h"}), on=["eid", "hk"]) \
               .merge(ms.rename(columns={"team": "ak", "miss": "miss_a", "n_reg": "reg_a"}), on=["eid", "ak"])
    print(f"games with both sides' history: {len(side):,} | avg regulars missing: home "
          f"{side['miss_h'].mean():.2f}, away {side['miss_a'].mean():.2f}")
    # our forecasts made BEFORE these games: backtest (2025/26) + live (2026/27)
    key = ["league", "date", "home", "away"]
    bt = storage.load_predictions("bt_all")
    lv = storage.load_predictions("live")
    pr = pd.concat([bt[bt["model"] == "final"], lv[lv["model"] == "final"]]).drop_duplicates(key, keep="last")
    d = side.merge(pr[key + ["p_home", "p_draw", "p_away"]], on=key)
    mm = m[key + ["open_h", "open_d", "open_a", "odds_h", "odds_d", "odds_a"]]
    d = d.merge(mm, on=key, how="left").sort_values("date").reset_index(drop=True)
    y = d["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    P = d[["p_home", "p_draw", "p_away"]].to_numpy(float)
    x = (d["miss_a"] - d["miss_h"]).to_numpy(float)
    cut = int(len(d) * 0.6)
    print(f"games with our forecast: {len(d):,} (fit on {cut:,} up to {d['date'].iloc[cut]:%d.%m.%Y}, "
          f"test on {len(d) - cut:,})")
    for name, M in (("наша модель", P), ("ранние коэф.", None), ("закрывающие коэф.", None)):
        if M is None:
            cols = ["open_h", "open_d", "open_a"] if "ранние" in name else ["odds_h", "odds_d", "odds_a"]
            ok = d[cols].notna().all(1).to_numpy() & (d[cols] > 1).all(1).to_numpy()
            M = np.where(ok[:, None], 1 / d[cols].to_numpy(float), P)
            M = M / M.sum(1, keepdims=True)
        f = lambda b: ll(tilt(M[:cut], b * x[:cut]), y[:cut])
        b = minimize_scalar(f, bounds=(-1, 1), method="bounded").x
        print(f"{name:18} beta={b:+.3f} | тест: без составов {ll(M[cut:], y[cut:]):.4f} → "
              f"с составами {ll(tilt(M[cut:], b * x[cut:]), y[cut:]):.4f}")
    big = np.abs(x[cut:]) >= 2
    if big.any():
        b = minimize_scalar(lambda b: ll(tilt(P[:cut], b * x[:cut]), y[:cut]), bounds=(-1, 1),
                            method="bounded").x
        print(f"матчи, где у одной команды нет 2+ основных (тест, {big.sum()}): "
              f"{ll(P[cut:][big], y[cut:][big]):.4f} → {ll(tilt(P[cut:][big], b * x[cut:][big]), y[cut:][big]):.4f}")
