r"""Do starting line-ups make the forecast better? (data: scripts/collect_lineups.py)

Accepted variant (2026-10-16): "core" (started >= 80% of the last 10 games within 120 days),
beta 0.03 -> soccer/lineups.py. For each team its "regular XI" is learnt from its previous K league games with a known line-up:
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
MAX_AGE = int(__import__('os').environ.get('LU_MAX_AGE', 120))  # days
COUNTRY = {v: k for k, v in Engine.LS_COUNTRY.items()}  # English -> Russian


def from_cache() -> pd.DataFrame:
    """Line-ups already downloaded by a collector that is still running."""
    import json
    raw = storage.DB_PATH.parent / "raw"
    rows = []
    for day in sorted((raw / "livescore_days").glob("*.json")):
        for e in json.loads(day.read_text(encoding="utf-8")):
            f = raw / "ls_lineups" / f"{str(e['id']).removeprefix('ls')}.json"
            if not f.exists():
                continue
            for t in (json.loads(f.read_text(encoding="utf-8")) or {}).get("Lu", []):
                for p in t.get("Ps", []):
                    if p.get("Pos") in (1, 2, 3, 4, 5):
                        rows.append({"eid": f.stem, "date": pd.Timestamp(day.stem), "competition": e["competition"],
                                     "home_raw": e["home_raw"], "away_raw": e["away_raw"],
                                     "side": t.get("Tnb"), "pid": str(p.get("Pid")), "pos": p.get("Pos"),
                                     "name": f"{p.get('Fn', '')} {p.get('Ln', '')}".strip()})
    return pd.DataFrame(rows)


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
    st = lu[lu["pos"].between(1, 4)].drop(columns=["date"]).merge(links, on="eid")
    st["team"] = np.where(st["side"] == 1, st["home"], st["away"])
    st["team"] = st["league"] + "|" + st["team"]
    games = st.drop_duplicates(["eid", "team"])[["eid", "team", "date"]].sort_values("date")
    xi = st.groupby(["eid", "team"])["pid"].apply(set).to_dict()
    hist: dict[str, list] = {}
    seen: dict[str, list] = {}  # the team's earlier "missing" values: its normal rotation
    rows = []
    for g in games.itertuples():
        allpast = hist.setdefault(g.team, [])
        today = xi[(g.eid, g.team)]
        # only recent games: last season's XI says little after the summer transfers
        past = [s for d, s in allpast if (g.date - d).days <= MAX_AGE]
        if len(past) >= MIN_GAMES:
            last = past[-K:]
            cnt = pd.Series([p for s in last for p in s]).value_counts() / len(last)
            reg = cnt[cnt >= W_MIN]
            core = cnt[cnt >= 0.8]
            miss = float(reg[~reg.index.isin(today)].sum())
            prev = seen.setdefault(g.team, [])
            rows.append({"eid": g.eid, "team": g.team, "miss": miss, "n_reg": len(reg),
                         "core": float(core[~core.index.isin(today)].sum()),
                         "rel": miss - (np.mean(prev[-10:]) if len(prev) >= 3 else 2.4)})
            prev.append(miss)
        if len(today) >= 9:  # a real line-up (some feeds list only a few players)
            allpast.append((g.date, today))
    return pd.DataFrame(rows)


def tilt(P: np.ndarray, s: np.ndarray) -> np.ndarray:
    Q = P * np.stack([np.exp(s), np.ones_like(s), np.exp(-s)], 1)
    return Q / Q.sum(1, keepdims=True)


def ll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    f = storage.DB_PATH.parent / "raw" / "lineups.csv"
    lu = pd.read_csv(f, dtype={"eid": str, "pid": str}) if f.exists() else from_cache()
    lu["date"] = pd.to_datetime(lu["date"])
    m = load_matches(seasons=[2025, 2026])
    links = link(lu, m)
    print(f"livescore games: {lu['eid'].nunique():,} | linked to our matches: {len(links):,}")
    ms = missing(lu, links)
    if "--save-history" in sys.argv:  # seed data/state/lineup_xi.json for the live site
        from soccer import lineups
        st = lu[lu["pos"].between(1, 4)].drop(columns=["date"]).merge(links, on="eid")
        hist = {}
        for (eid, side), g in st.sort_values("date").groupby(["eid", "side"], sort=False):
            r = g.iloc[0]
            lineups.add_game(hist, r["home"] if side == 1 else r["away"], r["date"],
                             dict(zip(g["pid"].astype(str), g["name"].fillna(""))))
        lineups.save_history(hist)
        print("history saved:", len(hist), "teams ->", lineups.XI_FILE)
    side = links.assign(hk=links["league"] + "|" + links["home"], ak=links["league"] + "|" + links["away"])
    ren = lambda s: {"team": f"{s}k", "miss": f"miss_{s}", "n_reg": f"reg_{s}", "core": f"core_{s}",
                     "rel": f"rel_{s}"}
    side = side.merge(ms.rename(columns=ren("h")), on=["eid", "hk"]) \
               .merge(ms.rename(columns=ren("a")), on=["eid", "ak"])
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
    cut = int(len(d) * 0.6)
    for feat in ("miss", "core", "rel"):
        xf = (d[f"{feat}_a"] - d[f"{feat}_h"]).to_numpy(float)
        b = minimize_scalar(lambda b: ll(tilt(P[:cut], b * xf[:cut]), y[:cut]), bounds=(-1, 1),
                            method="bounded").x
        print(f"вариант {feat:5}: подбор {ll(tilt(P[:cut], b * xf[:cut]), y[:cut]):.4f} "
              f"(без {ll(P[:cut], y[:cut]):.4f}) | тест {ll(P[cut:], y[cut:]):.4f} → "
              f"{ll(tilt(P[cut:], b * xf[cut:]), y[cut:]):.4f} | beta {b:+.3f}")
    x = (d["miss_a"] - d["miss_h"]).to_numpy(float)
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

    # the same check on ALL linked games that have the bookmaker's opening odds (no need for
    # our backtest forecast, which ends with the 2025/26 season)
    mo = side.merge(mm, on=key).dropna(subset=["open_h", "open_d", "open_a"]).sort_values("date")
    mo = mo[(mo[["open_h", "open_d", "open_a"]] > 1).all(1)].reset_index(drop=True)
    yo = mo["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    Mo = 1 / mo[["open_h", "open_d", "open_a"]].to_numpy(float)
    Mo = Mo / Mo.sum(1, keepdims=True)
    xo = (mo["miss_a"] - mo["miss_h"]).to_numpy(float)
    co = int(len(mo) * 0.6)
    bo = minimize_scalar(lambda b: ll(tilt(Mo[:co], b * xo[:co]), yo[:co]), bounds=(-1, 1), method="bounded").x
    print(f"ВСЕ матчи с ранними коэф. ({len(mo):,}, тест {len(mo) - co:,} с {mo['date'].iloc[co]:%d.%m.%Y}): "
          f"beta={bo:+.3f} | {ll(Mo[co:], yo[co:]):.4f} → {ll(tilt(Mo[co:], bo * xo[co:]), yo[co:]):.4f}")
    for lo, hi in ((0, 0.5), (0.5, 1.5), (1.5, 99)):
        s = (np.abs(xo[co:]) >= lo) & (np.abs(xo[co:]) < hi)
        print(f"   разница в отсутствующих {lo}–{hi}: {s.sum()} матчей, "
              f"{ll(Mo[co:][s], yo[co:][s]):.4f} → {ll(tilt(Mo[co:][s], bo * xo[co:][s]), yo[co:][s]):.4f}")
