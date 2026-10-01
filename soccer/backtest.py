"""Walk-forward backtest: every prediction uses only data strictly before the match."""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .data import implied_probs
from .models.dixon_coles import DixonColes
from .models.elo import Elo

PRED_COLS = ["match_id", "league", "season", "date", "home", "away", "hg", "ag", "result",
             "model", "p_home", "p_draw", "p_away", "p_over25", "xg_home", "xg_away"]


def _block_key(dates: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return dates.dt.normalize()
    return (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.normalize()  # Monday


def _rows(block: pd.DataFrame, model: str, probs, p_over=None, xgh=None, xga=None):
    out = block[PRED_COLS[:9]].copy()
    out["model"] = model
    out["p_home"], out["p_draw"], out["p_away"] = probs[:, 0], probs[:, 1], probs[:, 2]
    out["p_over25"] = np.nan if p_over is None else p_over
    out["xg_home"] = np.nan if xgh is None else xgh
    out["xg_away"] = np.nan if xga is None else xga
    return out


def market_predictions(matches: pd.DataFrame) -> pd.DataFrame:
    """Bookmaker implied probabilities — the benchmark every model must approach/beat."""
    m = matches.dropna(subset=["odds_h", "odds_d", "odds_a"])
    p = implied_probs(m["odds_h"], m["odds_d"], m["odds_a"])
    over = np.full(len(m), np.nan)
    o, u = m["odds_o25"].to_numpy(float), m["odds_u25"].to_numpy(float)
    ok = (o > 1) & (u > 1)
    over[ok] = (1 / o[ok]) / (1 / o[ok] + 1 / u[ok])
    return _rows(m, "market", p, over)


def _dc_league(lm: pd.DataFrame, test_seasons, dc_kwargs: dict, freq: str):
    """Walk-forward Dixon-Coles for one league (top-level so it can run in a subprocess)."""
    t0 = time.time()
    test = lm[lm["season"].isin(test_seasons)]
    participants = {s: set(g["home"]) | set(g["away"]) for s, g in test.groupby("season")}
    out, prev = [], None
    for key, block in test.groupby(_block_key(test["date"], freq)):
        season = int(block["season"].iloc[0])
        dc = DixonColes(**dc_kwargs).fit(lm, as_of=key, warm_start=prev, season=season,
                                         season_teams=participants[season])
        pr = dc.predict(block["home"], block["away"])
        out.append(_rows(block, dc.name, pr["probs"], pr["p_over25"],
                         pr["xg_home"], pr["xg_away"]))
        prev = dc
    return pd.concat(out, ignore_index=True), time.time() - t0


def walk_forward(matches: pd.DataFrame, test_seasons, leagues=None, dc_kwargs=None,
                 elo_kwargs=None, freq: str = "W", models=("dixon_coles", "elo", "market"),
                 verbose: bool = True, n_jobs: int = max(1, (os.cpu_count() or 2) - 2)
                 ) -> pd.DataFrame:
    leagues = leagues or sorted(matches["league"].unique())
    matches = matches[matches["league"].isin(leagues)]
    test_seasons = list(test_seasons)
    out = []

    if "market" in models:
        out.append(market_predictions(matches[matches["season"].isin(test_seasons)]))

    if "elo" in models:
        elo = Elo(**(elo_kwargs or {}))
        rated = elo.run(matches)
        for (league, season), test in rated[rated["season"].isin(test_seasons)].groupby(
                ["league", "season"]):
            # outcome model refitted at each league-season start, on earlier data only
            elo.fit_outcome_model(rated, as_of=test["date"].min(), leagues=[league])
            out.append(_rows(test, "elo", elo.probs_from_diff(
                elo.diff(test.elo_h, test.elo_a), league)))

    if "dixon_coles" in models:
        jobs = [(matches[matches["league"] == lg], test_seasons, dc_kwargs or {}, freq)
                for lg in leagues]
        if n_jobs > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
                results = list(ex.map(_dc_league, *zip(*jobs)))
        else:
            results = [_dc_league(*j) for j in jobs]
        for league, (df, secs) in zip(leagues, results):
            out.append(df)
            if verbose:
                print(f"  DC {league}: {len(df)} matches in {secs:.1f}s")

    preds = pd.concat(out, ignore_index=True)
    return preds.sort_values(["date", "match_id", "model"]).reset_index(drop=True)


def common_matches(preds: pd.DataFrame, models) -> pd.DataFrame:
    """Restrict to matches every listed model predicted (fair comparison)."""
    sub = preds[preds["model"].isin(models)]
    key = pd.MultiIndex.from_frame(sub[["league", "date", "home", "away"]])
    counts = sub.groupby(key)["model"].nunique()
    keep = counts.index[counts == len(models)]
    return sub[key.isin(keep)]
