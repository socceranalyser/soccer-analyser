"""Fit production models on all available data and record live forecasts."""
from __future__ import annotations

import pandas as pd

from . import storage
from .backtest import _rows, market_predictions
from .config import LEAGUES
from .data import load_fixtures, load_matches
from .models.dixon_coles import DixonColes
from .models.elo import Elo
from .tuned import DC_PARAMS, ELO_PARAMS

LIVE_RUN_ID = "live"
LIVE_DC = {**DC_PARAMS, "n_samples": 300}  # posterior predictive for displayed forecasts


def fit_dc(matches: pd.DataFrame, league: str, fixtures=None, **overrides) -> DixonColes:
    return DixonColes(**{**LIVE_DC, **overrides}).fit(
        matches[matches["league"] == league],
        season=current_season(matches, league, fixtures),
        season_teams=current_teams(matches, league, fixtures))


def fit_elo(matches: pd.DataFrame) -> Elo:
    return Elo(**ELO_PARAMS).fit(matches)


def current_season(matches: pd.DataFrame, league: str, fixtures=None) -> int:
    """Latest season of a league (calendar-year leagues differ from Aug-May ones)."""
    s = matches.loc[matches["league"] == league, "season"].max()
    if fixtures is not None and not fixtures.empty and (fixtures["league"] == league).any():
        s = max(s, fixtures.loc[fixtures["league"] == league, "season"].max())
    return int(s)


def current_teams(matches: pd.DataFrame, league: str, fixtures: pd.DataFrame | None = None):
    season = current_season(matches, league, fixtures)
    cur = matches[(matches["league"] == league) & (matches["season"] == season)]
    teams = set(cur["home"]) | set(cur["away"])
    if fixtures is not None and not fixtures.empty:
        f = fixtures[fixtures["league"] == league]
        teams |= set(f["home"]) | set(f["away"])
    return sorted(teams)


def predict_fixtures(matches: pd.DataFrame, fixtures: pd.DataFrame, elo: Elo | None = None,
                     dcs: dict | None = None) -> pd.DataFrame:
    """Rows in the storage format for every model, for each upcoming fixture."""
    if fixtures.empty:
        return pd.DataFrame()
    elo = elo or fit_elo(matches)
    dcs = dcs or {}
    fx = fixtures.copy()
    for c in ("match_id", "hg", "ag", "result"):
        if c not in fx:
            fx[c] = None
    out = []
    for league, block in fx.groupby("league"):
        dc = dcs.get(league) or fit_dc(matches, league, fixtures)
        pr = dc.predict(block["home"], block["away"])
        out.append(_rows(block, "dixon_coles", pr["probs"], pr["p_over25"],
                         pr["xg_home"], pr["xg_away"]))
        out.append(_rows(block, "elo",
                         elo.predict(block["home"], block["away"], league)["probs"]))
    mk = market_predictions(fx)
    if not mk.empty:
        out.append(mk)
    return pd.concat(out, ignore_index=True)


def record_live(refresh: bool = True) -> dict:
    """Refresh data, store results, forecast all upcoming fixtures of every league."""
    matches = load_matches(refresh=refresh)
    storage.save_results(matches)
    fixtures = load_fixtures(list(LEAGUES))
    if not fixtures.empty:
        fixtures = fixtures[fixtures["date"] >= pd.Timestamp.today().normalize()]
    preds = predict_fixtures(matches, fixtures)
    n = 0
    if not preds.empty:
        n = storage.save_run(LIVE_RUN_ID, "live", preds, description="live forecasts",
                             params={"dc": LIVE_DC, "elo": ELO_PARAMS}, replace_run=False)
    return {"matches": len(matches), "fixtures": len(fixtures), "saved": n}
