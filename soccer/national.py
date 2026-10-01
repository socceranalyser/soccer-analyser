"""National teams: data (martj42/international_results), Elo and Dixon-Coles.

Elo follows eloratings.net: K depends on the match importance, the goal-difference
multiplier is the club one, and there is no home bonus at neutral venues.
Dixon-Coles uses the same importance as a likelihood weight (friendlies count less).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit

from .config import RAW_DIR
from .data import _fetch
from .models.dixon_coles import DixonColes
from .models.elo import Elo, _gd_multiplier, _ordered_logit_probs

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
NAT_LEAGUE = "INT"

FINALS = {"FIFA World Cup"}
CONTINENTAL_FINALS = {"UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
                      "Gold Cup", "CONCACAF Championship", "Oceania Nations Cup",
                      "Confederations Cup"}


def importance(tournament: str) -> str:
    t = tournament or ""
    if t in FINALS:
        return "world_cup"
    if t in CONTINENTAL_FINALS:
        return "continental"
    if "qualification" in t or "Nations League" in t:
        return "qualifier"
    if t == "Friendly":
        return "friendly"
    return "other"


# eloratings.net K-factors
ELO_K = {"world_cup": 60, "continental": 50, "qualifier": 40, "other": 30, "friendly": 20}

# Russian labels for the dashboard
TOURNAMENT_RU = {
    "FIFA World Cup": "Чемпионат мира", "FIFA World Cup qualification": "Отбор ЧМ",
    "UEFA Euro": "Чемпионат Европы", "UEFA Euro qualification": "Отбор Евро",
    "UEFA Nations League": "Лига наций УЕФА", "Friendly": "Товарищеский",
    "Copa América": "Кубок Америки", "African Cup of Nations": "Кубок Африки",
    "African Cup of Nations qualification": "Отбор Кубка Африки",
    "AFC Asian Cup": "Кубок Азии", "AFC Asian Cup qualification": "Отбор Кубка Азии",
    "Gold Cup": "Золотой кубок КОНКАКАФ", "CONCACAF Nations League": "Лига наций КОНКАКАФ",
}


def load_international(refresh: bool = False, since: str = "1990-01-01") -> pd.DataFrame:
    path = _fetch(RESULTS_URL, RAW_DIR / NAT_LEAGUE / "results.csv", live=True, force=refresh)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(since)].dropna(subset=["home_score", "away_score"])
    out = pd.DataFrame({
        "league": NAT_LEAGUE, "date": df["date"], "season": df["date"].dt.year,
        "home": df["home_team"], "away": df["away_team"],
        "hg": df["home_score"].astype(int), "ag": df["away_score"].astype(int),
        "tournament": df["tournament"], "neutral": df["neutral"].astype(bool),
        "country": df["country"],
    })
    out["importance"] = out["tournament"].map(importance)
    out["result"] = np.select([out.hg > out.ag, out.hg == out.ag], ["H", "D"], "A")
    out["match_id"] = ("INT_" + out["date"].dt.strftime("%Y%m%d") + "_" + out["home"] + "_"
                       + out["away"]).str.replace(" ", "")
    return out.sort_values("date").drop_duplicates("match_id").reset_index(drop=True)


@dataclass
class NationalElo:
    home_adv: float = 100.0
    k_scale: float = 1.0
    start: float = 1500.0
    name: str = "elo"
    ratings: dict = field(default_factory=dict, init=False)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date", kind="stable")
        homes, aways = df["home"].tolist(), df["away"].tolist()
        gd = (df["hg"] - df["ag"]).to_numpy(float)
        score = np.where(gd > 0, 1.0, np.where(gd == 0, 0.5, 0.0))
        k = df["importance"].map(ELO_K).to_numpy(float) * self.k_scale * _gd_multiplier(gd)
        hadv = np.where(df["neutral"].to_numpy(bool), 0.0, self.home_adv)
        dates = df["date"].to_numpy()
        ratings: dict[str, float] = {}
        rh, ra = np.empty(len(df)), np.empty(len(df))
        start = 0
        for end in list(np.flatnonzero(dates[1:] != dates[:-1]) + 1) + [len(df)]:
            for i in range(start, end):
                rh[i] = ratings.setdefault(homes[i], self.start)
                ra[i] = ratings.setdefault(aways[i], self.start)
            for i in range(start, end):
                exp_h = 1 / (1 + 10 ** (-(rh[i] + hadv[i] - ra[i]) / 400))
                delta = k[i] * (score[i] - exp_h)
                ratings[homes[i]] += delta
                ratings[aways[i]] -= delta
            start = end
        self.ratings = ratings
        return df.assign(elo_h=rh, elo_a=ra)

    def diff(self, elo_h, elo_a, neutral) -> np.ndarray:
        h = np.where(np.asarray(neutral, bool), 0.0, self.home_adv)
        return (np.asarray(elo_h) + h - np.asarray(elo_a)) / 400.0

    def fit_outcome_model(self, rated: pd.DataFrame, as_of=None, since="2000-01-01"):
        df = rated[rated["date"] >= pd.Timestamp(since)]
        if as_of is not None:
            df = df[df["date"] < pd.Timestamp(as_of)]
        d = self.diff(df["elo_h"], df["elo_a"], df["neutral"])
        y = df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        self.logit = Elo._fit_logit(d, y)
        return self

    def probs(self, elo_h, elo_a, neutral) -> np.ndarray:
        return _ordered_logit_probs(self.logit, self.diff(elo_h, elo_a, neutral))

    def predict(self, homes, aways, neutral) -> dict:
        rh = np.array([self.ratings.get(t, self.start - 200) for t in homes])
        ra = np.array([self.ratings.get(t, self.start - 200) for t in aways])
        return {"probs": self.probs(rh, ra, neutral), "elo_h": rh, "elo_a": ra}

    def fit(self, df: pd.DataFrame, as_of=None):
        train = df if as_of is None else df[df["date"] < pd.Timestamp(as_of)]
        self.history = self.run(train)
        return self.fit_outcome_model(self.history)


# Dixon-Coles settings for national teams (tuned: see CLAUDE.md)
# tuned on 2019-2022 (monthly walk-forward): DC logloss 0.8667; Elo (defaults) 0.8601
NAT_DC = {"xi": 0.0005, "ridge": 1.0, "window_days": 365 * 8, "promoted_prior": False}
NAT_WEIGHTS = {"world_cup": 1.0, "continental": 1.0, "qualifier": 1.0, "other": 0.8,
               "friendly": 0.7}


def with_weights(df: pd.DataFrame, weights=None) -> pd.DataFrame:
    return df.assign(weight=df["importance"].map(weights or NAT_WEIGHTS))


def fit_national_dc(df: pd.DataFrame, as_of=None, weights=None, **kw) -> DixonColes:
    params = {**NAT_DC, **kw}
    return DixonColes(**params).fit(with_weights(df, weights), as_of=as_of,
                                    season=0, season_teams=())


def national_walk_forward(df: pd.DataFrame, start: str, end: str, dc_kwargs=None,
                          elo_kwargs=None, weights=None, models=("dixon_coles", "elo")):
    """Monthly refit; returns predictions in the common format (+ tournament column)."""
    from .backtest import _rows
    test = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] < pd.Timestamp(end))]
    out = []
    if "elo" in models:
        elo = NationalElo(**(elo_kwargs or {}))
        rated = elo.run(df)
        rt = rated[rated["match_id"].isin(test["match_id"])]
        for key, block in rt.groupby(rt["date"].dt.to_period("Y")):
            elo.fit_outcome_model(rated, as_of=block["date"].min())
            out.append(_rows(block, "elo", elo.probs(block.elo_h, block.elo_a, block.neutral))
                       .assign(tournament=block["tournament"].to_numpy()))
    if "dixon_coles" in models:
        prev = None
        for key, block in test.groupby(test["date"].dt.to_period("M")):
            dc = DixonColes(**{**NAT_DC, **(dc_kwargs or {})}).fit(
                with_weights(df, weights), as_of=key.start_time, warm_start=prev,
                season=0, season_teams=())
            pr = dc.predict(block["home"], block["away"], block["neutral"].to_numpy())
            out.append(_rows(block, "dixon_coles", pr["probs"], pr["p_over25"],
                             pr["xg_home"], pr["xg_away"])
                       .assign(tournament=block["tournament"].to_numpy()))
            prev = dc
    return pd.concat(out, ignore_index=True)
