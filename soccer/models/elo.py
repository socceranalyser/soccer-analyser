"""Goal-difference Elo (World Football Elo style) + ordered-logit mapping to 1X2.

All leagues are processed in ONE chronological pass with a single rating per team, so a
club keeps its rating when it moves between divisions (that is what makes tiers within a
country comparable; European cups will later link countries). Ratings are updated in
date batches: every match on day D uses ratings from before D (leak-free).

At the start of each league-season, ratings regress towards the league mean; a team
new to the league keeps its rating if it has one (promoted/relegated club), otherwise it
starts at the mean of the teams that left the league (or a tier-based default).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from ..config import LEAGUES

MIN_LEAGUE_MATCHES = 1500  # fewer than this -> use the pooled ordered-logit model


def _gd_multiplier(gd: np.ndarray) -> np.ndarray:
    gd = np.abs(gd)
    return np.where(gd <= 1, 1.0, np.where(gd == 2, 1.5, (11 + gd) / 8))


def _ordered_logit_probs(params, d) -> np.ndarray:
    beta, c1, gap = params
    c2 = c1 + np.exp(gap)
    d = np.asarray(d, float)
    p_a = expit(c1 - beta * d)
    p_ad = expit(c2 - beta * d)
    return np.stack([1 - p_ad, p_ad - p_a, p_a], axis=-1)  # [H, D, A]


@dataclass
class Elo:
    k: float = 15.0
    home_adv: float = 65.0
    season_regress: float = 0.0   # fraction pulled towards league mean each new season
    start: float = 1500.0
    tier_step: float = 150.0      # default start rating drop per tier below the top
    k_cup_mult: float = 1.0       # K multiplier for European cup matches
    country_spill: float = 0.0    # share of a cup rating change given to the club's country
    cup_start: float = 1450.0     # start rating for clubs from countries we do not cover
    name: str = "elo"

    ratings: dict = field(default_factory=dict, init=False)
    logits: dict = field(default_factory=dict, init=False)
    history: pd.DataFrame | None = field(default=None, init=False)

    # ---------------------------------------------------------------- ratings
    def run(self, matches: pd.DataFrame, cups: pd.DataFrame | None = None) -> pd.DataFrame:
        """Returns matches with pre-match ratings elo_h / elo_a (leak-free).

        cups: optional cross-country matches (European competitions) with columns
        date, home_key, away_key, hg, ag, neutral. They are rated in the same pass;
        their rated copy is stored in self.cup_history. A share `country_spill` of each
        cup rating change is applied to every club of the same country, so the model
        learns relative league strength.
        """
        dom = matches.assign(home=team_keys(matches["league"], matches["home"]),
                             away=team_keys(matches["league"], matches["away"]),
                             _cup=False, _neutral=False)
        parts = [dom[["date", "league", "season", "home", "away", "hg", "ag", "_cup",
                      "_neutral"]]]
        if cups is not None and len(cups):
            c = cups.dropna(subset=["hg", "ag"])
            parts.append(pd.DataFrame({
                "date": c["date"], "league": c["league"], "season": c["season"],
                "home": c["home_key"], "away": c["away_key"], "hg": c["hg"], "ag": c["ag"],
                "_cup": True, "_neutral": c["neutral"].astype(bool)}).set_axis(
                [f"cup{i}" for i in c.index]))
        df = pd.concat(parts).sort_values(["date", "_cup"], kind="stable")
        homes, aways = df["home"].tolist(), df["away"].tolist()
        leagues, seasons = df["league"].tolist(), df["season"].to_numpy()
        is_cup, neutral = df["_cup"].to_numpy(bool), df["_neutral"].to_numpy(bool)
        gd = (df["hg"] - df["ag"]).to_numpy(float)
        score = np.where(gd > 0, 1.0, np.where(gd == 0, 0.5, 0.0))
        k_mult = self.k * _gd_multiplier(gd) * np.where(is_cup, self.k_cup_mult, 1.0)
        hadv = np.where(neutral, 0.0, self.home_adv)
        dates = df["date"].to_numpy()
        dd = df[~is_cup]
        participants = {key: set(g["home"]) | set(g["away"])
                        for key, g in dd.groupby(["league", "season"])}
        country_of = {lg: v["country"] for lg, v in LEAGUES.items()}

        ratings: dict[str, float] = {}
        cur_season: dict[str, int] = {}
        cur_teams: dict[str, set] = {}
        country_teams: dict[str, set] = {}
        rh = np.empty(len(df))
        ra = np.empty(len(df))

        def new_season(lg: str, s: int):
            teams = participants[(lg, s)]
            prev = cur_teams.get(lg, set())
            if prev:
                mean = np.mean([ratings[t] for t in prev])
                left = prev - teams
                default = np.mean([ratings[t] for t in left]) if left else mean - 50
                for t in prev & teams:
                    ratings[t] += self.season_regress * (mean - ratings[t])
            else:
                default = self.start - self.tier_step * (LEAGUES.get(lg, {}).get("tier", 1) - 1)
            for t in teams - prev:
                ratings.setdefault(t, default)
            cur_season[lg], cur_teams[lg] = s, teams
            country = country_of.get(lg, lg)
            country_teams[country] = set().union(
                *(cur_teams[l] for l in cur_teams if country_of.get(l, l) == country))

        def spill(team: str, delta: float):
            members = country_teams.get(team.split("|", 1)[0])
            if members and self.country_spill:
                for t in members:
                    if t != team:
                        ratings[t] += self.country_spill * delta

        start = 0
        bounds = list(np.flatnonzero(dates[1:] != dates[:-1]) + 1) + [len(df)]
        for end in bounds:
            for i in range(start, end):
                if not is_cup[i] and cur_season.get(leagues[i]) != seasons[i]:
                    new_season(leagues[i], seasons[i])
            for i in range(start, end):
                if is_cup[i]:  # clubs from countries we do not cover start here
                    ratings.setdefault(homes[i], self.cup_start)
                    ratings.setdefault(aways[i], self.cup_start)
                rh[i], ra[i] = ratings[homes[i]], ratings[aways[i]]
            for i in range(start, end):
                exp_h = 1 / (1 + 10 ** (-(rh[i] + hadv[i] - ra[i]) / 400))
                delta = k_mult[i] * (score[i] - exp_h)
                ratings[homes[i]] += delta
                ratings[aways[i]] -= delta
                if is_cup[i]:
                    spill(homes[i], delta)
                    spill(aways[i], -delta)
            start = end
        self.ratings = ratings
        pre = pd.DataFrame({"elo_h": rh, "elo_a": ra}, index=df.index)
        out = matches.copy()
        out["elo_h"] = pre.loc[matches.index, "elo_h"].to_numpy() if len(matches) else []
        out["elo_a"] = pre.loc[matches.index, "elo_a"].to_numpy() if len(matches) else []
        if cups is not None and len(cups):
            c = cups.dropna(subset=["hg", "ag"]).copy()
            c["elo_h"] = pre.loc[[f"cup{i}" for i in c.index], "elo_h"].to_numpy()
            c["elo_a"] = pre.loc[[f"cup{i}" for i in c.index], "elo_a"].to_numpy()
            self.cup_history = c
        return out

    def diff(self, elo_h, elo_a, neutral=None) -> np.ndarray:
        h = self.home_adv if neutral is None else np.where(np.asarray(neutral, bool), 0.0,
                                                           self.home_adv)
        return (np.asarray(elo_h) + h - np.asarray(elo_a)) / 400.0

    # ---------------------------------------------------------- ordered logit
    @staticmethod
    def _fit_logit(d: np.ndarray, y: np.ndarray, x0=(2.0, -1.0, 0.0)) -> np.ndarray:
        def nll(p):
            probs = _ordered_logit_probs(p, d)
            return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).sum()

        return minimize(nll, x0=list(x0), method="Nelder-Mead",
                        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000}).x

    def fit_outcome_model(self, rated: pd.DataFrame, as_of=None, leagues=None,
                          burn_in_seasons: int = 1):
        """Ordered logit P(result | rating diff) per league (+ pooled fallback),
        fitted only on matches before as_of."""
        df = rated.dropna(subset=["elo_h"])
        if as_of is not None:
            df = df[df["date"] < pd.Timestamp(as_of)]
        first = df.groupby("league")["season"].transform("min")
        df = df[df["season"] >= first + burn_in_seasons]
        d_all = self.diff(df["elo_h"], df["elo_a"],
                          df["neutral"] if "neutral" in df else None)
        y_all = df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        leagues = list(leagues) if leagues is not None else list(df["league"].unique())
        lg = df["league"].to_numpy()
        counts = {league: int((lg == league).sum()) for league in leagues}
        # the pooled model is only needed as a fallback for leagues with little history
        if None not in self.logits or any(c < MIN_LEAGUE_MATCHES for c in counts.values()):
            self.logits[None] = self._fit_logit(d_all, y_all)
        for league in leagues:
            if counts[league] >= MIN_LEAGUE_MATCHES:
                m = lg == league
                self.logits[league] = self._fit_logit(d_all[m], y_all[m], self.logits[None])
            else:
                self.logits.pop(league, None)
        return self

    def probs_from_diff(self, d, league=None) -> np.ndarray:
        """[P(H), P(D), P(A)]; `league` may be a scalar or an array aligned with d."""
        d = np.asarray(d, float)
        if league is None or np.ndim(league) == 0:
            params = self.logits.get(league, self.logits[None])
            return _ordered_logit_probs(params, d)
        league = np.asarray(league)
        out = np.empty(d.shape + (3,))
        for lg in np.unique(league):
            m = league == lg
            out[m] = _ordered_logit_probs(self.logits.get(lg, self.logits[None]), d[m])
        return out

    # ------------------------------------------------------------ convenience
    def fit(self, matches: pd.DataFrame, as_of=None):
        train = matches if as_of is None else matches[matches["date"] < pd.Timestamp(as_of)]
        self.history = self.run(train)
        self.logits = {}
        self.fit_outcome_model(self.history)
        return self

    def rating(self, team: str, league: str, default=np.nan) -> float:
        return self.ratings.get(team_key(league, team), default)

    def predict(self, homes, aways, league: str) -> dict:
        mean = np.mean(list(self.ratings.values())) if self.ratings else self.start
        rh = np.array([self.rating(t, league, mean - 100) for t in homes])
        ra = np.array([self.rating(t, league, mean - 100) for t in aways])
        return {"probs": self.probs_from_diff(self.diff(rh, ra), league),
                "elo_h": rh, "elo_a": ra}

    def table(self, teams, league: str) -> pd.DataFrame:
        return (pd.DataFrame({"team": list(teams),
                              "elo": [self.rating(t, league) for t in teams]})
                .sort_values("elo", ascending=False).reset_index(drop=True))


def team_key(league: str, team: str) -> str:
    """Ratings are keyed by country so equal club names in two countries never clash."""
    return f"{LEAGUES.get(league, {}).get('country', league)}|{team}"


def team_keys(leagues: pd.Series, teams: pd.Series) -> pd.Series:
    country = leagues.map(lambda lg: LEAGUES.get(lg, {}).get("country", lg))
    return country + "|" + teams
