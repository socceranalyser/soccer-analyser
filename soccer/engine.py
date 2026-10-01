"""One object that knows every model and can forecast any fixture.

Headline 1X2 per competition type (each choice backed by a backtest, see CLAUDE.md):
  * domestic league : 50/50 average of Dixon-Coles and Elo (beats both on 2023-25)
  * European cups   : cross-league Elo (Dixon-Coles has no common scale across leagues)
  * national teams  : Elo (clearly better than Dixon-Coles on 2023-26)
The exact-score matrix (DC, or an Elo-driven Poisson model for cups) is rescaled to the
headline 1X2 so all numbers shown for a match are consistent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import LEAGUES
from .data import implied_probs, load_matches
from .euro import EURO_LEAGUE, load_euro
from .fixtures import (build_schedule, current_cup_matches, fresh_league_results,
                       fresh_national_results)
from .models.dixon_coles import DixonColes
from .models.elo import Elo
from .national import NationalElo, fit_national_dc, load_international
from .pipeline import LIVE_DC, current_season, current_teams
from .probability import markets, rescale_to_outcomes, score_matrix
from .tuned import ELO_PARAMS

ENSEMBLE_DC_WEIGHT = 0.5
NAT_LEAGUE_CODE = "INT"
MODEL_KEYS = {"Dixon-Coles": "dixon_coles", "Elo": "elo", "Elo (межлиговый)": "elo",
              "Букмекеры": "market"}


@dataclass
class EloGoals:
    """Poisson goals from an Elo difference: log λ = a0 + a1·d, log μ = b0 - b1·d."""
    params: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.6, 0.1, 0.6]))

    def fit(self, d: np.ndarray, hg: np.ndarray, ag: np.ndarray):
        def nll(p):
            el, em = p[0] + p[1] * d, p[2] - p[3] * d
            return -(hg * el - np.exp(el) + ag * em - np.exp(em)).sum()

        self.params = minimize(nll, self.params, method="BFGS").x
        return self

    def rates(self, d):
        a0, a1, b0, b1 = self.params
        return np.exp(a0 + a1 * np.asarray(d)), np.exp(b0 - b1 * np.asarray(d))


class Engine:
    def __init__(self, refresh: bool = False, verbose: bool = False):
        t0 = time.time()
        log = (lambda *a: print(*a, f"[{time.time() - t0:.0f}s]", flush=True)) if verbose \
            else (lambda *a: None)
        self.base_matches = load_matches(refresh=refresh)
        euro_hist = load_euro(self.base_matches, refresh=refresh)
        self.intl_base = load_international(refresh=refresh, since="1960-01-01")
        log("data loaded")

        # first pass: ratings to map schedule names, then the schedule itself
        tmp_elo = Elo(**ELO_PARAMS)
        tmp_elo.run(self.base_matches, euro_hist)
        self.schedule = build_schedule(self.base_matches, tmp_elo.ratings.keys(),
                                       set(self.intl_base["home"]) | set(self.intl_base["away"]),
                                       refresh=refresh)
        log("schedule built", len(self.schedule))

        fresh = fresh_league_results(self.base_matches, self.schedule)
        self.matches = (pd.concat([self.base_matches, fresh], ignore_index=True)
                        .sort_values("date").reset_index(drop=True)
                        if len(fresh) else self.base_matches)
        self.n_fresh = len(fresh)
        cur_cups = current_cup_matches(self.schedule)
        if len(cur_cups):
            euro_hist = euro_hist[euro_hist["season"] < cur_cups["season"].min()]
        self.euro = pd.concat([euro_hist, cur_cups], ignore_index=True).sort_values("date")
        fresh_int = fresh_national_results(self.intl_base, self.schedule)
        self.intl = (pd.concat([self.intl_base, fresh_int], ignore_index=True)
                     if len(fresh_int) else self.intl_base)

        self.elo = Elo(**ELO_PARAMS)
        self.elo.history = self.elo.run(self.matches, self.euro)
        self.elo.fit_outcome_model(self.elo.history)
        cup = self.elo.cup_history
        self.elo.fit_outcome_model(cup, leagues=[EURO_LEAGUE])
        d = self.elo.diff(cup["elo_h"], cup["elo_a"], cup["neutral"])
        self.cup_goals = EloGoals().fit(d, cup["hg"].to_numpy(float), cup["ag"].to_numpy(float))
        log("club elo fitted")

        self.nat_elo = NationalElo().fit(self.intl)
        self.nat_dc = fit_national_dc(self.intl, n_samples=200)
        log("national models fitted")
        self._dc: dict[str, DixonColes] = {}
        self.built_at = pd.Timestamp.now()

    # ------------------------------------------------------------------ models
    def dc(self, league: str) -> DixonColes:
        if league not in self._dc:
            self._dc[league] = DixonColes(**LIVE_DC).fit(
                self.matches[self.matches["league"] == league],
                season=current_season(self.matches, league),
                season_teams=current_teams(self.matches, league))
        return self._dc[league]

    # --------------------------------------------------------------- forecast
    def forecast(self, kind: str, competition: str, home: str, away: str, neutral=False,
                 home_key=None, away_key=None, odds=None) -> dict:
        """Full forecast for one match. Returns probs (headline), per-model probs,
        the score matrix and derived markets."""
        models = {}
        if kind == "league":
            dc = self.dc(competition)
            m_dc = dc.score_matrices([home], [away])[0]
            p_dc = markets(m_dc)
            models["Dixon-Coles"] = np.array([p_dc["p_home"], p_dc["p_draw"], p_dc["p_away"]])
            models["Elo"] = self.elo.predict([home], [away], competition)["probs"][0]
            head = ENSEMBLE_DC_WEIGHT * models["Dixon-Coles"] + \
                (1 - ENSEMBLE_DC_WEIGHT) * models["Elo"]
            m = rescale_to_outcomes(m_dc, head)
            elo_h, elo_a = self.elo.rating(home, competition), self.elo.rating(away, competition)
            known = home in dc._idx and away in dc._idx
        elif kind == "cup":
            hk, ak = home_key or f"?|{home}", away_key or f"?|{away}"
            rh = self.elo.ratings.get(hk, self.elo.cup_start)
            ra = self.elo.ratings.get(ak, self.elo.cup_start)
            d = self.elo.diff([rh], [ra], [neutral])
            head = self.elo.probs_from_diff(d, EURO_LEAGUE)[0]
            models["Elo (межлиговый)"] = head
            lam, mu = self.cup_goals.rates(d)
            m = rescale_to_outcomes(score_matrix(lam[0], mu[0], -0.05), head)
            elo_h, elo_a = rh, ra
            known = hk in self.elo.ratings and ak in self.elo.ratings
        else:  # national
            p = self.nat_elo.predict([home], [away], [neutral])
            head = p["probs"][0]
            models["Elo"] = head
            m_dc = self.nat_dc.score_matrices([home], [away], np.array([neutral]))[0]
            pd_ = markets(m_dc)
            models["Dixon-Coles"] = np.array([pd_["p_home"], pd_["p_draw"], pd_["p_away"]])
            m = rescale_to_outcomes(m_dc, head)
            elo_h, elo_a = p["elo_h"][0], p["elo_a"][0]
            known = home in self.nat_elo.ratings and away in self.nat_elo.ratings
        if odds is not None and np.all(np.isfinite(odds)) and np.all(np.asarray(odds) > 1):
            models["Букмекеры"] = implied_probs([odds[0]], [odds[1]], [odds[2]])[0]
        mk = markets(m)
        return {"probs": np.array([mk["p_home"], mk["p_draw"], mk["p_away"]]),
                "models": models, "matrix": m, "markets": mk, "elo_h": elo_h, "elo_a": elo_a,
                "known": known}

    def forecast_row(self, r) -> dict:
        odds = None
        if "odds_h" in r and pd.notna(r.get("odds_h")):
            odds = np.array([r["odds_h"], r["odds_d"], r["odds_a"]], float)
        return self.forecast(r["kind"], r["competition"], r["home"], r["away"],
                             bool(r.get("neutral", False)), r.get("home_key"),
                             r.get("away_key"), odds)

    # --------------------------------------------------------------- storage
    def save_results(self) -> int:
        """Every finished match we know of, for scoring stored forecasts."""
        from . import storage
        n = storage.save_results(self.matches)
        n += storage.save_results(self.intl.assign(league=NAT_LEAGUE_CODE))
        s = self.schedule[self.schedule["played"]]
        s = s.assign(league=s["competition"], hg=s["hg"].astype(int), ag=s["ag"].astype(int),
                     result=np.select([s.hg > s.ag, s.hg == s.ag], ["H", "D"], "A"))
        return n + storage.save_results(s)

    def record_days(self, dates) -> int:
        """Store forecasts for matches that have not kicked off yet (one row per model)."""
        from . import storage
        now = pd.Timestamp.now(tz=self.schedule["kickoff"].dropna().dt.tz).floor("min") \
            if self.schedule["kickoff"].notna().any() else None
        rows = []
        for date in dates:
            day = self.schedule[self.schedule["date"] == pd.Timestamp(date).normalize()]
            for _, r in day.iterrows():
                if r["played"] or (now is not None and pd.notna(r["kickoff"])
                                   and r["kickoff"] <= now):
                    continue
                if not (bool(r.get("home_ok", True)) and bool(r.get("away_ok", True))) \
                        and r["kind"] != "cup":
                    continue
                try:
                    f = self.forecast_row(r)
                except Exception:
                    continue
                base = {"league": r["competition"], "season": int(r["season"]) if pd.notna(
                    r.get("season")) else int(r["date"].year), "date": r["date"],
                        "home": r["home"], "away": r["away"]}
                mk = f["markets"]
                rows.append({**base, "model": "final", "p_home": mk["p_home"],
                             "p_draw": mk["p_draw"], "p_away": mk["p_away"],
                             "p_over25": mk["totals"][2.5], "xg_home": mk["xg_home"],
                             "xg_away": mk["xg_away"]})
                for name, p in f["models"].items():
                    rows.append({**base, "model": MODEL_KEYS.get(name, name), "p_home": p[0],
                                 "p_draw": p[1], "p_away": p[2], "p_over25": None,
                                 "xg_home": None, "xg_away": None})
        if not rows:
            return 0
        return storage.save_run("live", "live", pd.DataFrame(rows),
                                description="ежедневные прогнозы", replace_run=False)

    # ------------------------------------------------------------------ days
    def day(self, date, tz=None) -> pd.DataFrame:
        """All fixtures on a date (in time zone tz; default: this computer's) with headline
        forecasts, one row per match."""
        date = pd.Timestamp(date).normalize()
        s = self.schedule
        if tz is not None:
            kick = s["kickoff"].dt.tz_convert(tz)
            local_date = kick.dt.tz_localize(None).dt.normalize().fillna(s["date"])
            s = s.assign(kickoff=kick, date=local_date)
        s = s[s["date"] == date]
        rows = []
        for _, r in s.iterrows():
            ok = bool(r.get("home_ok", True)) and bool(r.get("away_ok", True))
            if r["kind"] == "national":
                ok = r["home"] in self.nat_elo.ratings and r["away"] in self.nat_elo.ratings
            f = self.forecast_row(r) if ok or r["kind"] == "cup" else None
            row = r.to_dict()
            if f is not None:
                mk = f["markets"]
                row.update(p_home=mk["p_home"], p_draw=mk["p_draw"], p_away=mk["p_away"],
                           xg_home=mk["xg_home"], xg_away=mk["xg_away"],
                           p_over25=mk["totals"][2.5], p_btts=mk["btts"],
                           top_score=mk["top_scores"][0][0], known=f["known"])
            rows.append(row)
        return pd.DataFrame(rows)
