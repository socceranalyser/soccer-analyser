"""One object that knows every model and can forecast any fixture.

Headline 1X2 per competition type (each choice backed by a backtest, see CLAUDE.md):
  * domestic league : 50/50 average of Dixon-Coles and Elo (beats both on 2023-25)
  * European cups   : cross-league Elo (Dixon-Coles has no common scale across leagues)
  * national teams  : Elo (clearly better than Dixon-Coles on 2023-26)
The exact-score matrix (DC, or an Elo-driven Poisson model for cups) is rescaled to the
headline 1X2 so all numbers shown for a match are consistent.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import DATA_DIR, LEAGUES
from .data import implied_probs, load_matches
from .euro import EURO_LEAGUE, load_euro
from .fixtures import (build_schedule, current_cup_matches, fresh_league_results,
                       fresh_national_results)
from .models.dixon_coles import DixonColes
from .models.elo import Elo
from .calibration import VectorScaling
from .national import (NationalElo, fit_national_dc, load_international,
                       national_walk_forward)
from .pipeline import LIVE_DC, current_season, current_teams
from .probability import markets, rescale_to_outcomes, score_matrix
from .tuned import ELO_PARAMS

ENSEMBLE_DC_WEIGHT = 0.5
# injuries/suspensions (scripts/test_injuries.py): +0.025 log-odds to the home win per extra
# player missing for the away side (and vice versa); fitted 2023/24, holdout 2024/25 -0.0016
INJURY_BETA = 0.025
INJURY_LEAGUES = {"E0", "SP1", "I1", "D1", "F1", "E1", "N1", "T1"}
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


@lru_cache(maxsize=1)
def _goals_cal() -> dict | None:
    try:
        return json.loads((DATA_DIR / "goals_calibration.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def calibrate_goals(m: np.ndarray, league: str) -> np.ndarray:
    """Platt-calibrated over-2.5 (scripts/tune_goals_calibration.py) applied as a goal tilt of
    the score matrix; 1X2 unchanged. The raw model was too sure of low-scoring games."""
    cal = _goals_cal()
    if cal is None:
        return m
    from .probability import goal_tilt, outcome_probs, over_prob
    p = float(np.clip(over_prob(m), 1e-6, 1 - 1e-6))
    z = cal["a"] + cal["b"] * np.log(p / (1 - p)) + cal["league"].get(league, 0.0)
    return goal_tilt(m, 1 / (1 + np.exp(-z)), outcome_probs(m))


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
        from .models.elo import team_keys
        club_keys = set(team_keys(self.base_matches["league"], self.base_matches["home"])) |             set(euro_hist["home_key"]) | set(euro_hist["away_key"])
        self.schedule = build_schedule(self.base_matches, club_keys,
                                       set(self.intl_base["home"]) | set(self.intl_base["away"]),
                                       refresh=refresh)
        self._add_recent_scores(log)
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
        # calibration learnt from the model's own out-of-sample forecasts (2015 -> now);
        # backtest: Elo test logloss 0.8643 -> 0.8630 (DC blends were rejected)
        self.nat_cal = self._national_calibration()
        log("national models fitted")
        self._dc: dict[str, DixonColes] = {}
        self._injuries: dict = {}
        self.built_at = pd.Timestamp.now()

    def _national_calibration(self, max_age_days: int = 7):
        """VectorScaling fitted on out-of-sample national forecasts; cached for a week
        (the walk-forward behind it takes ~20 s and changes little day to day)."""
        import json
        from .config import DATA_DIR
        path = DATA_DIR / "nat_calibration.json"
        try:
            cached = json.loads(path.read_text())
            if pd.Timestamp.now() - pd.Timestamp(cached["fitted"]) < pd.Timedelta(days=max_age_days):
                cal = VectorScaling(l2=1.0)
                cal.params = np.array(cached["params"])
                return cal
        except (OSError, ValueError, KeyError):
            pass
        try:
            oos = national_walk_forward(self.intl, "2015-01-01", "2100-01-01", models=("elo",))
            y = oos["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
            cal = VectorScaling(l2=1.0).fit(oos[["p_home", "p_draw", "p_away"]].to_numpy(), y)
            path.write_text(json.dumps({"fitted": str(pd.Timestamp.now()),
                                        "params": cal.params.tolist(), "n": int(len(y))}))
            return cal
        except Exception:
            return None

    def _add_recent_scores(self, log, days: int = 4):
        """Fill in scores of the last few days from misli.az (published within minutes),
        so ratings learn from results before the main sources catch up."""
        from .misli import attach_results
        from .scores import fetch_results
        today = pd.Timestamp.now().normalize()
        recent = self.schedule["date"].between(today - pd.Timedelta(days=days), today) & \
            ~self.schedule["played"]
        if not recent.any():
            return
        try:
            res = fetch_results([today - pd.Timedelta(days=d) for d in range(days + 1)])
        except Exception:
            return
        done = res[res["ended"]] if len(res) else res
        upd = attach_results(self.schedule[recent], done)
        from . import storage  # scores misli no longer serves -> our results table
        for idx in upd.index[~upd["played"]]:
            r = upd.loc[idx]
            found = storage.find_result(r["home"], r["away"], r["date"])
            if found is not None:
                upd.loc[idx, ["hg", "ag", "played"]] = [found[0], found[1], True]
        newly = upd["played"] & ~self.schedule.loc[recent, "played"]
        self.schedule.loc[upd.index, ["hg", "ag", "played"]] = upd[["hg", "ag", "played"]]
        log(f"misli scores added: {int(newly.sum())}")

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
                 home_key=None, away_key=None, odds=None, date=None) -> dict:
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
            absent = self.absences(competition, home, away, date)
            if absent is not None:  # backtest 2024/25: logloss 0.9860 -> 0.9844
                d = len(absent["away"]) - len(absent["home"])
                z = np.log(np.clip(head, 1e-9, 1)) + np.array([INJURY_BETA * d, 0.0,
                                                              -INJURY_BETA * d])
                head = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
            m = rescale_to_outcomes(m_dc, head)
            m = calibrate_goals(m, competition)  # test 2025-26: O/U 0.6832 -> 0.6810
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
            models["Elo"] = p["probs"][0]
            head = self.nat_cal.transform(p["probs"])[0] if self.nat_cal is not None \
                else p["probs"][0]
            m_dc = self.nat_dc.score_matrices([home], [away], np.array([neutral]))[0]
            pd_ = markets(m_dc)
            models["Dixon-Coles"] = np.array([pd_["p_home"], pd_["p_draw"], pd_["p_away"]])
            m = rescale_to_outcomes(m_dc, head)
            elo_h, elo_a = p["elo_h"][0], p["elo_a"][0]
            known = home in self.nat_elo.ratings and away in self.nat_elo.ratings
        from .analysis import category, live_calibrator
        cal = live_calibrator(category(competition))
        if cal is not None:  # learnt from our own live mistakes (only when it proved better)
            head = cal.transform(np.asarray(head)[None, :])[0]
            m = rescale_to_outcomes(m, head)
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
                             r.get("away_key"), odds, r.get("date"))

    # ------------------------------------------------------------- absences
    def absences(self, league: str, home: str, away: str, date) -> dict | None:
        """Players ruled out ('Missing Fixture') and doubtful for both teams (API-Football).
        None when there is no data (league not covered, no key, no date)."""
        from . import apifootball as af
        from .names import best_match
        if league not in INJURY_LEAGUES or date is None or pd.isna(date) or not af.available():
            return None
        day = pd.Timestamp(date).normalize()
        if day not in self._injuries:
            try:
                self._injuries[day] = af.injuries_on(day)
            except Exception:
                self._injuries[day] = pd.DataFrame()
        inj = self._injuries[day]
        if inj.empty:
            return None
        inj = inj[inj["league_id"].map(af.LEAGUE_IDS) == league]
        if inj.empty:
            return None
        names = list(inj["team"].unique())
        out = {}
        for side, team in (("home", home), ("away", away)):
            hit, _ = best_match(team, names, None, 0.72)
            rows = inj[inj["team"] == hit] if hit else inj.iloc[:0]
            rows = rows.drop_duplicates("player_id")
            fmt = lambda df: [f"{r.player} ({r.reason})" for r in df.itertuples()]
            out[side] = fmt(rows[rows["type"] == "Missing Fixture"])
            out[f"{side}_doubt"] = fmt(rows[rows["type"] == "Questionable"])
        return out

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
                             "p_over25": mk["totals"][2.5], "p_btts": mk["btts"],
                             "xg_home": mk["xg_home"],
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
        s = self.schedule.assign(sched_date=self.schedule["date"])  # storage key date
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
                           top_score=f"{mk['top_scores'][0][0]} ({mk['top_scores'][0][1]:.0%})",
                           known=f["known"])
            rows.append(row)
        return pd.DataFrame(rows)
