import numpy as np
import pandas as pd
import pytest

from soccer.euro import _ft_score, parse_openfootball
from soccer.models.dixon_coles import DixonColes
from soccer.models.elo import Elo
from soccer.names import CLUB_ALIASES, best_match
from soccer.probability import outcome_probs, rescale_to_outcomes, score_matrix

from test_core import synthetic_league

SAMPLE = """= UEFA Champions League 2023/24

▪ Group, Matchday 1
  Tue Sep 19 2023
    18:45  AC Milan (ITA)          v Newcastle United FC (ENG)  0-0
           BSC Young Boys (SUI)    v RB Leipzig (GER)         1-3 (1-1)
▪ Finals, Quarterfinals
  Tue Apr 9
    21:00  Manchester City FC (ENG) v Real Madrid CF (ESP)     3-4 pen. 1-1 a.e.t. (1-1, 0-1)
▪ Finals, Final
  Sat Jun 1
    21:00  Borussia Dortmund (GER) v Real Madrid CF (ESP)     0-2 (0-0)
"""


def test_openfootball_parser_scores_years_and_neutral_final():
    df = parse_openfootball(SAMPLE, 2023, "ЛЧ")
    assert len(df) == 4
    assert list(df["hg"]) == [0, 1, 1, 0] and list(df["ag"]) == [0, 3, 1, 2]  # 90-min scores
    assert df["date"].iloc[2] == pd.Timestamp("2024-04-09")  # year rolls over after December
    assert list(df["neutral"]) == [False, False, False, True]
    assert _ft_score("2-3 a.e.t. (1-0, 1-0)") == (1, 0)


def test_cup_name_matching_is_strict():
    keys = ["Нидерланды|Sparta", "CZE|AC Sparta Praha", "Италия|Inter", "Испания|Ath Madrid",
            "Германия|RB Leipzig"]
    m = lambda n: best_match(n, keys, CLUB_ALIASES, threshold=0.82, both_ways=False)[0]
    assert m("Sparta Praha") == "CZE|AC Sparta Praha"
    assert m("Inter Escaldes") is None
    assert m("Atleti") == "Испания|Ath Madrid"
    assert m("Leipzig") == "Германия|RB Leipzig"


def test_rescale_to_outcomes_hits_target():
    m = score_matrix(1.6, 1.0, -0.05)
    target = np.array([0.5, 0.3, 0.2])
    r = rescale_to_outcomes(m, target)
    assert r.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(outcome_probs(r), target, atol=1e-9)


def test_dixon_coles_neutral_removes_home_advantage():
    df, *_ = synthetic_league()
    dc = DixonColes().fit(df)
    lam_h, mu_h = dc.rates(["T0"], ["T1"])
    lam_n, mu_n = dc.rates(["T0"], ["T1"], neutral=[True])
    assert lam_n[0] < lam_h[0] and mu_n[0] == pytest.approx(mu_h[0])


def test_cup_results_spill_to_whole_country():
    df, *_ = synthetic_league(seasons=(2022,))
    cups = pd.DataFrame({"date": [df["date"].max() + pd.Timedelta(days=1)] * 3,
                         "league": "EUR", "season": 2022, "home_key": ["X|T0"] * 3,
                         "away_key": ["ZZZ|Foreign"] * 3, "hg": [3, 3, 3], "ag": [0, 0, 0],
                         "neutral": False})
    base = Elo(country_spill=0.0)
    base.run(df)
    spill = Elo(country_spill=0.5)
    spill.run(df, cups)
    # T1 never played abroad, yet its rating rises with its compatriot's cup wins
    assert spill.ratings["X|T1"] > base.ratings["X|T1"]
    assert spill.ratings["ZZZ|Foreign"] < spill.cup_start
