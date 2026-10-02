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


def test_plain_language_calls():
    from soccer.verdicts import btts_call, outcome_call, score_calls, total_call
    lab, cov, p = outcome_call([0.71, 0.2, 0.09], "Germany", "Serbia")
    assert lab == "Победа Germany" and cov == {0} and p == pytest.approx(0.71)
    lab, cov, p = outcome_call([0.25, 0.29, 0.46], "Greece", "Netherlands")  # no clear favourite
    assert lab == "Netherlands не проиграет" and cov == {1, 2} and p == pytest.approx(0.75)
    assert total_call(0.62)[:2] == ("Больше 2.5", True) and btts_call(0.4)[:2] == ("Нет", False)
    sc = score_calls([0.25, 0.29, 0.46], 0.6, 0.55, 2, 2)  # Greece 2:2 Netherlands
    assert sc == {"outcome": True, "total": True, "btts": True}


def test_coupon_suggestions():
    from soccer.coupons import market_probs, suggest
    rows = []
    for i in range(8):
        f = 1 + i * 0.15  # increasingly even matches
        rows.append({"event_id": i, "kickoff": pd.Timestamp("2030-01-01", tz="UTC"),
                     "home_raw": f"H{i}", "away_raw": f"A{i}", "competition_az": "X",
                     "mbs": 1 if i != 0 else 5, "o1": 1.25 * f, "ox": 5.0, "o2": 9.0 / f,
                     "o1x": 1.05, "o12": 1.15, "ox2": 3.2, "ou_line": 2.5, "o_over": 1.9,
                     "o_under": 1.9, "o_btts_yes": 1.8, "o_btts_no": 2.0,
                     **{f"p_{c}": 0.5 for c in ("o1", "ox", "o2", "o1x", "o12", "ox2", "o_over",
                                                "o_under", "o_btts_yes", "o_btts_no")}})
    df = pd.DataFrame(rows)
    mp = market_probs(df.iloc[1])
    assert mp["o1"] + mp["ox"] + mp["o2"] == pytest.approx(1.0)
    assert mp["o_over"] + mp["o_under"] == pytest.approx(1.0)
    coupons = suggest(df)
    assert coupons and coupons[0]["style"] == "safe"
    for cp in coupons:
        ids = list(cp["picks"]["event_id"])
        assert len(ids) == len(set(ids))            # one pick per match
        assert 0 not in ids                          # MBS 5 never fits a 3-4 match coupon
        assert cp["prob"] == pytest.approx(np.prod(cp["picks"]["p"]))
