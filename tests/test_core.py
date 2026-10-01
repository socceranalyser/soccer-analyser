import numpy as np
import pandas as pd
import pytest

from soccer.backtest import walk_forward
from soccer.calibration import VectorScaling
from soccer.metrics import log_loss, rps
from soccer.models.dixon_coles import DixonColes
from soccer.models.elo import Elo
from soccer.probability import markets, outcome_probs, score_matrix
from soccer.simulation import simulate_season


def synthetic_league(n_teams=12, seasons=(2020, 2021, 2022), seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.3, n_teams)
    rows = []
    for s in seasons:
        day = pd.Timestamp(f"{s}-08-10")
        pairs = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(pairs)
        for k, (h, a) in enumerate(pairs):
            date = day + pd.Timedelta(days=3 * (k // (n_teams // 2)))
            lam = np.exp(0.15 + 0.25 + att[h] - dfn[a])
            mu = np.exp(0.15 + att[a] - dfn[h])
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            rows.append({"league": "X", "season": s, "date": date, "home": teams[h],
                         "away": teams[a], "hg": hg, "ag": ag,
                         "odds_h": np.nan, "odds_d": np.nan, "odds_a": np.nan,
                         "odds_o25": np.nan, "odds_u25": np.nan})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    df["match_id"] = df["season"].astype(str) + df["home"] + df["away"]
    return df, att, dfn


def test_score_matrix_sums_to_one_and_dc_shifts_draws():
    m0 = score_matrix(1.4, 1.1, 0.0)
    m1 = score_matrix(1.4, 1.1, -0.1)
    assert m0.sum() == pytest.approx(1.0)
    assert m1.sum() == pytest.approx(1.0)
    assert outcome_probs(m1)[1] > outcome_probs(m0)[1]  # rho<0 inflates 0:0 / 1:1
    mk = markets(m0)
    assert mk["p_home"] + mk["p_draw"] + mk["p_away"] == pytest.approx(1.0)
    assert mk["xg_home"] == pytest.approx(1.4, abs=1e-3)


def test_dixon_coles_gradient_and_recovery():
    df, att, dfn = synthetic_league(seasons=(2018, 2019, 2020, 2021, 2022))
    dc = DixonColes(xi=0.0, ridge=0.5, window_days=10_000).fit(df)
    assert dc.converged
    est = dc.ratings().set_index("team").loc[[f"T{i}" for i in range(len(att))]]
    assert np.corrcoef(est["attack"], att)[0, 1] > 0.85
    assert np.corrcoef(est["defence"], dfn)[0, 1] > 0.85
    assert dc.theta[-2] == pytest.approx(0.25, abs=0.08)  # home advantage


def test_bayesian_predictive_is_less_extreme():
    df, *_ = synthetic_league()
    point = DixonColes().fit(df).predict(["T0"], ["T1"])["probs"]
    bayes = DixonColes(n_samples=500).fit(df).predict(["T0"], ["T1"])["probs"]
    assert bayes.sum() == pytest.approx(1.0)
    assert np.abs(bayes - point).max() < 0.03


def test_unknown_team_gets_weak_proxy():
    df, *_ = synthetic_league()
    dc = DixonColes().fit(df)
    p = dc.predict(["NewTeam"], ["T0"])["probs"][0]
    assert p.sum() == pytest.approx(1.0)


def test_walk_forward_has_no_lookahead():
    """Changing results AFTER a match must not change that match's prediction."""
    df, *_ = synthetic_league()
    test_day = df[df.season == 2022]["date"].sort_values().iloc[30]
    tampered = df.copy()
    future = tampered["date"] >= test_day
    tampered.loc[future, ["hg", "ag"]] = 9
    kw = dict(test_seasons=[2022], models=("dixon_coles", "elo"), verbose=False, freq="D")
    a = walk_forward(df, **kw)
    b = walk_forward(tampered, **kw)
    sel = lambda p: p[p["date"] == test_day].sort_values(["model", "match_id"])
    pa, pb = sel(a), sel(b)
    assert len(pa) > 0
    np.testing.assert_allclose(pa[["p_home", "p_draw", "p_away"]].to_numpy(),
                               pb[["p_home", "p_draw", "p_away"]].to_numpy())


def test_elo_ratings_are_zero_sum_within_league():
    df, *_ = synthetic_league(seasons=(2022,))
    elo = Elo(season_regress=0.0)
    elo.run(df)
    assert np.mean(list(elo.ratings.values())) == pytest.approx(1500.0)


def test_metrics_known_values():
    p = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    y = np.array([0, 2])
    assert rps(p[:1], y[:1]) == pytest.approx(0.0)
    assert rps(p[1:], y[1:]) == pytest.approx(((1 / 3) ** 2 + (2 / 3) ** 2) / 2)
    assert log_loss(p[1:], y[1:]) == pytest.approx(np.log(3))


def test_season_simulation_probabilities():
    df, *_ = synthetic_league(seasons=(2021, 2022))
    played = df[(df.season == 2022)].iloc[:40]
    dc = DixonColes().fit(df[df.season == 2021])
    res = simulate_season(dc, played, n_sims=2000)
    pp = res["position_probs"]
    np.testing.assert_allclose(pp.sum(axis=0), 1.0)
    np.testing.assert_allclose(pp.sum(axis=1), 1.0)


def test_vector_scaling_identity_on_calibrated_data():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([4, 3, 3], size=5000)
    y = np.array([rng.choice(3, p=row) for row in p])
    cal = VectorScaling().fit(p, y)
    assert np.abs(cal.transform(p) - p).max() < 0.05
