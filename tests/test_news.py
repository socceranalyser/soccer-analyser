import json

import numpy as np
import pandas as pd

from soccer import news


def test_strength_score_signs_and_caps():
    f = [{"team": "home", "impact": -2, "certainty": 1.0},
         {"team": "away", "impact": -9, "certainty": 0.5},   # impact capped at -3
         {"team": "both", "impact": 3, "certainty": 1.0}]    # ignored for 1X2
    assert news.strength_score(f) == -2 + 1.5


def test_adjust_shifts_towards_helped_side_and_is_bounded():
    p = (0.45, 0.27, 0.28)
    (h, d, a), over, btts = news.adjust(p, 3.0, 0.5, 1, 0.5, w=(0.06, 0.10, 0))
    assert abs(h + d + a - 1) < 1e-9 and h > p[0] and a < p[2]
    assert over > 0.5 and btts > 0.5
    (h2, _, a2), _, _ = news.adjust(p, 100.0, w=(0.06, 0.10, 0))  # capped at MAX_SHIFT
    assert np.log(h2 / a2) - np.log(p[0] / p[2]) <= 2 * news.MAX_SHIFT + 1e-9
    assert news.adjust(p, 0.0, w=(0.06, 0.1, 0))[0] == pytest_approx(p)


def pytest_approx(p):
    import pytest
    return pytest.approx(p)


def test_apply_vetoes_avoided_match(tmp_path, monkeypatch):
    f = tmp_path / "news.json"
    kick = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=5)).isoformat()
    f.write_text(json.dumps([{"event_id": 1, "kickoff": kick, "s": 2.0, "goals_shift": 0,
                              "avoid": True, "summary": "ротация"}]), encoding="utf-8")
    monkeypatch.setattr(news, "NEWS_FILE", f)
    monkeypatch.setattr(news, "LOG_FILE", tmp_path / "log.csv")
    ev = pd.DataFrame([{"event_id": 1, "p_o1": 0.5, "p_ox": 0.25, "p_o2": 0.25,
                        "p_o_over": 0.5, "p_o_btts_yes": 0.5},
                       {"event_id": 2, "p_o1": 0.5, "p_ox": 0.25, "p_o2": 0.25,
                        "p_o_over": 0.5, "p_o_btts_yes": 0.5}])
    out = news.apply(ev)
    assert out.loc[0, "news_avoid"] and out.loc[0, "p_o1"] > 0.5
    assert not out.loc[1, "news_avoid"] and out.loc[1, "p_o1"] == 0.5
    assert out.loc[0, "news_note"] == "ротация"
