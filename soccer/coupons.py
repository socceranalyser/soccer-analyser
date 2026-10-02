"""Ready-made accumulator suggestions from misli.az odds + model probabilities.

Three styles, each with distinct matches where possible:
  safe      - the most likely outcomes (odds >= 1.20 so the coupon is not all 1.03s)
  balanced  - likely outcomes (p >= 55%) with the best probability x odds
  bold      - bigger odds (>= 1.80) where the model sees the most value
Coupon probability assumes the matches are independent (true for different matches).

Probabilities: the bookmaker's (margin removed) sharpened by MARKET_POWER. A backtest on
47k league matches (scripts/tune_market_blend.py) gave the model a weight of ~0 next to
closing odds: market 1.0015 -> pooled 1.0011 log loss, model alone 1.0194. So the coupon
chance comes from the market; the model is shown as a second opinion only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .misli import MARKET_LABELS

ODDS_COLS = ["o1", "ox", "o2", "o1x", "o12", "ox2", "o_over", "o_under", "o_btts_yes",
             "o_btts_no"]

MARKET_POWER = 1.10  # fitted b in p ∝ market^b (model weight ~0)
GROUPS = [("o1", "ox", "o2"), ("o_over", "o_under"), ("o_btts_yes", "o_btts_no")]


def market_probs(r) -> dict:
    """Margin-free, slightly sharpened probabilities for every misli market of a match."""
    out = {}
    for g in GROUPS:
        odds = np.array([r.get(c) for c in g], float)
        if np.isnan(odds).any() or (odds <= 1).any():
            continue
        z = (1 / odds) ** MARKET_POWER
        out.update(zip(g, z / z.sum()))
    if {"o1", "ox", "o2"} <= out.keys():
        out.update(o1x=out["o1"] + out["ox"], o12=out["o1"] + out["o2"],
                   ox2=out["ox"] + out["o2"])
    return out


STYLES = {
    "safe": {"title": "🛡️ Купон дня — максимально надёжный", "size": 3, "min_odds": 1.20,
             "min_p": 0.0, "score": lambda p, o: p},
    "balanced": {"title": "⚖️ Сбалансированный", "size": 4, "min_odds": 1.40, "min_p": 0.55,
                 "score": lambda p, o: p * o},
    "bold": {"title": "🚀 Смелый — высокий коэффициент", "size": 3, "min_odds": 1.90,
             "min_p": 0.35, "score": lambda p, o: p * o},
}


def _label(col: str, line) -> str:
    line = 2.5 if line is None or pd.isna(line) else line
    return MARKET_LABELS[col].format(line=f"{line:g}")


def candidates(events: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, market outcome) with misli odds and model probability."""
    rows = []
    for _, r in events.iterrows():
        mkt = market_probs(r)
        for c in ODDS_COLS:
            o, p, pm = r.get(c), mkt.get(c), r.get(f"p_{c}")
            if pd.isna(o) or p is None or o <= 1:
                continue
            rows.append({"event_id": int(r["event_id"]), "kickoff": r["kickoff"],
                         "match": f"{r['home_raw']} — {r['away_raw']}",
                         "comp": r.get("competition_az", ""), "market": c,
                         "label": _label(c, r.get("ou_line")), "odds": float(o), "p": float(p),
                         "p_model": None if pd.isna(pm) else float(pm),
                         "mbs": int(r.get("mbs", 1)),
                         "home": r.get("home") if pd.notna(r.get("home")) else None,
                         "away": r.get("away") if pd.notna(r.get("away")) else None,
                         "line": None if pd.isna(r.get("ou_line")) else float(r["ou_line"])})
    return pd.DataFrame(rows)


def build(cands: pd.DataFrame, style: str, exclude=frozenset()) -> dict | None:
    st = STYLES[style]
    c = cands[(cands["odds"] >= st["min_odds"]) & (cands["p"] >= st["min_p"])
              & (cands["mbs"] <= st["size"]) & ~cands["event_id"].isin(exclude)
              & cands["p_model"].notna()]  # only matches our model also covers
    if c.empty:
        return None
    c = c.assign(score=[st["score"](p, o) for p, o in zip(c["p"], c["odds"])])
    best = c.sort_values("score", ascending=False).drop_duplicates("event_id")
    picks = best.head(st["size"])
    if len(picks) < st["size"]:
        return None
    total = float(np.prod(picks["odds"]))
    prob = float(np.prod(picks["p"]))
    return {"style": style, "title": st["title"], "picks": picks.drop(columns="score"),
            "total_odds": total, "prob": prob, "ev": prob * total - 1}


def suggest(events: pd.DataFrame, source: str = "model") -> list[dict]:
    """Safe, balanced and bold coupons, using different matches where possible.

    source="model": chances are our model's own analysis (bookmaker odds only set the price);
    source="market": chances from the bookmaker's margin-free odds.
    """
    cands = candidates(events)
    if cands.empty:
        return []
    cands = cands.assign(p_market=cands["p"])
    if source == "model":
        cands = cands[cands["p_model"].notna()].assign(p=lambda d: d["p_model"].astype(float))
    out, used = [], set()
    for style in ("safe", "balanced", "bold"):
        cp = build(cands, style, frozenset(used)) or build(cands, style)
        if cp:
            out.append(cp)
            used |= set(cp["picks"]["event_id"])
    return out


# ------------------------------------------------------------ results
def pick_result(pick: dict, scores: pd.DataFrame | None = None):
    """(hg, ag) of a coupon pick's match if it has finished, else None.

    Looks in our results table first, then in livescore/misli scores by name and kick-off.
    """
    from . import storage
    from .misli import attach_results
    kick = pd.Timestamp(pick["kickoff"])
    if pick.get("home") and pick.get("away"):
        res = storage.find_result(pick["home"], pick["away"], kick.tz_localize(None)
                                  if kick.tzinfo else kick)
        if res is not None:
            return res
    if scores is None:
        from .scores import fetch_results
        d = kick.tz_convert("UTC").tz_localize(None).normalize() if kick.tzinfo else kick.normalize()
        scores = fetch_results([d - pd.Timedelta(days=1), d, d + pd.Timedelta(days=1)])
    if scores is None or scores.empty:
        return None
    home_raw, _, away_raw = pick["match"].partition(" — ")
    row = pd.DataFrame([{"home": pick.get("home") or home_raw, "away": pick.get("away") or away_raw,
                         "home_src": home_raw, "away_src": away_raw,
                         "kickoff": kick if kick.tzinfo else kick.tz_localize("UTC"),
                         "played": False, "hg": np.nan, "ag": np.nan}])
    out = attach_results(row, scores[scores["ended"]])
    r = out.iloc[0]
    return (int(r["hg"]), int(r["ag"])) if r["played"] and pd.notna(r["hg"]) else None


def coupon_status(picks: list[dict], scores: pd.DataFrame | None = None) -> dict:
    """Per-pick results and overall state: 'won' / 'lost' / 'pending'."""
    from .misli import outcome_won
    rows = []
    for p in picks:
        res = pick_result(p, scores)
        won = None if res is None else outcome_won(p["market"], res[0], res[1], p.get("line"))
        rows.append({**p, "result": res, "won": won})
    if any(r["won"] is False for r in rows):
        state = "lost"
    elif rows and all(r["won"] is True for r in rows):
        state = "won"
    else:
        state = "pending"
    return {"picks": rows, "state": state}
