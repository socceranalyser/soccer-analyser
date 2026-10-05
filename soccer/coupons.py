"""Ready-made accumulator suggestions from misli.az odds + model probabilities.

Styles (1, 2 and 3 matches), each with distinct matches where possible; only outcomes that
both our model (with news) and the market rate highly. Coupon probability assumes the matches
are independent (true for different matches).

Chance = the bookmaker's margin-free probability (MARKET_POWER): next to the odds our model
adds no accuracy (scripts/tune_market_blend.py), it acts as a filter instead. History of
each style on 2023-25 with real odds: HISTORY (scripts/backtest_coupons.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .misli import MARKET_LABELS

ODDS_COLS = ["o1", "ox", "o2", "o1x", "o12", "ox2", "o_over", "o_under", "o_btts_yes",
             "o_btts_no"]

# 1.10 fits ALL outcomes best, but for the favourites a coupon picks it overstated the chance
# (scripts/backtest_coupons.py, 2023-25: promised 77.7% / came true 75.0%); with 1.0 the
# promise matches reality (75.8% / 75.8%).
MARKET_POWER = 1.0
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


# agree_min: both our model AND the market must give the outcome at least this chance.
# History (23k matches, scripts check 2026-10-03): when both agree at 75-85% the call came
# true 80.3%, at 85%+ 92%; when they disagree the hit rate drops to 63-66%.
# A ladder from "almost sure, small odds" to "rare, big odds". score(p, odds, p_model) ranks the
# outcomes that pass the filters (odds window + both our model AND the market >= agree_min).
# "value" = our model rates the outcome higher than the bookmaker: on higher odds this is where
# our own analysis helps (2-match 1.5-2.2: -7.2% vs -11.1% when ranked by plain likelihood).
STYLES = {
    "single": {"title": "💎 Ставка дня — одна самая надёжная игра", "size": 1, "min_odds": 1.25,
               "max_odds": 99, "min_p": 0.0, "agree_min": 0.75, "no_friendly": True,
               "score": lambda p, o, pm: p},
    "double": {"title": "✌️ Двойной — две надёжные игры", "size": 2, "min_odds": 1.30,
               "max_odds": 1.60, "min_p": 0.0, "agree_min": 0.62, "no_friendly": True,
               "score": lambda p, o, pm: min(p, pm)},
    "medium": {"title": "⚖️ Средний — две игры с хорошим коэффициентом", "size": 2,
               "min_odds": 1.50, "max_odds": 2.20, "min_p": 0.0, "agree_min": 0.50,
               "score": lambda p, o, pm: pm - p},
    "big": {"title": "🚀 Крупный — три игры, большой коэффициент", "size": 3, "min_odds": 1.50,
            "max_odds": 2.20, "min_p": 0.0, "agree_min": 0.50,
            "score": lambda p, o, pm: pm - p},
}
# How such coupons did on 2023-25 league matches with real odds (scripts/backtest_coupons.py):
# share of days the coupon came true, average total odds, longest losing streak (days), money
# result per stake. Promised chances matched reality within 1-2 points for every style.
HISTORY = {"single": {"hit": 0.758, "odds": 1.25, "streak": 3, "ret": -0.053},
           "double": {"hit": 0.500, "odds": 1.89, "streak": 10, "ret": -0.067},
           "medium": {"hit": 0.316, "odds": 2.99, "streak": 12, "ret": -0.072},
           "big": {"hit": 0.170, "odds": 5.16, "streak": 28, "ret": -0.141}}


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
                         "line": None if pd.isna(r.get("ou_line")) else float(r["ou_line"]),
                         "news": r.get("news_note") or "", "avoid": bool(r.get("news_avoid")),
                         # experimental line-ups: national friendlies stay out of reliable coupons
                         "friendly": "yoldaşlıq" in str(r.get("competition_az", "")).lower()})
    return pd.DataFrame(rows)


def build(cands: pd.DataFrame, style: str, exclude=frozenset(), agree: bool = False) -> dict | None:
    st = STYLES[style]
    c = cands[(cands["odds"] >= st["min_odds"]) & (cands["odds"] <= st["max_odds"])
              & (cands["p"] >= st["min_p"])
              & (cands["mbs"] <= st["size"]) & ~cands["event_id"].isin(exclude)
              & cands["p_model"].notna()  # only matches our model also covers
              & ~cands["avoid"].astype(bool)  # news analysis: unpredictable today
              & ~(cands["friendly"].astype(bool) & st.get("no_friendly", False))]
    if agree:  # our own analysis must confirm the outcome, not just the bookmaker's price
        c = c[np.minimum(c["p_model"].astype(float), c["p_market"].astype(float))
              >= st["agree_min"]]
    if c.empty:
        return None
    pm = c["p_model"].astype(float).fillna(c["p"]) if "p_model" in c else c["p"]
    c = c.assign(score=[st["score"](p, o, m) for p, o, m in zip(c["p"], c["odds"], pm)])
    best = c.sort_values("score", ascending=False).drop_duplicates("event_id")
    picks = best.head(st["size"])
    if len(picks) < st["size"]:
        return None
    picks = picks.assign(why=[_why(r) for r in picks.itertuples()])
    total = float(np.prod(picks["odds"]))
    prob = float(np.prod(picks["p"]))
    return {"style": style, "title": st["title"], "picks": picks.drop(columns="score"),
            "total_odds": total, "prob": prob, "ev": prob * total - 1}


def _why(r) -> str:
    pm, pq = float(r.p_model), float(r.p_market)
    if min(pm, pq) >= 0.70:
        lead = "мой анализ и рынок уверенно согласны"
    elif abs(pm - pq) <= 0.08:
        lead = "мой анализ и рынок согласны"
    elif pm > pq:
        lead = "мой анализ оценивает выше рынка"
    else:
        lead = "рынок оценивает выше моего анализа"
    news = getattr(r, "news", "")
    return f"{lead} (модель {pm:.0%}, рынок {pq:.0%})" + (f" · 📰 {news}" if news else "")


def suggest(events: pd.DataFrame, source: str = "combined") -> list[dict]:
    """Single, double and treble coupons, using different matches where possible.

    source="combined" (default): market chance, confirmed by our model (agree_min);
    source="model": our model alone; source="market": bookmaker's margin-free odds alone.
    """
    try:  # today's news analysis corrects the model and vetoes unpredictable matches
        from .news import apply
        events = apply(events)
    except Exception:
        pass
    cands = candidates(events)
    if cands.empty:
        return []
    cands = cands.assign(p_market=cands["p"])
    if source == "model":
        cands = cands[cands["p_model"].notna()].assign(p=lambda d: d["p_model"].astype(float))
    elif source == "combined":  # chance = honest market; our model (and news) must confirm it
        cands = cands[cands["p_model"].notna()]
    agree = source == "combined"
    out, used = [], set()
    for style in STYLES:
        cp = build(cands, style, frozenset(used), agree) or build(cands, style, agree=agree)
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
