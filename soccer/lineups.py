"""Starting line-ups: who of a team's core players is NOT in today's XI.

livescore.com publishes line-ups about an hour before kick-off. A team's regulars are learnt
from its last K league games: w_p = share of those games player p started; regulars have
w_p >= W_MIN. missing = sum of w_p of regulars absent from today's XI. The 1X2 forecast is
tilted towards the side missing less: p_home*e^s, p_away*e^-s with s = BETA*(miss_away -
miss_home) (scripts/test_lineups.py: fitted on 2025/26 games, checked on later ones).

History of recent XIs per team: data/state/lineup_xi.json, updated daily by the cloud job.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import DATA_DIR, LEAGUES

XI_FILE = DATA_DIR / "state" / "lineup_xi.json"
K, MIN_GAMES = 10, 5
W_MIN = 0.8      # the team's core: started 80%+ of its recent games
MAX_AGE = 120    # days: last season's XI says little after the summer transfers
BETA = 0.03      # see scripts/test_lineups.py (variant "core", 120 days)


# ------------------------------------------------------------------ the measure
def regulars(past: list[set]) -> pd.Series:
    """Share of the last K games each player started (only players at W_MIN or more)."""
    last = past[-K:]
    cnt = pd.Series([p for s in last for p in s], dtype=object).value_counts() / max(len(last), 1)
    return cnt[cnt >= W_MIN]


def missing(past: list[set], today: set) -> float | None:
    """Weighted number of regulars not in today's XI; None without enough history."""
    if len(past) < MIN_GAMES or len(today) < 9:
        return None
    reg = regulars(past)
    return float(reg[~reg.index.isin(list(today))].sum())


def absent_names(past_named: list[dict], today: set, top: int = 4) -> list[str]:
    """Names of the most regular players missing today (for the match card)."""
    if len(past_named) < MIN_GAMES:
        return []
    reg = regulars([set(g) for g in past_named])
    names = {}
    for g in past_named[-K:]:
        names.update(g)
    out = reg[~reg.index.isin(list(today))].sort_values(ascending=False)
    return [names.get(p, p) for p in out.index[:top]]


def tilt(p_home: float, p_draw: float, p_away: float, s: float) -> tuple[float, float, float]:
    q = np.array([p_home * np.exp(s), p_draw, p_away * np.exp(-s)])
    q = q / q.sum()
    return float(q[0]), float(q[1]), float(q[2])


# ------------------------------------------------------------------ history store
def load_history() -> dict:
    """{team: [{"date": "...", "xi": {pid: name}}, ...]} newest last."""
    if XI_FILE.exists():
        try:
            return json.loads(XI_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_history(hist: dict):
    XI_FILE.parent.mkdir(parents=True, exist_ok=True)
    XI_FILE.write_text(json.dumps(hist, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def add_game(hist: dict, team: str, date, xi: dict):
    """Record a team's starting XI ({pid: name}); keeps the last K+5 games, no duplicates."""
    if len(xi) < 9:
        return
    d = f"{pd.Timestamp(date):%Y-%m-%d}"
    games = [g for g in hist.get(team, []) if g["date"] != d]
    games.append({"date": d, "xi": xi})
    hist[team] = sorted(games, key=lambda g: g["date"])[-(K + 5):]


def past_xis(hist: dict, team: str, before) -> list[dict]:
    """XIs of the team's league games in the MAX_AGE days before `before`."""
    b = pd.Timestamp(before)
    b = b.tz_localize(None) if b.tzinfo else b
    lo, hi = f"{b - pd.Timedelta(days=MAX_AGE):%Y-%m-%d}", f"{b:%Y-%m-%d}"
    return [g["xi"] for g in hist.get(team, []) if lo <= g["date"] < hi]


# ------------------------------------------------------------------ livescore
def fetch_xi(eid: str) -> tuple[dict, dict] | None:
    """({pid: name} home starters, away starters) or None when not published yet."""
    from .livescore import _get
    try:
        d = _get(f"lineups/soccer/{str(eid).removeprefix('ls')}") or {}
    except Exception:
        return None
    sides = {}
    for t in d.get("Lu", []):
        sides[t.get("Tnb")] = {str(p.get("Pid")): f"{p.get('Fn', '')} {p.get('Ln', '')}".strip()
                               for p in t.get("Ps", []) if p.get("Pos") in (1, 2, 3, 4)}
    h, a = sides.get(1, {}), sides.get(2, {})
    return (h, a) if len(h) >= 9 and len(a) >= 9 else None


def link(events: list[dict], fixtures: pd.DataFrame, hours: float = 3) -> dict:
    """fixture index -> livescore event id (same country, kick-off within `hours`, both names)."""
    from .engine import Engine
    from .names import CLUB_ALIASES, similarity
    alias = lambda n: CLUB_ALIASES.get(n, n)
    out, by_country = {}, {}
    for e in events:
        by_country.setdefault(e["competition"].split(" · ")[0], []).append(e)
    for idx, f in fixtures.iterrows():
        meta = LEAGUES.get(f.get("competition"))
        country = Engine.LS_COUNTRY.get(meta["country"]) if meta else None
        best, score = None, 0.6
        for e in (by_country.get(country, []) if country else events):
            k = e.get("kickoff")
            if k is not None and pd.notna(f.get("kickoff")) and not pd.isna(k):
                if abs((pd.Timestamp(k) - f["kickoff"]).total_seconds()) > hours * 3600:
                    continue
            s = min(similarity(alias(e["home_raw"]), f["home"]), similarity(alias(e["away_raw"]), f["away"]))
            if s >= score:
                best, score = e["id"], s
        if best:
            out[idx] = str(best).removeprefix("ls")
    return out


def adjust(day: pd.DataFrame, now=None, hist: dict | None = None) -> pd.DataFrame:
    """For matches about to start (line-ups out): tilt 1X2 by missing regulars, add a note."""
    from .livescore import _events, _get
    now = now or pd.Timestamp.now(tz="UTC")
    out = day.copy()
    out["lineup_note"] = None
    if out.empty or "p_home" not in out:
        return out
    k = pd.to_datetime(out["kickoff"], utc=True)
    soon = out[(k > now) & (k < now + pd.Timedelta(hours=2)) & out["p_home"].notna()
               & ~out["played"].astype(bool)]
    if soon.empty:
        return out
    hist = load_history() if hist is None else hist
    try:
        evs = _events(_get(f"date/soccer/{now:%Y%m%d}/0?MD=1"))
        if (now + pd.Timedelta(hours=2)).date() != now.date():
            evs += _events(_get(f"date/soccer/{now + pd.Timedelta(days=1):%Y%m%d}/0?MD=1"))
    except Exception:
        return out
    for idx, eid in link(evs, soon).items():
        xi = fetch_xi(eid)
        if xi is None:
            continue
        r = out.loc[idx]
        ph, pa = past_xis(hist, r["home"], r["kickoff"]), past_xis(hist, r["away"], r["kickoff"])
        mh = missing([set(g) for g in ph], set(xi[0]))
        ma = missing([set(g) for g in pa], set(xi[1]))
        if mh is None or ma is None:
            out.at[idx, "lineup_note"] = "📋 составы вышли (истории мало — не учтены)"
            continue
        out.loc[idx, ["p_home", "p_draw", "p_away"]] = tilt(r["p_home"], r["p_draw"], r["p_away"],
                                                           BETA * (ma - mh))
        parts = []
        for team, past, today, m in ((r["home"], ph, xi[0], mh), (r["away"], pa, xi[1], ma)):
            names = absent_names(past, set(today), top=99)
            if m >= 1.5 and names:
                more = f" и ещё {len(names) - 3}" if len(names) > 3 else ""
                parts.append(f"у {team} нет {len(names)} основных ({', '.join(names[:3])}{more})")
        out.at[idx, "lineup_note"] = "📋 составы учтены" + (": " + "; ".join(parts) if parts else
                                                            " — обе команды в основном составе")
    return out


def update_history(matches: pd.DataFrame, days: int = 3, today=None) -> int:
    """Daily (cloud job): starting XIs of finished league games of the last `days` days."""
    from .livescore import _events, _get
    today = pd.Timestamp(today or pd.Timestamp.now(tz="UTC").normalize()).tz_localize(None)
    hist, added = load_history(), 0
    for k in range(1, days + 1):
        d = today - pd.Timedelta(days=k)
        try:
            evs = [e for e in _events(_get(f"date/soccer/{d:%Y%m%d}/0?MD=1")) if e.get("ended")]
        except Exception:
            continue
        fx = matches[(matches["date"] - d).abs() <= pd.Timedelta(days=1)]
        fx = fx.rename(columns={"league": "competition"}).assign(kickoff=pd.NaT)
        for idx, eid in link(evs, fx).items():
            xi = fetch_xi(eid)
            if xi is None:
                continue
            r = fx.loc[idx]
            add_game(hist, r["home"], r["date"], xi[0])
            add_game(hist, r["away"], r["date"], xi[1])
            added += 1
    if added:
        save_history(hist)
    return added
