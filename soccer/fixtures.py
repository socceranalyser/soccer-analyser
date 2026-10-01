"""Full-season schedules (fixturedownload.com) + football-data fixtures, for any date.

Also provides fresh results that football-data.co.uk has not published yet, the current
season of the European cups (openfootball only has finished seasons) and the latest
national-team results.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from .config import CALENDAR_YEAR, LEAGUES, RAW_DIR, current_season_start
from .data import _fetch, load_fixtures
from .names import CLUB_ALIASES, NATION_ALIASES, best_match

FEED_URL = "https://fixturedownload.com/feed/json/{slug}-{year}"

# competition code -> fixturedownload slug
FEEDS = {
    "E0": "epl", "E1": "championship", "D1": "bundesliga", "SP1": "la-liga", "I1": "serie-a",
    "F1": "ligue-1", "F2": "ligue-2", "N1": "eredivisie", "P1": "primeira-liga",
    "SC0": "scottish-premiership", "T1": "super-lig", "USA": "mls",
    "UCL": "champions-league", "UEL": "europa-league", "UECL": "conference-league",
    "UNL": "nations-league", "WC": "fifa-world-cup",
}
CUPS = {"UCL": "Лига чемпионов", "UEL": "Лига Европы", "UECL": "Лига конференций"}
NATIONAL = {"UNL": "Лига наций УЕФА", "WC": "Чемпионат мира"}
COMP_ORDER = ["UCL", "UEL", "UECL", "UNL", "WC"]


def competition_name(code: str) -> str:
    if code in LEAGUES:
        return LEAGUES[code]["name"]
    return CUPS.get(code) or NATIONAL.get(code) or code


def kind_of(code: str) -> str:
    return "cup" if code in CUPS else "national" if code in NATIONAL else "league"


def local_tz():
    return datetime.now().astimezone().tzinfo


def _feed_year(code: str) -> int:
    if code in CALENDAR_YEAR or code == "WC":
        return datetime.now().year
    return current_season_start()


def fetch_feed(code: str, refresh: bool = False) -> pd.DataFrame:
    year = _feed_year(code)
    path = RAW_DIR / "FD" / f"{FEEDS[code]}-{year}.json"
    p = _fetch(FEED_URL.format(slug=FEEDS[code], year=year), path, live=True, force=refresh)
    if p is None:
        return pd.DataFrame()
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    kick = pd.to_datetime(df["DateUtc"].str.replace("Z", "", regex=False), utc=True)
    out = pd.DataFrame({
        "competition": code, "kickoff": kick.dt.tz_convert(local_tz()),
        "home_src": df["HomeTeam"], "away_src": df["AwayTeam"],
        "hg": pd.to_numeric(df.get("HomeTeamScore"), errors="coerce"),
        "ag": pd.to_numeric(df.get("AwayTeamScore"), errors="coerce"),
        "round": df.get("RoundNumber"), "group": df.get("Group"), "venue": df.get("Location"),
        "season": year,
    })
    out["date"] = out["kickoff"].dt.tz_localize(None).dt.normalize()
    # placeholder pairings ("To be announced", "1A", "Winner Match 49") are not real fixtures
    bad = out["home_src"].str.contains(r"announced|Winner|Loser|^\d[A-Z]$|^[A-Z]\d$", regex=True,
                                       na=True)
    return out[~bad].reset_index(drop=True)


# --------------------------------------------------------------------- mapping
def _map_names(df: pd.DataFrame, candidates_for, aliases) -> pd.DataFrame:
    cache = {}
    for side in ("home", "away"):
        mapped, ok = [], []
        for comp, name in zip(df["competition"], df[f"{side}_src"]):
            key = (comp, name)
            if key not in cache:
                # cup candidates span all countries -> stricter matching
                cache[key] = (best_match(name, candidates_for(comp), aliases, threshold=0.82,
                                         both_ways=False) if comp in CUPS
                              else best_match(name, candidates_for(comp), aliases))
            m, _ = cache[key]
            mapped.append(m)
            ok.append(m is not None)
        df[f"{side}_mapped"] = mapped
        df[f"{side}_ok"] = ok
    return df


def build_schedule(matches: pd.DataFrame, club_keys, nations, refresh: bool = False,
                   codes=None) -> pd.DataFrame:
    """All known fixtures/results of the current seasons, mapped to our team names.

    club_keys: 'country|team' Elo keys of clubs (for European cups);
    nations:   national-team names known to the national model.
    """
    codes = codes or list(FEEDS)
    feeds = [fetch_feed(c, refresh) for c in codes]
    feeds = [f for f in feeds if not f.empty]
    sched = pd.concat(feeds, ignore_index=True) if feeds else pd.DataFrame()

    recent = matches[matches["season"] >= matches["season"].max() - 1]
    league_teams = {lg: sorted(set(g["home"]) | set(g["away"])) for lg, g in recent.groupby("league")}
    club_keys, nations = sorted(club_keys), sorted(nations)

    def candidates(comp):
        if comp in CUPS:
            return club_keys
        if comp in NATIONAL:
            return nations
        return league_teams.get(comp, [])

    if not sched.empty:
        sched = _map_names(sched, candidates, {**CLUB_ALIASES, **NATION_ALIASES})
        for side in ("home", "away"):
            src, m = sched[f"{side}_src"], sched[f"{side}_mapped"]
            is_cup = sched["competition"].isin(list(CUPS))
            # cups keep the full 'country|team' key; unknown clubs get a '?|' key
            sched[f"{side}_key"] = np.where(is_cup, m.fillna("?|" + src), None)
            sched[side] = np.where(m.notna(), m.astype(str).str.split("|", n=1).str[-1], src)

    # football-data fixtures (odds, and leagues without a schedule feed)
    fd = load_fixtures()
    if not fd.empty:
        # football-data kick-off times are UK local time
        stamp = pd.to_datetime(fd["date"].dt.strftime("%Y-%m-%d") + " " + fd["time"].fillna("15:00")
                               .astype(str), errors="coerce")
        kick = stamp.dt.tz_localize("Europe/London", ambiguous="NaT",
                                    nonexistent="NaT").dt.tz_convert(local_tz())
        fd = fd.assign(competition=fd["league"], home_src=fd["home"], away_src=fd["away"],
                       home_ok=True, away_ok=True, kickoff=kick,
                       date=kick.dt.tz_localize(None).dt.normalize().fillna(fd["date"]))
        if sched.empty:
            sched = fd
        else:
            odds = fd[["competition", "home", "away", "odds_h", "odds_d", "odds_a", "odds_o25",
                       "odds_u25"]].drop_duplicates(["competition", "home", "away"])
            sched = sched.merge(odds, on=["competition", "home", "away"], how="left")
            have = set(zip(sched["competition"], sched["home"], sched["away"]))
            fd = fd[[(c, h, a) not in have for c, h, a in zip(fd["competition"], fd["home"],
                                                                fd["away"])]]
            sched = pd.concat([sched, fd[[c for c in fd.columns if c in sched.columns]]],
                              ignore_index=True)
    if sched.empty:
        return sched
    sched["kind"] = sched["competition"].map(kind_of)
    sched["played"] = sched["hg"].notna() & sched["ag"].notna()
    sched["neutral"] = False
    sched["comp_name"] = sched["competition"].map(competition_name)
    return sched.sort_values(["kickoff", "competition"], na_position="last").reset_index(drop=True)


# ------------------------------------------------------------ fresh results
def fresh_league_results(matches: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Played domestic matches from the schedule that football-data has not published."""
    s = sched[(sched["kind"] == "league") & sched["played"] & sched["home_ok"] & sched["away_ok"]]
    if s.empty:
        return s
    out = []
    for lg, g in s.groupby("competition"):
        last = matches.loc[matches["league"] == lg, "date"].max()
        new = g[g["date"] > last]
        if new.empty:
            continue
        season = int(matches.loc[matches["league"] == lg, "season"].max())
        out.append(pd.DataFrame({
            "league": lg, "season": season, "date": new["date"], "home": new["home"],
            "away": new["away"], "hg": new["hg"].astype(int), "ag": new["ag"].astype(int)}))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    df["match_id"] = (df["league"] + "_" + df["date"].dt.strftime("%Y%m%d") + "_" + df["home"]
                      + "_" + df["away"]).str.replace(" ", "")
    df["source"] = "fixturedownload"
    return df


def current_cup_matches(sched: pd.DataFrame) -> pd.DataFrame:
    """This season's European cup matches in the format of soccer.euro.load_euro."""
    s = sched[sched["kind"] == "cup"]
    if s.empty:
        return s
    comp_ru = {"UCL": "ЛЧ", "UEL": "ЛЕ", "UECL": "ЛК"}
    df = pd.DataFrame({
        "date": s["date"], "season": s["season"], "competition": s["competition"].map(comp_ru),
        "stage": s["group"].fillna(""), "home_key": s["home_key"], "away_key": s["away_key"],
        "home": s["home"], "away": s["away"], "hg": s["hg"], "ag": s["ag"],
        "neutral": False, "league": "EUR"})
    played = df["hg"].notna()
    df["result"] = None
    df.loc[played, "result"] = np.select(
        [df.loc[played, "hg"] > df.loc[played, "ag"], df.loc[played, "hg"] == df.loc[played, "ag"]],
        ["H", "D"], "A")
    df["match_id"] = ("EUR_" + df["date"].dt.strftime("%Y%m%d") + "_" + df["home_key"] + "_"
                      + df["away_key"]).str.replace(" ", "")
    return df.reset_index(drop=True)


def fresh_national_results(intl: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    s = sched[(sched["kind"] == "national") & sched["played"]]
    if s.empty:
        return s
    last = intl["date"].max()
    s = s[s["date"] > last]
    if s.empty:
        return pd.DataFrame()
    tour = {"UNL": "UEFA Nations League", "WC": "FIFA World Cup"}
    df = pd.DataFrame({
        "league": "INT", "date": s["date"], "season": s["date"].dt.year, "home": s["home"],
        "away": s["away"], "hg": s["hg"].astype(int), "ag": s["ag"].astype(int),
        "tournament": s["competition"].map(tour), "neutral": False, "country": None})
    from .national import importance
    df["importance"] = df["tournament"].map(importance)
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], ["H", "D"], "A")
    df["match_id"] = ("INT_" + df["date"].dt.strftime("%Y%m%d") + "_" + df["home"] + "_"
                      + df["away"]).str.replace(" ", "")
    return df
