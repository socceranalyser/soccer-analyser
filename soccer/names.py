"""Matching team names between data sources."""
from __future__ import annotations

import difflib

from .euro import _norm

# exact source-name -> our-name fixes (applied before fuzzy matching)
CLUB_ALIASES = {
    "Atleti": "Ath Madrid", "Atlético Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
    "FC Cologne": "FC Koln", "Borussia Dortmund": "Dortmund", "Bayer Leverkusen": "Leverkusen",
    "Independiente Rivadavia": "Ind. Rivadavia",
    "Paris": "Paris SG", "Paris Saint-Germain": "Paris SG", "B. Dortmund": "Dortmund",
    "Man Utd": "Man United", "Leipzig": "RB Leipzig", "Bayern München": "Bayern Munich",
    "Real Betis": "Betis", "Sporting CP": "Sp Lisbon", "Sporting": "Sp Lisbon",
    "SC Braga": "Sp Braga", "PSV": "PSV Eindhoven", "Spurs": "Tottenham",
    "Nottingham Forest": "Nott'm Forest", "Wolverhampton": "Wolves", "M'gladbach": "M'gladbach",
    "Gladbach": "M'gladbach", "Celta Vigo": "Celta", "Real Sociedad": "Sociedad",
    "Rayo Vallecano": "Vallecano", "Espanyol": "Espanol", "Inter Milan": "Inter",
    "AC Milan": "Milan", "Hellas Verona": "Verona", "Olympiacos": "Olympiakos",
    "Fenerbahçe": "Fenerbahce", "Beşiktaş": "Besiktas", "Başakşehir": "Buyuksehyr",
    "Bodø/Glimt": "Bodo/Glimt", "Copenhagen": "FC Copenhagen", "Red Star": "Crvena Zvezda",
    "Union SG": "St. Gilloise", "Union St.-Gilloise": "St. Gilloise",
    "AEK Athens": "AEK", "GNK Dinamo": "Dinamo Zagreb", "Dinamo Zagreb": "Dinamo Zagreb",
    "RCD Espanyol de Barcelona": "Espanol",
    "Red Bull New York": "New York Red Bulls", "NY Red Bulls": "New York Red Bulls",
    "LA Galaxy": "Los Angeles Galaxy", "LAFC": "Los Angeles FC", "N.E.C.": "Nijmegen", "N.E.C. Nijmegen": "Nijmegen",
}
NATION_ALIASES = {
    "Türkiye": "Turkey", "Czechia": "Czech Republic", "USA": "United States",
    "Korea Republic": "South Korea", "IR Iran": "Iran", "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde", "Congo DR": "DR Congo", "China PR": "China",
    "Republic of Ireland": "Republic of Ireland", "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curaçao", "Kyrgyz Republic": "Kyrgyzstan",
}


def similarity(src: str, cand: str, both_ways: bool = True) -> float:
    """Normalised-name similarity with a bonus when one name's tokens contain the other's.

    both_ways=False only rewards a *source* name contained in the candidate ("Leipzig" ->
    "RB Leipzig"); the reverse ("Sparta Praha" -> "Sparta") is unsafe when candidates
    span many countries, because the longer name usually means a different club.
    """
    ts, tc = set(src.split()), set(cand.split())
    ratio = difflib.SequenceMatcher(None, src, cand).ratio()
    if ts and tc and (ts <= tc or (both_ways and tc <= ts)):
        return 0.9 + 0.1 * ratio
    return ratio


def best_match(name: str, candidates, aliases=None, threshold: float = 0.72,
               both_ways: bool = True):
    """Return (candidate, score) or (None, score). Candidates may be 'country|team' keys."""
    aliases = aliases or {}
    target = aliases.get(name, name)
    cands = list(candidates)
    plain = {c: c.split("|", 1)[-1] for c in cands}
    for c, p in plain.items():  # exact hit after aliasing
        if p == target:
            return c, 1.0
    nt = _norm(target)
    best, score = None, 0.0
    for c, p in plain.items():
        s = similarity(nt, _norm(p), both_ways)
        if s > score:
            best, score = c, s
    return (best, score) if score >= threshold else (None, score)
