"""European club competitions (Champions League, Europa League, Conference League).

History comes from openfootball/champions-league (plain-text, 2011/12 onwards; Europa
and Conference League files from 2021/22). Club names are mapped onto the names used by
football-data.co.uk so that European matches link the domestic ratings of different
countries — this is what makes a cross-league strength scale possible.
"""
from __future__ import annotations

import difflib
import functools
import re
import unicodedata

import pandas as pd

from .config import DATA_DIR, LEAGUES, RAW_DIR
from .data import _fetch

OPENFOOTBALL = "https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/{file}"
COMPETITIONS = {"cl.txt": "ЛЧ", "el.txt": "ЛЕ", "conf.txt": "ЛК"}
FIRST_EURO_SEASON = 2011
EURO_LEAGUE = "EUR"  # pseudo-league code for all European club matches

# openfootball 3-letter codes -> our country names (others stay as the code)
CODE_COUNTRY = {
    "ENG": "Англия", "SCO": "Шотландия", "GER": "Германия", "ITA": "Италия",
    "ESP": "Испания", "FRA": "Франция", "NED": "Нидерланды", "BEL": "Бельгия",
    "POR": "Португалия", "TUR": "Турция", "GRE": "Греция", "AUT": "Австрия",
    "DEN": "Дания", "FIN": "Финляндия", "IRL": "Ирландия", "NOR": "Норвегия",
    "POL": "Польша", "ROU": "Румыния", "RUS": "Россия", "SWE": "Швеция", "SUI": "Швейцария",
}

# substring rewrites applied to normalised openfootball names before matching
ALIASES = [
    ("manchester", "man"), ("athletic club", "ath bilbao"), ("athletic bilbao", "ath bilbao"),
    ("atletico de madrid", "ath madrid"), ("atletico madrid", "ath madrid"),
    ("internazionale milano", "inter"), ("internazionale", "inter"),
    ("paris saint germain", "paris sg"), ("sporting clube de portugal", "sp lisbon"),
    ("sporting cp", "sp lisbon"), ("sporting clube de braga", "sp braga"), ("braga", "sp braga"),
    ("munchen", "munich"), ("eintracht", "ein"), ("borussia monchengladbach", "m gladbach"),
    ("nottingham", "nott m"), ("wolverhampton wanderers", "wolves"),
    ("sport lisboa e benfica", "benfica"), ("moskva", "moscow"), ("kobenhavn", "copenhagen"),
    ("olympiacos", "olympiakos"), ("real sociedad de futbol", "sociedad"),
    ("real sociedad", "sociedad"), ("real betis balompie", "betis"), ("real betis", "betis"),
    ("saint gilloise", "st gilloise"), ("psv", "psv eindhoven"), ("tottenham hotspur", "tottenham"),
    ("brighton hove albion", "brighton"), ("west ham united", "west ham"),
    ("racing club de lens", "lens"), ("olympique lyonnais", "lyon"),
    ("olympique de marseille", "marseille"), ("stade rennais", "rennes"),
    ("zenit st petersburg", "zenit"), ("bodo glimt", "bodo glimt"), ("fenerbahce", "fenerbahce"),
    ("istanbul basaksehir", "basaksehir"), ("steaua bucuresti", "fcsb"), ("steaua", "fcsb"),
    ("stade brestois", "brest"), ("heart of midlothian", "hearts"),
]
STOPWORDS = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "sv", "vfl", "vfb", "bsc", "fk", "sk",
             "rc", "cd", "ud", "rcd", "sl", "ss", "us", "club", "de", "calcio", "football",
             "futbol", "1", "04", "05", "09", "1899", "1909", "1913", "1907", "1846", "kv",
             "krc", "rsc", "osc", "ogc", "aj", "bc", "jk", "if", "ff", "bk", "sad", "cp", "the",
             "and", "e", "y", "royal", "royale", "sporting club", "jsc", "hsc", "nk", "gnk"}


SPECIAL_LETTERS = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss",
                                 "ı": "i", "İ": "I", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
                                 "þ": "th", "ð": "d"})


@functools.lru_cache(maxsize=None)
def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name.translate(SPECIAL_LETTERS))
    s = s.encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = " ".join(s.split())
    for old, new in ALIASES:
        s = re.sub(rf"\b{old}\b", new, s)
    tokens = [t for t in s.split() if t not in STOPWORDS]
    return " ".join(tokens) or s


def _similarity(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if tb and tb <= ta:  # every token of the short domestic name appears in the long one
        # ...and among such candidates prefer the closer string ("lazio roma" -> Lazio, not Roma)
        return 0.9 + 0.1 * ratio
    return ratio


# ------------------------------------------------------------------------- parsing
DATE_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\s+([A-Z][a-z]{2})\w*\s+(\d{1,2})"
                     r"(?:\s+(\d{4}))?\s*$")
MATCH_RE = re.compile(r"^\s*(?:\d{1,2}[:.]\d{2}\s+)?(?P<home>.+?)\s+\((?P<hc>[A-Z]{3})\)\s+v\s+"
                      r"(?P<away>.+?)\s+\((?P<ac>[A-Z]{3})\)\s*(?P<score>.*)$")
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _ft_score(score: str):
    """90-minute score. 'a.e.t.'/'pen.' lines carry it as the first bracketed score."""
    score = score.strip()
    if not score or not re.match(r"\d", score):
        return None, None
    if "a.e.t" in score:
        m = re.search(r"\((\d+)-(\d+)", score)
    else:
        m = re.match(r"(\d+)-(\d+)", score)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def parse_openfootball(text: str, season: int, competition: str) -> pd.DataFrame:
    rows, year, last_month, stage = [], season, 7, ""
    for line in text.splitlines():
        if line.startswith("▪"):
            stage = line.lstrip("▪ ").strip()
            continue
        dm = DATE_RE.match(line)
        if dm:
            mon = MONTHS[dm.group(1)]
            if dm.group(3):
                year = int(dm.group(3))
            elif mon < last_month:
                year += 1
            last_month = mon
            date = pd.Timestamp(year=year, month=mon, day=int(dm.group(2)))
            continue
        mm = MATCH_RE.match(line)
        if mm:
            hg, ag = _ft_score(mm.group("score"))
            rows.append({"date": date, "season": season, "competition": competition,
                         "stage": stage, "home_raw": mm.group("home").strip(),
                         "away_raw": mm.group("away").strip(), "hc": mm.group("hc"),
                         "ac": mm.group("ac"), "hg": hg, "ag": ag,
                         # the final ("Final" / "Finals, Final") is played at a neutral venue
                         "neutral": stage.lower().split(",")[-1].strip() == "final"})
    return pd.DataFrame(rows)


def _season_dir(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def load_raw_euro(seasons=None, refresh: bool = False) -> pd.DataFrame:
    from .config import current_season_start
    seasons = seasons or range(FIRST_EURO_SEASON, current_season_start() + 1)
    frames = []
    for s in seasons:
        for file, comp in COMPETITIONS.items():
            path = RAW_DIR / EURO_LEAGUE / _season_dir(s) / file
            live = s >= current_season_start() - 1
            p = _fetch(OPENFOOTBALL.format(season=_season_dir(s), file=file), path,
                       live=live, force=refresh)
            if p is not None:
                frames.append(parse_openfootball(p.read_text(encoding="utf-8"), s, comp))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------------- name mapping
def build_name_map(raw: pd.DataFrame, domestic: pd.DataFrame) -> pd.DataFrame:
    """openfootball (name, code) -> football-data team name, per country, with a score."""
    country_of = {lg: v["country"] for lg, v in LEAGUES.items()}
    dom = domestic.assign(country=domestic["league"].map(country_of))
    teams_by_country_season = {
        (c, s): set(g["home"]) | set(g["away"]) for (c, s), g in dom.groupby(["country", "season"])}
    clubs = pd.concat([raw[["home_raw", "hc", "season"]].set_axis(["raw", "code", "season"], axis=1),
                       raw[["away_raw", "ac", "season"]].set_axis(["raw", "code", "season"], axis=1)])
    rows = []
    for (name, code), g in clubs.groupby(["raw", "code"]):
        country = CODE_COUNTRY.get(code)
        best, score = None, 0.0
        if country:
            cands = set()
            for s in set(g["season"]):
                for ds in (s - 1, s, s + 1):
                    cands |= teams_by_country_season.get((country, ds), set())
            n = _norm(name)
            for cand in cands:
                sc = _similarity(n, _norm(cand))
                if sc > score:
                    best, score = cand, sc
        rows.append({"raw": name, "code": code, "country": country or code,
                     "team": best if score >= 0.6 else None, "score": round(score, 3),
                     "n": len(g)})
    out = pd.DataFrame(rows)
    # clubs from countries we do not cover: merge spellings ("FK Shakhtar Donetsk" /
    # "Shakhtar Donetsk") under the most frequent one
    unc = out["team"].isna()
    out["_norm"] = out["raw"].map(_norm)
    canon = (out[unc].sort_values("n", ascending=False)
             .drop_duplicates(["code", "_norm"]).set_index(["code", "_norm"])["raw"])
    out.loc[unc, "team"] = [canon[(c, n)] for c, n in zip(out.loc[unc, "code"],
                                                           out.loc[unc, "_norm"])]
    out = out.drop(columns=["_norm", "n"])
    out.to_csv(DATA_DIR / "euro_name_map.csv", index=False, encoding="utf-8")
    return out


def _cached_name_map(raw: pd.DataFrame, domestic: pd.DataFrame) -> pd.DataFrame:
    """Reuse data/euro_name_map.csv when it already covers every (club, country) pair."""
    path = DATA_DIR / "euro_name_map.csv"
    need = set(zip(raw["home_raw"], raw["hc"])) | set(zip(raw["away_raw"], raw["ac"]))
    try:
        cached = pd.read_csv(path, encoding="utf-8")
        if need <= set(zip(cached["raw"], cached["code"])):
            return cached
    except (OSError, ValueError, KeyError):
        pass
    return build_name_map(raw, domestic)


def load_euro(domestic: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """European matches with keys compatible with the domestic Elo ('country|team')."""
    raw = load_raw_euro(refresh=refresh)
    if raw.empty:
        return raw
    nm = _cached_name_map(raw, domestic).set_index(["raw", "code"])

    def key(name, code):
        r = nm.loc[(name, code)]
        return f"{r['country']}|{r['team'] if pd.notna(r['team']) else name}"

    df = raw.copy()
    df["home_key"] = [key(n, c) for n, c in zip(df["home_raw"], df["hc"])]
    df["away_key"] = [key(n, c) for n, c in zip(df["away_raw"], df["ac"])]
    df["home"] = df["home_key"].str.split("|", n=1).str[1]
    df["away"] = df["away_key"].str.split("|", n=1).str[1]
    df["league"] = EURO_LEAGUE
    df["result"] = None
    played = df["hg"].notna()
    df.loc[played, "result"] = [("H" if h > a else "D" if h == a else "A")
                                for h, a in zip(df.loc[played, "hg"], df.loc[played, "ag"])]
    df["match_id"] = ("EUR_" + df["date"].dt.strftime("%Y%m%d") + "_" + df["home_key"] + "_"
                      + df["away_key"]).str.replace(" ", "")
    return df.sort_values("date").reset_index(drop=True)
