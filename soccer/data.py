"""Download and normalise football-data.co.uk CSVs.

Normalised match frame columns:
    league, season (start year), date, home, away, hg, ag, result ('H'/'D'/'A'),
    odds_h/odds_d/odds_a (closing where available), odds_src,
    odds_o25/odds_u25, plus raw stats kept for later factors
    (hxg, axg, hs, as_, hst, ast, hc, ac, hf, af, hy, ay, hr, ar, referee).
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from concurrent.futures import ThreadPoolExecutor

from .config import (BASE_URL, CALENDAR_YEAR, DATA_DIR, EXTRA_COUNTRY,
                     EXTRA_FIXTURES_URL, EXTRA_URL, FIXTURES_URL, LEAGUES, RAW_DIR,
                     all_seasons, current_season_start, season_code)

# extra-league files use different headers
EXTRA_RENAME = {"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"}
CACHE_PATH = DATA_DIR / "matches.pkl"

# 1X2 odds, in order of preference: sharp closing -> market average closing -> opening.
ODDS_PRIORITY = [
    ("PSC", "PSCH", "PSCD", "PSCA"),
    ("AvgC", "AvgCH", "AvgCD", "AvgCA"),
    ("B365C", "B365CH", "B365CD", "B365CA"),
    ("PS", "PSH", "PSD", "PSA"),
    ("Avg", "AvgH", "AvgD", "AvgA"),
    ("BbAv", "BbAvH", "BbAvD", "BbAvA"),
    ("B365", "B365H", "B365D", "B365A"),
]
OU_PRIORITY = [
    ("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5"), ("P>2.5", "P<2.5"),
    ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"), ("B365>2.5", "B365<2.5"),
]
OPEN_PRIORITY = [("B365", "B365H", "B365D", "B365A"), ("BW", "BWH", "BWD", "BWA"),
                 ("Avg", "AvgH", "AvgD", "AvgA"), ("BbAv", "BbAvH", "BbAvD", "BbAvA")]
OPEN_OU_PRIORITY = [("B365>2.5", "B365<2.5"), ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5")]
STAT_COLS = {
    "HxG": "hxg", "AxG": "axg", "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
    "HC": "hc", "AC": "ac", "HF": "hf", "AF": "af", "HY": "hy", "AY": "ay",
    "HR": "hr", "AR": "ar", "Referee": "referee", "Time": "time",
    "HTHG": "hthg", "HTAG": "htag",  # half-time score (checks the in-play model)
}
CACHE_VERSION = 2  # bump when the parsed columns change -> data/matches.pkl is rebuilt

CURRENT_SEASON_MAX_AGE_H = 6  # re-download the live season after this many hours


def _raw_path(code: str, season: int) -> Path:
    return RAW_DIR / code / f"{season_code(season)}.csv"


def _download(url: str) -> bytes:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "soccer-analyser/0.1"})
    resp.raise_for_status()
    return resp.content


def _fetch(url: str, path: Path, live: bool, force: bool) -> Path | None:
    """Cached download. Finished seasons never change; live files refresh after N hours.
    A '.missing' marker remembers 404s (e.g. a division not covered in early years)."""
    missing = path.with_suffix(".missing")
    if missing.exists() and not live:
        return None
    if path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if not live or (age_h < CURRENT_SEASON_MAX_AGE_H and not force):
            return path
    try:
        content = _download(url)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404 and not live:
            missing.parent.mkdir(parents=True, exist_ok=True)
            missing.touch()
        return path if path.exists() else None
    except requests.RequestException:
        return path if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def fetch_season(code: str, season: int, force: bool = False) -> Path | None:
    return _fetch(BASE_URL.format(season=season_code(season), code=code),
                  _raw_path(code, season), live=season == current_season_start(), force=force)


def fetch_extra(code: str, force: bool = False) -> Path | None:
    return _fetch(EXTRA_URL.format(code=code), RAW_DIR / code / "all.csv", live=True,
                  force=force)


def _read_csv(source) -> pd.DataFrame:
    raw = source.read_bytes() if isinstance(source, Path) else source
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sep = "\t" if "\t" in text.split("\n", 1)[0] else ","
    # Some old files have ragged rows; skip them rather than fail.
    return pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip", low_memory=False)


def _pick_triplet(df: pd.DataFrame, priority) -> tuple[pd.DataFrame, pd.Series]:
    out = pd.DataFrame(np.nan, index=df.index, columns=["h", "d", "a"])
    src = pd.Series(None, index=df.index, dtype=object)
    for name, h, d, a in priority:
        if not {h, d, a} <= set(df.columns):
            continue
        cand = df[[h, d, a]].apply(pd.to_numeric, errors="coerce")
        ok = out["h"].isna() & cand.notna().all(axis=1) & (cand > 1).all(axis=1)
        out.loc[ok] = cand.loc[ok].to_numpy()
        src.loc[ok] = name
    return out, src


def normalise(df: pd.DataFrame, code: str, season: int | None) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    out = pd.DataFrame({
        "league": code,
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed"),
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
    })
    if "FTHG" in df.columns:
        out["hg"] = pd.to_numeric(df["FTHG"], errors="coerce")
        out["ag"] = pd.to_numeric(df["FTAG"], errors="coerce")
    else:
        out["hg"] = np.nan
        out["ag"] = np.nan
    if season is None:
        season = out["date"].map(lambda d: d.year if code in CALENDAR_YEAR or d.month >= 7
                                 else d.year - 1)
    out["season"] = season

    odds, src = _pick_triplet(df, ODDS_PRIORITY)
    out["odds_h"], out["odds_d"], out["odds_a"] = odds["h"], odds["d"], odds["a"]
    out["odds_src"] = src
    out["odds_o25"] = np.nan
    out["odds_u25"] = np.nan
    for o, u in OU_PRIORITY:
        if {o, u} <= set(df.columns):
            co = pd.to_numeric(df[o], errors="coerce")
            cu = pd.to_numeric(df[u], errors="coerce")
            ok = out["odds_o25"].isna() & co.notna() & cu.notna()
            out.loc[ok, "odds_o25"] = co[ok]
            out.loc[ok, "odds_u25"] = cu[ok]
    # early (opening) odds of a mass-market bookmaker — closest to what a bettor sees on a
    # site like misli.az hours/days before kick-off (closing odds above are sharper)
    op, _ = _pick_triplet(df, OPEN_PRIORITY)
    out["open_h"], out["open_d"], out["open_a"] = op["h"], op["d"], op["a"]
    out["open_o25"] = np.nan
    out["open_u25"] = np.nan
    for o, u in OPEN_OU_PRIORITY:
        if {o, u} <= set(df.columns):
            co = pd.to_numeric(df[o], errors="coerce")
            cu = pd.to_numeric(df[u], errors="coerce")
            ok = out["open_o25"].isna() & co.notna() & cu.notna() & (co > 1) & (cu > 1)
            out.loc[ok, "open_o25"] = co[ok]
            out.loc[ok, "open_u25"] = cu[ok]
    for raw_col, col in STAT_COLS.items():
        if raw_col in df.columns:
            out[col] = df[raw_col] if col in ("referee", "time") else pd.to_numeric(
                df[raw_col], errors="coerce")
        else:
            out[col] = np.nan
    out = out.dropna(subset=["date"])
    out["result"] = np.select([out.hg > out.ag, out.hg == out.ag, out.hg < out.ag],
                              ["H", "D", "A"], default=None)
    return out


def _normalise_extra(df: pd.DataFrame, code: str) -> pd.DataFrame:
    df = df.rename(columns=EXTRA_RENAME)
    season = df["Season"].astype(str).str[:4].astype(int) if "Season" in df else None
    return normalise(df, code, season)


def _fetch_all(leagues, seasons, refresh: bool) -> list[tuple[str, int | None, Path]]:
    tasks = []
    for code in leagues:
        if LEAGUES[code]["source"] == "extra":
            tasks.append((code, None))
        else:
            tasks.extend((code, s) for s in seasons)

    def work(t):
        code, season = t
        path = fetch_extra(code, refresh) if season is None else fetch_season(code, season, refresh)
        return code, season, path

    with ThreadPoolExecutor(max_workers=8) as ex:
        return [r for r in ex.map(work, tasks) if r[2] is not None]


def load_matches(leagues=None, seasons=None, refresh: bool = False) -> pd.DataFrame:
    """All played matches for the given leagues/seasons (downloads what is missing)."""
    df = _load_all(refresh)
    if leagues is not None:
        df = df[df["league"].isin(list(leagues))]
    if seasons is not None:
        df = df[df["season"].isin(list(seasons))]
    return df.reset_index(drop=True) if (leagues is not None or seasons is not None) else df


def _load_all(refresh: bool) -> pd.DataFrame:
    """Every league; parsed frame cached in data/matches.pkl, rebuilt when raw files change."""
    leagues = list(LEAGUES)
    seasons = all_seasons()
    from .seed import ensure_raw
    ensure_raw()  # fresh install: unpack past seasons instead of downloading them
    files = _fetch_all(leagues, seasons, refresh)
    stamp = [("version", CACHE_VERSION)] + sorted((str(p), p.stat().st_mtime) for _, _, p in files)
    if CACHE_PATH.exists():
        try:
            cached = pd.read_pickle(CACHE_PATH)
            if cached.attrs.get("stamp") == stamp:
                return cached
        except Exception:
            pass
    frames = []
    for code, season, path in files:
        raw = _read_csv(path)
        frames.append(_normalise_extra(raw, code) if season is None
                      else normalise(raw, code, season))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season"].isin(seasons)]
    df = df.dropna(subset=["hg", "ag"])
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df = df.sort_values(["date", "league", "home"]).reset_index(drop=True)
    df["match_id"] = (df["league"] + "_" + df["date"].dt.strftime("%Y%m%d") + "_"
                      + df["home"] + "_" + df["away"]).str.replace(" ", "")
    df = df.drop_duplicates("match_id").reset_index(drop=True)
    df.attrs["stamp"] = stamp
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CACHE_PATH)
    return df


def load_fixtures(leagues=None) -> pd.DataFrame:
    """Upcoming fixtures (football-data.co.uk publishes them a few days ahead)."""
    leagues = set(leagues or LEAGUES)
    out = []
    try:
        df = _read_csv(_download(FIXTURES_URL))
        df = df[df["Div"].isin(leagues)]
        out += [normalise(df[df.Div == c], c, None) for c in df.Div.unique()]
    except (requests.RequestException, KeyError):
        pass
    try:
        df = _read_csv(_download(EXTRA_FIXTURES_URL)).rename(columns=EXTRA_RENAME)
        by_country = {v: k for k, v in EXTRA_COUNTRY.items()}
        df["code"] = df["Country"].map(by_country)
        df = df[df["code"].isin(leagues)]
        out += [normalise(df[df.code == c], c, None) for c in df.code.unique()]
    except (requests.RequestException, KeyError):
        pass
    if not out:
        return pd.DataFrame()
    return pd.concat(out).sort_values("date").reset_index(drop=True)


def implied_probs(odds_h, odds_d, odds_a) -> np.ndarray:
    """Bookmaker implied 1X2 probabilities with proportional margin removal."""
    inv = 1.0 / np.column_stack([odds_h, odds_d, odds_a]).astype(float)
    return inv / inv.sum(axis=1, keepdims=True)
