"""Scores from several sources at once (livescore.com + misli.az).

Results: both sources are combined; the fixture matcher picks the best name/time match,
so a match missing or oddly spelled in one source is still found in the other.
Live: livescore.com first (widest coverage, English names), misli.az as a fallback.
"""
from __future__ import annotations

import pandas as pd

from . import livescore, misli


def fetch_results(dates) -> pd.DataFrame:
    frames = []
    for src, mod in (("livescore", livescore), ("misli", misli)):
        try:
            df = mod.fetch_results(dates)
            if len(df):
                frames.append(df.assign(source=src))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[~_not_senior_men(df)].reset_index(drop=True)


def _not_senior_men(df: pd.DataFrame) -> pd.Series:
    """Women's, youth and reserve games: their clubs share the men's names ("Chelsea" in the
    WSL, "WFC Mançester Siti") and must never be matched to a men's fixture."""
    from .engine import Engine
    from .misli import EXCLUDE
    comp = df.get("competition", pd.Series("", index=df.index)).fillna("").astype(str)
    names = df["home_raw"].fillna("").astype(str) + " | " + df["away_raw"].fillna("").astype(str)
    team = df["home_raw"].fillna("").astype(str).str.contains(Engine.LS_NOT_MEN_TEAM) |         df["away_raw"].fillna("").astype(str).str.contains(Engine.LS_NOT_MEN_TEAM)
    return (comp.str.contains(Engine.LS_NOT_MEN) | names.str.contains(EXCLUDE) | team)


def fetch_live() -> pd.DataFrame:
    for mod in (livescore, misli):
        try:
            df = mod.fetch_live()
            if len(df):
                return df
        except Exception:
            continue
    return pd.DataFrame()
