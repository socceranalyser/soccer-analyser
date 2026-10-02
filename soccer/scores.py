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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_live() -> pd.DataFrame:
    for mod in (livescore, misli):
        try:
            df = mod.fetch_live()
            if len(df):
                return df
        except Exception:
            continue
    return pd.DataFrame()
