"""Seed for a fresh install (the cloud dashboard after every redeploy): historical raw data.

Downloading ~700 football-data files + cups + internationals + xG took 7-8 minutes on each
cold start. Past seasons never change, so they ship with the code as one archive
(data/seed/raw_seed.tar.xz, made by scripts/make_seed.py); files keep their original
timestamps, so the current season and the live files are still refreshed as usual.
"""
from __future__ import annotations

import tarfile

from .config import DATA_DIR, RAW_DIR

SEED = DATA_DIR / "seed" / "raw_seed.tar.xz"


def ensure_raw() -> int:
    """Unpack the seed when data/raw is (almost) empty; returns the number of files added."""
    if not SEED.exists():
        return 0
    have = sum(1 for _ in RAW_DIR.rglob("*")) if RAW_DIR.exists() else 0
    if have >= 100:
        return 0
    with tarfile.open(SEED, "r:xz") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.startswith("raw/")
                   and ".." not in m.name]
        tar.extractall(DATA_DIR, members=members)
    return len(members)
