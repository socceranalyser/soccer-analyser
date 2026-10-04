r"""Build data/seed/raw_seed.tar.xz from the local data/raw (all public sources; the API-Football
cache is left out). Rebuild once a season:  python scripts/make_seed.py"""
import tarfile

from soccer.config import RAW_DIR
from soccer.seed import SEED

SKIP = {"APIF"}

if __name__ == "__main__":
    SEED.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(SEED, "w:xz", preset=9) as tar:
        for p in sorted(RAW_DIR.rglob("*")):
            rel = p.relative_to(RAW_DIR)
            if p.is_file() and rel.parts[0] not in SKIP:
                tar.add(p, arcname=f"raw/{rel.as_posix()}")
                n += 1
    print(f"{n} files -> {SEED} ({SEED.stat().st_size / 1e6:.1f} MB)")
