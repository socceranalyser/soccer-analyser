"""Production-equivalent backtest over every competition; stored as run 'bt_all'.

Leagues: Dixon-Coles, Elo, bookmakers and the final 50/50 ensemble (seasons 2023-2025).
European cups: cross-league Elo (2023-2025). National teams: Elo + Dixon-Coles (2023-2026).
"final" is what the dashboard shows as the headline forecast.

    .venv\\Scripts\\python.exe scripts\\backtest_all.py
"""
from __future__ import annotations

import time

import pandas as pd

from soccer import storage
from soccer.backtest import common_matches, euro_walk_forward, walk_forward
from soccer.data import load_matches
from soccer.engine import ENSEMBLE_DC_WEIGHT
from soccer.euro import load_euro
from soccer.metrics import summary_table
from soccer.national import load_international, national_walk_forward
from soccer.tuned import DC_PARAMS, ELO_PARAMS

SEASONS = [2023, 2024, 2025]
EURO_CODE = {"ЛЧ": "UCL", "ЛЕ": "UEL", "ЛК": "UECL"}
KEY = ["league", "date", "home", "away"]


def ensemble(preds: pd.DataFrame) -> pd.DataFrame:
    dc = preds[preds["model"] == "dixon_coles"].set_index(KEY)
    el = preds[preds["model"] == "elo"].set_index(KEY)
    both = dc.index.intersection(el.index)
    fin = dc.loc[both].copy()
    for c in ("p_home", "p_draw", "p_away"):
        fin[c] = ENSEMBLE_DC_WEIGHT * dc.loc[both, c] + (1 - ENSEMBLE_DC_WEIGHT) * el.loc[both, c]
    return fin.reset_index().assign(model="final")


def main():
    t0 = time.time()
    matches = load_matches()
    euro = load_euro(matches)
    intl = load_international(since="1960-01-01")

    dom = walk_forward(matches, SEASONS, dc_kwargs=DC_PARAMS, elo_kwargs=ELO_PARAMS,
                       cups=euro, verbose=False)
    dom = pd.concat([dom, ensemble(dom)], ignore_index=True)
    print(f"leagues done [{time.time() - t0:.0f}s]")

    eu = euro_walk_forward(matches, euro, SEASONS, ELO_PARAMS)
    comp = euro.set_index("match_id")["competition"].map(EURO_CODE)
    eu["league"] = eu["match_id"].map(comp)
    eu = pd.concat([eu, eu.assign(model="final")], ignore_index=True)
    print(f"european cups done [{time.time() - t0:.0f}s]")

    nat = national_walk_forward(intl, "2023-01-01", "2026-12-31")
    nat = pd.concat([nat, nat[nat["model"] == "elo"].assign(model="final")], ignore_index=True)
    print(f"national teams done [{time.time() - t0:.0f}s]")

    allp = pd.concat([dom, eu, nat.drop(columns=["tournament"])], ignore_index=True)
    storage.save_results(matches)
    storage.save_results(euro.dropna(subset=["hg"]).assign(
        league=euro["competition"].map(EURO_CODE), hg=lambda d: d["hg"].astype(int),
        ag=lambda d: d["ag"].astype(int)))
    storage.save_results(intl)
    n = storage.save_run("bt_all", "backtest", allp,
                         description="все турниры: лиги 2023–25, еврокубки 2023–25, сборные 2023–26",
                         params={"dc": DC_PARAMS, "elo": ELO_PARAMS})
    pd.set_option("display.width", 200)
    for name, part in (("ЛИГИ", dom), ("ЕВРОКУБКИ", eu), ("СБОРНЫЕ", nat)):
        models = sorted(part["model"].unique())
        print(f"\n== {name} ==")
        print(summary_table(common_matches(part, models)).round(4).to_string(index=False))
    print(f"\nsaved {n:,} rows as 'bt_all' [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
