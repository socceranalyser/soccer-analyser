r"""Re-tune the European-cup Elo settings (country_spill, k_cup_mult) on validation seasons
2019-2022, confirm on 2023-2025.   python scripts/tune_euro.py"""
import itertools
import sys

from soccer.backtest import euro_walk_forward
from soccer.data import load_matches
from soccer.euro import load_euro
from soccer.metrics import summary_table
from soccer.tuned import ELO_PARAMS

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    m = load_matches()
    e = load_euro(m)
    res = {}
    for spill, kc in itertools.product([0.2, 0.35, 0.5, 0.65], [1.0, 1.5, 2.0, 3.0]):
        p = euro_walk_forward(m, e, [2019, 2020, 2021, 2022],
                              {**ELO_PARAMS, "country_spill": spill, "k_cup_mult": kc})
        res[(spill, kc)] = summary_table(p)["log_loss"].iloc[0]
        print(f"spill {spill:.2f} k_cup {kc:.1f}: validation {res[(spill, kc)]:.4f}", flush=True)
    best = min(res, key=res.get)
    for name, prm in (("current", (ELO_PARAMS["country_spill"], ELO_PARAMS["k_cup_mult"])),
                      ("best", best)):
        p = euro_walk_forward(m, e, [2023, 2024, 2025],
                              {**ELO_PARAMS, "country_spill": prm[0], "k_cup_mult": prm[1]})
        print(f"TEST 2023-25 {name} {prm}: {summary_table(p)['log_loss'].iloc[0]:.4f}")
