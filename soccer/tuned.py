"""Hyperparameters currently in production (update after scripts/tune.py + backtest)."""

# tuned on 2019-2022 top-5 (scripts/tune.py dc: 108-config grid, then xi/mult follow-up):
# logloss 0.99173 (untuned defaults: 0.99389)
DC_PARAMS = {"xi": 0.0025, "ridge": 6.0, "window_days": 365 * 5, "n_samples": 0,
             "newcomer_shift": 0.0, "newcomer_ridge_mult": 6.0}
# tuned on 2019-2022 (scripts/tune.py elo): logloss 0.99533 vs 0.9958 for k=20/regress=0.2
# European cups (euro walk-forward, val 2019-22): spill 0.35 / k_cup 2.0 -> logloss 0.9725
# (no spill 1.0007); domestic logloss unchanged (1.02342 -> 1.02355)
# xG (understat, top-5 leagues): DC learns from 0.1*goals + 0.9*xG. scripts/tune_xg.py:
# validation 2019-22 ensemble 0.9918 -> 0.9890; test 2023-25 0.9832 -> 0.9817, totals 0.6755 -> 0.6750
XG_WEIGHT = 0.9
ELO_PARAMS = {"k": 15.0, "home_adv": 65.0, "season_regress": 0.0,
              "country_spill": 0.35, "k_cup_mult": 2.0}
