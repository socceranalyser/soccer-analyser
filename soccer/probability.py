"""Score-matrix maths: Dixon-Coles correction and derived markets."""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10  # matrix is (MAX_GOALS+1) x (MAX_GOALS+1)


def dc_tau_matrix(lam, mu, rho, n: int = MAX_GOALS + 1) -> np.ndarray:
    """Dixon-Coles low-score correction, broadcast over leading dims of lam/mu."""
    lam = np.asarray(lam, float)[..., None, None]
    mu = np.asarray(mu, float)[..., None, None]
    tau = np.ones(lam.shape[:-2] + (n, n))
    tau[..., 0, 0] = (1 - lam * mu * rho)[..., 0, 0]
    tau[..., 0, 1] = (1 + lam * rho)[..., 0, 0]
    tau[..., 1, 0] = (1 + mu * rho)[..., 0, 0]
    tau[..., 1, 1] = 1 - rho
    return tau


def score_matrix(lam, mu, rho: float = 0.0, n: int = MAX_GOALS + 1) -> np.ndarray:
    """P(home=i, away=j). lam/mu may be scalars or arrays (matrices stacked on leading dims)."""
    goals = np.arange(n)
    lam_a = np.asarray(lam, float)
    mu_a = np.asarray(mu, float)
    ph = poisson.pmf(goals, lam_a[..., None])
    pa = poisson.pmf(goals, mu_a[..., None])
    m = ph[..., :, None] * pa[..., None, :]
    if rho:
        m = m * dc_tau_matrix(lam_a, mu_a, rho, n)
    m = np.clip(m, 0, None)
    return m / m.sum(axis=(-2, -1), keepdims=True)


def outcome_probs(m: np.ndarray) -> np.ndarray:
    """[P(home), P(draw), P(away)] from a score matrix (supports stacked matrices)."""
    home = np.tril(m, -1).sum(axis=(-2, -1))
    draw = np.trace(m, axis1=-2, axis2=-1)
    away = np.triu(m, 1).sum(axis=(-2, -1))
    return np.stack([home, draw, away], axis=-1)


def total_goals_dist(m: np.ndarray) -> np.ndarray:
    n = m.shape[-1]
    i, j = np.indices((n, n))
    tot = i + j
    return np.stack([m[..., tot == k].sum(axis=-1) for k in range(2 * n - 1)], axis=-1)


def over_prob(m: np.ndarray, line: float = 2.5) -> np.ndarray:
    n = m.shape[-1]
    i, j = np.indices((n, n))
    return (m * (i + j > line)).sum(axis=(-2, -1))


def btts_prob(m: np.ndarray) -> np.ndarray:
    return m[..., 1:, 1:].sum(axis=(-2, -1))


def top_scores(m: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
    flat = m.ravel()
    idx = np.argsort(flat)[::-1][:k]
    n = m.shape[-1]
    return [(f"{i // n}:{i % n}", float(flat[i])) for i in idx]


def markets(m: np.ndarray) -> dict:
    """Everything the dashboard shows for a single match."""
    n = m.shape[-1]
    goals = np.arange(n)
    p = outcome_probs(m)
    return {
        "p_home": float(p[0]), "p_draw": float(p[1]), "p_away": float(p[2]),
        "xg_home": float((m.sum(axis=1) * goals).sum()),
        "xg_away": float((m.sum(axis=0) * goals).sum()),
        "totals": {line: float(over_prob(m, line)) for line in (0.5, 1.5, 2.5, 3.5, 4.5)},
        "btts": float(btts_prob(m)),
        "home_clean_sheet": float(m[:, 0].sum()),
        "away_clean_sheet": float(m[0, :].sum()),
        "top_scores": top_scores(m, 10),
    }
