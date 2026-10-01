"""Dixon-Coles bivariate Poisson model with time decay and a Bayesian (Gaussian) prior.

log λ_home = c + home + att[h] - def[a]
log μ_away = c + att[a] - def[h]
P(x, y) = τ(x, y; λ, μ, ρ) · Pois(x; λ) · Pois(y; μ)

* Time decay: match weight w = exp(-xi · days_ago)  (Dixon & Coles, 1997).
* Prior: att, def ~ N(0, 1/ridge) -> the fit is a MAP estimate (shrinks teams with
  little data towards the league average).
* Laplace approximation: posterior ≈ N(θ_MAP, H⁻¹) with H the Fisher information
  plus prior precision. Used for posterior-predictive match probabilities
  (n_samples > 0) and for parameter uncertainty in Monte Carlo season simulation.
* Teams unseen in training (promoted with no recent top-flight history) get the
  average parameters of the weakest `proxy_k` teams.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..probability import outcome_probs, over_prob, score_matrix


@dataclass
class DixonColes:
    xi: float = 0.0019          # time decay per day (half-life ≈ 1 year)
    ridge: float = 10.0         # prior precision for att/def
    window_days: int = 365 * 4  # training window
    n_samples: int = 0          # >0 -> Bayesian posterior predictive with this many draws
    proxy_k: int = 3            # weakest teams averaged to get a proxy for unknown teams
    promoted_prior: bool = True  # newcomers' prior mean = relegated teams' mean ...
    newcomer_shift: float = 0.0       # ... minus this (on both attack and defence)
    newcomer_ridge_mult: float = 1.0  # prior precision multiplier for newcomers
    seed: int = 0
    name: str = "dixon_coles"

    teams: list = field(default_factory=list, init=False)
    theta: np.ndarray | None = field(default=None, init=False)
    cov: np.ndarray | None = field(default=None, init=False)

    # ------------------------------------------------------------------ fitting
    def fit(self, matches: pd.DataFrame, as_of=None, warm_start: "DixonColes | None" = None,
            season_teams=None, season: int | None = None):
        """season / season_teams: the season being predicted and its participants (public
        before it starts). Newcomers get a prior centred on last season's relegated teams."""
        as_of = pd.Timestamp(as_of) if as_of is not None else matches["date"].max() + pd.Timedelta(days=1)
        train = matches[(matches["date"] < as_of)
                        & (matches["date"] >= as_of - pd.Timedelta(days=self.window_days))]
        self.as_of = as_of
        if season is None:
            season = int(train["season"].max()) if len(train) else 0
        prev = set(train.loc[train["season"] == season - 1, "home"])
        if season_teams is None:
            season_teams = set(train.loc[train["season"] == season, "home"]) | set(
                train.loc[train["season"] == season, "away"])
        season_teams = set(season_teams)
        newcomers = season_teams - prev if prev and season_teams else set()
        relegated = prev - season_teams if newcomers else set()
        self.teams = sorted(set(train["home"]) | set(train["away"]) | season_teams)
        self._idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        # Gaussian prior precision Q = ridge·DᵀD, D = I - M; M maps newcomers to the
        # mean of the relegated teams (others are centred on 0 = league average).
        # penalty = ridge/2 · Σ_i W_i ((Dθ)_i + b_i)²  for θ = att and θ = def separately
        D = np.eye(n)
        W = np.ones(n)
        b = np.zeros(n)
        if self.promoted_prior and newcomers and relegated:
            rel_idx = [self._idx[t] for t in relegated]
            for t in newcomers:
                i = self._idx[t]
                D[i, rel_idx] -= 1.0 / len(rel_idx)
                W[i] = self.newcomer_ridge_mult
                b[i] = self.newcomer_shift
        Q = self.ridge * D.T @ (W[:, None] * D)
        lin = self.ridge * D.T @ (W * b)  # gradient of the shift term
        self._Q = Q
        self.newcomers = sorted(newcomers)
        h = train["home"].map(self._idx).to_numpy()
        a = train["away"].map(self._idx).to_numpy()
        x = train["hg"].to_numpy(float)
        y = train["ag"].to_numpy(float)
        w = np.exp(-self.xi * (as_of - train["date"]).dt.days.to_numpy(float))
        if "weight" in train:  # match importance (e.g. friendlies count less)
            w = w * train["weight"].to_numpy(float)
        # home factor: 0 at a neutral venue
        hf = 1.0 - train["neutral"].to_numpy(float) if "neutral" in train else np.ones(len(train))
        self._hf = hf

        theta0 = np.zeros(2 * n + 3)
        theta0[2 * n:] = [0.1, 0.25, -0.05]
        if warm_start is not None and warm_start.theta is not None:
            for t, i in self._idx.items():
                j = warm_start._idx.get(t)
                if j is not None:
                    theta0[i] = warm_start.theta[j]
                    theta0[n + i] = warm_start.theta[warm_start.n + j]
            theta0[2 * n:] = warm_start.theta[-3:]

        m00, m01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
        m10, m11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
        def objective(theta):
            att, dfn = theta[:n], theta[n:2 * n]
            c, home, rho = theta[2 * n:]
            eta1 = c + home * hf + att[h] - dfn[a]
            eta2 = c + att[a] - dfn[h]
            lam, mu = np.exp(eta1), np.exp(eta2)
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            tau = np.maximum(tau, 1e-10)
            ll = w * (np.log(tau) + x * eta1 - lam + y * eta2 - mu)

            g1 = w * (x - lam)
            g2 = w * (y - mu)
            t00 = w[m00] * (-lam[m00] * mu[m00] * rho) / tau[m00]
            g1[m00] += t00
            g2[m00] += t00
            g1[m01] += w[m01] * lam[m01] * rho / tau[m01]
            g2[m10] += w[m10] * mu[m10] * rho / tau[m10]
            grho = (np.sum(w[m00] * -lam[m00] * mu[m00] / tau[m00])
                    + np.sum(w[m01] * lam[m01] / tau[m01])
                    + np.sum(w[m10] * mu[m10] / tau[m10])
                    + np.sum(w[m11] * -1 / tau[m11]))
            g_att = np.bincount(h, g1, n) + np.bincount(a, g2, n)
            g_def = -np.bincount(a, g1, n) - np.bincount(h, g2, n)
            grad = -np.concatenate([g_att, g_def, [g1.sum() + g2.sum(), (g1 * hf).sum(), grho]])
            q_att, q_def = Q @ att, Q @ dfn
            grad[:n] += q_att + lin
            grad[n:2 * n] += q_def + lin
            obj = -ll.sum() + 0.5 * (att @ q_att + dfn @ q_def) + lin @ (att + dfn)
            return obj, grad

        bounds = [(None, None)] * (2 * n + 2) + [(-0.25, 0.25)]
        res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "gtol": 1e-6})
        self.theta = res.x
        self.n = n
        self.converged = bool(res.success)
        self.cov = self._laplace_cov(h, a, w)
        strength = self.theta[:n] + self.theta[n:2 * n]
        self._proxy = np.argsort(strength)[: min(self.proxy_k, n)]
        return self

    def _laplace_cov(self, h, a, w) -> np.ndarray:
        """Inverse of (Fisher information + prior precision) for [att, def, c, home]."""
        n = self.n
        att, dfn = self.theta[:n], self.theta[n:2 * n]
        c, home, _ = self.theta[2 * n:]
        hf = self._hf
        lam = np.exp(c + home * hf + att[h] - dfn[a])
        mu = np.exp(c + att[a] - dfn[h])
        m = len(h)
        p = 2 * n + 2
        rows = np.arange(m)
        x1 = np.zeros((m, p))
        x1[rows, h] = 1
        x1[rows, n + a] -= 1
        x1[:, 2 * n] = 1
        x1[:, 2 * n + 1] = hf
        x2 = np.zeros((m, p))
        x2[rows, a] = 1
        x2[rows, n + h] -= 1
        x2[:, 2 * n] = 1
        info = (x1 * (w * lam)[:, None]).T @ x1 + (x2 * (w * mu)[:, None]).T @ x2
        info[:n, :n] += self._Q
        info[n:2 * n, n:2 * n] += self._Q
        return np.linalg.pinv(info)

    # --------------------------------------------------------------- prediction
    def _params(self, theta: np.ndarray):
        """att/def arrays with an extra trailing 'proxy' slot for unknown teams."""
        n = self.n
        att, dfn = theta[..., :n], theta[..., n:2 * n]
        att = np.concatenate([att, att[..., self._proxy].mean(-1, keepdims=True)], -1)
        dfn = np.concatenate([dfn, dfn[..., self._proxy].mean(-1, keepdims=True)], -1)
        return att, dfn, theta[..., 2 * n], theta[..., 2 * n + 1]

    def _index(self, teams) -> np.ndarray:
        return np.array([self._idx.get(t, self.n) for t in teams])

    def rates(self, homes, aways, theta=None, neutral=None):
        theta = self.theta if theta is None else theta
        att, dfn, c, home = self._params(theta)
        hi, ai = self._index(homes), self._index(aways)
        c, home = np.asarray(c)[..., None], np.asarray(home)[..., None]
        hf = 1.0 if neutral is None else 1.0 - np.asarray(neutral, float)
        lam = np.exp(c + home * hf + att[..., hi] - dfn[..., ai])
        mu = np.exp(c + att[..., ai] - dfn[..., hi])
        return lam, mu

    @property
    def rho(self) -> float:
        return float(self.theta[-1])

    def sample_theta(self, n_samples: int, rng=None) -> np.ndarray:
        """Draws from the Laplace posterior (rho kept at its MAP value)."""
        rng = rng or np.random.default_rng(self.seed)
        draws = rng.multivariate_normal(self.theta[:-1], self.cov, size=n_samples,
                                        method="cholesky" if self._pd() else "svd")
        return np.column_stack([draws, np.full(n_samples, self.theta[-1])])

    def _pd(self) -> bool:
        try:
            np.linalg.cholesky(self.cov)
            return True
        except np.linalg.LinAlgError:
            return False

    def score_matrices(self, homes, aways, neutral=None) -> np.ndarray:
        """(n_matches, G, G) score matrices; posterior-predictive if n_samples > 0."""
        if self.n_samples > 0:
            lam, mu = self.rates(homes, aways, self.sample_theta(self.n_samples), neutral)
            return score_matrix(lam, mu, self.rho).mean(axis=0)
        lam, mu = self.rates(homes, aways, neutral=neutral)
        return score_matrix(lam, mu, self.rho)

    def predict(self, homes, aways, neutral=None) -> dict:
        m = self.score_matrices(list(homes), list(aways), neutral)
        n = m.shape[-1]
        g = np.arange(n)
        return {
            "probs": outcome_probs(m),
            "p_over25": over_prob(m, 2.5),
            "xg_home": (m.sum(-1) * g).sum(-1),
            "xg_away": (m.sum(-2) * g).sum(-1),
            "matrices": m,
        }

    def ratings(self) -> pd.DataFrame:
        n = self.n
        att, dfn = self.theta[:n], self.theta[n:2 * n]
        sd = np.sqrt(np.clip(np.diag(self.cov), 0, None))
        df = pd.DataFrame({"team": self.teams, "attack": att, "defence": dfn,
                           "attack_sd": sd[:n], "defence_sd": sd[n:2 * n]})
        df["strength"] = df["attack"] + df["defence"]
        return df.sort_values("strength", ascending=False).reset_index(drop=True)
