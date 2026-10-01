"""Post-hoc probability calibration for 1X2 forecasts.

VectorScaling: p'_k ∝ exp(a_k · log p_k + b_k)  (Guo et al. 2017 / Kull et al. 2019).
With a_k = 1, b_k = 0 it is the identity, so it can only help if the backtest says so.
Must always be fitted on *earlier* out-of-sample predictions than those it is applied to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax


@dataclass
class VectorScaling:
    l2: float = 1.0  # shrinkage towards the identity map

    def fit(self, p: np.ndarray, y: np.ndarray):
        lp = np.log(np.clip(p, 1e-12, 1))
        n = len(y)

        def nll(params):
            a, b = params[:3], np.r_[params[3:], 0.0]
            z = log_softmax(lp * a + b, axis=1)
            reg = self.l2 * (np.sum((a - 1) ** 2) + np.sum(b ** 2))
            return -z[np.arange(n), y].mean() + reg / n

        res = minimize(nll, x0=[1, 1, 1, 0, 0], method="L-BFGS-B")
        self.params = res.x
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        a, b = self.params[:3], np.r_[self.params[3:], 0.0]
        z = np.log(np.clip(p, 1e-12, 1)) * a + b
        return np.exp(log_softmax(z, axis=1))
