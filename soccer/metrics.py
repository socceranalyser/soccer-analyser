"""Scoring rules and calibration diagnostics for probabilistic forecasts.

Probabilities are arrays (n, 3) in [home, draw, away] order; outcomes are ints 0/1/2
in the same order (0 = home win).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def _onehot(y: np.ndarray, k: int = 3) -> np.ndarray:
    return np.eye(k)[y]


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, None))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sum((p - _onehot(y, p.shape[1])) ** 2, axis=1)))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score — the standard ordinal score for 1X2 (lower is better)."""
    cp = np.cumsum(p, axis=1)[:, :-1]
    co = np.cumsum(_onehot(y, p.shape[1]), axis=1)[:, :-1]
    return float(np.mean(np.sum((cp - co) ** 2, axis=1) / (p.shape[1] - 1)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(p.argmax(axis=1) == y))


def binary_log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Binned predicted vs observed frequency, per outcome class (for 1X2) or binary."""
    if p.ndim == 1:
        p, y, labels = p[:, None], y[:, None].astype(float), ["yes"]
    else:
        y = _onehot(y, p.shape[1])
        labels = ["H", "D", "A"]
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for k, lab in enumerate(labels):
        b = np.clip(np.digitize(p[:, k], edges) - 1, 0, bins - 1)
        for i in range(bins):
            m = b == i
            if m.sum() == 0:
                continue
            rows.append({"outcome": lab, "bin": i, "n": int(m.sum()),
                         "predicted": float(p[m, k].mean()), "observed": float(y[m, k].mean())})
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error averaged over outcome classes."""
    rel = reliability(p, y, bins)
    total = rel.groupby("outcome")["n"].transform("sum")
    return float((rel["n"] / total * (rel["predicted"] - rel["observed"]).abs())
                 .groupby(rel["outcome"]).sum().mean())


def summarize(df: pd.DataFrame) -> dict:
    """df needs p_home/p_draw/p_away/result and optionally p_over25 + hg/ag."""
    df = df.dropna(subset=["p_home", "result"])
    p = df[["p_home", "p_draw", "p_away"]].to_numpy(float)
    p = p / p.sum(axis=1, keepdims=True)
    y = df["result"].map(OUTCOME_INDEX).to_numpy()
    out = {"n": len(df), "log_loss": log_loss(p, y), "rps": rps(p, y),
           "brier": brier(p, y), "accuracy": accuracy(p, y), "ece": ece(p, y)}
    if "p_over25" in df and df["p_over25"].notna().any():
        d = df.dropna(subset=["p_over25"])
        yo = ((d["hg"] + d["ag"]) > 2.5).to_numpy(float)
        out["ou25_log_loss"] = binary_log_loss(d["p_over25"].to_numpy(float), yo)
    else:
        out["ou25_log_loss"] = np.nan
    return out


def summary_table(preds: pd.DataFrame, by=("model",)) -> pd.DataFrame:
    rows = []
    for key, g in preds.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **summarize(g)})
    return pd.DataFrame(rows)
