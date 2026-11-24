"""Decision-level propagation (J-CBB, uHAC, independent propagation) and scoring helpers."""
from __future__ import annotations


import argparse


import json


import time


import numpy as np


import pandas as pd


from . import exp_retest as E


from .registry import ROOT, Run


def projected_influence_fast(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    Z = E._std(X)
    T, N = Z.shape
    C = Z.T @ Z / T
    i, j = np.triu_indices(N, 1)
    r = np.clip(C[i, j], -0.999999, 0.999999)
    G = np.zeros((N, N))
    G[i, j] = g / (1.0 - r ** 2)
    G = G + G.T
    Rm = np.zeros((N, N))
    Rm[i, j] = r
    Rm = Rm + Rm.T
    c = 0.5 * (G * Rm).sum(1)
    return 0.5 * np.einsum("ti,ti->t", Z @ G, Z) - (Z ** 2) @ c


def lrv(u: np.ndarray, kernel: str, bw: float) -> float:
    u = u - u.mean()
    T = len(u)
    H = int(min(np.ceil(bw), T - 1))
    g = np.array([u[: T - h] @ u[h:] / T for h in range(H + 1)])
    h = np.arange(1, H + 1)
    k = np.clip(1 - h / bw, 0, None) if kernel == "bartlett" else 0.5 * (1 + np.cos(np.pi * h / bw))
    return float(g[0] + 2 * np.sum(k * g[1:]))


def decision_sigma(X: np.ndarray, g: np.ndarray, b: float) -> dict:
    u = projected_influence_fast(X, g)
    T = len(u)
    return {"uhac_bartlett": np.sqrt(max(lrv(u, "bartlett", 2 * b), 0) / T),
            "uhac_tukey": np.sqrt(max(lrv(u, "tukey", np.sqrt(T)), 0) / T),
            "u_iid": np.sqrt(u.var() / T)}


def _metrics(p, f):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"brier": float(np.mean((p - f) ** 2)), "logloss": float(-np.mean(f * np.log(p) + (1 - f) * np.log(1 - p))),
           "mean_p": float(p.mean()), "obs": float(f.mean())}
    if 0 < f.sum() < len(f):
        out |= {"auc": float(roc_auc_score(f, p)), "pr_auc": float(average_precision_score(f, p))}
        lo = np.log(p / (1 - p))[:, None]
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(lo, f)
        out |= {"cal_slope": float(cal.coef_[0, 0]), "cal_intercept": float(cal.intercept_[0])}
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "p": p, "f": f}).groupby("q")
        nk, pk, ok = g.size().values, g.p.mean().values, g.f.mean().values
        out |= {"reliability": float(np.sum(nk * (pk - ok) ** 2) / len(f)), "resolution": float(np.sum(nk * (ok - f.mean()) ** 2) / len(f)),
                "uncertainty": float(f.mean() * (1 - f.mean()))}
    return out


def _paired(a, b, f, n=10000, seed=E.SEED):
    rng = np.random.default_rng(seed)
    d = (a - f) ** 2 - (b - f) ** 2
    bo = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"diff": float(d.mean()), "ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])],
            "p_two_sided": float(min(1.0, 2 * min(np.mean(bo <= 0), np.mean(bo >= 0)))), "n": int(len(d))}
