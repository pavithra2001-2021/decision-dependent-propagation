"""Scoring metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_EPS = 1e-7


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(y, p):
    p = np.clip(np.asarray(p), _EPS, 1 - _EPS)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins: int = 10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(np.abs(y[k == b].mean() - p[k == b].mean()) * (k == b).mean() for b in range(bins) if (k == b).any()))


def calibration_slope_intercept(y, p):
    from sklearn.linear_model import LogisticRegression
    lo = np.log(np.clip(p, _EPS, 1 - _EPS) / np.clip(1 - p, _EPS, 1))
    m = LogisticRegression(C=1e6).fit(lo[:, None], y)
    citl = float(np.mean(y) - np.mean(p))
    return float(m.coef_[0, 0]), citl


def binary_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= 0.5
    sens = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else np.nan
    two = len(np.unique(y)) == 2
    return {"auc": float(roc_auc_score(y, p)) if two else np.nan,
            "pr_auc": float(average_precision_score(y, p)) if two else np.nan,
            "brier": brier(y, p), "logloss": logloss(y, p), "ece": ece(y, p),
            "bacc": float(np.nanmean([sens, spec])), "sens": sens, "spec": spec,
            "p_sd": float(np.std(p))}


def site_weighted(df: pd.DataFrame, pcol: str = "p") -> dict:
    per = {s: binary_metrics(g.y.values, g[pcol].values) for s, g in df.groupby("site")}
    keys = next(iter(per.values())).keys()
    out = {f"sw_{k}": float(np.nanmean([v[k] for v in per.values()])) for k in keys}
    out.update({f"pooled_{k}": v for k, v in binary_metrics(df.y.values, df[pcol].values).items()})
    return out
