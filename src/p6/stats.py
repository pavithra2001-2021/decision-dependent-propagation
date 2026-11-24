"""Statistical helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .metrics import brier, logloss


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _site_metric(y, p, metric):
    if metric == "brier":
        return brier(y, p)
    if metric == "logloss":
        return logloss(y, p)
    if metric == "auc":
        return fast_auc(y, p)
    raise ValueError(metric)


def site_weighted_diff(df: pd.DataFrame, pa: str, pb: str, metric: str) -> float:
    vals = [_site_metric(g.y.values, g[pa].values, metric) - _site_metric(g.y.values, g[pb].values, metric)
            for _, g in df.groupby("site")]
    return float(np.nanmean(vals))


def paired_bootstrap(df: pd.DataFrame, pa: str, pb: str, metric: str = "brier", n_boot: int = 2000,
                     scheme: str = "hierarchical", seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    groups = {s: (g.y.values, g[pa].values, g[pb].values) for s, g in df.groupby("site")}
    sites = list(groups)
    est = site_weighted_diff(df, pa, pb, metric)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(sites, size=len(sites), replace=True) if scheme == "hierarchical" else sites
        vals = []
        for s in draw:
            y, a, c = groups[s]
            k = rng.integers(0, len(y), size=len(y))
            vals.append(_site_metric(y[k], a[k], metric) - _site_metric(y[k], c[k], metric))
        boots[b] = np.nanmean(vals)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"metric": metric, "diff": est, "ci95": [float(lo), float(hi)], "scheme": scheme,
            "p_two_sided": float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))),
            "n_sites": len(sites), "n_boot": n_boot}


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    adj[order] = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return adj


def per_site_diff(df: pd.DataFrame, pa: str, pb: str, metric: str) -> pd.Series:
    return pd.Series({s: _site_metric(g.y.values, g[pa].values, metric) - _site_metric(g.y.values, g[pb].values, metric)
                      for s, g in df.groupby("site")})
