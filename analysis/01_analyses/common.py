"""Shared loaders and helpers for the analyses."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = REV.parent
sys.path.insert(0, str(ROOT / "src"))

from p6 import exp_confirm as EC
from p6 import exp_retest as ER
from p6 import uncertainty as U
from p6.exp_mechanism import decision_sigma
from p6.stats import holm

SEED_C = EC.SEED
SEED_B = ER.SEED
R = 100
DERIV = EC.DERIV
RESULTS = ROOT / "results"
paired = EC.paired
nlpd = EC.nlpd
mean_ci = EC.mean_ci


def nyu2_cohort() -> pd.DataFrame:
    return EC.load_cohort(True)


def nyu2_load(pid: str, r: int) -> np.ndarray:
    import scipy.io as sio
    return sio.loadmat(DERIV / f"{pid}_run{r}" / f"ROISignals_{pid}_run{r}.mat")["ROISignals"][:, :116].astype(np.float64)


def point_z(X: np.ndarray) -> np.ndarray:
    T, N = X.shape
    iu = np.triu_indices(N, 1)
    Z = ER._std(X)
    r = np.clip((Z.T @ Z / T)[iu], -0.999999, 0.999999)
    return np.arctanh(r)


def nyu2_ages(C: pd.DataFrame) -> pd.Series:
    d = HERE / "external_metadata" / "nyu2" / "sessions"
    out = {}
    for pid in C.pid:
        f = d / f"{pid}_sessions.tsv"
        age = np.nan
        if f.exists():
            t = pd.read_csv(f, sep="\t", dtype=str)
            row = t[t.session_id == "ses-1"]
            if len(row):
                age = pd.to_numeric(row.iloc[0].get("age_at_scan_1"), errors="coerce")
        out[pid] = age
    return C.pid.map(out)


def d1_fold_weights(ZA: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    n, E_ = ZA.shape
    W1, c1, fold = np.zeros((n, E_)), np.zeros(n), np.zeros(n, int)
    lams = [0.01, 0.1, 1.0, 10.0]
    fit = lambda X, t, lam: LogisticRegression(C=1.0 / (2 * len(t) * lam), max_iter=5000, tol=1e-6).fit(X, t)
    for f, (tr, te) in enumerate(StratifiedKFold(10, shuffle=True, random_state=SEED_C).split(ZA, y)):
        ll = {}
        for lam in lams:
            v = []
            for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED_C).split(ZA[tr], y[tr]):
                p = np.clip(fit(ZA[tr][a], y[tr][a], lam).predict_proba(ZA[tr][b])[:, 1], 1e-6, 1 - 1e-6)
                v.append(-np.mean(y[tr][b] * np.log(p) + (1 - y[tr][b]) * np.log(1 - p)))
            ll[lam] = np.mean(v)
        m = fit(ZA[tr], y[tr], min(ll, key=ll.get))
        W1[te], c1[te], fold[te] = m.coef_[0], m.intercept_[0], f
    return W1, c1, fold


def nyu2_outer_folds(y: np.ndarray):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(10, shuffle=True, random_state=SEED_C).split(np.zeros((len(y), 1)), y))


def fixed_weights(E_: int, seed: int):
    w2 = np.full(E_, 1.0 / E_)
    w3 = np.random.default_rng(seed).standard_normal(E_) / np.sqrt(E_)
    return w2, w3


def nyu2_point_all(C: pd.DataFrame, T: int | None = None):
    ZA, ZB = [], []
    for pid in C.pid:
        A, B = nyu2_load(pid, 1), nyu2_load(pid, 2)
        if T is not None:
            A, B = A[:T], B[:T]
        ZA.append(point_z(A)); ZB.append(point_z(B))
    return np.stack(ZA), np.stack(ZB)


def tcp_primary(pe: str) -> pd.DataFrame:
    S = pd.read_csv(ROOT / ".cache" / "tcp" / f"stats_{pe}.csv")
    prim = S[(S.fd_A <= ER.FD_MAX) & (S.fd_B <= ER.FD_MAX)].reset_index(drop=True)
    return prim[prim.Group.isin(["Patient", "GenPop"])].reset_index(drop=True)


def tcp_fold_weights(pe: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    S = tcp_primary(pe)
    A = np.load(ROOT / ".cache" / "tcp" / f"arrays_{pe}.npz")
    subs_all = list(A["subs"])
    pos = {s: i for i, s in enumerate(subs_all)}
    ix = np.array([pos[s] for s in S["sub"]])
    XA = A["zA"][ix].astype(np.float64)
    y = (S.Group == "Patient").astype(int).values
    strat = S.Group.astype(str) + "_" + S.Site.astype(str)
    lams = [0.01, 0.1, 1.0, 10.0]
    W, c, lamv = np.zeros((len(S), XA.shape[1])), np.zeros(len(S)), np.zeros(len(S))
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=SEED_B).split(XA, strat):
        def fit(Xt, yt, lam):
            return LogisticRegression(C=1.0 / (2 * len(yt) * lam), max_iter=5000, tol=1e-6).fit(Xt, yt)
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED_B)
        ll = {}
        for lam in lams:
            v = []
            for a, b in inner.split(XA[tr], strat.values[tr]):
                p = np.clip(fit(XA[tr][a], y[tr][a], lam).predict_proba(XA[tr][b])[:, 1], 1e-6, 1 - 1e-6)
                v.append(-np.mean(y[tr][b] * np.log(p) + (1 - y[tr][b]) * np.log(1 - p)))
            ll[lam] = np.mean(v)
        lam = min(ll, key=ll.get)
        m = fit(XA[tr], y[tr], lam)
        W[te], c[te], lamv[te] = m.coef_[0], float(m.intercept_[0]), lam
    del XA, A
    return S, W, c, lamv, y


def stream_cbb(X: np.ndarray, rng: np.random.Generator, Wm: np.ndarray, n_rep: int = R, chunk: int = 10):
    T, N = X.shape
    iu = np.triu_indices(N, 1)
    b = U.fc_block_length(X, source="roi", kind="cb", n_series=min(256, N), rng=rng)
    idx = U.cbb_indices(T, b, n_rep, rng)
    L, s1, s2, n = [], np.zeros(len(iu[0])), np.zeros(len(iu[0])), 0
    for s in range(0, n_rep, chunk):
        Zr = ER._ztri(ER._std(X[idx[s:s + chunk]]), iu)
        L.append(Zr @ Wm)
        s1 += Zr.sum(0); s2 += (Zr ** 2).sum(0); n += Zr.shape[0]
        del Zr
    var = (s2 - s1 ** 2 / n) / (n - 1)
    return b, np.concatenate(L), var


def xdf_var(X: np.ndarray) -> np.ndarray:
    T, N = X.shape
    iu = np.triu_indices(N, 1)
    Z = ER._std(X)
    r = np.clip((Z.T @ Z / T)[iu], -0.999999, 0.999999)
    return ER.xdf_z(X, r, iu)


def tcp_load(sub: str, pe: str, run: int) -> np.ndarray:
    return ER.load_run(sub, pe, run)


def jdump(obj, path: Path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=1, default=conv))


CACHE = HERE / "_cache"


def d1_weights_cached(C: pd.DataFrame, ZA: np.ndarray, y: np.ndarray):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "d1_weights.npz"
    if f.exists():
        a = np.load(f)
        if list(a["pids"]) == list(C.pid):
            return a["W"], a["c"], a["fold"]
    W, c, fold = d1_fold_weights(ZA, y)
    np.savez(f, W=W, c=c, fold=fold, pids=np.array(C.pid, dtype=str))
    return W, c, fold


def tcp_weights_cached(pe: str = "AP"):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"tcp_weights_{pe}.npz"
    if f.exists():
        a = np.load(f, allow_pickle=True)
        S = tcp_primary(pe)
        assert list(a["subs"]) == list(S["sub"])
        return S, a["W"].astype(np.float64), a["c"], a["lam"], a["y"]
    S, W, c, lam, y = tcp_fold_weights(pe)
    A = np.load(ROOT / ".cache" / "tcp" / f"arrays_{pe}.npz")
    pos = {s: i for i, s in enumerate(A["subs"])}
    zA = A["zA"]
    lA = np.array([float(zA[pos[s]].astype(np.float64) @ W[i] + c[i]) for i, s in enumerate(S["sub"])])
    Fl = pd.read_csv(RESULTS / f"discovery_flips_{pe}.csv").set_index("sub").loc[S["sub"]]
    rel = float(np.max(np.abs(lA - Fl.lA.values) / np.maximum(np.abs(Fl.lA.values), 1e-12)))
    del A, zA
    np.savez(f, W=W.astype(np.float32), c=c, lam=lam, y=y, subs=np.array(S["sub"], dtype=str), repro_rel=rel)
    return S, W.astype(np.float32).astype(np.float64), c, lam, y
