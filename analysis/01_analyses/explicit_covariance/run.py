"""Explicit HAC covariance of all features and its identity with uHAC."""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
C = K.nyu2_cohort()
y = (C.sex == "female").astype(int).values
ZA, ZB = K.nyu2_point_all(C)
W1, c1, _ = K.d1_weights_cached(C, ZA, y)
w2, w3 = K.fixed_weights(ZA.shape[1], K.SEED_C)
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
res = []
for k in range(3):
    pid = C.pid[k]
    X = K.nyu2_load(pid, 1)
    mA = K.EC.run_measurement(X, np.random.default_rng(K.SEED_C + 10 * k + 1), K.R)
    b = mA["b"]
    T, N = X.shape
    t0 = time.perf_counter()
    Z = K.ER._std(X)
    i, j = np.triu_indices(N, 1)
    r = np.clip((Z.T @ Z / T)[i, j], -0.999999, 0.999999)
    IF = (Z[:, i] * Z[:, j] - 0.5 * r * (Z[:, i] ** 2 + Z[:, j] ** 2)) / (1.0 - r ** 2)
    IF -= IF.mean(0)
    bw = 2 * b
    Hm = int(min(np.ceil(bw), T - 1))
    lag = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    Kmat = np.where(lag <= Hm, np.clip(1 - lag / bw, 0, None), 0.0)
    Sig = IF.T @ (Kmat @ IF) / T ** 2
    t_sig = time.perf_counter() - t0
    rec = {"pid": pid, "T": T, "N": N, "E": int(len(r)), "b": float(b), "bandwidth": float(bw), "H": Hm,
           "sigma_bytes": int(Sig.nbytes), "t_form_sigma_s": t_sig}
    for D, w in (("D1", W1[k]), ("D2", w2), ("D3", w3)):
        t1 = time.perf_counter()
        su = float(K.decision_sigma(X, w, b)["uhac_bartlett"])
        tu = time.perf_counter() - t1
        q = float(w @ Sig @ w)
        dvar = np.diag(Sig)
        arch = P[(P.pid == pid) & (P.D == D)].iloc[0]
        rec |= {f"{D}_uHAC_var": su ** 2, f"{D}_explicit_var": q, f"{D}_rel_diff": abs(q - su ** 2) / su ** 2, f"{D}_t_uHAC_s": tu,
                f"{D}_kappa_HAC": q / float(np.sum(w ** 2 * dvar)),
                f"{D}_kappa_boot_original": float((arch.sA_J_cbb / arch.sA_I_cbb) ** 2),
                f"{D}_uHAC_original_rel_diff": abs(su - arch.sA_uHAC) / arch.sA_uHAC}
    dvar = np.diag(Sig).copy()
    del Sig
    gc.collect()
    A = IF / np.sqrt(dvar)
    ev = np.sort(np.linalg.eigvals(Kmat @ (A @ A.T)).real / T ** 2)[::-1]
    ev = np.clip(ev, 0, None)
    Etot = float(len(r))
    cum = np.cumsum(ev) / ev.sum()
    rec |= {"trace_check": float(ev.sum()), "lambda1": float(ev[0]), "lambda1_over_E": float(ev[0] / Etot),
            "n_eig_90pct": int(np.searchsorted(cum, 0.9) + 1), "rank_numerical": int(np.sum(ev > 1e-10 * ev[0])),
            "lambda_min_full": 0.0}
    for D in ("D1", "D2", "D3"):
        rec[f"{D}_within_bounds"] = bool(0.0 <= rec[f"{D}_kappa_HAC"] <= ev[0] * (1 + 1e-9))
    res.append(rec)
    print({kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in rec.items()}, flush=True)
    del IF, A, Kmat
    gc.collect()
K.jdump({"participants": res}, OUT / "summary.json")
