"""Tuning sensitivity and runtime of uHAC and the block bootstrap."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K
from p6.exp_mechanism import decision_sigma

OUT = Path(__file__).resolve().parent / "results"


def time_run(X, w, seed):
    T, N = X.shape
    iu = np.triu_indices(N, 1)
    t = {}
    t0 = time.perf_counter()
    b = K.U.fc_block_length(X, source="roi", kind="cb", n_series=min(256, N), rng=np.random.default_rng(seed))
    t["block_length_ms"] = 1000 * (time.perf_counter() - t0)
    t1 = time.perf_counter()
    decision_sigma(X, w, b)
    t["uhac_decision_ms"] = 1000 * (time.perf_counter() - t1)
    t["uhac_end_to_end_ms"] = t["block_length_ms"] + t["uhac_decision_ms"]
    t2 = time.perf_counter()
    idx = K.U.cbb_indices(T, b, K.R, np.random.default_rng(seed + 1))
    L = np.concatenate([blk @ w for blk in K.ER.replicate_z(X, idx, iu)])
    L.std(ddof=1)
    t["jcbb_end_to_end_ms"] = t["block_length_ms"] + 1000 * (time.perf_counter() - t2)
    t3 = time.perf_counter()
    sx = K.xdf_var(X)
    float(np.sqrt(np.sum(w ** 2 * sx)))
    t["xdf_end_to_end_ms"] = 1000 * (time.perf_counter() - t3)
    return t


res = {}
C = K.nyu2_cohort()
y = (C.sex == "female").astype(int).values
ZA, _ = K.nyu2_point_all(C)
W1, _, _ = K.d1_weights_cached(C, ZA, y)
rows = [time_run(K.nyu2_load(pid, 1), W1[k], 2026112306 + k) for k, pid in enumerate(C.pid[:20])]
res["nyu2"] = {k: float(np.median([r[k] for r in rows])) for k in rows[0]} | {"n_runs": len(rows), "T": 170, "N": 116}
S, W, c, lam, yy = K.tcp_weights_cached("AP") if (K.CACHE / "tcp_weights_AP.npz").exists() else (None,) * 5
if S is None:
    S, W, c, lam, yy = K.tcp_weights_cached("AP")
rows = [time_run(K.tcp_load(sub, "AP", 1), W[k], 2026112307 + k) for k, sub in enumerate(S["sub"][:5])]
res["tcp"] = {k: float(np.median([r[k] for r in rows])) for k in rows[0]} | {"n_runs": len(rows), "T": 488, "N": 434}
K.jdump(res, OUT / "summary_timing.json")
print(res)
