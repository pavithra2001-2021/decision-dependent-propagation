"""Analyses with both runs truncated to shorter lengths."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
t0 = time.time()
C = K.nyu2_cohort()
y = (C.sex == "female").astype(int).values
ZA, ZB = K.nyu2_point_all(C)
W1, c1, _ = K.d1_weights_cached(C, ZA, y)
w2, w3 = K.fixed_weights(ZA.shape[1], K.SEED_C)
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
rows = []
for k, pid in enumerate(C.pid):
    XA, XB = K.nyu2_load(pid, 1), K.nyu2_load(pid, 2)
    for T in (60, 100, 170):
        rng = np.random.default_rng(K.SEED_C + 10 * k + 1) if T == 170 else np.random.default_rng([K.SEED_C, 6, T, k, 1])
        mA = K.EC.run_measurement(XA[:T], rng, K.R)
        zB = K.point_z(XB[:T])
        for D, w, c in (("D1", W1[k], c1[k]), ("D2", w2, 0.0), ("D3", w3, 0.0)):
            s = K.EC.decision_sds(mA, w)
            rows.append({"pid": pid, "T": T, "D": D, "lA": float(mA["z"] @ w + c), "lB": float(zB @ w + c), "b": float(mA["b"])}
                        | {f"s_{m}": s[m] for m in ("J_cbb", "uHAC", "I_xdf", "I_cbb", "J_iid")})
        del mA
    if k % 20 == 0:
        print(f"{k}/{len(C)} ({time.time()-t0:.0f}s)", flush=True)
D = pd.DataFrame(rows)
chk = D[D["T"] == 170].merge(P, on=["pid", "D"], suffixes=("", "_arch"))
rep = {c: float(np.max(np.abs(chk[c] - chk[a]) / np.maximum(np.abs(chk[a]), 1e-12)))
       for c, a in (("lA", "lA_arch"), ("lB", "lB_arch"), ("s_J_cbb", "sA_J_cbb"), ("s_I_xdf", "sA_I_xdf"), ("s_uHAC", "sA_uHAC"))}
print("reproduction at T=170:", rep)
assert max(rep.values()) < 1e-6
D.to_csv(OUT / "participants_nyu2.csv.gz", index=False, compression="gzip")
K.jdump({"reproduction_T170_max_rel": rep, "runtime_s": time.time() - t0}, OUT / "run_nyu2_meta.json")
print("saved", time.time() - t0)
