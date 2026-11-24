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
S, W, c, lam, y = K.tcp_weights_cached("AP")
assert float(np.load(K.CACHE / "tcp_weights_AP.npz")["repro_rel"]) < 1e-6
E_ = W.shape[1]
w2, w3 = K.fixed_weights(E_, K.SEED_B)
TS = (150, 300, 488)
rows = []
for k, sub in enumerate(S["sub"]):
    Wm = np.column_stack([W[k], w2, w3])
    cv = np.array([c[k], 0.0, 0.0])
    X1, X2 = K.tcp_load(sub, "AP", 1), K.tcp_load(sub, "AP", 2)
    for T in TS:
        A, B = X1[:T], X2[:T]
        lA, lB = K.point_z(A) @ Wm + cv, K.point_z(B) @ Wm + cv
        b, Lr, ev = K.stream_cbb(A, np.random.default_rng([K.SEED_B, 6, T, k]), Wm)
        sx = K.xdf_var(A)
        rec = {"sub": sub, "T": T, "T_A": A.shape[0], "T_B": B.shape[0], "b": float(b)}
        for j, nm in enumerate(("C", "G", "Rd")):
            w = Wm[:, j]
            rec |= {f"lA_{nm}": lA[j], f"lB_{nm}": lB[j], f"sJ_{nm}": float(Lr[:, j].std(ddof=1)),
                    f"sIcbb_{nm}": float(np.sqrt(np.sum(w ** 2 * ev))), f"sIxdf_{nm}": float(np.sqrt(np.sum(w ** 2 * sx))),
                    f"suHAC_{nm}": float(K.decision_sigma(A, w, b)["uhac_bartlett"])}
        rows.append(rec)
    if k % 20 == 0:
        print(f"{k}/{len(S)} ({time.time()-t0:.0f}s)", flush=True)
D = pd.DataFrame(rows)
D.to_csv(OUT / "participants_tcp.csv.gz", index=False, compression="gzip")
print("saved", time.time() - t0)
