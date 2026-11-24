"""Within-run and across-run change of half-run differences."""
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
H = 85
rows = []
for k, pid in enumerate(C.pid):
    XA, XB = K.nyu2_load(pid, 1), K.nyu2_load(pid, 2)
    assert XA.shape[0] >= 2 * H and XB.shape[0] >= H
    mH = K.EC.run_measurement(XA[:H], np.random.default_rng([K.SEED_C, 4, k, 1]), K.R)
    z12, z21 = K.point_z(XA[H:2 * H]), K.point_z(XB[:H])
    for D, w, c in (("D1", W1[k], c1[k]), ("D2", w2, 0.0), ("D3", w3, 0.0)):
        s = K.EC.decision_sds(mH, w)
        l11 = float(mH["z"] @ w + c)
        rows.append({"pid": pid, "D": D, "l11": l11, "dwithin": float(z12 @ w + c) - l11, "dacross": float(z21 @ w + c) - l11,
                     "b": float(mH["b"])} | {f"s_{m}": s[m] for m in ("J_cbb", "uHAC", "I_xdf", "I_cbb", "J_iid")})
    del mH
    if k % 20 == 0:
        print(f"{k}/{len(C)} ({time.time()-t0:.0f}s)", flush=True)
D = pd.DataFrame(rows)
D.to_csv(OUT / "participants_nyu2.csv.gz", index=False, compression="gzip")


def boot_ratio(num, den, n=10000, seed=K.SEED_C):
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        i = rng.integers(0, len(num), len(num))
        b.append(np.mean(num[i]) / np.mean(den[i]))
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


summ = {"n": len(C), "half_length": H}
for Dn in ("D1", "D2", "D3"):
    g = D[D.D == Dn]
    dw2, da2 = g.dwithin.values ** 2, g.dacross.values ** 2
    r = {"mean_dwithin2": float(dw2.mean()), "mean_dacross2": float(da2.mean()),
         "ratio_across_over_within": float(da2.mean() / dw2.mean()), "ratio_ci95": boot_ratio(da2, dw2)}
    for m in ("J_cbb", "uHAC", "I_xdf", "I_cbb", "J_iid"):
        v = 2 * g[f"s_{m}"].values ** 2
        r[f"vcr_within_{m}"] = float(v.mean() / dw2.mean()); r[f"vcr_within_{m}_ci95"] = boot_ratio(v, dw2)
        r[f"vcr_across_{m}"] = float(v.mean() / da2.mean()); r[f"vcr_across_{m}_ci95"] = boot_ratio(v, da2)
        r[f"cov_within_{m}"] = float(np.mean(np.sqrt(dw2) <= 1.96 * np.sqrt(v)))
        r[f"cov_across_{m}"] = float(np.mean(np.sqrt(da2) <= 1.96 * np.sqrt(v)))
    summ[Dn] = r
summ["runtime_s"] = time.time() - t0
K.jdump(summ, OUT / "summary_nyu2.json")
for Dn in ("D1", "D2", "D3"):
    r = summ[Dn]
    print(Dn, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in r.items()
               if any(s in k for s in ("ratio", "J_cbb", "I_xdf"))})
