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
S, W, c, lam, y = K.tcp_weights_cached("AP")
repro = float(np.load(K.CACHE / "tcp_weights_AP.npz")["repro_rel"])
print(f"TCP AP cohort {len(S)}; reproduction max rel diff of l_A vs original = {repro:.2e} ({time.time()-t0:.0f}s)", flush=True)
assert repro < 1e-6, "TCP classifier reproduction failed"
E_ = W.shape[1]
w2, w3 = K.fixed_weights(E_, K.SEED_B)
fdPA1, fdPA2 = K.ER.mean_fd("PA", 1), K.ER.mean_fd("PA", 2)
H = 244


def dec(X, Wm, cv):
    return K.point_z(X) @ Wm + cv


def unc(X, Wm, rng):
    b, Lr, ev = K.stream_cbb(X, rng, Wm)
    sx = K.xdf_var(X)
    out = {}
    for j, nm in enumerate(("C", "G", "Rd")):
        w = Wm[:, j]
        out[f"sJ_{nm}"] = float(Lr[:, j].std(ddof=1))
        out[f"sIcbb_{nm}"] = float(np.sqrt(np.sum(w ** 2 * ev)))
        out[f"sIxdf_{nm}"] = float(np.sqrt(np.sum(w ** 2 * sx)))
        out[f"suHAC_{nm}"] = float(K.decision_sigma(X, w, b)["uhac_bartlett"])
    out["b"] = float(b)
    return out


rows = []
for k, sub in enumerate(S["sub"]):
    Wm = np.column_stack([W[k], w2, w3])
    cv = np.array([c[k], 0.0, 0.0])
    files = {(pe, r): K.ER.run_file(sub, pe, r) for pe in ("AP", "PA") for r in (1, 2)}
    rec = {"sub": sub, "y": int(y[k])}
    X1, X2 = K.tcp_load(sub, "AP", 1), K.tcp_load(sub, "AP", 2)
    rec["T_AP1"], rec["T_AP2"] = X1.shape[0], X2.shape[0]
    lf1, lf2 = dec(X1, Wm, cv), dec(X2, Wm, cv)
    if X1.shape[0] >= 2 * H and X2.shape[0] >= H:
        u = unc(X1[:H], Wm, np.random.default_rng([K.SEED_B, 4, k, 1]))
        l11, l12, l21 = dec(X1[:H], Wm, cv), dec(X1[H:2 * H], Wm, cv), dec(X2[:H], Wm, cv)
        for j, nm in enumerate(("C", "G", "Rd")):
            rec |= {f"AP_dwithin_{nm}": l12[j] - l11[j], f"AP_dacross_{nm}": l21[j] - l11[j], f"AP_dfull_{nm}": lf2[j] - lf1[j],
                    f"AP_lA_H1_{nm}": l11[j]}
        rec |= {f"AP_{kk}": v for kk, v in u.items()}
    pa_ok = files[("PA", 1)].exists() and files[("PA", 2)].exists() and fdPA1.get(sub, np.inf) <= K.ER.FD_MAX \
        and fdPA2.get(sub, np.inf) <= K.ER.FD_MAX
    rec["PA_eligible"] = bool(pa_ok)
    if pa_ok:
        P1, P2 = K.tcp_load(sub, "PA", 1), K.tcp_load(sub, "PA", 2)
        if all(np.isfinite(X).all() and (X.std(0) > 1e-8).all() for X in (P1, P2)) and P1.shape[0] >= 2 * H and P2.shape[0] >= H:
            u = unc(P1[:H], Wm, np.random.default_rng([K.SEED_B, 4, k, 2]))
            m11, m12, m21 = dec(P1[:H], Wm, cv), dec(P1[H:2 * H], Wm, cv), dec(P2[:H], Wm, cv)
            pf1, pf2 = dec(P1, Wm, cv), dec(P2, Wm, cv)
            for j, nm in enumerate(("C", "G", "Rd")):
                rec |= {f"PA_dwithin_{nm}": m12[j] - m11[j], f"PA_dacross_{nm}": m21[j] - m11[j], f"PA_dfull_{nm}": pf2[j] - pf1[j]}
            rec |= {f"PA_{kk}": v for kk, v in u.items()}
            rec["T_PA1"], rec["T_PA2"] = P1.shape[0], P2.shape[0]
        else:
            rec["PA_eligible"] = False
    rows.append(rec)
    if k % 20 == 0:
        print(f"{k}/{len(S)} ({time.time()-t0:.0f}s)", flush=True)
D = pd.DataFrame(rows)
D.to_csv(OUT / "participants_tcp.csv.gz", index=False, compression="gzip")


def boot_ratio(num, den, n=10000, seed=K.SEED_B):
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        i = rng.integers(0, len(num), len(num))
        b.append(np.mean(num[i]) / np.mean(den[i]))
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


summ = {"cohort_AP": len(S), "reproduction_lA_max_rel": repro, "half_length": H}
from scipy.stats import spearmanr
for pe in ("AP", "PA"):
    Dp = D[D[f"{pe}_dwithin_C"].notna()].reset_index(drop=True)
    summ[pe] = {"n": int(len(Dp))}
    for nm in ("C", "G", "Rd"):
        dw2, da2 = Dp[f"{pe}_dwithin_{nm}"].values ** 2, Dp[f"{pe}_dacross_{nm}"].values ** 2
        r = {"mean_dwithin2": float(dw2.mean()), "mean_dacross2": float(da2.mean()),
             "ratio_across_over_within": float(da2.mean() / dw2.mean()), "ratio_ci95": boot_ratio(da2, dw2)}
        for m, col in (("J_cbb", "sJ"), ("uHAC", "suHAC"), ("I_xdf", "sIxdf"), ("I_cbb", "sIcbb")):
            v = 2 * Dp[f"{pe}_{col}_{nm}"].values ** 2
            r[f"vcr_within_{m}"] = float(v.mean() / dw2.mean())
            r[f"vcr_within_{m}_ci95"] = boot_ratio(v, dw2)
            r[f"vcr_across_{m}"] = float(v.mean() / da2.mean())
            r[f"vcr_across_{m}_ci95"] = boot_ratio(v, da2)
            r[f"cov_within_{m}"] = float(np.mean(np.sqrt(dw2) <= 1.96 * np.sqrt(v)))
            r[f"cov_across_{m}"] = float(np.mean(np.sqrt(da2) <= 1.96 * np.sqrt(v)))
        summ[pe][nm] = r
both = D[D.AP_dfull_C.notna() & D.PA_dfull_C.notna()]
summ["full_length_consistency"] = {"n": int(len(both))}
for nm in ("C", "G", "Rd"):
    rho = spearmanr(np.abs(both[f"AP_dfull_{nm}"]), np.abs(both[f"PA_dfull_{nm}"]))
    rng = np.random.default_rng(K.SEED_B)
    a, b = np.abs(both[f"AP_dfull_{nm}"].values), np.abs(both[f"PA_dfull_{nm}"].values)
    bo = []
    for _ in range(10000):
        i = rng.integers(0, len(a), len(a))
        bo.append(spearmanr(a[i], b[i]).statistic)
    summ["full_length_consistency"][nm] = {"spearman_absdAP_absdPA": float(rho.statistic), "p": float(rho.pvalue),
                                           "ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])]}
summ["runtime_s"] = time.time() - t0
K.jdump(summ, OUT / "summary_tcp.json")
import json
print(json.dumps(summ, indent=1)[:6000])
