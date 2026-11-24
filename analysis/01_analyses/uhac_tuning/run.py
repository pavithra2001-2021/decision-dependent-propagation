"""Tuning sensitivity and runtime of uHAC and the block bootstrap."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K
from p6.exp_mechanism import lrv, projected_influence_fast

HERE = Path(__file__).resolve().parent
NA = HERE.parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
t0 = time.time()
S = {}

P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
cnt = {"original_confirmation_runA": int((P.sA_uHAC <= 0).sum()), "original_confirmation_runB": int((P.sB_uHAC <= 0).sum()),
       "original_cells": int(2 * len(P))}
tabs = {"model_families": NA / "model_families/results/participants.csv.gz",
        "scan_length_nyu2": NA / "scan_length/results/participants_nyu2.csv.gz",
        "scan_length_tcp": NA / "scan_length/results/participants_tcp.csv.gz",
        "split_half_nyu2": NA / "split_halves/results/participants_nyu2.csv.gz",
        "split_half_tcp": NA / "split_halves/results/participants_tcp.csv.gz",
        "cpm": NA / "cpm_transfer/results/participants.csv.gz",
        "ridge_extended": NA / "supplementary_checks/results/participants_ridge_extended.csv.gz"}
for k, f in tabs.items():
    D = pd.read_csv(f)
    cols = [c for c in D.columns if "uHAC" in c and ("_s_" in c or c.startswith("s_") or c.startswith("suHAC") or "suHAC" in c)]
    vals = D[cols].values.ravel()
    vals = vals[np.isfinite(vals)]
    cnt[k] = {"cells": int(len(vals)), "zero": int((vals <= 0).sum())}
S["vhat_negative_counts"] = cnt
print(cnt, flush=True)

C = K.nyu2_cohort()
y = (C.sex == "female").astype(int).values
ZA, ZB = K.nyu2_point_all(C)
W1, c1, _ = K.d1_weights_cached(C, ZA, y)
w2, w3 = K.fixed_weights(ZA.shape[1], K.SEED_C)
iu = np.triu_indices(116, 1)
MULT = {"b/2": 0.5, "b": 1.0, "2b": 2.0, "4b": 4.0}
rows = []
for k, pid in enumerate(C.pid):
    XA = K.nyu2_load(pid, 1)
    mA = K.EC.run_measurement(XA, np.random.default_rng(K.SEED_C + 10 * k + 1), K.R)
    b, T = mA["b"], XA.shape[0]
    Z = K.ER._std(XA)
    r = np.clip((Z.T @ Z / T)[iu], -0.999999, 0.999999)
    hac = {m: K.ER.hac_bartlett_z(XA, r, iu, mult * b) for m, mult in MULT.items()}
    devs = {"b": mA["dev"]["cbb"]}
    for mi, (m, mult) in enumerate(MULT.items()):
        if m == "b":
            continue
        idx = K.U.cbb_indices(T, mult * b, K.R, np.random.default_rng([K.SEED_C, 7, mi, k]))
        rep = np.concatenate(list(K.ER.replicate_z(XA, idx, iu)))
        devs[m] = rep - rep.mean(0)
    for D, w, c in (("D1", W1[k], c1[k]), ("D2", w2, 0.0), ("D3", w3, 0.0)):
        u = projected_influence_fast(XA, w)
        rec = {"pid": pid, "D": D, "lA": float(ZA[k] @ w + c), "lB": float(ZB[k] @ w + c), "b": b,
               "sIxdf": float(np.sqrt(np.sum(w ** 2 * mA["s2"]["xdf"])))}
        for m, mult in MULT.items():
            V = lrv(u, "bartlett", mult * b)
            rec[f"V_{m}"] = V
            rec[f"sU_{m}"] = float(np.sqrt(max(V, 0) / T))
            rec[f"sIhac_{m}"] = float(np.sqrt(np.sum(w ** 2 * hac[m])))
            dv = devs[m] @ w
            rec[f"sJ_{m}"] = float(dv.std(ddof=1))
            rec[f"sIcbb_{m}"] = float(np.sqrt(np.sum(w ** 2 * devs[m].var(0, ddof=1))))
        uc = u - u.mean()
        rho = float(uc[:-1] @ uc[1:] / (uc @ uc))
        alpha = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
        bwA = 1.1447 * (alpha * T) ** (1 / 3)
        VA = lrv(u, "bartlett", max(bwA, 1e-6))
        rec |= {"andrews_bw": bwA, "andrews_rho": rho, "V_andrews": VA, "sU_andrews": float(np.sqrt(max(VA, 0) / T)),
                "sIhac_andrews": float(np.sqrt(np.sum(w ** 2 * K.ER.hac_bartlett_z(XA, r, iu, max(bwA, 1e-6))))),
                "sU_default_check": float(K.decision_sigma(XA, w, b)["uhac_bartlett"])}
        rows.append(rec)
    del mA, devs, hac
    if k % 20 == 0:
        print(f"{k}/{len(C)} {time.time()-t0:.0f}s", flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT / "participants.csv.gz", index=False, compression="gzip")
chk = R[R.D == "D1"].merge(P[P.D == "D1"][["pid", "sA_uHAC", "sA_J_cbb"]], on="pid")
S["check_default_uhac_rel"] = float(np.max(np.abs(chk["sU_2b"] - chk.sA_uHAC) / chk.sA_uHAC))
S["check_Jcbb_b_rel"] = float(np.max(np.abs(chk["sJ_b"] - chk.sA_J_cbb) / chk.sA_J_cbb))
out = {}
for D in ("D1", "D2", "D3"):
    g = R[R.D == D]
    d = (g.lB - g.lA).values
    vI = 2 * g.sIxdf.values ** 2
    o = {}
    for m in list(MULT) + ["andrews"]:
        vU = 2 * g[f"sU_{m}"].values ** 2
        okU = vU > 0
        e = {"bandwidth_label": m, "n_V_negative": int((g[f"V_{m}"] < 0).sum()),
             "median_uhac_over_Jcbb_default": float(np.median(g[f"sU_{m}"] / g["sJ_b"])),
             "kappa_hac_median": float(np.median((g[f"sU_{m}"] / g[f"sIhac_{m}"]) ** 2)),
             "vcr_uhac": float(vU.mean() / np.mean(d ** 2)), "cov_uhac": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(vU)))}
        if okU.all():
            e["nlpd_uhac_minus_Ixdf"] = K.paired(K.nlpd(np.abs(d), vU) - K.nlpd(np.abs(d), vI))
        if m in MULT:
            vJ = 2 * g[f"sJ_{m}"].values ** 2
            e |= {"kappa_cbb_median": float(np.median((g[f"sJ_{m}"] / g[f"sIcbb_{m}"]) ** 2)),
                  "nlpd_Jcbb_minus_Ixdf": K.paired(K.nlpd(np.abs(d), vJ) - K.nlpd(np.abs(d), vI)),
                  "vcr_Jcbb": float(vJ.mean() / np.mean(d ** 2)), "cov_Jcbb": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(vJ)))}
        o[m] = e
    o["andrews_bw_median_iqr"] = [float(g.andrews_bw.median()), float(g.andrews_bw.quantile(.25)), float(g.andrews_bw.quantile(.75))]
    o["block_length_b_median"] = float(g.b.median())
    out[D] = o
S["sensitivity"] = out
S["runtime_s"] = time.time() - t0
K.jdump(S, OUT / "summary.json")
print("checks", S["check_default_uhac_rel"], S["check_Jcbb_b_rel"])
for D in out:
    print(D, "b median", out[D]["block_length_b_median"], "andrews bw", [round(v, 2) for v in out[D]["andrews_bw_median_iqr"]])
    for m in list(MULT) + ["andrews"]:
        e = out[D][m]
        s = f"  {m:8s} Vneg {e['n_V_negative']:3d} U/J {e['median_uhac_over_Jcbb_default']:.3f} kHAC {e['kappa_hac_median']:.2f} vcrU {e['vcr_uhac']:.3f}"
        if "nlpd_uhac_minus_Ixdf" in e:
            s += f" NLPD U-I {e['nlpd_uhac_minus_Ixdf']['median']:.2f}"
        if "kappa_cbb_median" in e:
            s += f" | kCBB {e['kappa_cbb_median']:.2f} NLPD J-I {e['nlpd_Jcbb_minus_Ixdf']['median']:.2f} {[round(v,2) for v in e['nlpd_Jcbb_minus_Ixdf']['ci95']]} vcrJ {e['vcr_Jcbb']:.3f}"
        print(s)
