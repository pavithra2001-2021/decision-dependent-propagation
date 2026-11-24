"""Simulation at the dimensions of the confirmation data."""
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
TR, T, N = 2.0, 170, 116
iu = np.triu_indices(N, 1)
E_ = len(iu[0])
C = K.nyu2_cohort()
y = (C.sex == "female").astype(int).values
ZA, _ = K.nyu2_point_all(C)
W1, c1, fold = K.d1_weights_cached(C, ZA, y)
Rm = np.zeros((N, N))
for pid in C.pid:
    Z = K.ER._std(K.nyu2_load(pid, 1))
    Rm += Z.T @ Z / Z.shape[0]
Rm /= len(C.pid)
Sig0 = Rm + 1e-3 * np.eye(N)
L = np.linalg.cholesky(Sig0)
w_rand = np.random.default_rng(2026112301).standard_normal(E_)
w_glob = np.ones(E_)
w_train = W1[np.where(fold == 0)[0][0]]
Wm = np.column_stack([w / np.linalg.norm(w) for w in (w_rand, w_glob, w_train)])
DN = ["random", "global_mean", "trained"]
freqs = np.fft.rfftfreq(T, d=TR)
mask = (freqs >= 0.01) & (freqs <= 0.08)


def bandpass(X):
    t = np.arange(T)
    A = np.column_stack([np.ones(T), t])
    X = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
    F = np.fft.rfft(X, axis=0)
    F[~mask] = 0
    return np.fft.irfft(F, n=T, axis=0)


def realise(rng, phi, bp):
    burn = 200
    e = rng.standard_normal((T + burn, N)) @ L.T
    x = np.zeros((T + burn, N))
    for t in range(1, T + burn):
        x[t] = phi * x[t - 1] + e[t]
    X = x[burn:]
    return bandpass(X) if bp else X


CONDS = {"C1_ar05_nofilter": (0.5, False), "C2_white_bandpass": (0.0, True), "C3_ar05_bandpass": (0.5, True)}
M_TRUTH, M_EST = 2000, 100
summary, est_rows = {"conditions": {}, "mask_frequencies": int(mask.sum())}, []
for ci, (cname, (phi, bp)) in enumerate(CONDS.items()):
    s1, s2 = np.zeros(E_), np.zeros(E_)
    dv = np.zeros((M_TRUTH, 3))
    batch_kappa, batch_sd = [], []
    for i in range(M_TRUTH):
        X = realise(np.random.default_rng([2026112303, ci, i]), phi, bp)
        z = K.point_z(X)
        s1 += z; s2 += z ** 2
        dv[i] = z @ Wm
        if (i + 1) % 100 == 0:
            pass
    varz = (s2 - s1 ** 2 / M_TRUTH) / (M_TRUTH - 1)
    sd_true = dv.std(0, ddof=1)
    kappa_true = sd_true ** 2 / ((Wm ** 2).T @ varz)
    b = dv.reshape(20, 100, 3).std(1, ddof=1)
    se_sd = b.std(0, ddof=1) / np.sqrt(20)
    bk = b ** 2 / ((Wm ** 2).T @ varz)
    se_kappa = bk.std(0, ddof=1) / np.sqrt(20)
    print(f"{cname}: truth done {time.time()-t0:.0f}s kappa_true {np.round(kappa_true,2)}", flush=True)
    for i in range(M_EST):
        X = realise(np.random.default_rng([2026112303, ci, i]), phi, bp)
        bL, Lr, ev = K.stream_cbb(X, np.random.default_rng([2026112304, ci, i]), Wm)
        sx = K.xdf_var(X)
        for j, nm in enumerate(DN):
            w = Wm[:, j]
            sJ = float(Lr[:, j].std(ddof=1))
            sIc = float(np.sqrt(np.sum(w ** 2 * ev)))
            sU = float(K.decision_sigma(X, w, bL)["uhac_bartlett"])
            sX = float(np.sqrt(np.sum(w ** 2 * sx)))
            est_rows.append({"condition": cname, "rep": i, "decision": nm, "b": bL, "sd_true": sd_true[j], "kappa_true": kappa_true[j],
                             "sJ": sJ, "sU": sU, "sIcbb": sIc, "sIxdf": sX, "kappa_hat": (sJ / sIc) ** 2})
    print(f"{cname}: estimates done {time.time()-t0:.0f}s", flush=True)
    summary["conditions"][cname] = {"phi": phi, "bandpass": bp, "decisions": {
        nm: {"sd_true": float(sd_true[j]), "sd_true_se": float(se_sd[j]), "kappa_true": float(kappa_true[j]), "kappa_true_se": float(se_kappa[j])}
        for j, nm in enumerate(DN)}}
E = pd.DataFrame(est_rows)
E.to_csv(OUT / "sim_estimates.csv", index=False)
for cname in CONDS:
    for nm in DN:
        g = E[(E.condition == cname) & (E.decision == nm)]
        r = {}
        for lab, col in (("J_cbb", "sJ"), ("uHAC", "sU"), ("I_cbb", "sIcbb"), ("I_xdf", "sIxdf")):
            q = g[col] / g.sd_true
            r[lab] = {"mean_sd_ratio": float(q.mean()), "se": float(q.std(ddof=1) / np.sqrt(len(q))), "median_sd_ratio": float(q.median()),
                      "mean_var_ratio": float((q ** 2).mean()), "se_var": float((q ** 2).std(ddof=1) / np.sqrt(len(q)))}
        kq = g.kappa_hat / g.kappa_true
        r["kappa"] = {"mean_ratio": float(kq.mean()), "se": float(kq.std(ddof=1) / np.sqrt(len(kq))), "median_ratio": float(kq.median()),
                      "mean_kappa_hat": float(g.kappa_hat.mean())}
        summary["conditions"][cname]["decisions"][nm]["estimates"] = r

P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
w2, w3 = K.fixed_weights(E_, K.SEED_C)
rel = {d: {"kappa": [], "sJ": []} for d in ("D1", "D2", "D3")}
rng0 = np.random.default_rng(2026112305)
B = 2000
for k, pid in enumerate(C.pid):
    mA = K.EC.run_measurement(K.nyu2_load(pid, 1), np.random.default_rng(K.SEED_C + 10 * k + 1), K.R)
    dev = mA["dev"]["cbb"]
    Wd = np.column_stack([W1[k], w2, w3])
    Lr = dev @ Wd
    q = (dev ** 2) @ (Wd ** 2)
    kap, sj = [], []
    for s in range(0, B, 500):
        idx = rng0.integers(0, K.R, (500, K.R))
        cnt = np.zeros((500, K.R))
        np.add.at(cnt, (np.repeat(np.arange(500), K.R), idx.ravel()), 1)
        mean_dev = cnt @ dev / K.R
        sum_q = cnt @ q
        var_w = (sum_q - K.R * (mean_dev ** 2) @ (Wd ** 2)) / (K.R - 1)
        Lb = np.take_along_axis(np.repeat(Lr[None], 500, 0), idx[:, :, None].repeat(3, 2), 1)
        sJb = Lb.std(1, ddof=1)
        kap.append(sJb ** 2 / var_w); sj.append(sJb)
        del mean_dev
    kap, sj = np.concatenate(kap), np.concatenate(sj)
    k0 = (Lr.std(0, ddof=1) ** 2) / ((Wd ** 2).T @ dev.var(0, ddof=1))
    for j, d in enumerate(("D1", "D2", "D3")):
        rel[d]["kappa"].append(kap[:, j].std(ddof=1) / k0[j])
        rel[d]["sJ"].append(sj[:, j].std(ddof=1) / Lr[:, j].std(ddof=1))
    del mA, dev
summary["real_data_mc_se"] = {d: {"median_rel_se_kappa": float(np.median(v["kappa"])), "median_rel_se_sJ": float(np.median(v["sJ"])),
                                  "iqr_rel_se_kappa": [float(np.percentile(v["kappa"], 25)), float(np.percentile(v["kappa"], 75))]}
                              for d, v in rel.items()}
summary["runtime_s"] = time.time() - t0
K.jdump(summary, OUT / "summary.json")
for cname in CONDS:
    for nm in DN:
        r = summary["conditions"][cname]["decisions"][nm]
        e = r["estimates"]
        print(f"{cname:18s} {nm:12s} kappa_true {r['kappa_true']:8.2f}±{r['kappa_true_se']:.2f} | J {e['J_cbb']['mean_sd_ratio']:.3f}±{e['J_cbb']['se']:.3f} "
              f"U {e['uHAC']['mean_sd_ratio']:.3f}±{e['uHAC']['se']:.3f} Icbb {e['I_cbb']['mean_sd_ratio']:.3f} Ixdf {e['I_xdf']['mean_sd_ratio']:.3f} "
              f"| kappa ratio {e['kappa']['mean_ratio']:.3f}±{e['kappa']['se']:.3f}")
print(summary["real_data_mc_se"])
