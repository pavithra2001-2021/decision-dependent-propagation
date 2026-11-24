"""Direct calibration, power benchmark, discovery-cohort replication, mean-signal regression and region-specific simulation."""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
t0 = time.time()
C = K.nyu2_cohort()
y = (C.sex == "female").astype(int).values
ZA, _ = K.nyu2_point_all(C)
W1, c1, fold = K.d1_weights_cached(C, ZA, y)
del ZA
N, T, TR, BURN = 116, 170, 2.0, 300
iu = np.triu_indices(N, 1)
E_ = len(iu[0])
w_rand = np.random.default_rng(2026112301).standard_normal(E_)
w_glob = np.ones(E_)
w_train = W1[np.where(fold == 0)[0][0]]
Wm = np.column_stack([w / np.linalg.norm(w) for w in (w_rand, w_glob, w_train)])
DN = ["random", "global_mean", "trained"]
SRC = np.random.default_rng(2026112310).choice(len(C), 10, replace=False)
MULT = (6.0, 12.0, 24.0)
M_TRUTH, M_EST, PMAX = 2000, 100, 8
freqs = np.fft.rfftfreq(T, d=TR)
band = (freqs >= 0.01) & (freqs <= 0.08)


def fit_ar(x):
    best = None
    for p in range(1, PMAX + 1):
        Y = x[PMAX:]
        Xl = np.column_stack([x[PMAX - k:len(x) - k] for k in range(1, p + 1)])
        a, *_ = np.linalg.lstsq(Xl, Y, rcond=None)
        rss = np.sum((Y - Xl @ a) ** 2)
        bic = len(Y) * np.log(rss / len(Y)) + p * np.log(len(Y))
        if best is None or bic < best[0]:
            best = (bic, p)
    p = best[1]
    while p >= 1:
        Y = x[PMAX:]
        Xl = np.column_stack([x[PMAX - k:len(x) - k] for k in range(1, p + 1)])
        a, *_ = np.linalg.lstsq(Xl, Y, rcond=None)
        roots = np.roots(np.r_[1.0, -a])
        if np.max(np.abs(roots)) < 0.99:
            return a
        p -= 1
    return np.array([0.0])


def friston24(rp):
    lag = np.vstack([rp[:1] * 0, rp[:-1]])
    return np.column_stack([rp, lag, rp ** 2, lag ** 2])


def lag1(X):
    Xc = X - X.mean(0)
    return (Xc[1:] * Xc[:-1]).sum(0) / (Xc ** 2).sum(0)


def cbb_fixed(X, b, rng, R=K.R, chunk=10):
    idx = K.U.cbb_indices(X.shape[0], b, R, rng)
    L, s1, s2, nn = [], np.zeros(E_), np.zeros(E_), 0
    for s in range(0, R, chunk):
        Zr = K.ER._ztri(K.ER._std(X[idx[s:s + chunk]]), iu)
        L.append(Zr @ Wm)
        s1 += Zr.sum(0); s2 += (Zr ** 2).sum(0); nn += Zr.shape[0]
    return np.concatenate(L), (s2 - s1 ** 2 / nn) / (nn - 1)


summary, rows = {"sources": {}, "multipliers": list(MULT)}, []
for si, k in enumerate(SRC):
    pid = C.pid.iloc[k]
    X0 = K.nyu2_load(pid, 1)
    X0 = X0 - X0.mean(0)
    rp = np.loadtxt(glob.glob(str(K.DERIV / f"{pid}_run1" / "rp_*.txt"))[0])
    assert rp.shape[0] == T and X0.shape[0] == T
    A = [fit_ar(X0[:, i]) for i in range(N)]
    orders = np.array([len(a) for a in A])
    P_ = orders.max()
    Acoef = np.zeros((P_, N))
    for i, a in enumerate(A):
        Acoef[:len(a), i] = a
    resid = np.column_stack([X0[PMAX:, i] - np.column_stack([X0[PMAX - kk:T - kk, i] for kk in range(1, len(A[i]) + 1)]) @ A[i] for i in range(N)])
    S = np.cov(resid.T)
    S = 0.9 * S + 0.1 * np.diag(np.diag(S))
    Lc = np.linalg.cholesky(S)
    Nuis = np.column_stack([np.ones(T), np.arange(T), friston24(rp)])
    Qn, _ = np.linalg.qr(Nuis)

    def realise(rng):
        e = rng.standard_normal((T + BURN, N)) @ Lc.T
        x = np.zeros((T + BURN, N))
        for t in range(P_, T + BURN):
            x[t] = np.einsum("kn,kn->n", Acoef, x[t - P_:t][::-1]) + e[t]
        X = x[BURN:]
        X = X - Qn @ (Qn.T @ X)
        F = np.fft.rfft(X, axis=0)
        F[~band] = 0
        return np.fft.irfft(F, n=T, axis=0)

    s1, s2 = np.zeros(E_), np.zeros(E_)
    dv = np.zeros((M_TRUTH, 3))
    for i in range(M_TRUTH):
        X = realise(np.random.default_rng([2026112314, si, i]))
        if i == 0:
            a1 = lag1(X)
        z = K.point_z(X)
        s1 += z; s2 += z ** 2
        dv[i] = z @ Wm
    varz = (s2 - s1 ** 2 / M_TRUTH) / (M_TRUTH - 1)
    sd_true = dv.std(0, ddof=1)
    kappa_true = sd_true ** 2 / ((Wm ** 2).T @ varz)
    bb = dv.reshape(20, 100, 3).std(1, ddof=1)
    se_kappa = (bb ** 2 / ((Wm ** 2).T @ varz)).std(0, ddof=1) / np.sqrt(20)
    summary["sources"][pid] = {"ar_order_counts": {int(p): int((orders == p).sum()) for p in np.unique(orders)},
                               "lag1_autocorr_sim_p5_p50_p95": [float(v) for v in np.percentile(a1, [5, 50, 95])],
                               "kappa_true": {nm: float(kappa_true[j]) for j, nm in enumerate(DN)},
                               "kappa_true_se": {nm: float(se_kappa[j]) for j, nm in enumerate(DN)},
                               "sd_true": {nm: float(sd_true[j]) for j, nm in enumerate(DN)}}
    print(f"source {si} {pid}: orders {summary['sources'][pid]['ar_order_counts']} truth {time.time()-t0:.0f}s kappa {np.round(kappa_true, 2)} "
          f"lag1 {np.round(np.percentile(a1, [5, 50, 95]), 2)}", flush=True)
    for i in range(M_EST):
        X = realise(np.random.default_rng([2026112314, si, i]))
        rng = np.random.default_rng([2026112315, si, i])
        b0 = K.U.fc_block_length(X, source="roi", kind="cb", n_series=min(256, N), rng=rng)
        sx = K.xdf_var(X)
        for m in MULT:
            Lr, ev = cbb_fixed(X, m, np.random.default_rng([2026112316, si, i, int(m * 10)]))
            for j, nm in enumerate(DN):
                w = Wm[:, j]
                sJ = float(Lr[:, j].std(ddof=1))
                sIc = float(np.sqrt(np.sum(w ** 2 * ev)))
                sU = float(K.decision_sigma(X, w, m)["uhac_bartlett"])
                sX = float(np.sqrt(np.sum(w ** 2 * sx)))
                rows.append({"source": pid, "rep": i, "mult": m, "b0": b0, "decision": nm, "sd_true": sd_true[j], "kappa_true": kappa_true[j],
                             "sJ": sJ, "sU": sU, "sIcbb": sIc, "sIxdf": sX, "kappa_hat": (sJ / sIc) ** 2})
    print(f"source {si}: estimates {time.time()-t0:.0f}s", flush=True)
E = pd.DataFrame(rows)
E.to_csv(OUT / "sim_estimates_fixed_blocks.csv.gz", index=False)


def agg(g):
    r = {}
    for lab, col in (("J_cbb", "sJ"), ("uHAC", "sU"), ("I_cbb", "sIcbb"), ("I_xdf", "sIxdf")):
        q = g[col] / g.sd_true
        per = q.groupby(g.source).mean()
        r[lab] = {"mean_sd_ratio": float(q.mean()), "se_between_sources": float(per.std(ddof=1) / np.sqrt(len(per))),
                  "range_over_sources": [float(per.min()), float(per.max())]}
    kq = g.kappa_hat / g.kappa_true
    per = kq.groupby(g.source).mean()
    r["kappa"] = {"mean_ratio": float(kq.mean()), "se_between_sources": float(per.std(ddof=1) / np.sqrt(len(per))),
                  "range_over_sources": [float(per.min()), float(per.max())]}
    return r


summary["pooled"] = {f"{m}": {nm: agg(E[(E.mult == m) & (E.decision == nm)]) for nm in DN} for m in MULT}
kt = pd.DataFrame({pid: s["kappa_true"] for pid, s in summary["sources"].items()}).T
summary["kappa_true_across_sources"] = {nm: {"min": float(kt[nm].min()), "median": float(kt[nm].median()), "max": float(kt[nm].max())} for nm in DN}
summary["median_b0"] = float(E.b0.median())
lag = np.array([s["lag1_autocorr_sim_p5_p50_p95"] for s in summary["sources"].values()])
summary["lag1_p5_p95_median_over_sources"] = [float(np.median(lag[:, 0])), float(np.median(lag[:, 2]))]
allord = {}
for s in summary["sources"].values():
    for p, c in s["ar_order_counts"].items():
        allord[p] = allord.get(p, 0) + c
summary["ar_order_counts_all"] = allord
summary["runtime_s"] = time.time() - t0


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "\\textminus{}")


mv = {}
TAG = {"random": "Rw", "global_mean": "Gm", "trained": "Tr"}
MT = {6.0: "Six", 12.0: "Twelve", 24.0: "TwentyFour"}
for m in MULT:
    for nm in DN:
        r = summary["pooled"][f"{m}"][nm]
        g = TAG[nm] + MT[m]
        mv |= {f"Fb{g}K": fmt(r["kappa"]["mean_ratio"]), f"Fb{g}J": fmt(r["J_cbb"]["mean_sd_ratio"]), f"Fb{g}U": fmt(r["uHAC"]["mean_sd_ratio"]),
               f"Fb{g}Ic": fmt(r["I_cbb"]["mean_sd_ratio"]), f"Fb{g}Ix": fmt(r["I_xdf"]["mean_sd_ratio"])}
for m in MULT:
    rr = [summary["pooled"][f"{m}"][nm] for nm in DN]
    s = MT[m]
    mv |= {f"FbK{s}Lo": fmt(min(r["kappa"]["mean_ratio"] for r in rr)), f"FbK{s}Hi": fmt(max(r["kappa"]["mean_ratio"] for r in rr)),
           f"FbJ{s}Lo": fmt(min(r["J_cbb"]["mean_sd_ratio"] for r in rr)), f"FbJ{s}Hi": fmt(max(r["J_cbb"]["mean_sd_ratio"] for r in rr)),
           f"FbU{s}Lo": fmt(min(r["uHAC"]["mean_sd_ratio"] for r in rr)), f"FbU{s}Hi": fmt(max(r["uHAC"]["mean_sd_ratio"] for r in rr))}
mv |= {"FbBzero": fmt(summary["median_b0"], 1), "FbLagLo": fmt(summary["lag1_p5_p95_median_over_sources"][0]),
       "FbLagHi": fmt(summary["lag1_p5_p95_median_over_sources"][1]),
       "FbOrdMin": str(min(allord)), "FbOrdMax": str(max(allord))}
for nm in DN:
    s = summary["kappa_true_across_sources"][nm]
    nd = 2 if s["max"] < 10 else (1 if s["max"] < 100 else 0)
    mv |= {f"FbKt{TAG[nm]}Lo": fmt(s["min"], nd), f"FbKt{TAG[nm]}Hi": fmt(s["max"], nd)}
summary["macro_values"] = mv
K.jdump(summary, OUT / "summary_fixed_blocks.json")
for m in MULT:
    for nm in DN:
        r = summary["pooled"][f"{m}"][nm]
        print(f"x{m} {nm:12s} kappa {r['kappa']['mean_ratio']:.3f}±{r['kappa']['se_between_sources']:.3f} J {r['J_cbb']['mean_sd_ratio']:.3f} "
              f"U {r['uHAC']['mean_sd_ratio']:.3f} Icbb {r['I_cbb']['mean_sd_ratio']:.3f} Ixdf {r['I_xdf']['mean_sd_ratio']:.3f}")
print(summary["kappa_true_across_sources"], summary["median_b0"], allord, summary["runtime_s"])
