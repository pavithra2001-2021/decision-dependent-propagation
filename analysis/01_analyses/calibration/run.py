"""Calibration repairs and transfer of a calibration scale to the fixed decisions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
METH = ["J_cbb", "uHAC", "I_xdf", "I_cbb", "J_iid"]

P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
G = {D: P[P.D == D].reset_index(drop=True) for D in ("D1", "D2", "D3")}
y = G["D1"].y.values
assert all((G[D].pid.values == G["D1"].pid.values).all() for D in G)
folds = K.nyu2_outer_folds(y)
n = len(y)


def fit_add(d2, v0):
    f = lambda t: np.mean(0.5 * np.log(2 * np.pi * (v0 + t)) + d2 / (2 * (v0 + t)))
    r = minimize_scalar(f, bounds=(0.0, 100 * np.mean(d2)), method="bounded", options={"xatol": 1e-12 * np.mean(d2)})
    return float(r.x) if f(r.x) < f(0.0) else 0.0


def conf_q(scores, alpha=0.05):
    s = np.sort(scores)
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return float(s[min(k, len(s)) - 1])


V = {}
HW = {}
fitted = []
for D in ("D1", "D2", "D3"):
    g = G[D]
    d2 = ((g.lB - g.lA) ** 2).values
    V[(D, "CONST", "-")] = np.zeros(n)
    HW[(D, "CONFraw", "-")] = np.zeros(n)
    for m in METH:
        v0 = 2 * g[f"sA_{m}"].values ** 2
        V[(D, "RAW", m)] = v0
        for kind in ("SCALE", "ADD"):
            V[(D, kind, m)] = np.zeros(n)
        HW[(D, "CONF", m)] = np.zeros(n)
    for f, (tr, te) in enumerate(folds):
        V[(D, "CONST", "-")][te] = d2[tr].mean()
        HW[(D, "CONFraw", "-")][te] = conf_q(np.sqrt(d2[tr]))
        for m in METH:
            v0 = V[(D, "RAW", m)]
            c = float(np.mean(d2[tr] / v0[tr]))
            t2 = fit_add(d2[tr], v0[tr])
            V[(D, "SCALE", m)][te] = c * v0[te]
            V[(D, "ADD", m)][te] = v0[te] + t2
            q = conf_q(np.sqrt(d2[tr] / v0[tr]))
            HW[(D, "CONF", m)][te] = q * np.sqrt(v0[te])
            fitted.append({"D": D, "fold": f, "method": m, "scale_c": c, "add_tau2_ml": t2,
                           "add_tau2_moment": float(max(0.0, np.mean(d2[tr] - v0[tr]))), "conf_q_std": q,
                           "mean_d2_train": float(d2[tr].mean())})
for D in ("D2", "D3"):
    for m in METH:
        for kind in ("SCALE", "ADD"):
            V[(D, f"T{kind}", m)] = np.zeros(n)
        v0 = V[(D, "RAW", m)]
        g1 = G["D1"]
        d2_1 = ((g1.lB - g1.lA) ** 2).values
        v0_1 = V[("D1", "RAW", m)]
        for tr, te in folds:
            c = float(np.mean(d2_1[tr] / v0_1[tr]))
            t2 = fit_add(d2_1[tr], v0_1[tr])
            V[(D, "TSCALE", m)][te] = c * v0[te]
            V[(D, "TADD", m)][te] = v0[te] + t2
pd.DataFrame(fitted).to_csv(OUT / "fitted_constants_by_fold.csv", index=False)


def flip_metrics(lA, f, v):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    p = np.clip(norm.cdf(-np.abs(lA) / np.sqrt(v)), 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p))[:, None]
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(lo, f)
    return {"brier": float(np.mean((p - f) ** 2)), "logloss": float(-np.mean(f * np.log(p) + (1 - f) * np.log(1 - p))),
            "mean_p": float(p.mean()), "cal_slope": float(cal.coef_[0, 0]), "auc": float(roc_auc_score(f, p))}


rows = []
for (D, kind, m), v in V.items():
    g = G[D]
    d = (g.lB - g.lA).values
    nl = K.nlpd(np.abs(d), v)
    r = {"D": D, "kind": kind, "method": m, "mean_nlpd": float(nl.mean()), "median_nlpd": float(np.median(nl)),
         "coverage95": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(v))), "vcr": float(v.mean() / np.mean(d ** 2)),
         "spearman_sd_absd": float(spearmanr(np.sqrt(v), np.abs(d)).statistic) if np.std(v) > 0 else np.nan}
    if D == "D1":
        f = (np.sign(g.lA) != np.sign(g.lB)).astype(int).values
        r |= {f"flip_{k}": x for k, x in flip_metrics(g.lA.values, f, v).items()}
    rows.append(r)
for (D, kind, m), h in HW.items():
    d = (G[D].lB - G[D].lA).values
    rows.append({"D": D, "kind": kind, "method": m, "coverage95": float(np.mean(np.abs(d) <= h)),
                 "median_halfwidth": float(np.median(h)), "mean_halfwidth": float(np.mean(h))})
M = pd.DataFrame(rows)
for (D, kind, m) in V:
    M.loc[(M.D == D) & (M.kind == kind) & (M.method == m), "median_halfwidth"] = float(np.median(1.96 * np.sqrt(V[(D, kind, m)])))
M.to_csv(OUT / "metrics.csv", index=False)


def nl(D, kind, m):
    g = G[D]
    return K.nlpd(np.abs((g.lB - g.lA).values), V[(D, kind, m)])


fam = {"C0_D1_RAW_Jcbb_minus_CONST": nl("D1", "RAW", "J_cbb") - nl("D1", "CONST", "-"),
       "C1_D1_SCALE_Jcbb_minus_CONST": nl("D1", "SCALE", "J_cbb") - nl("D1", "CONST", "-"),
       "C2_D1_ADD_Jcbb_minus_CONST": nl("D1", "ADD", "J_cbb") - nl("D1", "CONST", "-"),
       "C3_D1_SCALE_Jcbb_minus_SCALE_Ixdf": nl("D1", "SCALE", "J_cbb") - nl("D1", "SCALE", "I_xdf"),
       "C4_D1_SCALE_uHAC_minus_SCALE_Jcbb": nl("D1", "SCALE", "uHAC") - nl("D1", "SCALE", "J_cbb"),
       "C5_D2_TSCALE_Jcbb_minus_TSCALE_Ixdf": nl("D2", "TSCALE", "J_cbb") - nl("D2", "TSCALE", "I_xdf"),
       "C6_D3_TSCALE_Jcbb_minus_TSCALE_Ixdf": nl("D3", "TSCALE", "J_cbb") - nl("D3", "TSCALE", "I_xdf")}
res = {k: K.paired(v) | {"mean": float(np.mean(v)), "mean_ci95": K.mean_ci(v)["ci95"]} for k, v in fam.items()}
adj = K.holm([r["wilcoxon_p"] for r in res.values()])
for (k, r), a in zip(res.items(), adj):
    r["p_holm"] = float(a)
desc = {"D2_SCALE_Jcbb_minus_CONST": K.paired(nl("D2", "SCALE", "J_cbb") - nl("D2", "CONST", "-")),
        "D3_SCALE_Jcbb_minus_CONST": K.paired(nl("D3", "SCALE", "J_cbb") - nl("D3", "CONST", "-")),
        "D1_ADD_Jcbb_minus_ADD_Ixdf": K.paired(nl("D1", "ADD", "J_cbb") - nl("D1", "ADD", "I_xdf")),
        "D2_TADD_Jcbb_minus_TADD_Ixdf": K.paired(nl("D2", "TADD", "J_cbb") - nl("D2", "TADD", "I_xdf")),
        "D3_TADD_Jcbb_minus_TADD_Ixdf": K.paired(nl("D3", "TADD", "J_cbb") - nl("D3", "TADD", "I_xdf"))}
FC = pd.DataFrame(fitted)
summ = {"n": n, "family_holm": res, "descriptive": desc,
        "fitted_constants_median_over_folds": FC.groupby(["D", "method"])[["scale_c", "add_tau2_ml", "add_tau2_moment", "conf_q_std"]]
        .median().reset_index().to_dict(orient="records")}
K.jdump(summ, OUT / "summary.json")
pd.set_option("display.width", 250)
print(M[M.D == "D1"].round(3).to_string())
print(M[(M.D != "D1") & M.kind.isin(["RAW", "CONST", "SCALE", "TSCALE", "TADD", "CONF", "CONFraw"]) & M.method.isin(["J_cbb", "I_xdf", "-"])].round(3).to_string())
for k, r in res.items():
    print(k, round(r["median"], 4), [round(x, 4) for x in r["ci95"]], round(r["frac_negative"], 3), f"{r['p_holm']:.2g}")
