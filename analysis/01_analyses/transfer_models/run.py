"""Transfer of the classifier-fitted calibration scale to the trained decisions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
D1 = P[P.D == "D1"].reset_index(drop=True)
A3 = pd.read_csv(HERE.parent / "model_families" / "results" / "participants.csv.gz")
assert (A3.pid.values == D1.pid.values).all()
y = D1.y.values
folds = K.nyu2_outer_folds(y)
n = len(y)
d2_1 = ((D1.lB - D1.lA) ** 2).values
src = {"J_cbb": "sA_J_cbb", "uHAC": "sA_uHAC", "I_xdf": "sA_I_xdf"}
C = {m: np.zeros(n) for m in src}
for tr, te in folds:
    for m, col in src.items():
        v0 = 2 * D1[col].values ** 2
        C[m][te] = np.mean(d2_1[tr] / v0[tr])
res = {"transfer": {}}
pv = []
for M, lab in (("M2", "linear_svm_sex"), ("M3", "ridge_age"), ("M4", "rbf_svm_sex")):
    d = (A3[f"{M}_lB"] - A3[f"{M}_lA"]).values
    r = {}
    NL = {}
    for m in src:
        v = C[m] * 2 * A3[f"{M}_s_{m}"].values ** 2
        NL[m] = K.nlpd(np.abs(d), v)
        r[m] = {"coverage95": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(v))), "vcr": float(v.mean() / np.mean(d ** 2)),
                "mean_nlpd": float(NL[m].mean())}
    r["nlpd_Jcbb_minus_Ixdf"] = K.paired(NL["J_cbb"] - NL["I_xdf"])
    pv.append(r["nlpd_Jcbb_minus_Ixdf"]["wilcoxon_p"])
    res["transfer"][lab] = r
for lab, a in zip(res["transfer"], K.holm(pv)):
    res["transfer"][lab]["nlpd_Jcbb_minus_Ixdf"]["p_holm"] = float(a)
res["median_scale_c"] = {m: float(np.median(C[m])) for m in src}

age = A3.age.values
d = (D1.lB - D1.lA).values
diff = (D1.nlpd_J_cbb - D1.nlpd_I_xdf).values
res["age_subgroups"] = {}
for g, mask in (("under18", age < 18), ("adult", age >= 18)):
    sub = D1[mask]
    dd = d[mask]
    kap = (sub.sA_J_cbb / sub.sA_I_cbb) ** 2
    res["age_subgroups"][g] = {
        "n": int(mask.sum()), "nlpd_Jcbb_minus_Ixdf": K.paired(diff[mask]),
        "vcr_Jcbb": float(np.mean(2 * sub.sA_J_cbb ** 2) / np.mean(dd ** 2)),
        "vcr_Ixdf": float(np.mean(2 * sub.sA_I_xdf ** 2) / np.mean(dd ** 2)),
        "cov_Jcbb": float(np.mean(np.abs(dd) <= 1.96 * np.sqrt(2) * sub.sA_J_cbb)),
        "cov_Ixdf": float(np.mean(np.abs(dd) <= 1.96 * np.sqrt(2) * sub.sA_I_xdf)),
        "kappa_median": float(np.median(kap)), "mean_fd_runA_note": "FD not stored in the original table"}
mw = mannwhitneyu(diff[age < 18], diff[age >= 18])
res["age_subgroups"]["mannwhitney_p"] = float(mw.pvalue)
K.jdump(res, OUT / "summary.json")
for lab, r in res["transfer"].items():
    print(lab, {m: {k: round(v, 3) for k, v in r[m].items()} for m in src}, round(r["nlpd_Jcbb_minus_Ixdf"]["median"], 3),
          [round(x, 3) for x in r["nlpd_Jcbb_minus_Ixdf"]["ci95"]], f"{r['nlpd_Jcbb_minus_Ixdf']['p_holm']:.2g}")
print(res["median_scale_c"])
for g in ("under18", "adult"):
    r = res["age_subgroups"][g]
    print(g, r["n"], round(r["nlpd_Jcbb_minus_Ixdf"]["median"], 2), [round(x, 2) for x in r["nlpd_Jcbb_minus_Ixdf"]["ci95"]],
          round(r["nlpd_Jcbb_minus_Ixdf"]["frac_negative"], 2), round(r["vcr_Jcbb"], 3), round(r["vcr_Ixdf"], 4), round(r["cov_Jcbb"], 2),
          round(r["kappa_median"], 1))
print("MW p", res["age_subgroups"]["mannwhitney_p"])
