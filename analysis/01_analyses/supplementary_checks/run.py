"""Extended ridge grid, fold-level estimates, age composition and NLPD decomposition."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
S = {}
C = K.nyu2_cohort()
n = len(C)
y = (C.sex == "female").astype(int).values
age = K.nyu2_ages(C).values.astype(float)
ZA, ZB = K.nyu2_point_all(C)
folds = K.nyu2_outer_folds(y)

GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4]
W3, c3, chosen, fold_of = np.zeros_like(ZA), np.zeros(n), [], np.zeros(n, int)
for f, (tr, te) in enumerate(folds):
    best, bs = None, -np.inf
    for a in GRID:
        v = []
        for i, j in StratifiedKFold(5, shuffle=True, random_state=K.SEED_C).split(ZA[tr], y[tr]):
            m = Ridge(alpha=a).fit(ZA[tr][i], age[tr][i])
            v.append(-np.mean((m.predict(ZA[tr][j]) - age[tr][j]) ** 2))
        if np.mean(v) > bs:
            best, bs = a, np.mean(v)
    m = Ridge(alpha=best).fit(ZA[tr], age[tr])
    W3[te], c3[te], fold_of[te] = m.coef_, m.intercept_, f
    chosen.append(best)
pred = np.array([ZA[k] @ W3[k] + c3[k] for k in range(n)])
base = np.zeros(n)
for tr, te in folds:
    base[te] = age[tr].mean()


def perf(msk):
    a, p, b = age[msk], pred[msk], base[msk]
    return {"n": int(msk.sum()), "r": float(pearsonr(a, p).statistic), "r2": float(1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)),
            "mae": float(np.mean(np.abs(a - p))), "mae_mean_predictor": float(np.mean(np.abs(a - b)))}


S["ridge_extended"] = {"grid": GRID, "chosen_alpha_per_fold": chosen, "n_at_lower_boundary": int(sum(a == GRID[0] for a in chosen)),
                       "overall": perf(np.ones(n, bool)), "under18": perf(age < 18), "adult": perf(age >= 18)}
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
D1 = P[P.D == "D1"].reset_index(drop=True)
rows = []
for k, pid in enumerate(C.pid):
    mA = K.EC.run_measurement(K.nyu2_load(pid, 1), np.random.default_rng(K.SEED_C + 10 * k + 1), K.R)
    s = K.EC.decision_sds(mA, W3[k])
    rows.append({"pid": pid, "lA": pred[k], "lB": float(ZB[k] @ W3[k] + c3[k])} | {f"s_{m}": s[m] for m in ("J_cbb", "uHAC", "I_xdf", "I_cbb")})
    del mA
R = pd.DataFrame(rows)
R.to_csv(OUT / "participants_ridge_extended.csv.gz", index=False, compression="gzip")
d = (R.lB - R.lA).values
vJ, vI = 2 * R.s_J_cbb.values ** 2, 2 * R.s_I_xdf.values ** 2
kap = (R.s_J_cbb / R.s_I_cbb) ** 2
cJ, cI = np.zeros(n), np.zeros(n)
d1 = ((D1.lB - D1.lA) ** 2).values
for tr, te in folds:
    cJ[te] = np.mean(d1[tr] / (2 * D1.sA_J_cbb.values[tr] ** 2)); cI[te] = np.mean(d1[tr] / (2 * D1.sA_I_xdf.values[tr] ** 2))
S["ridge_extended"]["propagation"] = {
    "kappa_median_iqr": [float(kap.median()), float(kap.quantile(.25)), float(kap.quantile(.75))],
    "sd_ratio_median": float(np.median(R.s_J_cbb / R.s_I_xdf)), "uhac_over_Jcbb_median": float(np.median(R.s_uHAC / R.s_J_cbb)),
    "nlpd_Jcbb_minus_Ixdf": K.paired(K.nlpd(np.abs(d), vJ) - K.nlpd(np.abs(d), vI)),
    "coverage_J": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(vJ))), "coverage_I": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(vI))),
    "transfer_coverage_J": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(cJ * vJ))), "transfer_coverage_I": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(cI * vI))),
    "transfer_vcr_J": float(np.mean(cJ * vJ) / np.mean(d ** 2)), "transfer_vcr_I": float(np.mean(cI * vI) / np.mean(d ** 2))}

diff = (D1.nlpd_J_cbb - D1.nlpd_I_xdf).values
sdr = (D1.sA_J_cbb / D1.sA_I_xdf).values
kp = ((D1.sA_J_cbb / D1.sA_I_cbb) ** 2).values
fl = []
for f, (tr, te) in enumerate(folds):
    fl.append({"fold": f + 1, "n": int(len(te)), "median_nlpd_diff": float(np.median(diff[te])), "frac_favour_J": float(np.mean(diff[te] < 0)),
               "median_sd_ratio": float(np.median(sdr[te])), "median_kappa": float(np.median(kp[te])),
               "loo_median_nlpd_diff": float(np.median(np.delete(diff, te)))})
FL = pd.DataFrame(fl)
FL.to_csv(OUT / "fold_level.csv", index=False)
S["fold_level"] = {"folds_negative_median": int((FL.median_nlpd_diff < 0).sum()), "range_median_nlpd_diff": [float(FL.median_nlpd_diff.min()), float(FL.median_nlpd_diff.max())],
                   "range_loo_median": [float(FL.loo_median_nlpd_diff.min()), float(FL.loo_median_nlpd_diff.max())],
                   "range_median_sd_ratio": [float(FL.median_sd_ratio.min()), float(FL.median_sd_ratio.max())],
                   "range_median_kappa": [float(FL.median_kappa.min()), float(FL.median_kappa.max())]}

A3 = pd.read_csv(HERE.parent / "model_families/results/participants.csv.gz")
M2 = pd.read_csv(HERE.parent / "cpm_transfer/results/participants.csv.gz")
SETS = {}
for D, nm in (("D1", "ridge_logistic"), ("D2", "global_mean"), ("D3", "random")):
    g = P[P.D == D].reset_index(drop=True)
    SETS[nm] = (g.lA.values, g.lB.values, g.sA_J_cbb.values, g.sA_I_xdf.values)
for M, nm in (("M2", "linear_svm"), ("M4", "rbf_svm")):
    SETS[nm] = (A3[f"{M}_lA"].values, A3[f"{M}_lB"].values, A3[f"{M}_s_J_cbb"].values, A3[f"{M}_s_I_xdf"].values)
SETS["ridge_age_extended"] = (R.lA.values, R.lB.values, R.s_J_cbb.values, R.s_I_xdf.values)
for dname in ("cpm_sex", "pos_strength", "cpm_age"):
    SETS[dname] = (M2[f"{dname}_lA"].values, M2[f"{dname}_lB"].values, M2[f"{dname}_s_J_cbb"].values, M2[f"{dname}_s_I_xdf"].values)
sp = {}
for nm, (lA, lB, sJ, sI) in SETS.items():
    z = np.abs(lB - lA)
    rng = np.random.default_rng(K.SEED_C)
    idx = [rng.integers(0, n, n) for _ in range(10000)]
    sp[nm] = {}
    for lab, x in (("J_cbb", sJ), ("I_xdf", sI)):
        bo = [spearmanr(x[i], z[i]).statistic for i in idx]
        sp[nm][lab] = {"rho": float(spearmanr(x, z).statistic), "ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])]}
S["spearman"] = sp

C181 = K.EC.load_cohort(False)
a181 = K.nyu2_ages(C181).values.astype(float)
keep = C181.pid.isin(C.pid).values


def agesum(a):
    return {"n": int(len(a)), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
            "under10": int((a < 10).sum()), "under12": int((a < 12).sum()), "under18": int((a < 18).sum())}


fdm = np.maximum(C181.fd1.values, C181.fd2.values)
grp = pd.cut(a181, [0, 10, 12, 18, 100], right=False, labels=["<10", "10-12", "12-18", ">=18"])
S["age_motion"] = {"verified_181": agesum(a181), "eligible_158": agesum(a181[keep]), "excluded_23": agesum(a181[~keep]),
                   "exclusion_rate_by_group": {str(g): float(np.mean(~keep[grp == g])) for g in grp.categories},
                   "mean_fd_by_group": {str(g): float(np.mean((C181.fd1.values + C181.fd2.values)[grp == g] / 2)) for g in grp.categories},
                   "n_by_group_181": {str(g): int(np.sum(grp == g)) for g in grp.categories}}

dec = {}
for nm, (lA, lB, sJ, sI) in SETS.items():
    dd = lB - lA
    vj, vi = 2 * sJ ** 2, 2 * sI ** 2
    t1 = 0.5 * np.log(vj / vi)
    t2 = 0.5 * dd ** 2 * (1 / vj - 1 / vi)
    dec[nm] = {"median_logvar_term": float(np.median(t1)), "median_residual_term": float(np.median(t2)),
               "mean_logvar_term": float(t1.mean()), "mean_residual_term": float(t2.mean()),
               "median_total": float(np.median(t1 + t2)), "median_variance_ratio": float(np.median(vj / vi))}
S["nlpd_decomposition"] = dec
K.jdump(S, OUT / "summary.json")
import json
print(json.dumps({k: S[k] for k in ("ridge_extended", "fold_level", "age_motion")}, indent=1)[:5000])
for nm in dec:
    print(nm, {k: round(v, 3) for k, v in dec[nm].items()}, sp[nm]["J_cbb"])
