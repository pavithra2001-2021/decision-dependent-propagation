"""Further decisions (linear SVM, ridge regression of age, RBF-SVM): propagation and forecasts."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
t0 = time.time()
C = K.nyu2_cohort()
n = len(C)
y = (C.sex == "female").astype(int).values
age = K.nyu2_ages(C).values.astype(float)
print(f"cohort n={n}; age available {np.isfinite(age).sum()} (missing {np.isnan(age).sum()}); "
      f"age range {np.nanmin(age):.2f}-{np.nanmax(age):.2f}, median {np.nanmedian(age):.2f}", flush=True)
ZA, ZB = K.nyu2_point_all(C)
E_ = ZA.shape[1]
folds = K.nyu2_outer_folds(y)
print(f"point FC done {time.time()-t0:.0f}s", flush=True)

W1, c1, fold_of = K.d1_fold_weights(ZA, y)
print(f"M1 fitted {time.time()-t0:.0f}s", flush=True)


def inner_select(Xtr, ytr, strat, make, grid, score):
    best, bs = None, -np.inf
    for g in grid:
        v = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=K.SEED_C).split(Xtr, strat):
            m = make(g).fit(Xtr[a], ytr[a])
            v.append(score(m, Xtr[b], ytr[b]))
        if np.mean(v) > bs:
            best, bs = g, np.mean(v)
    return best


auc_sc = lambda m, X, t: roc_auc_score(t, m.decision_function(X))
mse_sc = lambda m, X, t: -np.mean((m.predict(X) - t) ** 2)
W2, c2, W3, c3 = np.zeros((n, E_)), np.zeros(n), np.full((n, E_), np.nan), np.full(n, np.nan)
rbf, hyp = {}, []
for f, (tr, te) in enumerate(folds):
    Cs = inner_select(ZA[tr], y[tr], y[tr], lambda c: SVC(kernel="linear", C=c), [1e-3, 1e-2, 1e-1, 1.0, 10.0], auc_sc)
    m = SVC(kernel="linear", C=Cs).fit(ZA[tr], y[tr])
    W2[te], c2[te] = m.coef_[0], m.intercept_[0]
    tra = tr[np.isfinite(age[tr])]
    al = inner_select(ZA[tra], age[tra], y[tra], lambda a: Ridge(alpha=a), [1.0, 10.0, 1e2, 1e3, 1e4], mse_sc)
    m3 = Ridge(alpha=al).fit(ZA[tra], age[tra])
    W3[te], c3[te] = m3.coef_, m3.intercept_
    Cr = inner_select(ZA[tr], y[tr], y[tr], lambda c: SVC(kernel="rbf", gamma="scale", C=c), [0.1, 1.0, 10.0, 100.0], auc_sc)
    m4 = SVC(kernel="rbf", gamma="scale", C=Cr).fit(ZA[tr], y[tr])
    rbf[f] = {"sv": m4.support_vectors_.copy(), "a": m4.dual_coef_[0].copy(), "b": float(m4.intercept_[0]), "g": float(m4._gamma)}
    hyp.append({"fold": f, "M2_C": Cs, "M3_alpha": al, "M4_C": Cr, "M4_gamma": float(m4._gamma), "M4_nsv": int(len(m4.support_))})
    print(f"fold {f}: linSVM C={Cs}, ridge alpha={al}, rbf C={Cr} ({time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(hyp).to_csv(OUT / "hyperparameters.csv", index=False)


def rbf_f(Zq, M):
    d2 = (Zq ** 2).sum(1)[:, None] + (M["sv"] ** 2).sum(1)[None, :] - 2 * Zq @ M["sv"].T
    return np.exp(-M["g"] * np.maximum(d2, 0)) @ M["a"] + M["b"]


def rbf_grad(z, M):
    k = np.exp(-M["g"] * np.maximum(((M["sv"] - z) ** 2).sum(1), 0))
    return (-2 * M["g"]) * ((M["a"] * k) @ (z[None, :] - M["sv"]))


P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
P1 = P[P.D == "D1"].set_index("pid")
rows = []
for k, pid in enumerate(C.pid):
    XA = K.nyu2_load(pid, 1)
    mA = K.EC.run_measurement(XA, np.random.default_rng(K.SEED_C + 10 * k + 1), K.R)
    assert np.allclose(mA["z"], ZA[k])
    rec = {"pid": pid, "y": int(y[k]), "age": age[k], "fold": int(fold_of[k])}
    for M, w, c in (("M1", W1[k], c1[k]), ("M2", W2[k], c2[k]), ("M3", W3[k], c3[k])):
        if not np.isfinite(c):
            continue
        s = K.EC.decision_sds(mA, w)
        rec |= {f"{M}_lA": float(ZA[k] @ w + c), f"{M}_lB": float(ZB[k] @ w + c)}
        rec |= {f"{M}_s_{m}": s[m] for m in K.EC.METHODS}
        if M == "M1":
            rec["M1_t_uHAC"] = s["_t_uHAC"]
    Mr = rbf[int(fold_of[k])]
    lA, lB = float(rbf_f(ZA[k][None], Mr)[0]), float(rbf_f(ZB[k][None], Mr)[0])
    g = rbf_grad(ZA[k], Mr)
    rc, ri = rbf_f(ZA[k] + mA["dev"]["cbb"], Mr), rbf_f(ZA[k] + mA["dev"]["iid"], Mr)
    s2 = mA["s2"]
    rec |= {"M4_lA": lA, "M4_lB": lB, "M4_s_J_cbb": float(rc.std(ddof=1)), "M4_s_J_iid": float(ri.std(ddof=1)),
            "M4_s_J_cbb_lin": float((mA["dev"]["cbb"] @ g).std(ddof=1)),
            "M4_s_uHAC": float(K.decision_sigma(XA, g, mA["b"])["uhac_bartlett"]),
            "M4_s_I_xdf": float(np.sqrt(np.sum(g ** 2 * s2["xdf"]))), "M4_s_I_cbb": float(np.sqrt(np.sum(g ** 2 * s2["cbb"]))),
            "M4_s_I_xdf_adaptive": float(np.sqrt(np.sum(g ** 2 * s2["xdf_adaptive"]))),
            "M4_gradnorm": float(np.linalg.norm(g)), "M4_rep_mean_shift": float(rc.mean() - lA)}
    rows.append(rec)
    del mA
    if k % 20 == 0:
        print(f"participant {k}/{n} ({time.time()-t0:.0f}s)", flush=True)
D = pd.DataFrame(rows)
D.to_csv(OUT / "participants.csv.gz", index=False, compression="gzip")

rep = {}
for col_new, col_old in (("M1_lA", "lA"), ("M1_lB", "lB"), ("M1_s_J_cbb", "sA_J_cbb"), ("M1_s_I_xdf", "sA_I_xdf"),
                         ("M1_s_uHAC", "sA_uHAC"), ("M1_s_I_cbb", "sA_I_cbb"), ("M1_s_J_iid", "sA_J_iid")):
    a, b = D.set_index("pid")[col_new], P1.loc[D.pid, col_old]
    rep[col_new] = float(np.max(np.abs(a.values - b.values) / np.maximum(np.abs(b.values), 1e-12)))
print("reproduction max relative differences:", rep, flush=True)
if max(rep.values()) > 1e-6:
    K.jdump({"reproduction": rep, "status": "FAILED"}, OUT / "summary.json")
    raise SystemExit("reproduction of original D1 failed; analysis stopped (see summary)")


def outcome(M, Dm, classifier):
    lA, lB = Dm[f"{M}_lA"].values, Dm[f"{M}_lB"].values
    d = lB - lA
    o = {"n": int(len(Dm))}
    if classifier:
        yy = Dm.y.values
        o |= {"auc_runA": float(roc_auc_score(yy, lA)), "auc_runB": float(roc_auc_score(yy, lB))}
        f = (np.sign(lA) != np.sign(lB)).astype(int)
        o["flip_rate"] = float(f.mean())
        PJ = norm.cdf(-np.abs(lA) / (np.sqrt(2) * Dm[f"{M}_s_J_cbb"].values))
        PI = norm.cdf(-np.abs(lA) / (np.sqrt(2) * Dm[f"{M}_s_I_xdf"].values))
        db = (PJ - f) ** 2 - (PI - f) ** 2
        rng = np.random.default_rng(K.SEED_C)
        bo = np.array([db[rng.integers(0, len(db), len(db))].mean() for _ in range(10000)])
        o |= {"brier_J": float(np.mean((PJ - f) ** 2)), "brier_I": float(np.mean((PI - f) ** 2)),
              "brier_diff_J_minus_I": float(db.mean()), "brier_diff_ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])],
              "mean_PJ": float(PJ.mean())}
    else:
        ya = Dm.age.values
        o |= {"pearson_r_runA": float(pearsonr(ya, lA).statistic), "pearson_p_runA": float(pearsonr(ya, lA).pvalue),
              "r2_runA": float(1 - np.sum((ya - lA) ** 2) / np.sum((ya - ya.mean()) ** 2)),
              "mae_runA": float(np.mean(np.abs(ya - lA))), "pearson_r_runB": float(pearsonr(ya, lB).statistic)}
        base = np.zeros(len(Dm))
        for f_ in range(10):
            te = Dm.fold.values == f_
            base[te] = Dm.age.values[~te].mean()
        o["mae_mean_predictor"] = float(np.mean(np.abs(ya - base)))
        o["r2_mean_predictor"] = float(1 - np.sum((ya - base) ** 2) / np.sum((ya - ya.mean()) ** 2))
    meths = [c.replace(f"{M}_s_", "") for c in Dm.columns if c.startswith(f"{M}_s_")]
    o["vcr"] = {m: float(np.mean(2 * Dm[f"{M}_s_{m}"] ** 2) / np.mean(d ** 2)) for m in meths}
    o["coverage95"] = {m: float(np.mean(np.abs(d) <= 1.96 * np.sqrt(2) * Dm[f"{M}_s_{m}"])) for m in meths}
    o["sd_ratio_Jcbb_over_Ixdf_median"] = float(np.median(Dm[f"{M}_s_J_cbb"] / Dm[f"{M}_s_I_xdf"]))
    kap = (Dm[f"{M}_s_J_cbb"] / Dm[f"{M}_s_I_cbb"]) ** 2
    o["kappa_median_iqr"] = [float(np.median(kap)), float(np.percentile(kap, 25)), float(np.percentile(kap, 75))]
    o["uhac_over_Jcbb_median"] = float(np.median(Dm[f"{M}_s_uHAC"] / Dm[f"{M}_s_J_cbb"]))
    o["spearman_sJ_absd"] = float(spearmanr(Dm[f"{M}_s_J_cbb"], np.abs(d)).statistic)
    nl = lambda m: K.nlpd(np.abs(d), 2 * Dm[f"{M}_s_{m}"].values ** 2)
    o["nlpd_Jcbb_minus_Ixdf"] = K.paired(nl("J_cbb") - nl("I_xdf"))
    o["nlpd_Jcbb_minus_Icbb"] = K.paired(nl("J_cbb") - nl("I_cbb"))
    o["nlpd_uHAC_minus_Jcbb"] = K.paired(nl("uHAC") - nl("J_cbb"))
    if M == "M4":
        o["linearisation_ratio_nonlin_over_lin_median_iqr"] = [float(np.median(Dm.M4_s_J_cbb / Dm.M4_s_J_cbb_lin)),
                                                               float(np.percentile(Dm.M4_s_J_cbb / Dm.M4_s_J_cbb_lin, 25)),
                                                               float(np.percentile(Dm.M4_s_J_cbb / Dm.M4_s_J_cbb_lin, 75))]
        o["nlpd_uHAClin_minus_Jcbb_nonlin"] = o["nlpd_uHAC_minus_Jcbb"]
        o["nlpd_Jcbblin_minus_Jcbb_nonlin"] = K.paired(nl("J_cbb_lin") - nl("J_cbb"))
    return o


S = {"reproduction_max_rel_diff_M1_vs_original_D1": rep, "n_cohort": n, "n_age": int(np.isfinite(age).sum()),
     "age_summary": {"min": float(np.nanmin(age)), "max": float(np.nanmax(age)), "median": float(np.nanmedian(age)),
                     "mean": float(np.nanmean(age)), "sd": float(np.nanstd(age, ddof=1)),
                     "n_under_18": int(np.sum(age < 18))},
     "M1_ridge_logistic_sex": outcome("M1", D, True), "M2_linear_svm_sex": outcome("M2", D, True),
     "M3_ridge_age": outcome("M3", D[np.isfinite(D.age) & D.M3_lA.notna()], False), "M4_rbf_svm_sex": outcome("M4", D, True),
     "hyperparameters": hyp, "runtime_s": time.time() - t0}
fam = ["M2_linear_svm_sex", "M3_ridge_age", "M4_rbf_svm_sex"]
adj = K.holm([S[m]["nlpd_Jcbb_minus_Ixdf"]["wilcoxon_p"] for m in fam])
for m, a in zip(fam, adj):
    S[m]["nlpd_Jcbb_minus_Ixdf"]["p_holm"] = float(a)
K.jdump(S, OUT / "summary.json")
for m in ("M1_ridge_logistic_sex",) + tuple(fam):
    o = S[m]
    print(m, {k: o[k] for k in o if k in ("auc_runA", "auc_runB", "pearson_r_runA", "r2_runA", "mae_runA", "mae_mean_predictor", "flip_rate",
                                         "sd_ratio_Jcbb_over_Ixdf_median", "kappa_median_iqr", "uhac_over_Jcbb_median", "spearman_sJ_absd")})
    print("   NLPD J-I", o["nlpd_Jcbb_minus_Ixdf"], "\n   VCR", {k: round(v, 3) for k, v in o["vcr"].items()},
          "\n   cov", {k: round(v, 3) for k, v in o["coverage95"].items()})
print("done", time.time() - t0)
