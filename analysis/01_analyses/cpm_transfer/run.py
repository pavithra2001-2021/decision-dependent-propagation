"""Connectome-based predictive models: propagation, forecasts and transfer."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
t0 = time.time()
C = K.nyu2_cohort()
n = len(C)
y = (C.sex == "female").astype(int).values
age = K.nyu2_ages(C).values.astype(float)
ZA, ZB = K.nyu2_point_all(C)
E_ = ZA.shape[1]
folds = K.nyu2_outer_folds(y)
P_THR = 0.01


def select(Z, t):
    m = len(t)
    Zs = (Z - Z.mean(0)) / Z.std(0)
    ts = (t - t.mean()) / t.std()
    r = Zs.T @ ts / m
    tt = r * np.sqrt((m - 2) / np.maximum(1 - r ** 2, 1e-12))
    p = 2 * stats.t.sf(np.abs(tt), m - 2)
    return (r > 0) & (p < P_THR), (r < 0) & (p < P_THR)


W = {k: np.zeros((n, E_)) for k in ("cpm_sex", "pos_strength", "cpm_age")}
c = {k: np.zeros(n) for k in W}
fold_of = np.zeros(n, int)
sizes = []
for f, (tr, te) in enumerate(folds):
    fold_of[te] = f
    Pn, Nn = select(ZA[tr], y[tr].astype(float))
    S = np.column_stack([ZA[tr][:, Pn].sum(1), ZA[tr][:, Nn].sum(1)])
    use = [j for j, msk in enumerate((Pn, Nn)) if msk.sum() > 0]
    m = LogisticRegression(penalty=None, max_iter=20000).fit(S[:, use], y[tr])
    w = np.zeros(E_)
    for b, j in zip(m.coef_[0], use):
        w[(Pn, Nn)[j]] = b
    W["cpm_sex"][te], c["cpm_sex"][te] = w, m.intercept_[0]
    wp = np.zeros(E_); wp[Pn] = 1.0
    W["pos_strength"][te] = wp
    Pa, Na = select(ZA[tr], age[tr])
    Sa = np.column_stack([ZA[tr][:, Pa].sum(1), ZA[tr][:, Na].sum(1)])
    usea = [j for j, msk in enumerate((Pa, Na)) if msk.sum() > 0]
    ma = LinearRegression().fit(Sa[:, usea], age[tr])
    wa = np.zeros(E_)
    for b, j in zip(ma.coef_, usea):
        wa[(Pa, Na)[j]] = b
    W["cpm_age"][te], c["cpm_age"][te] = wa, ma.intercept_
    sizes.append({"fold": f, "sex_pos": int(Pn.sum()), "sex_neg": int(Nn.sum()), "age_pos": int(Pa.sum()), "age_neg": int(Na.sum()),
                  "sex_beta_pos": float(m.coef_[0][use.index(0)]) if 0 in use else None,
                  "sex_beta_neg": float(m.coef_[0][use.index(1)]) if 1 in use else None})
print(pd.DataFrame(sizes).to_string(), flush=True)

rows = []
METH = ("J_cbb", "uHAC", "I_xdf", "I_cbb", "J_iid")
for k_, pid in enumerate(C.pid):
    mA = K.EC.run_measurement(K.nyu2_load(pid, 1), np.random.default_rng(K.SEED_C + 10 * k_ + 1), K.R)
    rec = {"pid": pid, "y": int(y[k_]), "age": age[k_], "fold": int(fold_of[k_])}
    for d in W:
        w = W[d][k_]
        s = K.EC.decision_sds(mA, w)
        rec |= {f"{d}_lA": float(ZA[k_] @ w + c[d][k_]), f"{d}_lB": float(ZB[k_] @ w + c[d][k_])} | {f"{d}_s_{m}": s[m] for m in METH}
    rows.append(rec)
    del mA
    if k_ % 40 == 0:
        print(f"{k_}/{n} {time.time()-t0:.0f}s", flush=True)
D = pd.DataFrame(rows)
D.to_csv(OUT / "participants.csv.gz", index=False, compression="gzip")

P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
D1 = P[P.D == "D1"].reset_index(drop=True)
assert (D1.pid.values == D.pid.values).all()
d1 = ((D1.lB - D1.lA) ** 2).values
src = {"J_cbb": "sA_J_cbb", "uHAC": "sA_uHAC", "I_xdf": "sA_I_xdf"}
Cs = {m: np.zeros(n) for m in src}
for tr, te in folds:
    for m, col in src.items():
        Cs[m][te] = np.mean(d1[tr] / (2 * D1[col].values[tr] ** 2))

S = {"p_threshold": P_THR, "network_sizes": sizes, "median_scale_from_classifier": {m: float(np.median(v)) for m, v in Cs.items()}}
pv = []
for d in W:
    lA, lB = D[f"{d}_lA"].values, D[f"{d}_lB"].values
    dd = lB - lA
    o = {}
    if d == "cpm_sex":
        o |= {"auc_runA": float(roc_auc_score(y, lA)), "auc_runB": float(roc_auc_score(y, lB))}
    if d == "cpm_age":
        o |= {"pearson_r_runA": float(stats.pearsonr(age, lA).statistic), "mae_runA": float(np.mean(np.abs(age - lA)))}
    kap = (D[f"{d}_s_J_cbb"] / D[f"{d}_s_I_cbb"]) ** 2
    o["kappa_median_iqr"] = [float(kap.median()), float(kap.quantile(.25)), float(kap.quantile(.75))]
    o["sd_ratio_Jcbb_over_Ixdf_median"] = float(np.median(D[f"{d}_s_J_cbb"] / D[f"{d}_s_I_xdf"]))
    o["uhac_over_Jcbb_median"] = float(np.median(D[f"{d}_s_uHAC"] / D[f"{d}_s_J_cbb"]))
    v = {m: 2 * D[f"{d}_s_{m}"].values ** 2 for m in METH}
    o["vcr"] = {m: float(v[m].mean() / np.mean(dd ** 2)) for m in METH}
    o["coverage95"] = {m: float(np.mean(np.abs(dd) <= 1.96 * np.sqrt(v[m]))) for m in METH}
    o["nlpd_Jcbb_minus_Ixdf"] = K.paired(K.nlpd(np.abs(dd), v["J_cbb"]) - K.nlpd(np.abs(dd), v["I_xdf"]))
    o["spearman_sJ_absd"] = float(stats.spearmanr(D[f"{d}_s_J_cbb"], np.abs(dd)).statistic)
    tr_ = {}
    for m in src:
        vt = Cs[m] * v[m]
        tr_[m] = {"coverage95": float(np.mean(np.abs(dd) <= 1.96 * np.sqrt(vt))), "vcr": float(vt.mean() / np.mean(dd ** 2)),
                  "mean_nlpd": float(np.mean(K.nlpd(np.abs(dd), vt))), "nlpd": K.nlpd(np.abs(dd), vt)}
    o["transfer"] = {m: {k2: x for k2, x in tr_[m].items() if k2 != "nlpd"} for m in src}
    o["transfer_nlpd_Jcbb_minus_Ixdf"] = K.paired(tr_["J_cbb"]["nlpd"] - tr_["I_xdf"]["nlpd"])
    pv.append(o["transfer_nlpd_Jcbb_minus_Ixdf"]["wilcoxon_p"])
    own = {}
    for m in ("J_cbb", "I_xdf"):
        vo = np.zeros(n)
        for tr, te in folds:
            vo[te] = np.mean(dd[tr] ** 2 / v[m][tr]) * v[m][te]
        own[m] = {"coverage95": float(np.mean(np.abs(dd) <= 1.96 * np.sqrt(vo))), "mean_nlpd": float(np.mean(K.nlpd(np.abs(dd), vo)))}
    o["own_scale_reference"] = own
    S[d] = o
for d, a in zip(W, K.holm(pv)):
    S[d]["transfer_nlpd_Jcbb_minus_Ixdf"]["p_holm"] = float(a)
    t = S[d]["transfer"]
    S[d]["transfer_gap_reproduces"] = bool(0.85 <= t["J_cbb"]["coverage95"] <= 0.99 and S[d]["transfer_nlpd_Jcbb_minus_Ixdf"]["median"] < 0
                                         and a < 0.05)
S["runtime_s"] = time.time() - t0
K.jdump(S, OUT / "summary.json")
for d in W:
    o = S[d]
    print(d, {k2: o[k2] for k2 in o if k2 in ("auc_runA", "auc_runB", "pearson_r_runA", "kappa_median_iqr", "sd_ratio_Jcbb_over_Ixdf_median",
                                                "uhac_over_Jcbb_median", "spearman_sJ_absd")})
    print("   NLPD J-I", {k2: o["nlpd_Jcbb_minus_Ixdf"][k2] for k2 in ("median", "ci95", "frac_negative")})
    print("   cov", {m: round(x, 3) for m, x in o["coverage95"].items()}, "vcr", {m: round(x, 3) for m, x in o["vcr"].items()})
    print("   transfer", o["transfer"], "\n   transfer NLPD J-I", {k2: o["transfer_nlpd_Jcbb_minus_Ixdf"][k2] for k2 in ("median", "ci95", "p_holm")},
          "reproduces:", o["transfer_gap_reproduces"], "\n   own", o["own_scale_reference"])
