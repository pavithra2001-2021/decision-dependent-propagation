"""Selective prediction by the forecast flip probability."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
d1 = P[P.D == "D1"]
A3 = pd.read_csv(HERE.parent / "model_families" / "results" / "participants.csv.gz")
CL = {"NYU2_ridge_logistic": (d1.lA.values, d1.lB.values, d1.sA_J_cbb.values, d1.sA_I_xdf.values),
      "NYU2_linear_svm": (A3.M2_lA.values, A3.M2_lB.values, A3.M2_s_J_cbb.values, A3.M2_s_I_xdf.values),
      "NYU2_rbf_svm": (A3.M4_lA.values, A3.M4_lB.values, A3.M4_s_J_cbb.values, A3.M4_s_I_xdf.values)}
for pe in ("AP", "PA"):
    B = pd.read_csv(K.RESULTS / f"discovery_decisions_{pe}.csv.gz")
    B = B[B.model == "ridge_nested"]
    CL[f"TCP_{pe}_ridge_logistic"] = (B.lA.values, B.lB.values, B.sigma_J_cbb.values, B.sigma_I_xdf.values)

QS = np.round(np.linspace(0.5, 1.0, 51), 2)


def risk_curve(score, f):
    o = np.argsort(-score, kind="stable")
    n = len(f)
    out = []
    for q in QS:
        keep = o[n - int(round(q * n)):]
        out.append(f[keep].mean())
    return np.array(out)


def metrics(lA, f, sJ, sI):
    sc = {"margin": -np.abs(lA), "joint": norm.cdf(-np.abs(lA) / (np.sqrt(2) * sJ)),
          "independent": norm.cdf(-np.abs(lA) / (np.sqrt(2) * sI))}
    m = {}
    for k, s in sc.items():
        rc = risk_curve(s, f)
        m[k] = {"auc": float(roc_auc_score(f, s)), "aurc": float(rc.mean()), "risk_at_80": float(rc[QS == 0.8][0]), "curve": rc}
    return m


summ, curves, pvals = {}, [], []
for name, (lA, lB, sJ, sI) in CL.items():
    f = (np.sign(lA) != np.sign(lB)).astype(int)
    m = metrics(lA, f, sJ, sI)
    rng = np.random.default_rng(K.SEED_C)
    bo = {"auc_joint_minus_margin": [], "auc_joint_minus_independent": [], "aurc_joint_minus_margin": [],
          "aurc_joint_minus_independent": []}
    for _ in range(10000):
        i = rng.integers(0, len(f), len(f))
        if f[i].sum() in (0, len(i)):
            continue
        mb = metrics(lA[i], f[i], sJ[i], sI[i])
        for k in ("margin", "independent"):
            bo[f"auc_joint_minus_{k}"].append(mb["joint"]["auc"] - mb[k]["auc"])
            bo[f"aurc_joint_minus_{k}"].append(mb["joint"]["aurc"] - mb[k]["aurc"])
    r = {"n": int(len(f)), "flip_rate": float(f.mean())}
    for k in ("margin", "joint", "independent"):
        r[k] = {kk: v for kk, v in m[k].items() if kk != "curve"}
        curves += [{"classifier": name, "score": k, "retained": q, "flip_rate": v} for q, v in zip(QS, m[k]["curve"])]
    for key, arr in bo.items():
        a = np.array(arr)
        point = (m["joint"]["auc"] - m[key.split("_minus_")[1]]["auc"]) if key.startswith("auc") else \
            (m["joint"]["aurc"] - m[key.split("_minus_")[1]]["aurc"])
        r[key] = {"diff": float(point), "ci95": [float(v) for v in np.percentile(a, [2.5, 97.5])],
                  "p_two_sided": float(min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0)))), "n_boot": int(len(a))}
    pvals.append(r["aurc_joint_minus_margin"]["p_two_sided"])
    summ[name] = r
adj = K.holm(pvals)
for (name, r), a in zip(summ.items(), adj):
    r["aurc_joint_minus_margin"]["p_holm"] = float(a)
K.jdump(summ, OUT / "summary.json")
pd.DataFrame(curves).to_csv(OUT / "risk_coverage.csv", index=False)
for name, r in summ.items():
    print(name, r["n"], round(r["flip_rate"], 3), "AUC m/j/i", round(r["margin"]["auc"], 3), round(r["joint"]["auc"], 3),
          round(r["independent"]["auc"], 3), "AURC m/j/i", round(r["margin"]["aurc"], 4), round(r["joint"]["aurc"], 4),
          round(r["independent"]["aurc"], 4), "dAURC j-m", round(r["aurc_joint_minus_margin"]["diff"], 4),
          [round(x, 4) for x in r["aurc_joint_minus_margin"]["ci95"]], round(r["aurc_joint_minus_margin"]["p_holm"], 3))
