"""Information beyond scale within a decision."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
A3 = pd.read_csv(HERE.parent / "model_families/results/participants.csv.gz")
M2 = pd.read_csv(HERE.parent / "cpm_transfer/results/participants.csv.gz")
y = P[P.D == "D1"].y.values
folds = K.nyu2_outer_folds(y)
DEC = {}
for D, nm in (("D1", "ridge_logistic"), ("D2", "global_mean"), ("D3", "random")):
    g = P[P.D == D].reset_index(drop=True)
    DEC[nm] = (g.lA.values, g.lB.values, g.sA_J_cbb.values, g.sA_I_xdf.values)
for M, nm in (("M2", "linear_svm"), ("M3", "ridge_age"), ("M4", "rbf_svm")):
    DEC[nm] = (A3[f"{M}_lA"].values, A3[f"{M}_lB"].values, A3[f"{M}_s_J_cbb"].values, A3[f"{M}_s_I_xdf"].values)
for d in ("cpm_sex", "pos_strength", "cpm_age"):
    DEC[d] = (M2[f"{d}_lA"].values, M2[f"{d}_lB"].values, M2[f"{d}_s_J_cbb"].values, M2[f"{d}_s_I_xdf"].values)


def crossfit(dd, v):
    out = np.zeros(len(dd))
    for tr, te in folds:
        out[te] = (np.mean(dd[tr] ** 2 / v[tr]) * v[te]) if v is not None else np.mean(dd[tr] ** 2)
    return out


res, p1, p2 = {}, [], []
for nm, (lA, lB, sJ, sI) in DEC.items():
    dd = lB - lA
    vc, vj, vi = crossfit(dd, None), crossfit(dd, 2 * sJ ** 2), crossfit(dd, 2 * sI ** 2)
    nl = lambda v: K.nlpd(np.abs(dd), v)
    a, b = nl(vj) - nl(vc), nl(vj) - nl(vi)
    rng = np.random.default_rng(K.SEED_C)
    x, z = sJ, np.abs(dd)
    bo = [spearmanr(x[i], z[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(10000))]
    res[nm] = {"scaledJ_minus_constant": K.paired(a) | {"mean": float(a.mean())},
               "scaledJ_minus_scaledI": K.paired(b) | {"mean": float(b.mean())},
               "mean_nlpd": {"constant": float(nl(vc).mean()), "scaled_J": float(nl(vj).mean()), "scaled_I": float(nl(vi).mean())},
               "spearman_sJ_absd": float(spearmanr(x, z).statistic), "spearman_ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])]}
    p1.append(res[nm]["scaledJ_minus_constant"]["wilcoxon_p"]); p2.append(res[nm]["scaledJ_minus_scaledI"]["wilcoxon_p"])
for (nm, r), a, b in zip(res.items(), K.holm(p1), K.holm(p2)):
    r["scaledJ_minus_constant"]["p_holm"] = float(a)
    r["scaledJ_minus_scaledI"]["p_holm"] = float(b)
    r["informative_beyond_scale"] = bool(r["scaledJ_minus_constant"]["median"] < 0 and a < 0.05)
K.jdump(res, OUT / "summary.json")
for nm, r in res.items():
    c1, c2 = r["scaledJ_minus_constant"], r["scaledJ_minus_scaledI"]
    print(f"{nm:15s} J-const med {c1['median']:+.3f} {[round(v,3) for v in c1['ci95']]} mean {c1['mean']:+.3f} p {c1['p_holm']:.2g} | "
          f"J-I med {c2['median']:+.3f} {[round(v,3) for v in c2['ci95']]} p {c2['p_holm']:.2g} | rho {r['spearman_sJ_absd']:.2f} "
          f"{[round(v,2) for v in r['spearman_ci95']]} info={r['informative_beyond_scale']}")
