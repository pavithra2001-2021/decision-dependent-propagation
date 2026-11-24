"""Domain-agnostic illustration with simulated sensor channels."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
KCH, T, PHI = 20, 200, 0.6
iu = np.triu_indices(KCH, 1)
E_ = len(iu[0])
master = np.random.default_rng(2026112302)
lam0 = master.uniform(0.4, 0.9, KCH)


def window(rng, cls=0):
    lam = lam0.copy()
    if cls == 1:
        lam[:5] += 0.25
    burn = 100
    g = rng.standard_normal(T + burn)
    e = rng.standard_normal((T + burn, KCH))
    x = np.zeros((T + burn, KCH))
    for t in range(1, T + burn):
        x[t] = PHI * x[t - 1] + lam * g[t] + e[t]
    return x[burn:]


w_rand = master.standard_normal(E_)
w_sum = np.ones(E_)
w_con = np.r_[np.ones(E_ // 2), -np.ones(E_ - E_ // 2)]
Xtr = [window(np.random.default_rng([2026112302, 1, i]), cls=i % 2) for i in range(400)]
Ztr = np.stack([K.point_z(X) for X in Xtr])
ytr = np.array([i % 2 for i in range(400)])
clf = LogisticRegression(C=1.0, max_iter=5000).fit(Ztr, ytr)
w_trn = clf.coef_[0]
Wm = np.column_stack([w / np.linalg.norm(w) for w in (w_rand, w_sum, w_con, w_trn)])
DN = ["random", "sum", "contrast", "trained"]

Zt = np.stack([K.point_z(window(np.random.default_rng([2026112302, 2, i]))) for i in range(4000)])
varz = Zt.var(0, ddof=1)
sd_true = (Zt @ Wm).std(0, ddof=1)
kappa_true = sd_true ** 2 / ((Wm ** 2).T @ varz)

rows = []
for i in range(300):
    X1 = window(np.random.default_rng([2026112302, 3, i]))
    X2 = window(np.random.default_rng([2026112302, 4, i]))
    b, Lr, ev = K.stream_cbb(X1, np.random.default_rng([2026112302, 5, i]), Wm)
    d = (K.point_z(X2) - K.point_z(X1)) @ Wm
    sJ = Lr.std(0, ddof=1)
    sI = np.sqrt((Wm ** 2).T @ ev)
    sU = np.array([K.decision_sigma(X1, Wm[:, j], b)["uhac_bartlett"] for j in range(4)])
    rows.append((d, sJ, sI, sU))
d = np.stack([r[0] for r in rows]); sJ = np.stack([r[1] for r in rows]); sI = np.stack([r[2] for r in rows]); sU = np.stack([r[3] for r in rows])
res = {"K": KCH, "T": T, "E": E_, "decisions": {}}
vJ, vI = 2 * sJ ** 2, 2 * sI ** 2
for j, nm in enumerate(DN):
    res["decisions"][nm] = {"kappa_true": float(kappa_true[j]), "kappa_hat_median": float(np.median((sJ[:, j] / sI[:, j]) ** 2)),
                            "sd_ratio_J_over_true_median": float(np.median(sJ[:, j]) / sd_true[j]),
                            "sd_ratio_uHAC_over_J_median": float(np.median(sU[:, j] / sJ[:, j])),
                            "vcr_joint_raw": float(vJ[:, j].mean() / np.mean(d[:, j] ** 2)),
                            "vcr_indep_raw": float(vI[:, j].mean() / np.mean(d[:, j] ** 2)),
                            "cov_joint_raw": float(np.mean(np.abs(d[:, j]) <= 1.96 * np.sqrt(vJ[:, j]))),
                            "cov_indep_raw": float(np.mean(np.abs(d[:, j]) <= 1.96 * np.sqrt(vI[:, j])))}
jt = DN.index("trained")
cJ, cI = np.zeros(300), np.zeros(300)
for tr, te in KFold(10, shuffle=True, random_state=2026112302).split(np.arange(300)):
    cJ[te] = np.mean(d[tr, jt] ** 2 / vJ[tr, jt]); cI[te] = np.mean(d[tr, jt] ** 2 / vI[tr, jt])
for j, nm in enumerate(DN):
    if nm == "trained":
        continue
    res["decisions"][nm]["transfer"] = {
        "cov_joint": float(np.mean(np.abs(d[:, j]) <= 1.96 * np.sqrt(cJ * vJ[:, j]))), "vcr_joint": float(np.mean(cJ * vJ[:, j]) / np.mean(d[:, j] ** 2)),
        "cov_indep": float(np.mean(np.abs(d[:, j]) <= 1.96 * np.sqrt(cI * vI[:, j]))), "vcr_indep": float(np.mean(cI * vI[:, j]) / np.mean(d[:, j] ** 2))}
res["median_scale_joint"], res["median_scale_indep"] = float(np.median(cJ)), float(np.median(cI))
res["classifier_train_accuracy"] = float(clf.score(Ztr, ytr))
K.jdump(res, OUT / "summary_demo.json")
for nm, r in res["decisions"].items():
    print(nm, {k: (round(v, 3) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in r.items()})
print("scales", res["median_scale_joint"], res["median_scale_indep"], "acc", res["classifier_train_accuracy"])
