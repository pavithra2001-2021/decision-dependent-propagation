"""Transfer of a calibration scale across occasions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DER = HERE.parent / "second_session" / "derived"
S2 = pd.read_csv(HERE.parent / "second_session/results/participants.csv")
S2 = S2[S2.eligible].reset_index(drop=True)
S2["pid"] = [f"sub-{int(p):07d}" for p in S2.pid]
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
C = K.nyu2_cohort()
y = (C.sex == "female").astype(int).values
ZA, _ = K.nyu2_point_all(C)
W1, c1, _f = K.d1_weights_cached(C, ZA, y)
del ZA
w2, w3 = K.fixed_weights(W1.shape[1], K.SEED_C)
pos = {pid: i for i, pid in enumerate(C.pid)}

rows = []
for _, r in S2.iterrows():
    p = r.pid
    k = pos[p]
    n = f"{p}_ses2_run1"
    X = sio.loadmat(DER / n / f"ROISignals_{n}.mat")["ROISignals"][:, :116].astype(np.float64)
    sx = K.xdf_var(X)
    rec = {"pid": p}
    for D, w in (("D1", W1[k]), ("D2", w2), ("D3", w3)):
        a = P[(P.pid == p) & (P.D == D)].iloc[0]
        assert abs(a.lA - r[f"{D}_l11"]) < 1e-8 * max(1, abs(a.lA))
        rec |= {f"{D}_d1": float(r[f"{D}_l12"] - r[f"{D}_l11"]), f"{D}_d2": float(r[f"{D}_l22"] - r[f"{D}_l21"]),
                f"{D}_sJ1": float(a.sA_J_cbb), f"{D}_sI1": float(a.sA_I_xdf),
                f"{D}_sJ2": float(r[f"{D}_sJ21"]), f"{D}_sI2": float(np.sqrt(np.sum(w ** 2 * sx)))}
    rows.append(rec)
T = pd.DataFrame(rows)
T.to_csv(OUT / "participants.csv", index=False)
others = sorted(set(P.pid) - set(T.pid))


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "\\textminus{}")


def fci(ci, nd=2):
    return "[" + ", ".join(fmt(v, nd) for v in ci) + "]"


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


res = {"n": int(len(T)), "n_other": len(others), "fits": {}}
mv = {"N": str(len(T)), "NOther": str(len(others))}
TAG = {"D1": "Lr", "D2": "Gm", "D3": "Rw"}
for fit in ("same", "other"):
    res["fits"][fit] = {}
    pj, pc = [], []
    for D in ("D1", "D2", "D3"):
        if fit == "same":
            d1, sJ1, sI1 = T[f"{D}_d1"].values, T[f"{D}_sJ1"].values, T[f"{D}_sI1"].values
        else:
            g = P[(P.D == D) & P.pid.isin(others)]
            d1, sJ1, sI1 = (g.lB - g.lA).values, g.sA_J_cbb.values, g.sA_I_xdf.values
        cJ = np.mean(d1 ** 2) / np.mean(2 * sJ1 ** 2)
        cI = np.mean(d1 ** 2) / np.mean(2 * sI1 ** 2)
        cJr = np.mean(d1 ** 2 / (2 * sJ1 ** 2))
        cIr = np.mean(d1 ** 2 / (2 * sI1 ** 2))
        const = np.mean(d1 ** 2)
        d2, sJ2, sI2 = T[f"{D}_d2"].values, T[f"{D}_sJ2"].values, T[f"{D}_sI2"].values
        V = {"transfer_J": cJ * 2 * sJ2 ** 2, "transfer_I": cI * 2 * sI2 ** 2, "transfer_constant": np.full(len(d2), const),
             "raw_J": 2 * sJ2 ** 2, "raw_I": 2 * sI2 ** 2,
             "transfer_J_meanratio": cJr * 2 * sJ2 ** 2, "transfer_I_meanratio": cIr * 2 * sI2 ** 2}
        NL = {m: K.nlpd(np.abs(d2), v) for m, v in V.items()}
        r = {"scale_J": float(cJ), "scale_I": float(cI), "constant": float(const)}
        for m, v in V.items():
            r[m] = {"coverage95": float(np.mean(np.abs(d2) <= 1.96 * np.sqrt(v))), "vcr": float(v.mean() / np.mean(d2 ** 2)),
                    "mean_nlpd": float(NL[m].mean())}
        r["J_minus_I"] = K.paired(NL["transfer_J"] - NL["transfer_I"])
        r["J_minus_const"] = K.paired(NL["transfer_J"] - NL["transfer_constant"]) | {"mean": float((NL["transfer_J"] - NL["transfer_constant"]).mean())}
        pj.append(r["J_minus_I"]["wilcoxon_p"]); pc.append(r["J_minus_const"]["wilcoxon_p"])
        res["fits"][fit][D] = r
    for D, a, b in zip(("D1", "D2", "D3"), K.holm(pj), K.holm(pc)):
        res["fits"][fit][D]["J_minus_I"]["p_holm"] = float(a)
        res["fits"][fit][D]["J_minus_const"]["p_holm"] = float(b)
    F = "S" if fit == "same" else "O"
    for D in ("D1", "D2", "D3"):
        r, t = res["fits"][fit][D], TAG[D]
        for m, s in (("transfer_J", "J"), ("transfer_I", "I"), ("transfer_constant", "C"), ("raw_J", "RJ"), ("raw_I", "RI")):
            mv[f"{F}{t}{s}Cov"] = f"{100 * r[m]['coverage95']:.0f}"
            mv[f"{F}{t}{s}Vcr"] = fmt(r[m]["vcr"], 2 if r[m]["vcr"] >= 0.01 else 3)
            mv[f"{F}{t}{s}Nl"] = fmt(r[m]["mean_nlpd"])
        mv[f"{F}{t}JmI"] = fmt(r["J_minus_I"]["median"], 3); mv[f"{F}{t}JmICI"] = fci(r["J_minus_I"]["ci95"], 3)
        mv[f"{F}{t}JmIP"] = pfmt(r["J_minus_I"]["p_holm"])
        mv[f"{F}{t}JmC"] = fmt(r["J_minus_const"]["median"], 3); mv[f"{F}{t}JmCCI"] = fci(r["J_minus_const"]["ci95"], 3)
        mv[f"{F}{t}JmCP"] = pfmt(r["J_minus_const"]["p_holm"]); mv[f"{F}{t}JmCMean"] = fmt(r["J_minus_const"]["mean"], 3)
        mv[f"{F}{t}JmrCov"] = f"{100 * r['transfer_J_meanratio']['coverage95']:.0f}"
J = [res["fits"]["same"][D]["transfer_J"] for D in ("D1", "D2", "D3")]
I = [res["fits"]["same"][D]["transfer_I"] for D in ("D1", "D2", "D3")]
mv |= {"SJCovLo": f"{100 * min(x['coverage95'] for x in J):.0f}", "SJCovHi": f"{100 * max(x['coverage95'] for x in J):.0f}",
       "SICovLo": f"{100 * min(x['coverage95'] for x in I):.0f}", "SICovHi": f"{100 * max(x['coverage95'] for x in I):.0f}",
       "SJVcrLo": fmt(min(x["vcr"] for x in J)), "SJVcrHi": fmt(max(x["vcr"] for x in J)),
       "SIVcrLo": fmt(min(x["vcr"] for x in I)), "SIVcrHi": fmt(max(x["vcr"] for x in I))}
res["macro_values"] = mv
K.jdump(res, OUT / "summary.json")
for fit in ("same", "other"):
    for D in ("D1", "D2", "D3"):
        r = res["fits"][fit][D]
        print(fit, D, " | ".join(f"{m}: cov {r[m]['coverage95']:.2f} vcr {r[m]['vcr']:.3f} nl {r[m]['mean_nlpd']:.3f}"
                                  for m in ("transfer_J", "transfer_I", "transfer_constant", "raw_J")),
              f"| J-I {r['J_minus_I']['median']:.3f} p {r['J_minus_I']['p_holm']:.3g} | J-C {r['J_minus_const']['median']:.3f} mean {r['J_minus_const']['mean']:.3f} p {r['J_minus_const']['p_holm']:.3g}")
