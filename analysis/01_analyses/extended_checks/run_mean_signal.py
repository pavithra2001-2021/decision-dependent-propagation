"""Direct calibration, power benchmark, discovery-cohort replication, mean-signal regression and region-specific simulation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
NA = HERE.parent


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "\\textminus{}")


def fci(ci, nd=2):
    return "[" + ", ".join(fmt(v, nd) for v in ci) + "]"


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def crossfit(dd, v, fl):
    out = np.zeros(len(dd))
    for tr, te in fl:
        out[te] = (np.mean(dd[tr] ** 2 / v[tr]) * v[te]) if v is not None else np.mean(dd[tr] ** 2)
    return out


def transfer_block(dec, fl):
    d0, sJ0, sI0 = dec["classifier"]
    n = len(d0)
    cJ, cI = np.zeros(n), np.zeros(n)
    for tr, te in fl:
        cJ[te] = np.mean(d0[tr] ** 2 / (2 * sJ0[tr] ** 2)); cI[te] = np.mean(d0[tr] ** 2 / (2 * sI0[tr] ** 2))
    out = {}
    for t in ("global_mean", "random"):
        d, sJ, sI = dec[t]
        V = {"transfer_J": cJ * 2 * sJ ** 2, "transfer_I": cI * 2 * sI ** 2, "direct_constant": crossfit(d, None, fl),
             "direct_J": crossfit(d, 2 * sJ ** 2, fl), "direct_I": crossfit(d, 2 * sI ** 2, fl)}
        NL = {k: K.nlpd(np.abs(d), v) for k, v in V.items()}
        r = {k: {"coverage95": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(v))), "vcr": float(v.mean() / np.mean(d ** 2)),
                 "mean_nlpd": float(NL[k].mean())} for k, v in V.items()}
        r["tJ_minus_tI"] = K.paired(NL["transfer_J"] - NL["transfer_I"])
        r["tJ_minus_dConst"] = K.paired(NL["transfer_J"] - NL["direct_constant"]) | {"mean": float((NL["transfer_J"] - NL["direct_constant"]).mean())}
        out[t] = r
    return out


QS = np.round(np.linspace(0.5, 1.0, 51), 2)


def aurc(score, f):
    o = np.argsort(-score, kind="stable")
    m = len(f)
    return float(np.mean([f[o[m - int(round(q * m)):]].mean() for q in QS]))


def stats(lA, d, s, fl):
    out = {"spearman": float(spearmanr(s, np.abs(d)).statistic)}
    g = K.nlpd(np.abs(d), crossfit(d, 2 * s ** 2, fl)) - K.nlpd(np.abs(d), crossfit(d, None, fl))
    out["nlpd_gain_mean"] = float(np.mean(g))
    if lA is not None:
        f = (np.sign(lA) != np.sign(lA + d)).astype(int)
        if 0 < f.sum() < len(f):
            sj, sm = norm.cdf(-np.abs(lA) / (np.sqrt(2) * s)), -np.abs(lA)
            out["dauc"] = float(roc_auc_score(f, sj) - roc_auc_score(f, sm))
            out["daurc"] = aurc(sj, f) - aurc(sm, f)
    return out


def benchmark(lA, d, s, fl, tag, nsim=2000):
    obs = stats(lA, d, s, fl)
    v = 2 * s ** 2
    cbar = np.mean(d ** 2 / v)
    r = {"observed": obs}
    for hi, h in enumerate(("H_rel", "H_const")):
        lst = []
        for k in range(nsim):
            rng = np.random.default_rng([K.SEED_C, 77, 100 + tag, k, hi])
            ds = rng.standard_normal(len(d)) * (np.sqrt(cbar * v) if h == "H_rel" else np.sqrt(np.mean(d ** 2)))
            lst.append(stats(lA, ds, s, fl))
        df = pd.DataFrame(lst)
        r[h] = {c: {"mean": float(df[c].mean()), "ci95": [float(x) for x in np.nanpercentile(df[c], [2.5, 97.5])]} for c in df.columns}
    inrel = r["H_rel"]["spearman"]["ci95"][0] <= obs["spearman"] <= r["H_rel"]["spearman"]["ci95"][1]
    inconst = r["H_const"]["spearman"]["ci95"][0] <= obs["spearman"] <= r["H_const"]["spearman"]["ci95"][1]
    r["classification"] = ("underpowered" if inrel and inconst else "uninformative" if inconst else "informative" if inrel else "outside both")
    return r


res, mv = {"tcp": {}, "nyu2_msr": {}}, {}
A4 = pd.read_csv(NA / "split_halves/results/participants_tcp.csv.gz")
A6 = pd.read_csv(NA / "scan_length/results/participants_tcp.csv.gz")
A6 = A6[A6["T"] == 488].merge(A4[["sub", "y"]], on="sub")
assert len(A6) == len(A4)
for pe in ("AP", "PA"):
    if pe == "AP":
        T_ = A6
        dec = {"classifier": ((T_.lB_C - T_.lA_C).values, T_.sJ_C.values, T_.sIxdf_C.values),
               "global_mean": ((T_.lB_G - T_.lA_G).values, T_.sJ_G.values, T_.sIxdf_G.values),
               "random": ((T_.lB_Rd - T_.lA_Rd).values, T_.sJ_Rd.values, T_.sIxdf_Rd.values)}
        m4 = A4.set_index("sub").loc[T_["sub"]]
        assert np.allclose(dec["classifier"][0], m4.AP_dfull_C.values, rtol=1e-6, atol=1e-8)
        lA, sU, yv = T_.lA_C.values, T_.suHAC_C.values, T_.y.values
    else:
        T_ = A4[A4.PA_eligible.astype(bool)].reset_index(drop=True)
        dec = {"classifier": (T_.PA_dfull_C.values, T_.PA_sJ_C.values, T_.PA_sIxdf_C.values),
               "global_mean": (T_.PA_dfull_G.values, T_.PA_sJ_G.values, T_.PA_sIxdf_G.values),
               "random": (T_.PA_dfull_Rd.values, T_.PA_sJ_Rd.values, T_.PA_sIxdf_Rd.values)}
        lA, sU, yv = None, T_.PA_suHAC_C.values, T_.y.values
    fl = list(StratifiedKFold(10, shuffle=True, random_state=20261024).split(np.zeros((len(yv), 1)), yv))
    tr = transfer_block(dec, fl)
    d0, sJ0, _ = dec["classifier"]
    bJ = benchmark(lA, d0, sJ0, fl, 0 if pe == "AP" else 2)
    bU = benchmark(lA, d0, sU, fl, 1 if pe == "AP" else 3)
    res["tcp"][pe] = {"n": int(len(yv)), "transfer": tr, "benchmark_J": bJ, "benchmark_uHAC": bU}
    mv[f"Tc{pe}N"] = str(len(yv))
    for t, g in (("global_mean", "Gm"), ("random", "Rw")):
        r = tr[t]
        for m, s in (("transfer_J", "TJ"), ("transfer_I", "TI"), ("direct_constant", "DC")):
            mv[f"Tc{pe}{g}{s}Cov"] = f"{100 * r[m]['coverage95']:.0f}"
            mv[f"Tc{pe}{g}{s}Vcr"] = fmt(r[m]["vcr"], 2 if r[m]["vcr"] >= 0.01 else 3)
            mv[f"Tc{pe}{g}{s}Nl"] = fmt(r[m]["mean_nlpd"])
        mv[f"Tc{pe}{g}JmI"] = fmt(r["tJ_minus_tI"]["median"], 3)
        mv[f"Tc{pe}{g}JmICI"] = fci(r["tJ_minus_tI"]["ci95"], 3)
        mv[f"Tc{pe}{g}JmIP"] = pfmt(r["tJ_minus_tI"]["wilcoxon_p"])
    for b, s in ((bJ, "J"), (bU, "U")):
        mv[f"Tc{pe}Pw{s}Obs"] = fmt(b["observed"]["spearman"])
        mv[f"Tc{pe}Pw{s}Rel"] = fmt(b["H_rel"]["spearman"]["mean"])
        mv[f"Tc{pe}Pw{s}RelCI"] = fci(b["H_rel"]["spearman"]["ci95"])
        mv[f"Tc{pe}Pw{s}ConstCI"] = fci(b["H_const"]["spearman"]["ci95"])
        mv[f"Tc{pe}Pw{s}Class"] = b["classification"]
    print(pe, len(yv), {t: (round(tr[t]["transfer_J"]["coverage95"], 3), round(tr[t]["transfer_I"]["coverage95"], 3),
                            round(tr[t]["transfer_J"]["vcr"], 3), round(tr[t]["transfer_I"]["vcr"], 4)) for t in tr},
          bJ["classification"], round(bJ["observed"]["spearman"], 3), [round(x, 3) for x in bJ["H_rel"]["spearman"]["ci95"]], flush=True)

C = K.nyu2_cohort()
yy = (C.sex == "female").astype(int).values
ZA, _ = K.nyu2_point_all(C)
W1, c1, _f = K.d1_weights_cached(C, ZA, yy)
del ZA
E_ = W1.shape[1]
w2, w3 = K.fixed_weights(E_, K.SEED_C)
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")


def msr(X):
    A = np.column_stack([np.ones(len(X)), X.mean(1)])
    return X - A @ np.linalg.lstsq(A, X, rcond=None)[0]


rows = []
for k, pid in enumerate(C.pid):
    XA, XB = msr(K.nyu2_load(pid, 1)), msr(K.nyu2_load(pid, 2))
    zA, zB = K.point_z(XA), K.point_z(XB)
    Wm = np.column_stack([W1[k], w2, w3])
    b, Lr, ev = K.stream_cbb(XA, np.random.default_rng([K.SEED_C, 88, k]), Wm)
    sx = K.xdf_var(XA)
    rec = {"pid": pid, "b": b}
    for j, (D, cc) in enumerate((("D1", c1[k]), ("D2", 0.0), ("D3", 0.0))):
        w = Wm[:, j]
        rec |= {f"{D}_lA": float(zA @ w + cc), f"{D}_lB": float(zB @ w + cc), f"{D}_sJ": float(Lr[:, j].std(ddof=1)),
                f"{D}_sIcbb": float(np.sqrt(np.sum(w ** 2 * ev))), f"{D}_sIxdf": float(np.sqrt(np.sum(w ** 2 * sx)))}
    rows.append(rec)
G = pd.DataFrame(rows)
G.to_csv(OUT / "participants_mean_signal.csv.gz", index=False)
folds = K.nyu2_outer_folds(yy)
out = {"kappa_median": {}, "kappa_median_original": {}}
for D in ("D1", "D2", "D3"):
    out["kappa_median"][D] = float(np.median((G[f"{D}_sJ"] / G[f"{D}_sIcbb"]) ** 2))
    g = P[P.D == D]
    out["kappa_median_original"][D] = float(np.median((g.sA_J_cbb / g.sA_I_cbb) ** 2))
dec = {nm: ((G[f"{D}_lB"] - G[f"{D}_lA"]).values, G[f"{D}_sJ"].values, G[f"{D}_sIxdf"].values)
       for D, nm in (("D1", "classifier"), ("D2", "global_mean"), ("D3", "random"))}
out["transfer"] = transfer_block(dec, folds)
d1o = (P[P.D == "D1"].lB - P[P.D == "D1"].lA).values
out["classifier_change_sd_ratio_msr_over_original"] = float(np.std(dec["classifier"][0], ddof=1) / np.std(d1o, ddof=1))
res["nyu2_msr"] = out
for D, g in (("D1", "Lr"), ("D2", "Gm"), ("D3", "Rw")):
    v = out["kappa_median"][D]
    mv[f"Ms{g}K"] = fmt(v, 2 if v < 10 else 1) if v < 100 else f"{v:.0f}"
for t, g in (("global_mean", "Gm"), ("random", "Rw")):
    r = out["transfer"][t]
    mv |= {f"Ms{g}TJCov": f"{100 * r['transfer_J']['coverage95']:.0f}", f"Ms{g}TICov": f"{100 * r['transfer_I']['coverage95']:.0f}",
           f"Ms{g}TJVcr": fmt(r["transfer_J"]["vcr"]), f"Ms{g}TIVcr": fmt(r["transfer_I"]["vcr"], 2 if r["transfer_I"]["vcr"] >= 0.01 else 3)}
res["macro_values"] = mv
K.jdump(res, OUT / "summary_mean_signal.json")
print("kappa msr", out["kappa_median"], "original", out["kappa_median_original"])
print({t: {m: (round(out["transfer"][t][m]["coverage95"], 3), round(out["transfer"][t][m]["vcr"], 3)) for m in ("transfer_J", "transfer_I", "direct_constant")}
       for t in ("global_mean", "random")})
