"""Direct calibration, power benchmark, discovery-cohort replication, mean-signal regression and region-specific simulation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
NA = HERE.parent
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")
A3 = pd.read_csv(NA / "model_families/results/participants.csv.gz")
M2 = pd.read_csv(NA / "cpm_transfer/results/participants.csv.gz")
D1 = P[P.D == "D1"].reset_index(drop=True)
assert (A3.pid.values == D1.pid.values).all() and (M2.pid.values == D1.pid.values).all()
y = D1.y.values
folds = K.nyu2_outer_folds(y)
n = len(y)

DEC = {}
for D, nm in (("D1", "classifier"), ("D2", "global_mean"), ("D3", "random")):
    g = P[P.D == D].reset_index(drop=True)
    DEC[nm] = (g.lA.values, g.lB.values, g.sA_J_cbb.values, g.sA_I_xdf.values, g.sA_uHAC.values)
for M, nm in (("M2", "linear_svm"), ("M3", "ridge_age"), ("M4", "rbf_svm")):
    DEC[nm] = (A3[f"{M}_lA"].values, A3[f"{M}_lB"].values, A3[f"{M}_s_J_cbb"].values, A3[f"{M}_s_I_xdf"].values, A3[f"{M}_s_uHAC"].values)
for d in ("cpm_sex", "pos_strength", "cpm_age"):
    DEC[d] = (M2[f"{d}_lA"].values, M2[f"{d}_lB"].values, M2[f"{d}_s_J_cbb"].values, M2[f"{d}_s_I_xdf"].values, M2[f"{d}_s_uHAC"].values)
TARGETS = [k for k in DEC if k != "classifier"]
TRAINED = ["linear_svm", "ridge_age", "rbf_svm", "cpm_sex", "pos_strength", "cpm_age"]


def crossfit(dd, v, fl=folds):
    out = np.zeros(len(dd))
    for tr, te in fl:
        out[te] = (np.mean(dd[tr] ** 2 / v[tr]) * v[te]) if v is not None else np.mean(dd[tr] ** 2)
    return out


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "\\textminus{}")


def fci(ci, nd=2):
    return "[" + ", ".join(fmt(v, nd) for v in ci) + "]"


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


res = {"A": {}, "B": {}, "E": {}}
mv = {}
TAG = {"global_mean": "Gm", "random": "Rw", "linear_svm": "Sv", "ridge_age": "Ra", "rbf_svm": "Rb", "cpm_sex": "Cs",
       "pos_strength": "Cp", "cpm_age": "Ca", "classifier": "Lr"}

lA0, lB0, sJ0, sI0, _ = DEC["classifier"]
d0 = lB0 - lA0
Cs = {m: np.zeros(n) for m in ("J", "I")}
for tr, te in folds:
    Cs["J"][te] = np.mean(d0[tr] ** 2 / (2 * sJ0[tr] ** 2))
    Cs["I"][te] = np.mean(d0[tr] ** 2 / (2 * sI0[tr] ** 2))
pA1, pA2 = [], []
for t in TARGETS:
    lA, lB, sJ, sI, _ = DEC[t]
    d = lB - lA
    V = {"transfer_J": Cs["J"] * 2 * sJ ** 2, "transfer_I": Cs["I"] * 2 * sI ** 2, "direct_constant": crossfit(d, None),
         "direct_J": crossfit(d, 2 * sJ ** 2), "direct_I": crossfit(d, 2 * sI ** 2)}
    NL = {k: K.nlpd(np.abs(d), v) for k, v in V.items()}
    r = {k: {"coverage95": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(v))), "vcr": float(v.mean() / np.mean(d ** 2)),
             "mean_nlpd": float(NL[k].mean()), "median_nlpd": float(np.median(NL[k]))} for k, v in V.items()}
    r["tJ_minus_dConst"] = K.paired(NL["transfer_J"] - NL["direct_constant"]) | {"mean": float((NL["transfer_J"] - NL["direct_constant"]).mean())}
    r["tJ_minus_dJ"] = K.paired(NL["transfer_J"] - NL["direct_J"]) | {"mean": float((NL["transfer_J"] - NL["direct_J"]).mean())}
    r["tJ_minus_tI"] = K.paired(NL["transfer_J"] - NL["transfer_I"])
    pA1.append(r["tJ_minus_dConst"]["wilcoxon_p"]); pA2.append(r["tJ_minus_dJ"]["wilcoxon_p"])
    res["A"][t] = r
for t, a, b in zip(TARGETS, K.holm(pA1), K.holm(pA2)):
    res["A"][t]["tJ_minus_dConst"]["p_holm"] = float(a)
    res["A"][t]["tJ_minus_dJ"]["p_holm"] = float(b)
sub = {}
for m in ("transfer_J", "transfer_I", "direct_constant"):
    cov = [res["A"][t][m]["coverage95"] for t in TRAINED]
    vcr = [res["A"][t][m]["vcr"] for t in TRAINED]
    sub[m] = {"coverage_range": [min(cov), max(cov)], "vcr_range": [min(vcr), max(vcr)], "vcr_spread": max(vcr) / min(vcr)}
sub["n_trained_nlpd_favours_joint"] = int(sum(res["A"][t]["tJ_minus_tI"]["median"] < 0 and res["A"][t]["tJ_minus_tI"]["wilcoxon_p"] < 0.05 for t in TRAINED))
sub["n_trained_nlpd_favours_indep"] = int(sum(res["A"][t]["tJ_minus_tI"]["median"] > 0 and res["A"][t]["tJ_minus_tI"]["wilcoxon_p"] < 0.05 for t in TRAINED))
res["A"]["trained_subset"] = sub
n_dc_better = sum(res["A"][t]["tJ_minus_dConst"]["mean"] > 0 for t in TARGETS)
res["A"]["n_targets_direct_constant_better_mean"] = int(n_dc_better)
mv |= {"TrSubCovJLo": f"{100 * sub['transfer_J']['coverage_range'][0]:.0f}", "TrSubCovJHi": f"{100 * sub['transfer_J']['coverage_range'][1]:.0f}",
       "TrSubCovILo": f"{100 * sub['transfer_I']['coverage_range'][0]:.0f}", "TrSubCovIHi": f"{100 * sub['transfer_I']['coverage_range'][1]:.0f}",
       "TrSubVcrJLo": fmt(sub["transfer_J"]["vcr_range"][0]), "TrSubVcrJHi": fmt(sub["transfer_J"]["vcr_range"][1]),
       "TrSubVcrILo": fmt(sub["transfer_I"]["vcr_range"][0]), "TrSubVcrIHi": fmt(sub["transfer_I"]["vcr_range"][1]),
       "TrSubSpreadJ": fmt(sub["transfer_J"]["vcr_spread"], 1), "TrSubSpreadI": fmt(sub["transfer_I"]["vcr_spread"], 1),
       "TrSubNJoint": str(sub["n_trained_nlpd_favours_joint"]), "TrSubNIndep": str(sub["n_trained_nlpd_favours_indep"]),
       "DcNBetter": str(n_dc_better), "DcN": str(len(TARGETS))}
for t in TARGETS:
    r, g = res["A"][t], TAG[t]
    for m, s in (("transfer_J", "TJ"), ("transfer_I", "TI"), ("direct_constant", "DC"), ("direct_J", "DJ"), ("direct_I", "DI")):
        mv[f"Dc{g}{s}Cov"] = f"{100 * r[m]['coverage95']:.0f}"
        mv[f"Dc{g}{s}Vcr"] = fmt(r[m]["vcr"])
        mv[f"Dc{g}{s}Nl"] = fmt(r[m]["mean_nlpd"])
    mv[f"Dc{g}Diff"] = fmt(r["tJ_minus_dConst"]["mean"], 3)
    mv[f"Dc{g}DiffMed"] = fmt(r["tJ_minus_dConst"]["median"], 3)
    mv[f"Dc{g}DiffP"] = pfmt(r["tJ_minus_dConst"]["p_holm"])

SEX = {"classifier", "linear_svm", "rbf_svm"}
QS = np.round(np.linspace(0.5, 1.0, 51), 2)


def aurc(score, f):
    o = np.argsort(-score, kind="stable")
    m = len(f)
    return float(np.mean([f[o[m - int(round(q * m)):]].mean() for q in QS]))


def stats(lA, d, s):
    out = {"spearman": float(spearmanr(s, np.abs(d)).statistic)}
    g = K.nlpd(np.abs(d), crossfit(d, 2 * s ** 2)) - K.nlpd(np.abs(d), crossfit(d, None))
    out["nlpd_gain_median"], out["nlpd_gain_mean"] = float(np.median(g)), float(np.mean(g))
    if lA is not None:
        f = (np.sign(lA) != np.sign(lA + d)).astype(int)
        if 0 < f.sum() < len(f):
            sj, sm = norm.cdf(-np.abs(lA) / (np.sqrt(2) * s)), -np.abs(lA)
            out["dauc"] = float(roc_auc_score(f, sj) - roc_auc_score(f, sm))
            out["daurc"] = aurc(sj, f) - aurc(sm, f)
    return out


NSIM = 2000
bench_rows = []
for di, (nm, (lA, lB, sJ, sI, sU)) in enumerate(DEC.items()):
    d = lB - lA
    res["B"][nm] = {}
    for meth, s in (("J_cbb", sJ), ("uHAC", sU)):
        obs = stats(lA if nm in SEX else None, d, s)
        v = 2 * s ** 2
        cbar = np.mean(d ** 2 / v)
        sims = {"H_rel": [], "H_const": []}
        for h in sims:
            for k in range(NSIM):
                rng = np.random.default_rng([K.SEED_C, 77, di, k, 0 if h == "H_rel" else 1, 0 if meth == "J_cbb" else 1])
                ds = rng.standard_normal(n) * (np.sqrt(cbar * v) if h == "H_rel" else np.sqrt(np.mean(d ** 2)))
                sims[h].append(stats(lA if nm in SEX else None, ds, s))
        r = {"observed": obs}
        for h, lst in sims.items():
            df = pd.DataFrame(lst)
            r[h] = {c: {"mean": float(df[c].mean()), "ci95": [float(x) for x in np.nanpercentile(df[c], [2.5, 97.5])]} for c in df.columns}
        inrel = r["H_rel"]["spearman"]["ci95"][0] <= obs["spearman"] <= r["H_rel"]["spearman"]["ci95"][1]
        inconst = r["H_const"]["spearman"]["ci95"][0] <= obs["spearman"] <= r["H_const"]["spearman"]["ci95"][1]
        r["classification"] = ("too little between-individual spread to detect" if inrel and inconst else
                               "no individual information, adequate power" if inconst else
                               "informative" if inrel else "outside both")
        r["sd_log_s"] = float(np.std(np.log(s), ddof=1))
        res["B"][nm][meth] = r
        bench_rows.append({"decision": nm, "method": meth, "obs_spearman": obs["spearman"],
                           "Hrel_spearman_mean": r["H_rel"]["spearman"]["mean"], "Hrel_lo": r["H_rel"]["spearman"]["ci95"][0],
                           "Hrel_hi": r["H_rel"]["spearman"]["ci95"][1], "Hconst_lo": r["H_const"]["spearman"]["ci95"][0],
                           "Hconst_hi": r["H_const"]["spearman"]["ci95"][1], "obs_gain_mean": obs["nlpd_gain_mean"],
                           "Hrel_gain_mean": r["H_rel"]["nlpd_gain_mean"]["mean"], "sd_log_s": r["sd_log_s"], "class": r["classification"]})
    print(nm, {m: res["B"][nm][m]["classification"] for m in ("J_cbb", "uHAC")}, flush=True)
pd.DataFrame(bench_rows).to_csv(OUT / "power_benchmark.csv", index=False)
CLS = {"too little between-individual spread to detect": "underpowered", "no individual information, adequate power": "uninformative",
       "informative": "informative", "outside both": "outside both"}
for nm in DEC:
    g = TAG[nm]
    for meth, s in (("J_cbb", "J"), ("uHAC", "U")):
        r = res["B"][nm][meth]
        mv |= {f"Pw{g}{s}Obs": fmt(r["observed"]["spearman"]), f"Pw{g}{s}Rel": fmt(r["H_rel"]["spearman"]["mean"]),
               f"Pw{g}{s}RelCI": fci(r["H_rel"]["spearman"]["ci95"]), f"Pw{g}{s}ConstCI": fci(r["H_const"]["spearman"]["ci95"]),
               f"Pw{g}{s}GainObs": fmt(r["observed"]["nlpd_gain_mean"], 3), f"Pw{g}{s}GainRel": fmt(r["H_rel"]["nlpd_gain_mean"]["mean"], 3),
               f"Pw{g}{s}Class": CLS[r["classification"]], f"Pw{g}{s}SdLog": fmt(r["sd_log_s"])}
        if "dauc" in r["observed"]:
            mv |= {f"Pw{g}{s}DaucObs": fmt(r["observed"]["dauc"], 3), f"Pw{g}{s}DaucRel": fmt(r["H_rel"]["dauc"]["mean"], 3),
                   f"Pw{g}{s}DaucRelCI": fci(r["H_rel"]["dauc"]["ci95"], 3), f"Pw{g}{s}DaurcObs": fmt(r["observed"]["daurc"], 4),
                   f"Pw{g}{s}DaurcRel": fmt(r["H_rel"]["daurc"]["mean"], 4), f"Pw{g}{s}DaurcRelCI": fci(r["H_rel"]["daurc"]["ci95"], 4)}
for meth, s in (("J_cbb", "J"), ("uHAC", "U")):
    cl = [res["B"][nm][meth]["classification"] for nm in DEC]
    mv[f"PwN{s}Under"] = str(sum(c.startswith("too little") for c in cl))
    mv[f"PwN{s}Uninf"] = str(sum(c.startswith("no individual") for c in cl))
    mv[f"PwN{s}Inf"] = str(sum(c == "informative" for c in cl))
    mv[f"PwN{s}Out"] = str(sum(c == "outside both" for c in cl))

pe = []
for nm, (lA, lB, *_rest) in DEC.items():
    d = lB - lA
    z = d.mean() / d.std(ddof=1)
    rng = np.random.default_rng(K.SEED_C)
    bo = []
    for _ in range(10000):
        i = rng.integers(0, n, n)
        bo.append(d[i].mean() / d[i].std(ddof=1))
    res["E"].setdefault("mean_shift", {})[nm] = {"mean_over_sd": float(z), "ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])],
                                                 "wilcoxon_p": float(wilcoxon(d).pvalue)}
    pe.append(res["E"]["mean_shift"][nm]["wilcoxon_p"])
for nm, a in zip(DEC, K.holm(pe)):
    res["E"]["mean_shift"][nm]["p_holm"] = float(a)
    mv[f"Shift{TAG[nm]}"] = fmt(res["E"]["mean_shift"][nm]["mean_over_sd"])
    mv[f"Shift{TAG[nm]}CI"] = fci(res["E"]["mean_shift"][nm]["ci95"])
    mv[f"Shift{TAG[nm]}P"] = pfmt(a)
ms = res["E"]["mean_shift"]
mv["ShiftMaxAbs"] = fmt(max(abs(v["mean_over_sd"]) for v in ms.values()))
mv["ShiftNSig"] = str(sum(v["p_holm"] < 0.05 for v in ms.values()))
A4 = pd.read_csv(NA / "split_halves/results/participants_tcp.csv.gz")
BA = pd.read_csv(K.RESULTS / "discovery_decisions_AP.csv.gz"); BA = BA[BA.model == "ridge_nested"]
BP = pd.read_csv(K.RESULTS / "discovery_decisions_PA.csv.gz"); BP = BP[BP.model == "ridge_nested"]
res["E"]["tcp_sizes"] = {"AP_pairs": int(len(BA)), "PA_pairs": int(len(BP)), "both_sets_overlap": int(len(set(BA["sub"]) & set(BP["sub"]))),
                         "A4_rows_AP_cohort": int(len(A4)), "A4_PA_eligible_within_AP_cohort": int(A4.PA_eligible.sum())}
t = res["E"]["tcp_sizes"]
mv |= {"TcpAP": str(t["AP_pairs"]), "TcpPA": str(t["PA_pairs"]), "TcpBoth": str(t["both_sets_overlap"]), "TcpPAinAP": str(t["A4_PA_eligible_within_AP_cohort"])}
g1 = P[P.D == "D1"]
res["E"]["classifier_ratios"] = {"median_kappa_consistent": float(np.median((g1.sA_J_cbb / g1.sA_I_cbb) ** 2)),
                                 "median_var_ratio_Jcbb_Ixdf": float(np.median((g1.sA_J_cbb / g1.sA_I_xdf) ** 2))}
mv |= {"KapLr": fmt(res["E"]["classifier_ratios"]["median_kappa_consistent"], 1), "VrLr": fmt(res["E"]["classifier_ratios"]["median_var_ratio_Jcbb_Ixdf"], 1)}

res["macro_values"] = mv
K.jdump(res, OUT / "summary_direct_power_discovery.json")
print(json.dumps({k: v for k, v in mv.items() if k.startswith(("TrSub", "PwN", "Shift", "Tcp", "Kap", "Vr", "DcN"))}, indent=0))
for t in TARGETS:
    r = res["A"][t]
    print(f"{t:13s} tJ cov {r['transfer_J']['coverage95']:.2f} nl {r['transfer_J']['mean_nlpd']:.3f} | dC cov {r['direct_constant']['coverage95']:.2f} "
          f"nl {r['direct_constant']['mean_nlpd']:.3f} | dJ nl {r['direct_J']['mean_nlpd']:.3f} | dI nl {r['direct_I']['mean_nlpd']:.3f} | "
          f"tJ-dC mean {r['tJ_minus_dConst']['mean']:+.3f} med {r['tJ_minus_dConst']['median']:+.3f} p {r['tJ_minus_dConst']['p_holm']:.3g}")
print(pd.DataFrame(bench_rows).round(3).to_string())
