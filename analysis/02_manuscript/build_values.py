"""Numerical values (LaTeX macros) of the manuscript from the result files."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = REV.parent
NA = REV / "01_analyses"
sys.path.insert(0, str(ROOT / "src"))
from p6 import manuscript_assets as MA

f, ci, sci, pct = MA.f, MA.ci, MA.sci, MA.pct
V, PROV = {}, []


def put(name, val, src, field):
    assert name not in V, name
    V[name] = val
    PROV.append({"macro": name, "value": val, "source_file": src, "field": field})


def j(rel):
    return json.loads((REV / rel).read_text())


def pfmt(p):
    return sci(p)


for k, v in MA.values().items():
    put(k, v, "results (original; p6.manuscript_assets.values)", k)

a3p = "01_analyses/model_families/results/summary.json"
A3 = j(a3p)
ag = A3["age_summary"]
put("nAgeMin", f(ag["min"], 1), a3p, "age_summary.min"); put("nAgeMax", f(ag["max"], 1), a3p, "age_summary.max")
put("nAgeMed", f(ag["median"], 1), a3p, "age_summary.median"); put("nAgeUnder", str(ag["n_under_18"]), a3p, "age_summary.n_under_18")
put("nAgeN", str(A3["n_age"]), a3p, "n_age")
rep = max(A3["reproduction_max_rel_diff_M1_vs_original_D1"].values())
put("nReproDOne", sci(rep), a3p, "max(reproduction_max_rel_diff_M1_vs_original_D1)")
hyp = pd.DataFrame(A3["hyperparameters"])
put("nRidgeBoundary", str(int((hyp.M3_alpha == 1.0).sum())), a3p, "count(hyperparameters.M3_alpha == 1)")
for tag, key in (("Lr", "M1_ridge_logistic_sex"), ("Svm", "M2_linear_svm_sex"), ("Age", "M3_ridge_age"), ("Rbf", "M4_rbf_svm_sex")):
    o = A3[key]
    nl = o["nlpd_Jcbb_minus_Ixdf"]
    put(f"n{tag}Nlpd", f(nl["median"]), a3p, f"{key}.nlpd_Jcbb_minus_Ixdf.median")
    put(f"n{tag}NlpdCI", ci(*nl["ci95"]), a3p, f"{key}.nlpd_Jcbb_minus_Ixdf.ci95")
    put(f"n{tag}Frac", pct(nl["frac_negative"]), a3p, f"{key}.nlpd_Jcbb_minus_Ixdf.frac_negative")
    put(f"n{tag}K", str(int(round(nl["frac_negative"] * nl["n"]))), a3p, f"{key}.nlpd frac_negative*n")
    if "p_holm" in nl:
        put(f"n{tag}P", pfmt(nl["p_holm"]), a3p, f"{key}.nlpd_Jcbb_minus_Ixdf.p_holm")
    k = o["kappa_median_iqr"]
    put(f"n{tag}Kappa", f(k[0], 1), a3p, f"{key}.kappa_median_iqr[0]"); put(f"n{tag}KappaIQR", ci(k[1], k[2], 1), a3p, f"{key}.kappa_median_iqr[1:3]")
    put(f"n{tag}SdRatio", f(o["sd_ratio_Jcbb_over_Ixdf_median"]), a3p, f"{key}.sd_ratio_Jcbb_over_Ixdf_median")
    put(f"n{tag}VcrJ", f(o["vcr"]["J_cbb"]), a3p, f"{key}.vcr.J_cbb"); put(f"n{tag}VcrI", f(o["vcr"]["I_xdf"], 3), a3p, f"{key}.vcr.I_xdf")
    put(f"n{tag}VcrU", f(o["vcr"]["uHAC"]), a3p, f"{key}.vcr.uHAC")
    put(f"n{tag}CovJ", pct(o["coverage95"]["J_cbb"]), a3p, f"{key}.coverage95.J_cbb"); put(f"n{tag}CovI", pct(o["coverage95"]["I_xdf"]), a3p, f"{key}.coverage95.I_xdf")
    put(f"n{tag}UhacRatio", f(o["uhac_over_Jcbb_median"]), a3p, f"{key}.uhac_over_Jcbb_median")
    put(f"n{tag}Spear", f(o["spearman_sJ_absd"]), a3p, f"{key}.spearman_sJ_absd")
    un = o["nlpd_uHAC_minus_Jcbb"]
    put(f"n{tag}UhacNlpd", f(un["median"], 3), a3p, f"{key}.nlpd_uHAC_minus_Jcbb.median"); put(f"n{tag}UhacNlpdCI", ci(*un["ci95"], 3), a3p, f"{key}.nlpd_uHAC_minus_Jcbb.ci95")
    if "auc_runA" in o:
        put(f"n{tag}AucA", f(o["auc_runA"]), a3p, f"{key}.auc_runA"); put(f"n{tag}AucB", f(o["auc_runB"]), a3p, f"{key}.auc_runB")
        put(f"n{tag}Flip", pct(o["flip_rate"]), a3p, f"{key}.flip_rate")
        put(f"n{tag}BrJ", f(o["brier_J"], 3), a3p, f"{key}.brier_J"); put(f"n{tag}BrI", f(o["brier_I"], 3), a3p, f"{key}.brier_I")
        put(f"n{tag}BrDiff", f(o["brier_diff_J_minus_I"], 3), a3p, f"{key}.brier_diff_J_minus_I")
        put(f"n{tag}BrDiffCI", ci(*o["brier_diff_ci95"], 3), a3p, f"{key}.brier_diff_ci95")
o = A3["M3_ridge_age"]
put("nAgeR", f(o["pearson_r_runA"]), a3p, "M3_ridge_age.pearson_r_runA"); put("nAgeRB", f(o["pearson_r_runB"]), a3p, "M3_ridge_age.pearson_r_runB")
put("nAgeRsq", f(o["r2_runA"]), a3p, "M3_ridge_age.r2_runA"); put("nAgeMae", f(o["mae_runA"], 1), a3p, "M3_ridge_age.mae_runA")
put("nAgeMaeBase", f(o["mae_mean_predictor"], 1), a3p, "M3_ridge_age.mae_mean_predictor")
o = A3["M4_rbf_svm_sex"]
li = o["linearisation_ratio_nonlin_over_lin_median_iqr"]
put("nRbfLin", f(li[0]), a3p, "M4.linearisation_ratio[0]"); put("nRbfLinIQR", ci(li[1], li[2]), a3p, "M4.linearisation_ratio[1:3]")
put("nRbfVcrLin", f(o["vcr"]["J_cbb_lin"]), a3p, "M4.vcr.J_cbb_lin"); put("nRbfCovLin", pct(o["coverage95"]["J_cbb_lin"]), a3p, "M4.coverage95.J_cbb_lin")
put("nRbfCovU", pct(o["coverage95"]["uHAC"]), a3p, "M4.coverage95.uHAC")
ll = o["nlpd_Jcbblin_minus_Jcbb_nonlin"]
put("nRbfLinNlpd", f(ll["median"], 3), a3p, "M4.nlpd_Jcbblin_minus_Jcbb_nonlin.median"); put("nRbfLinNlpdCI", ci(*ll["ci95"], 3), a3p, "M4.nlpd_Jcbblin_minus_Jcbb_nonlin.ci95")

a3c = "01_analyses/model_families/results/participants.csv.gz"
P3 = pd.read_csv(REV / a3c)
kl = (P3.M4_s_J_cbb_lin / P3.M4_s_I_cbb) ** 2
put("nRbfKappaLin", f(float(kl.median()), 1), a3c, "median((M4_s_J_cbb_lin/M4_s_I_cbb)^2)")
put("nRbfKappaLinIQR", ci(float(kl.quantile(.25)), float(kl.quantile(.75)), 1), a3c, "IQR((M4_s_J_cbb_lin/M4_s_I_cbb)^2)")

a12 = "01_analyses/calibration/results/summary.json"
a12m = "01_analyses/calibration/results/metrics.csv"
S12 = j(a12)
M12 = pd.read_csv(REV / a12m)
names = {"C0_D1_RAW_Jcbb_minus_CONST": "Czero", "C1_D1_SCALE_Jcbb_minus_CONST": "Cone", "C2_D1_ADD_Jcbb_minus_CONST": "Ctwo",
         "C3_D1_SCALE_Jcbb_minus_SCALE_Ixdf": "Cthree", "C4_D1_SCALE_uHAC_minus_SCALE_Jcbb": "Cfour",
         "C5_D2_TSCALE_Jcbb_minus_TSCALE_Ixdf": "Cfive", "C6_D3_TSCALE_Jcbb_minus_TSCALE_Ixdf": "Csix"}
for k, nm in names.items():
    r = S12["family_holm"][k]
    d = 3 if abs(r["median"]) < 0.1 else 2
    put(f"n{nm}Med", f(r["median"], d), a12, f"family_holm.{k}.median"); put(f"n{nm}CI", ci(*r["ci95"], d), a12, f"family_holm.{k}.ci95")
    put(f"n{nm}P", pfmt(r["p_holm"]), a12, f"family_holm.{k}.p_holm"); put(f"n{nm}Frac", pct(r["frac_negative"]), a12, f"family_holm.{k}.frac_negative")
    put(f"n{nm}Mean", f(r["mean"], 3 if abs(r["mean"]) < 0.1 else 2), a12, f"family_holm.{k}.mean")
    put(f"n{nm}MeanCI", ci(*r["mean_ci95"], 3 if abs(r["mean"]) < 0.1 else 2), a12, f"family_holm.{k}.mean_ci95")


def mrow(D, kind, m):
    r = M12[(M12.D == D) & (M12.kind == kind) & (M12.method == m)]
    assert len(r) == 1, (D, kind, m)
    return r.iloc[0]


tagD = {"D1": "One", "D2": "Two", "D3": "Three"}
tagK = {"RAW": "Raw", "CONST": "Const", "SCALE": "Scale", "ADD": "Add", "TSCALE": "Tscale", "TADD": "Tadd", "CONF": "Conf", "CONFraw": "Confraw"}
tagM = {"J_cbb": "J", "I_xdf": "I", "uHAC": "U", "I_cbb": "Ic", "J_iid": "Ji", "-": ""}
for D in ("D1", "D2", "D3"):
    for kind in ("RAW", "CONST", "SCALE", "ADD", "TSCALE", "TADD", "CONF", "CONFraw"):
        for m in ("J_cbb", "I_xdf", "uHAC", "I_cbb", "J_iid", "-"):
            r = M12[(M12.D == D) & (M12.kind == kind) & (M12.method == m)]
            if len(r) != 1:
                continue
            r = r.iloc[0]
            base = f"n{tagK[kind]}{tagM[m]}{tagD[D]}"
            src = f"{D}/{kind}/{m}"
            put(base + "Cov", pct(r.coverage95), a12m, src + ".coverage95")
            if kind.startswith("CONF"):
                put(base + "Hw", f(r.median_halfwidth), a12m, src + ".median_halfwidth")
                continue
            put(base + "Nlpd", f(r.mean_nlpd), a12m, src + ".mean_nlpd")
            put(base + "Vcr", f(r.vcr), a12m, src + ".vcr")
            put(base + "Hw", f(r.median_halfwidth), a12m, src + ".median_halfwidth (1.96 sqrt v)")
            if D == "D1":
                put(base + "Brier", f(r.flip_brier, 3), a12m, src + ".flip_brier")
                put(base + "Slope", f(r.flip_cal_slope), a12m, src + ".flip_cal_slope")
                put(base + "MeanP", pct(r.flip_mean_p, 1), a12m, src + ".flip_mean_p")
                put(base + "Auc", f(r.flip_auc), a12m, src + ".flip_auc")
FCm = pd.DataFrame(S12["fitted_constants_median_over_folds"])
for m, t in (("J_cbb", "J"), ("I_xdf", "I"), ("uHAC", "U")):
    r = FCm[(FCm.D == "D1") & (FCm.method == m)].iloc[0]
    put(f"nScaleC{t}", f(r.scale_c, 1), a12, f"fitted_constants_median_over_folds[D1,{m}].scale_c")
    put(f"nConfQ{t}", f(r.conf_q_std), a12, f"fitted_constants_median_over_folds[D1,{m}].conf_q_std")

a2b = "01_analyses/transfer_models/results/summary.json"
S2b = j(a2b)
for key, t in (("linear_svm_sex", "Svm"), ("ridge_age", "Age"), ("rbf_svm_sex", "Rbf")):
    r = S2b["transfer"][key]
    put(f"nTr{t}CovJ", pct(r["J_cbb"]["coverage95"]), a2b, f"transfer.{key}.J_cbb.coverage95")
    put(f"nTr{t}CovI", pct(r["I_xdf"]["coverage95"]), a2b, f"transfer.{key}.I_xdf.coverage95")
    put(f"nTr{t}VcrJ", f(r["J_cbb"]["vcr"]), a2b, f"transfer.{key}.J_cbb.vcr")
    put(f"nTr{t}VcrI", f(r["I_xdf"]["vcr"]), a2b, f"transfer.{key}.I_xdf.vcr")
    nl = r["nlpd_Jcbb_minus_Ixdf"]
    put(f"nTr{t}Nlpd", f(nl["median"], 3), a2b, f"transfer.{key}.nlpd.median"); put(f"nTr{t}NlpdCI", ci(*nl["ci95"], 3), a2b, f"transfer.{key}.nlpd.ci95")
    put(f"nTr{t}P", pfmt(nl["p_holm"]), a2b, f"transfer.{key}.nlpd.p_holm")
for g, t in (("under18", "Minor"), ("adult", "Adult")):
    r = S2b["age_subgroups"][g]
    nl = r["nlpd_Jcbb_minus_Ixdf"]
    put(f"nSub{t}N", str(r["n"]), a2b, f"age_subgroups.{g}.n")
    put(f"nSub{t}Med", f(nl["median"]), a2b, f"age_subgroups.{g}.nlpd.median"); put(f"nSub{t}CI", ci(*nl["ci95"]), a2b, f"age_subgroups.{g}.nlpd.ci95")
    put(f"nSub{t}Frac", pct(nl["frac_negative"]), a2b, f"age_subgroups.{g}.nlpd.frac_negative")
    put(f"nSub{t}Vcr", f(r["vcr_Jcbb"]), a2b, f"age_subgroups.{g}.vcr_Jcbb"); put(f"nSub{t}Cov", pct(r["cov_Jcbb"]), a2b, f"age_subgroups.{g}.cov_Jcbb")
    put(f"nSub{t}Kappa", f(r["kappa_median"], 1), a2b, f"age_subgroups.{g}.kappa_median")
put("nSubP", f(S2b["age_subgroups"]["mannwhitney_p"]), a2b, "age_subgroups.mannwhitney_p")

a4n = "01_analyses/split_halves/results/summary_nyu2.json"
a4t = "01_analyses/split_halves/results/summary_tcp.json"
S4n, S4t = j(a4n), j(a4t)
for D in ("D1", "D2", "D3"):
    r = S4n[D]
    b = f"nHalf{tagD[D]}"
    put(b + "Within", f(r["vcr_within_J_cbb"]), a4n, f"{D}.vcr_within_J_cbb"); put(b + "WithinCI", ci(*r["vcr_within_J_cbb_ci95"]), a4n, f"{D}.vcr_within_J_cbb_ci95")
    put(b + "Across", f(r["vcr_across_J_cbb"]), a4n, f"{D}.vcr_across_J_cbb"); put(b + "AcrossCI", ci(*r["vcr_across_J_cbb_ci95"]), a4n, f"{D}.vcr_across_J_cbb_ci95")
    put(b + "Ratio", f(r["ratio_across_over_within"]), a4n, f"{D}.ratio_across_over_within"); put(b + "RatioCI", ci(*r["ratio_ci95"]), a4n, f"{D}.ratio_ci95")
    put(b + "CovW", pct(r["cov_within_J_cbb"]), a4n, f"{D}.cov_within_J_cbb"); put(b + "CovA", pct(r["cov_across_J_cbb"]), a4n, f"{D}.cov_across_J_cbb")
    put(b + "WithinI", f(r["vcr_within_I_xdf"], 3), a4n, f"{D}.vcr_within_I_xdf")
put("nHalfN", str(S4n["n"]), a4n, "n")
put("nReproTcp", sci(S4t["reproduction_lA_max_rel"]), a4t, "reproduction_lA_max_rel")
for pe in ("AP", "PA"):
    put(f"nTcp{pe}HalfN", str(S4t[pe]["n"]), a4t, f"{pe}.n")
    for nm, D in (("C", "One"), ("G", "Two"), ("Rd", "Three")):
        r = S4t[pe][nm]
        b = f"nTcp{pe}Half{D}"
        put(b + "Within", f(r["vcr_within_J_cbb"]), a4t, f"{pe}.{nm}.vcr_within_J_cbb"); put(b + "WithinCI", ci(*r["vcr_within_J_cbb_ci95"]), a4t, f"{pe}.{nm}.vcr_within_J_cbb_ci95")
        put(b + "Across", f(r["vcr_across_J_cbb"], 3 if r["vcr_across_J_cbb"] < 0.1 else 2), a4t, f"{pe}.{nm}.vcr_across_J_cbb")
        put(b + "AcrossCI", ci(*r["vcr_across_J_cbb_ci95"], 3 if r["vcr_across_J_cbb"] < 0.1 else 2), a4t, f"{pe}.{nm}.vcr_across_J_cbb_ci95")
        put(b + "Ratio", f(r["ratio_across_over_within"], 1 if r["ratio_across_over_within"] > 10 else 2), a4t, f"{pe}.{nm}.ratio_across_over_within")
        put(b + "RatioCI", ci(*r["ratio_ci95"], 1 if r["ratio_across_over_within"] > 10 else 2), a4t, f"{pe}.{nm}.ratio_ci95")
fc = S4t["full_length_consistency"]
put("nConsN", str(fc["n"]), a4t, "full_length_consistency.n")
for nm, D in (("C", "One"), ("G", "Two"), ("Rd", "Three")):
    put(f"nCons{D}", f(fc[nm]["spearman_absdAP_absdPA"]), a4t, f"full_length_consistency.{nm}.spearman")
    put(f"nCons{D}CI", ci(*fc[nm]["ci95"]), a4t, f"full_length_consistency.{nm}.ci95")

a6 = "01_analyses/scan_length/results/summary.json"
S6 = j(a6)
for coh, ct in (("NYU2", "Nyu"), ("TCP_AP", "Tcp")):
    if coh not in S6:
        continue
    for D in ("D1", "D2", "D3"):
        for T, r in S6[coh][D].items():
            if T == "decomposition":
                continue
            b = f"nLen{ct}{tagD[D]}T{ {'60': 'a', '100': 'b', '170': 'c', '150': 'a', '300': 'b', '488': 'c'}[T] }".replace(" ", "")
            put(b + "Vcr", f(r["vcr_J_cbb"]), a6, f"{coh}.{D}.{T}.vcr_J_cbb"); put(b + "VcrU", f(r["vcr_uHAC"]), a6, f"{coh}.{D}.{T}.vcr_uHAC")
            put(b + "VcrI", f(r["vcr_I_xdf"], 4 if r["vcr_I_xdf"] < 0.01 else 3), a6, f"{coh}.{D}.{T}.vcr_I_xdf")
            put(b + "Cov", pct(r["cov_J_cbb"]), a6, f"{coh}.{D}.{T}.cov_J_cbb"); put(b + "CovI", pct(r["cov_I_xdf"]), a6, f"{coh}.{D}.{T}.cov_I_xdf")
            put(b + "Kappa", f(r["kappa_median"], 0 if r["kappa_median"] > 100 else (1 if r["kappa_median"] > 5 else 2)), a6, f"{coh}.{D}.{T}.kappa_median")
            nl = r["nlpd_Jcbb_minus_Ixdf"]
            dd = 0 if abs(nl["median"]) > 100 else (2 if abs(nl["median"]) > 0.1 else 3)
            put(b + "Nlpd", f(nl["median"], dd), a6, f"{coh}.{D}.{T}.nlpd.median"); put(b + "NlpdCI", ci(*nl["ci95"], dd), a6, f"{coh}.{D}.{T}.nlpd.ci95")
            put(b + "Frac", pct(nl["frac_negative"]), a6, f"{coh}.{D}.{T}.nlpd.frac_negative")
        dc = S6[coh][D]["decomposition"]
        b = f"nLen{ct}{tagD[D]}"
        put(b + "TauShare", pct(dc["tau2_share_of_d2_at_Tmax"]), a6, f"{coh}.{D}.decomposition.tau2_share_of_d2_at_Tmax")
        put(b + "Tau", f(dc["d2_tau2"], 3 if dc["d2_tau2"] < 0.1 else 2), a6, f"{coh}.{D}.decomposition.d2_tau2")
        put(b + "TauCI", ci(*dc["d2_tau2_ci95"], 3 if dc["d2_tau2"] < 0.1 else 2), a6, f"{coh}.{D}.decomposition.d2_tau2_ci95")
        put(b + "SjInt", f(dc["sJ_intercept"], 4 if abs(dc["sJ_intercept"]) < 0.01 else 3), a6, f"{coh}.{D}.decomposition.sJ_intercept")
        put(b + "SjIntCI", ci(*dc["sJ_intercept_ci95"], 4 if abs(dc["sJ_intercept"]) < 0.01 else 3), a6, f"{coh}.{D}.decomposition.sJ_intercept_ci95")

a7 = "01_analyses/selective_prediction/results/summary.json"
S7 = j(a7)
for key, t in (("NYU2_ridge_logistic", "Lr"), ("NYU2_linear_svm", "Svm"), ("NYU2_rbf_svm", "Rbf"), ("TCP_AP_ridge_logistic", "Ap"),
               ("TCP_PA_ridge_logistic", "Pa")):
    r = S7[key]
    b = f"nSel{t}"
    put(b + "AucM", f(r["margin"]["auc"]), a7, f"{key}.margin.auc"); put(b + "AucJ", f(r["joint"]["auc"]), a7, f"{key}.joint.auc")
    put(b + "AucI", f(r["independent"]["auc"]), a7, f"{key}.independent.auc")
    put(b + "AurcM", f(r["margin"]["aurc"], 3), a7, f"{key}.margin.aurc"); put(b + "AurcJ", f(r["joint"]["aurc"], 3), a7, f"{key}.joint.aurc")
    dd = r["aurc_joint_minus_margin"]
    put(b + "dAurc", f(dd["diff"], 3), a7, f"{key}.aurc_joint_minus_margin.diff"); put(b + "dAurcCI", ci(*dd["ci95"], 3), a7, f"{key}.aurc_joint_minus_margin.ci95")
    put(b + "dAurcP", pfmt(dd["p_holm"]), a7, f"{key}.aurc_joint_minus_margin.p_holm")
    da = r["auc_joint_minus_margin"]
    put(b + "dAuc", f(da["diff"], 3), a7, f"{key}.auc_joint_minus_margin.diff"); put(b + "dAucCI", ci(*da["ci95"], 3), a7, f"{key}.auc_joint_minus_margin.ci95")
    put(b + "Flip", pct(r["flip_rate"]), a7, f"{key}.flip_rate")
    put(b + "RiskM", pct(r["margin"]["risk_at_80"], 1), a7, f"{key}.margin.risk_at_80"); put(b + "RiskJ", pct(r["joint"]["risk_at_80"], 1), a7, f"{key}.joint.risk_at_80")

a8 = "01_analyses/explicit_covariance/results/summary.json"
S8 = j(a8)["participants"]
rel = max(p[f"{D}_rel_diff"] for p in S8 for D in ("D1", "D2", "D3"))
put("nSigRel", sci(max(rel, 1e-16)), a8, "max over participants and decisions of *_rel_diff")
put("nSigMB", f(S8[0]["sigma_bytes"] / 1e6, 0), a8, "sigma_bytes/1e6")
put("nSigT", f(float(np.median([p["t_form_sigma_s"] for p in S8])), 2), a8, "median t_form_sigma_s")
tu = [p[f"{D}_t_uHAC_s"] for p in S8 for D in ("D1", "D2", "D3")]
put("nSigTu", f(1000 * float(np.median(tu)), 1), a8, "median *_t_uHAC_s * 1000 (ms)")
put("nSigLamLo", pct(min(p["lambda1_over_E"] for p in S8)), a8, "min lambda1_over_E (%)")
put("nSigLamHi", pct(max(p["lambda1_over_E"] for p in S8)), a8, "max lambda1_over_E (%)")
put("nSigEigLo", str(min(p["n_eig_90pct"] for p in S8)), a8, "min n_eig_90pct"); put("nSigEigHi", str(max(p["n_eig_90pct"] for p in S8)), a8, "max n_eig_90pct")
put("nSigRank", str(max(p["rank_numerical"] for p in S8)), a8, "max rank_numerical")
for D in ("D1", "D2", "D3"):
    kh = [p[f"{D}_kappa_HAC"] for p in S8]
    kb = [p[f"{D}_kappa_boot_original"] for p in S8]
    dd = 0 if min(kh) > 100 else (1 if min(kh) > 5 else 2)
    put(f"nSigK{tagD[D]}", ", ".join(f(x, dd) for x in kh), a8, f"{D}_kappa_HAC (3 participants)")
    put(f"nSigKb{tagD[D]}", ", ".join(f(x, dd) for x in kb), a8, f"{D}_kappa_boot_original (3 participants)")

lines = ["% generated by analysis/02_manuscript/build_values.py -- do not edit by hand"]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in V.items()]
(HERE / "values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
with open(HERE / "macro_provenance.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["macro", "value", "source_file", "field"])
    w.writeheader(); w.writerows(PROV)

S = json.loads((ROOT / "results" / "decision_simulation.json").read_text())
cols = ["cbb_joint", "uhac_bartlett", "iid_joint", "cbb_diag", "xdf_diag", "analytic_diag"]
rows = []
for wn, wl in (("dense_random", "random dense"), ("global_mean", "global mean")):
    for fa in (0.0, 0.6):
        for phi in (0.0, 0.5, 0.8):
            r = S[f"phi={phi}|factor={fa}|w={wn}"]
            rows.append(f"{wl} & {fa:.1f} & {phi:.1f} & " + " & ".join(f"{r[c]:.2f}" for c in cols) + r" \\")
(HERE / "tab_sim.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
B = json.loads((ROOT / "results" / "edge_simulation.json").read_text())
nm = {"white": "white", "ar_mod": "AR moderate", "ar_high": "AR high", "bandpass": "band-pass", "bp_het": "band-pass + AR"}
rows = []
for c in B["cells"]:
    if c.get("N") != 40:
        continue
    m = c["methods"]
    rows.append(f"{nm[c['condition']]} & {c['T']} & " + " & ".join(
        f"{m[k]['median_se_ratio']:.2f} ({100 * m[k]['coverage95']:.0f})" for k in ("naive", "iid", "cbb_roi")) + r" \\")
(HERE / "tab_b1.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(len(V), "macros;", sum(1 for k in V if k.startswith("n")), "new")
