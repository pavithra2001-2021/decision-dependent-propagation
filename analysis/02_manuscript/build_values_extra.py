"""Numerical values of the additional analyses (LaTeX macros) and table rows."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = REV.parent
NA = "01_analyses"
V, PROV = {}, []


def f(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "\\textminus{}") if (x < 0 and float(s) != 0) else s.replace("-", "")


def ci(a, b, d=2):
    return f"[{f(a, d)}, {f(b, d)}]"


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


def pv(p):
    return "<0.001" if p < 0.001 else ("1.00" if p >= 0.995 else f"{p:.3f}")


def put(name, val, src, field):
    assert name not in V, name
    V[name] = val
    PROV.append({"macro": name, "value": val, "source_file": src, "field": field})


def j(rel):
    return json.loads((REV / rel).read_text())


def dd(x):
    return 0 if abs(x) >= 100 else (1 if abs(x) >= 10 else 2)


C = json.loads((ROOT / "results/confirmation_results.json").read_text())
arch = "results/confirmation_results.json"
put("rPrimP", pv(C["PRIMARY"]["wilcoxon_p"]), arch, "PRIMARY.wilcoxon_p (formatted)")
for k, t in (("S1_I_cbb", "Icbb"), ("S2_J_iidm", "Jiidm"), ("S3_J_iid", "Jiid"), ("S6_D2", "Glob"), ("S6_D3", "Rand"), ("S11_I_xdf_adaptive", "Ixdfa")):
    put(f"rSec{t}P", pv(C["secondary"][k]["p_holm"]), arch, f"secondary.{k}.p_holm (formatted)")
P = pd.read_csv(ROOT / "results/confirmation_participants.csv.gz")
pp = "results/confirmation_participants.csv.gz"
D1 = P[P.D == "D1"]
for col, t in (("sA_I_xdf", "Ixdf"), ("sA_I_xdf_adaptive", "Ixdfa"), ("sA_I_cbb", "Icbb"), ("sA_J_iidm", "Jiidm"), ("sA_J_iid", "Jiid")):
    r = float(np.median((D1.sA_J_cbb / D1[col]) ** 2))
    put(f"rVr{t}", f(r, dd(r)), pp, f"median((sA_J_cbb/{col})^2), D1")
for D, t in (("D2", "Glob"), ("D3", "Rand")):
    g = P[P.D == D]
    r = float(np.median((g.sA_J_cbb / g.sA_I_xdf) ** 2))
    put(f"rVr{t}", f(r, dd(r)), pp, f"median((sA_J_cbb/sA_I_xdf)^2), {D}")
put("rBrDiffP", pv(C["S9_flips"]["brier_J_cbb_minus_I_xdf"]["p_two_sided"]), arch, "S9_flips.brier_J_cbb_minus_I_xdf.p_two_sided")

m2p = f"{NA}/cpm_transfer/results/summary.json"
M2 = j(m2p)
ns = pd.DataFrame(M2["network_sizes"])
put("rCpmSexPos", f"{ns.sex_pos.median():.0f}", m2p, "median network_sizes.sex_pos")
put("rCpmSexNeg", f"{ns.sex_neg.median():.0f}", m2p, "median network_sizes.sex_neg")
put("rCpmAgePos", f"{ns.age_pos.median():.0f}", m2p, "median network_sizes.age_pos")
put("rCpmAgeNeg", f"{ns.age_neg.median():.0f}", m2p, "median network_sizes.age_neg")
for key, t in (("cpm_sex", "Cs"), ("pos_strength", "Cp"), ("cpm_age", "Ca")):
    o = M2[key]
    if "auc_runA" in o:
        put(f"r{t}AucA", f(o["auc_runA"]), m2p, f"{key}.auc_runA"); put(f"r{t}AucB", f(o["auc_runB"]), m2p, f"{key}.auc_runB")
    if "pearson_r_runA" in o:
        put(f"r{t}R", f(o["pearson_r_runA"]), m2p, f"{key}.pearson_r_runA"); put(f"r{t}Mae", f(o["mae_runA"], 1), m2p, f"{key}.mae_runA")
    k = o["kappa_median_iqr"]
    put(f"r{t}Kappa", f(k[0], 1), m2p, f"{key}.kappa_median_iqr[0]"); put(f"r{t}KappaIQR", ci(k[1], k[2], 1), m2p, f"{key}.kappa_median_iqr[1:3]")
    put(f"r{t}SdRatio", f(o["sd_ratio_Jcbb_over_Ixdf_median"]), m2p, f"{key}.sd_ratio")
    put(f"r{t}UhacRatio", f(o["uhac_over_Jcbb_median"]), m2p, f"{key}.uhac_over_Jcbb_median")
    nl = o["nlpd_Jcbb_minus_Ixdf"]
    put(f"r{t}Nlpd", f(nl["median"]), m2p, f"{key}.nlpd.median"); put(f"r{t}NlpdCI", ci(*nl["ci95"]), m2p, f"{key}.nlpd.ci95")
    put(f"r{t}Frac", pct(nl["frac_negative"]), m2p, f"{key}.nlpd.frac_negative")
    put(f"r{t}CovJ", pct(o["coverage95"]["J_cbb"]), m2p, f"{key}.coverage95.J_cbb"); put(f"r{t}CovI", pct(o["coverage95"]["I_xdf"]), m2p, f"{key}.coverage95.I_xdf")
    put(f"r{t}VcrJ", f(o["vcr"]["J_cbb"]), m2p, f"{key}.vcr.J_cbb"); put(f"r{t}VcrI", f(o["vcr"]["I_xdf"], 3), m2p, f"{key}.vcr.I_xdf")
    tr = o["transfer"]
    put(f"r{t}TrCovJ", pct(tr["J_cbb"]["coverage95"]), m2p, f"{key}.transfer.J_cbb.coverage95")
    put(f"r{t}TrCovI", pct(tr["I_xdf"]["coverage95"]), m2p, f"{key}.transfer.I_xdf.coverage95")
    put(f"r{t}TrVcrJ", f(tr["J_cbb"]["vcr"]), m2p, f"{key}.transfer.J_cbb.vcr"); put(f"r{t}TrVcrI", f(tr["I_xdf"]["vcr"]), m2p, f"{key}.transfer.I_xdf.vcr")
    tn = o["transfer_nlpd_Jcbb_minus_Ixdf"]
    put(f"r{t}TrNlpd", f(tn["median"], 3), m2p, f"{key}.transfer_nlpd.median"); put(f"r{t}TrNlpdCI", ci(*tn["ci95"], 3), m2p, f"{key}.transfer_nlpd.ci95")
    put(f"r{t}TrP", pv(tn["p_holm"]), m2p, f"{key}.transfer_nlpd.p_holm")
put("rScaleJ", f(M2["median_scale_from_classifier"]["J_cbb"], 1), m2p, "median_scale_from_classifier.J_cbb")
put("rScaleI", f(M2["median_scale_from_classifier"]["I_xdf"], 1), m2p, "median_scale_from_classifier.I_xdf")

m2b = f"{NA}/within_decision_information/results/summary.json"
B = j(m2b)
TAG = {"ridge_logistic": "Lr", "global_mean": "Gm", "random": "Rw", "linear_svm": "Sv", "ridge_age": "Ag", "rbf_svm": "Rb",
       "cpm_sex": "Cs", "pos_strength": "Cp", "cpm_age": "Ca"}
for key, t in TAG.items():
    r = B[key]
    a = r["scaledJ_minus_constant"]
    put(f"rIn{t}Med", f(a["median"], 3), m2b, f"{key}.scaledJ_minus_constant.median"); put(f"rIn{t}CI", ci(*a["ci95"], 3), m2b, f"{key}.scaledJ_minus_constant.ci95")
    put(f"rIn{t}P", pv(a["p_holm"]), m2b, f"{key}.scaledJ_minus_constant.p_holm"); put(f"rIn{t}Mean", f(a["mean"], 3), m2b, f"{key}.scaledJ_minus_constant.mean")
    b = r["scaledJ_minus_scaledI"]
    put(f"rJi{t}Med", f(b["median"], 3), m2b, f"{key}.scaledJ_minus_scaledI.median"); put(f"rJi{t}CI", ci(*b["ci95"], 3), m2b, f"{key}.scaledJ_minus_scaledI.ci95")
    put(f"rJi{t}P", pv(b["p_holm"]), m2b, f"{key}.scaledJ_minus_scaledI.p_holm")
    put(f"rSp{t}", f(r["spearman_sJ_absd"]), m2b, f"{key}.spearman_sJ_absd"); put(f"rSp{t}CI", ci(*r["spearman_ci95"]), m2b, f"{key}.spearman_ci95")

m3 = f"{NA}/simulation_confirmation_regime/results/summary.json"
S3 = j(m3)
for cname, ct in (("C1_ar05_nofilter", "A"), ("C2_white_bandpass", "B"), ("C3_ar05_bandpass", "C")):
    for nm, dt in (("random", "Rw"), ("global_mean", "Gm"), ("trained", "Tr")):
        r = S3["conditions"][cname]["decisions"][nm]
        e = r["estimates"]
        put(f"rSim{ct}{dt}Kt", f(r["kappa_true"], dd(r["kappa_true"])), m3, f"{cname}.{nm}.kappa_true")
        put(f"rSim{ct}{dt}Kr", f(e["kappa"]["mean_ratio"]), m3, f"{cname}.{nm}.kappa.mean_ratio")
        put(f"rSim{ct}{dt}KrSe", f(e["kappa"]["se"]), m3, f"{cname}.{nm}.kappa.se")
        put(f"rSim{ct}{dt}J", f(e["J_cbb"]["mean_sd_ratio"]), m3, f"{cname}.{nm}.J_cbb.mean_sd_ratio")
        put(f"rSim{ct}{dt}JSe", f(e["J_cbb"]["se"]), m3, f"{cname}.{nm}.J_cbb.se")
        put(f"rSim{ct}{dt}U", f(e["uHAC"]["mean_sd_ratio"]), m3, f"{cname}.{nm}.uHAC.mean_sd_ratio")
        put(f"rSim{ct}{dt}USe", f(e["uHAC"]["se"]), m3, f"{cname}.{nm}.uHAC.se")
        put(f"rSim{ct}{dt}Ic", f(e["I_cbb"]["mean_sd_ratio"]), m3, f"{cname}.{nm}.I_cbb.mean_sd_ratio")
        put(f"rSim{ct}{dt}Ix", f(e["I_xdf"]["mean_sd_ratio"]), m3, f"{cname}.{nm}.I_xdf.mean_sd_ratio")
        put(f"rSim{ct}{dt}JV", f(e["J_cbb"]["mean_var_ratio"]), m3, f"{cname}.{nm}.J_cbb.mean_var_ratio")
allK = [S3["conditions"][c]["decisions"][d]["estimates"]["kappa"]["mean_ratio"] for c in S3["conditions"] for d in ("random", "global_mean", "trained")]
put("rSimKrLo", f(min(allK)), m3, "min kappa mean_ratio over conditions and decisions")
put("rSimKrHi", f(max(allK)), m3, "max kappa mean_ratio over conditions and decisions")
bpJ = [S3["conditions"][c]["decisions"][d]["estimates"]["J_cbb"]["mean_sd_ratio"] for c in ("C2_white_bandpass", "C3_ar05_bandpass") for d in ("random", "global_mean", "trained")]
bpU = [S3["conditions"][c]["decisions"][d]["estimates"]["uHAC"]["mean_sd_ratio"] for c in ("C2_white_bandpass", "C3_ar05_bandpass") for d in ("random", "global_mean", "trained")]
bpV = [S3["conditions"][c]["decisions"][d]["estimates"]["J_cbb"]["mean_var_ratio"] for c in ("C2_white_bandpass", "C3_ar05_bandpass") for d in ("random", "global_mean", "trained")]
put("rSimBpJLo", f(min(bpJ)), m3, "min J_cbb mean_sd_ratio, band-pass conditions"); put("rSimBpJHi", f(max(bpJ)), m3, "max J_cbb mean_sd_ratio, band-pass")
put("rSimBpULo", f(min(bpU)), m3, "min uHAC mean_sd_ratio, band-pass"); put("rSimBpUHi", f(max(bpU)), m3, "max uHAC mean_sd_ratio, band-pass")
put("rSimBpVLo", f(min(bpV)), m3, "min J_cbb mean_var_ratio, band-pass"); put("rSimBpVHi", f(max(bpV)), m3, "max J_cbb mean_var_ratio, band-pass")
put("rSimFreq", str(S3["mask_frequencies"]), m3, "mask_frequencies")
for nm, dt in (("random", "Rw"), ("trained", "Tr"), ("global_mean", "Gm")):
    kt = [S3["conditions"][c]["decisions"][nm]["kappa_true"] for c in S3["conditions"]]
    put(f"rSimKt{dt}Lo", f(min(kt), dd(min(kt))), m3, f"min kappa_true over conditions, {nm}")
    put(f"rSimKt{dt}Hi", f(max(kt), dd(max(kt))), m3, f"max kappa_true over conditions, {nm}")
seK = [S3["conditions"][c]["decisions"][d]["estimates"]["kappa"]["se"] for c in S3["conditions"] for d in ("random", "global_mean", "trained")]
put("rSimKrSeMax", f(max(seK), 3), m3, "max Monte Carlo SE of the kappa ratio")
for d, t in (("D1", "Lr"), ("D2", "Gm"), ("D3", "Rw")):
    r = S3["real_data_mc_se"][d]
    put(f"rMcK{t}", pct(r["median_rel_se_kappa"]), m3, f"real_data_mc_se.{d}.median_rel_se_kappa")
    put(f"rMcS{t}", f"{100 * r['median_rel_se_sJ']:.1f}", m3, f"real_data_mc_se.{d}.median_rel_se_sJ")

m7 = f"{NA}/uhac_tuning/results/summary.json"
S7 = j(m7)
cnt = S7["vhat_negative_counts"]
cells = cnt["original_cells"] + sum(v["cells"] for k, v in cnt.items() if isinstance(v, dict))
zeros = cnt["original_confirmation_runA"] + cnt["original_confirmation_runB"] + sum(v["zero"] for k, v in cnt.items() if isinstance(v, dict))
put("rVnegCells", f"{cells:,}", m7, "total cells over all tables"); put("rVnegZero", str(zeros), m7, "total zero uHAC SDs")
sens = S7["sensitivity"]
for D, t in (("D1", "Lr"), ("D2", "Gm"), ("D3", "Rw")):
    s = sens[D]
    for m, mt in (("b/2", "h"), ("b", "o"), ("2b", "t"), ("4b", "q")):
        e = s[m]
        put(f"rBw{t}{mt}KJ", f(e["kappa_cbb_median"], dd(e["kappa_cbb_median"])), m7, f"sensitivity.{D}.{m}.kappa_cbb_median")
        put(f"rBw{t}{mt}KU", f(e["kappa_hac_median"], dd(e["kappa_hac_median"])), m7, f"sensitivity.{D}.{m}.kappa_hac_median")
        put(f"rBw{t}{mt}N", f(e["nlpd_Jcbb_minus_Ixdf"]["median"]), m7, f"sensitivity.{D}.{m}.nlpd_Jcbb_minus_Ixdf.median")
        put(f"rBw{t}{mt}UJ", f(e["median_uhac_over_Jcbb_default"]), m7, f"sensitivity.{D}.{m}.median_uhac_over_Jcbb_default")
        put(f"rBw{t}{mt}VJ", f(e["vcr_Jcbb"]), m7, f"sensitivity.{D}.{m}.vcr_Jcbb")
        put(f"rBw{t}{mt}VU", f(e["vcr_uhac"]), m7, f"sensitivity.{D}.{m}.vcr_uhac")
    e = s["andrews"]
    put(f"rBw{t}aKU", f(e["kappa_hac_median"], dd(e["kappa_hac_median"])), m7, f"sensitivity.{D}.andrews.kappa_hac_median")
    put(f"rBw{t}aUJ", f(e["median_uhac_over_Jcbb_default"]), m7, f"sensitivity.{D}.andrews.median_uhac_over_Jcbb_default")
    put(f"rBw{t}aBw", f(s["andrews_bw_median_iqr"][0], 1), m7, f"sensitivity.{D}.andrews_bw_median_iqr[0]")
    put(f"rBw{t}B", f(s["block_length_b_median"], 1), m7, f"sensitivity.{D}.block_length_b_median")
nls = [sens["D1"][m]["nlpd_Jcbb_minus_Ixdf"]["median"] for m in ("b/2", "b", "2b", "4b")]
put("rBwPrimLo", f(min(nls)), m7, "min primary-endpoint median over block lengths"); put("rBwPrimHi", f(max(nls)), m7, "max primary-endpoint median over block lengths")
tf = REV / NA / "uhac_tuning/results/summary_timing.json"
if tf.exists():
    T7 = json.loads(tf.read_text())
    tp = f"{NA}/uhac_tuning/results/summary_timing.json"
    for ds, t in (("nyu2", "N"), ("tcp", "T")):
        r = T7[ds]
        for k2, t2 in (("block_length_ms", "Bl"), ("uhac_end_to_end_ms", "U"), ("jcbb_end_to_end_ms", "J"), ("xdf_end_to_end_ms", "X"), ("uhac_decision_ms", "Ud")):
            put(f"rTm{t}{t2}", f(r[k2], 0 if r[k2] >= 10 else 1), tp, f"{ds}.{k2}")
        put(f"rTm{t}Sp", f(r["jcbb_end_to_end_ms"] / r["uhac_end_to_end_ms"], 1), tp, f"{ds}.jcbb/uhac end-to-end")

mm = f"{NA}/supplementary_checks/results/summary.json"
Sm = j(mm)
re_ = Sm["ridge_extended"]
put("rRidgeBound", str(re_["n_at_lower_boundary"]), mm, "ridge_extended.n_at_lower_boundary")
put("rRidgeR", f(re_["overall"]["r"]), mm, "ridge_extended.overall.r"); put("rRidgeMae", f(re_["overall"]["mae"], 1), mm, "ridge_extended.overall.mae")
put("rRidgeMinorR", f(re_["under18"]["r"]), mm, "ridge_extended.under18.r"); put("rRidgeMinorMae", f(re_["under18"]["mae"], 1), mm, "ridge_extended.under18.mae")
put("rRidgeMinorBase", f(re_["under18"]["mae_mean_predictor"], 1), mm, "ridge_extended.under18.mae_mean_predictor")
put("rRidgeAdultR", f(re_["adult"]["r"]), mm, "ridge_extended.adult.r")
pr = re_["propagation"]
put("rRidgeKappa", f(pr["kappa_median_iqr"][0], 1), mm, "ridge_extended.propagation.kappa[0]"); put("rRidgeKappaIQR", ci(*pr["kappa_median_iqr"][1:], 1), mm, "ridge_extended.propagation.kappa[1:3]")
put("rRidgeNlpd", f(pr["nlpd_Jcbb_minus_Ixdf"]["median"]), mm, "ridge_extended.propagation.nlpd.median"); put("rRidgeNlpdCI", ci(*pr["nlpd_Jcbb_minus_Ixdf"]["ci95"]), mm, "ridge_extended.propagation.nlpd.ci95")
put("rRidgeCovJ", pct(pr["coverage_J"]), mm, "ridge_extended.propagation.coverage_J"); put("rRidgeCovI", pct(pr["coverage_I"]), mm, "ridge_extended.propagation.coverage_I")
put("rRidgeTrCovJ", pct(pr["transfer_coverage_J"]), mm, "ridge_extended.propagation.transfer_coverage_J"); put("rRidgeTrCovI", pct(pr["transfer_coverage_I"]), mm, "ridge_extended.propagation.transfer_coverage_I")
put("rRidgeTrVcrJ", f(pr["transfer_vcr_J"]), mm, "ridge_extended.propagation.transfer_vcr_J"); put("rRidgeTrVcrI", f(pr["transfer_vcr_I"]), mm, "ridge_extended.propagation.transfer_vcr_I")
put("rRidgeUhac", f(pr["uhac_over_Jcbb_median"]), mm, "ridge_extended.propagation.uhac_over_Jcbb_median")
fl = Sm["fold_level"]
put("rFoldNeg", str(fl["folds_negative_median"]), mm, "fold_level.folds_negative_median")
put("rFoldLo", f(fl["range_median_nlpd_diff"][0]), mm, "fold_level.range_median_nlpd_diff[0]"); put("rFoldHi", f(fl["range_median_nlpd_diff"][1]), mm, "fold_level.range_median_nlpd_diff[1]")
put("rFoldLooLo", f(fl["range_loo_median"][0]), mm, "fold_level.range_loo_median[0]"); put("rFoldLooHi", f(fl["range_loo_median"][1]), mm, "fold_level.range_loo_median[1]")
put("rFoldKLo", f(fl["range_median_kappa"][0], 1), mm, "fold_level.range_median_kappa[0]"); put("rFoldKHi", f(fl["range_median_kappa"][1], 1), mm, "fold_level.range_median_kappa[1]")
_m8 = REV / NA / "fold_cluster/results/summary.json"
if _m8.exists():
    put("rrFcCI", ci(*json.loads(_m8.read_text())["fold_cluster_ci95"]), f"{NA}/fold_cluster/results/summary.json", "fold_cluster_ci95")
sp = Sm["spearman"]
put("rSpLrI", f(sp["ridge_logistic"]["I_xdf"]["rho"]), mm, "spearman.ridge_logistic.I_xdf.rho"); put("rSpLrICI", ci(*sp["ridge_logistic"]["I_xdf"]["ci95"]), mm, "spearman.ridge_logistic.I_xdf.ci95")
am = Sm["age_motion"]
for grp, t in (("verified_181", "V"), ("eligible_158", "E"), ("excluded_23", "X")):
    a = am[grp]
    for k2, t2 in (("under10", "Ten"), ("under12", "Twelve"), ("under18", "Eighteen")):
        put(f"rAge{t}{t2}", str(a[k2]), mm, f"age_motion.{grp}.{k2}")
    put(f"rAge{t}Med", f(a["median"], 1), mm, f"age_motion.{grp}.median")
for g_, t in (("<10", "Ten"), ("10-12", "Twelve"), ("12-18", "Teen"), (">=18", "Adult")):
    put(f"rExcl{t}", pct(am["exclusion_rate_by_group"][g_]), mm, f"age_motion.exclusion_rate_by_group[{g_}]")
    put(f"rFd{t}", f(am["mean_fd_by_group"][g_], 3), mm, f"age_motion.mean_fd_by_group[{g_}]")
dec = Sm["nlpd_decomposition"]
for key, t in (("ridge_logistic", "Lr"), ("global_mean", "Gm"), ("random", "Rw"), ("linear_svm", "Sv"), ("ridge_age_extended", "Ag"), ("rbf_svm", "Rb"),
               ("cpm_sex", "Cs"), ("pos_strength", "Cp"), ("cpm_age", "Ca")):
    r = dec[key]
    put(f"rDc{t}Lv", f(r["median_logvar_term"]), mm, f"nlpd_decomposition.{key}.median_logvar_term")
    put(f"rDc{t}Rs", f(r["median_residual_term"], dd(r["median_residual_term"])), mm, f"nlpd_decomposition.{key}.median_residual_term")
    put(f"rDc{t}Vr", f(r["median_variance_ratio"], dd(r["median_variance_ratio"])), mm, f"nlpd_decomposition.{key}.median_variance_ratio")
    if key in ("linear_svm", "rbf_svm", "ridge_age_extended"):
        put(f"rSp{t}J", f(sp[key]["J_cbb"]["rho"]), mm, f"spearman.{key}.J_cbb.rho"); put(f"rSp{t}JCI", ci(*sp[key]["J_cbb"]["ci95"]), mm, f"spearman.{key}.J_cbb.ci95")

gd = f"{NA}/sensor_demo/results/summary_demo.json"
G = j(gd)
for nm, t in (("random", "Rw"), ("sum", "Su"), ("contrast", "Co"), ("trained", "Tr")):
    r = G["decisions"][nm]
    put(f"rDm{t}Kt", f(r["kappa_true"], dd(r["kappa_true"])), gd, f"decisions.{nm}.kappa_true")
    put(f"rDm{t}Kh", f(r["kappa_hat_median"], dd(r["kappa_hat_median"])), gd, f"decisions.{nm}.kappa_hat_median")
    put(f"rDm{t}VJ", f(r["vcr_joint_raw"]), gd, f"decisions.{nm}.vcr_joint_raw"); put(f"rDm{t}VI", f(r["vcr_indep_raw"], 3), gd, f"decisions.{nm}.vcr_indep_raw")
    put(f"rDm{t}SJ", f(r["sd_ratio_J_over_true_median"]), gd, f"decisions.{nm}.sd_ratio_J_over_true_median")
    if "transfer" in r:
        tr = r["transfer"]
        put(f"rDm{t}TCJ", pct(tr["cov_joint"]), gd, f"decisions.{nm}.transfer.cov_joint"); put(f"rDm{t}TCI", pct(tr["cov_indep"]), gd, f"decisions.{nm}.transfer.cov_indep")
        put(f"rDm{t}TVJ", f(tr["vcr_joint"]), gd, f"decisions.{nm}.transfer.vcr_joint"); put(f"rDm{t}TVI", f(tr["vcr_indep"]), gd, f"decisions.{nm}.transfer.vcr_indep")
put("rDmE", str(G["E"]), gd, "E"); put("rDmK", str(G["K"]), gd, "K"); put("rDmT", str(G["T"]), gd, "T")

tcj = [G["decisions"][n]["transfer"]["cov_joint"] for n in ("random", "sum", "contrast")]
put("rDmTCJLo", pct(min(tcj)), gd, "min transfer cov_joint (random, sum, contrast)"); put("rDmTCJHi", pct(max(tcj)), gd, "max transfer cov_joint")

A2 = json.loads((REV / NA / "calibration/results/summary.json").read_text())
M12 = pd.read_csv(REV / NA / "calibration/results/metrics.csv")
A2b = j(f"{NA}/transfer_models/results/summary.json")
covJ = [M12[(M12.D == D) & (M12.kind == "TSCALE") & (M12.method == "J_cbb")].coverage95.iloc[0] for D in ("D2", "D3")]
covI = [M12[(M12.D == D) & (M12.kind == "TSCALE") & (M12.method == "I_xdf")].coverage95.iloc[0] for D in ("D2", "D3")]
vcrJ = [M12[(M12.D == D) & (M12.kind == "TSCALE") & (M12.method == "J_cbb")].vcr.iloc[0] for D in ("D2", "D3")]
vcrI = [M12[(M12.D == D) & (M12.kind == "TSCALE") & (M12.method == "I_xdf")].vcr.iloc[0] for D in ("D2", "D3")]
for key in ("linear_svm_sex", "rbf_svm_sex"):
    covJ.append(A2b["transfer"][key]["J_cbb"]["coverage95"]); covI.append(A2b["transfer"][key]["I_xdf"]["coverage95"])
    vcrJ.append(A2b["transfer"][key]["J_cbb"]["vcr"]); vcrI.append(A2b["transfer"][key]["I_xdf"]["vcr"])
covJ.append(pr["transfer_coverage_J"]); covI.append(pr["transfer_coverage_I"]); vcrJ.append(pr["transfer_vcr_J"]); vcrI.append(pr["transfer_vcr_I"])
for key in ("cpm_sex", "pos_strength", "cpm_age"):
    covJ.append(M2[key]["transfer"]["J_cbb"]["coverage95"]); covI.append(M2[key]["transfer"]["I_xdf"]["coverage95"])
    vcrJ.append(M2[key]["transfer"]["J_cbb"]["vcr"]); vcrI.append(M2[key]["transfer"]["I_xdf"]["vcr"])
src = "A1A2 metrics (D2, D3 TSCALE); A2b (linear SVM, RBF-SVM); minor (ridge extended); M2 (CPM x3)"
put("rTrN", str(len(covJ)), src, "number of target decisions")
put("rTrCovJLo", pct(min(covJ)), src, "min transferred J-CBB coverage"); put("rTrCovJHi", pct(max(covJ)), src, "max transferred J-CBB coverage")
put("rTrCovILo", pct(min(covI)), src, "min transferred I-xDF coverage"); put("rTrCovIHi", pct(max(covI)), src, "max transferred I-xDF coverage")
put("rTrVcrJLo", f(min(vcrJ)), src, "min transferred J-CBB VCR"); put("rTrVcrJHi", f(max(vcrJ)), src, "max transferred J-CBB VCR")
put("rTrVcrILo", f(min(vcrI), 3), src, "min transferred I-xDF VCR"); put("rTrVcrIHi", f(max(vcrI), 1), src, "max transferred I-xDF VCR")
put("rTrSpreadJ", f(max(vcrJ) / min(vcrJ), 1), src, "max/min transferred J-CBB VCR")
put("rTrSpreadI", f"{max(vcrI) / min(vcrI):,.0f}", src, "max/min transferred I-xDF VCR")
put("rCfiveP", pv(A2["family_holm"]["C5_D2_TSCALE_Jcbb_minus_TSCALE_Ixdf"]["p_holm"]), f"{NA}/calibration/results/summary.json", "family_holm.C5.p_holm")
put("rCsixP", pv(A2["family_holm"]["C6_D3_TSCALE_Jcbb_minus_TSCALE_Ixdf"]["p_holm"]), f"{NA}/calibration/results/summary.json", "family_holm.C6.p_holm")

s2 = REV / NA / "second_session/results/summary.json"
if s2.exists():
    S4 = json.loads(s2.read_text())
    sp4 = f"{NA}/second_session/results/summary.json"
    for k2, t2 in S4.get("macro_values", {}).items():
        put(f"rSes{k2}", t2, sp4, f"macro_values.{k2}")

for fn in ("summary_direct_power_discovery.json", "summary_region_specific.json", "summary_fixed_blocks.json", "summary_mean_signal.json"):
    fp = REV / NA / "extended_checks/results" / fn
    if fp.exists():
        S5 = json.loads(fp.read_text())
        for k2, t2 in S5.get("macro_values", {}).items():
            put(f"rr{k2}", t2, f"{NA}/extended_checks/results/{fn}", f"macro_values.{k2}")
_fo = REV / NA / "occasion_transfer/results/summary.json"
if _fo.exists():
    for k2, t2 in json.loads(_fo.read_text()).get("macro_values", {}).items():
        put(f"rOc{k2}", t2, f"{NA}/occasion_transfer/results/summary.json", f"macro_values.{k2}")
_fa = REV / NA / "extended_checks/results/summary_direct_power_discovery.json"
if _fa.exists():
    _ra = json.loads(_fa.read_text())["A"]["ridge_age"]["tJ_minus_tI"]
    _src = f"{NA}/extended_checks/results/summary_direct_power_discovery.json"
    put("rrRaTrNlpd", f(_ra["median"], 3), _src, "A.ridge_age.tJ_minus_tI.median")
    put("rrRaTrNlpdCI", ci(*_ra["ci95"], 3), _src, "A.ridge_age.tJ_minus_tI.ci95")
    put("rrRaTrNlpdP", pv(_ra["wilcoxon_p"]), _src, "A.ridge_age.tJ_minus_tI.wilcoxon_p (unadjusted)")
_a6 = REV / NA / "scan_length/results/summary_table.csv"
if _a6.exists():
    import pandas as _pd
    _t = _pd.read_csv(_a6)
    _dn = {"D1": "classifier", "D2": "global mean", "D3": "random weights"}
    _rows, _prev = [], None
    for _, r_ in _t.iterrows():
        coh = "NYU\\_2" if r_["cohort"] == "NYU2" else "TCP AP"
        lab = (coh if (r_["cohort"], r_["D"]) != _prev and (_prev is None or _prev[0] != r_["cohort"]) else "")
        dec = _dn[r_["D"]] if (r_["cohort"], r_["D"]) != _prev else ""
        _prev = (r_["cohort"], r_["D"])
        kap = r_["kappa_median"]
        kf = f"{kap:.0f}" if kap >= 100 else (f"{kap:.1f}" if kap >= 10 else f"{kap:.2f}")
        nl = lambda x: (f"{x:.0f}" if abs(x) >= 10 else f"{x:.2f}").replace("-", "\\textminus{}")
        vi = f"{r_['vcr_I_xdf']:.3f}" if r_["vcr_I_xdf"] >= 0.01 else f"{r_['vcr_I_xdf']:.4f}"
        _rows.append(f"{lab} & {dec} & {int(r_['T'])} & {r_['vcr_J_cbb']:.2f} & {r_['vcr_uHAC']:.2f} & {vi} & "
                     f"{100 * r_['cov_J_cbb']:.0f} & {kf} & {nl(r_['nlpd_median'])} [{nl(r_['nlpd_ci_lo'])}, {nl(r_['nlpd_ci_hi'])}]\\\\")
    (HERE / "tab_len_full.tex").write_text("% generated by build_values_extra.py from scan_length/results/summary_table.csv\n" + "\n".join(_rows) + "\n", encoding="utf-8")
_kr = []
for fn in ("summary_region_specific.json", "summary_fixed_blocks.json"):
    fp = REV / NA / "extended_checks/results" / fn
    if fp.exists():
        for mm in json.loads(fp.read_text())["pooled"].values():
            _kr += [d["kappa"]["mean_ratio"] for d in mm.values()]
if _kr:
    put("rrKallLo", f"{min(_kr):.2f}", f"{NA}/extended_checks/results/summary_region_specific.json + summary_fixed_blocks.json", "min pooled kappa ratio over settings")
    put("rrKallHi", f"{max(_kr):.2f}", f"{NA}/extended_checks/results/summary_region_specific.json + summary_fixed_blocks.json", "max pooled kappa ratio over settings")

for k in [k for k in list(V) if k.endswith("P") and not k.startswith("rrTcp") and (V[k].startswith("<") or V[k][:1].isdigit())]:
    put(k + "eq", V[k] if V[k].startswith("<") else "=" + V[k], next(r["source_file"] for r in PROV if r["macro"] == k), f"text form of {k}")
SES = ["NWith", "NFour", "NCand", "NProc", "N", "AgeMed", "AgeMin", "AgeMax", "Under", "Minutes", "NMin", "NDays", "NTask"]
SES += [f"{a}{t}{s}" for t in ("Lr", "Gm", "Rw") for a in ("Share", "Ratio", "VcrW", "VcrB", "Cons") for s in ("", "CI")]
SES += [f"Pos{a}{t}{s}" for t in ("Lr", "Gm", "Rw") for a in ("XRatio", "Share", "VcrE", "Stab") for s in ("", "CI")]
SES += [f"Pos{a}{t}" for t in ("Lr", "Gm", "Rw") for a in ("Shift", "ShiftP", "FracUp")]
for k in ["rSes" + s for s in SES]:
    if k not in V:
        put(k, r"\textbf{??}", "PENDING", "placeholder")
for k in [f"rTm{d}{x}" for d in ("N", "T") for x in ("Bl", "U", "J", "X", "Ud", "Sp")]:
    if k not in V:
        put(k, r"\textbf{??}", "PENDING", "placeholder")

lines = ["% generated by analysis/02_manuscript/build_values_extra.py -- do not edit by hand"]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in V.items()]
(HERE / "values_extra.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
with open(HERE / "macro_provenance.csv", "a", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["macro", "value", "source_file", "field"])
    w.writerows(PROV)
print(len(V), "review-round macros")
