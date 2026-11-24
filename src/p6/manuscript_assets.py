"""Numerical values of the discovery and confirmation studies from the result files."""
from __future__ import annotations


import argparse


import json


import platform


import re


import time


from pathlib import Path


import numpy as np


import pandas as pd


from .registry import ROOT


RES = ROOT / "results"


def _j(name):
    return json.loads((RES / name).read_text())


def f(x, d=2):
    return f"{x:.{d}f}".replace("-", "\\textminus{}") if x < 0 else f"{x:.{d}f}"


def sci(p):
    if p >= 1e-3:
        return f"{p:.3f}" if p < 0.995 else "1.00"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def ci(lo, hi, d=2):
    return f"[{f(lo, d)}, {f(hi, d)}]"


def pct(x, d=0):
    return f"{100 * x:.{d}f}"


def values() -> dict:
    C = _j("confirmation_results.json")
    P = pd.read_csv(RES / "confirmation_participants.csv.gz")
    v = {}
    qc = json.loads((ROOT / "data/external/derived/corr_nyu2/qc.json").read_text())
    v["nFinal"] = str(C["n"])
    v["nNoFD"] = str(C["sensitivity"]["no_fd_rule"]["n"])
    D1 = P[P.D == "D1"]
    v["nFemale"], v["nMale"] = str(int((D1.y == 1).sum())), str(int((D1.y == 0).sum()))
    pr = C["PRIMARY"]
    v |= {"primMed": f(pr["median"]), "primCI": ci(*pr["ci95"]), "primP": sci(pr["wilcoxon_p"]),
          "primFrac": pct(pr["frac_negative"]), "primK": str(int(round(pr["frac_negative"] * pr["n"])))}
    s13 = C["estimation"]["S13_mean_nlpd_diff"]
    v |= {"meanDiff": f(s13["mean"], 1), "meanCI": ci(*s13["ci95"], 1)}
    for key, tag in (("no_fd_rule", "noFD"), ("R20", "Rtw"), ("swapped_roles", "swap")):
        s = C["sensitivity"][key]
        v |= {f"{tag}Med": f(s["median"]), f"{tag}CI": ci(*s["ci95"]), f"{tag}P": sci(s["wilcoxon_p"]), f"{tag}Frac": pct(s["frac_negative"])}
    a3 = C["estimation"]["A3_sigmaA2_plus_sigmaB2_primary"]
    v |= {"aThMed": f(a3["median"]), "aThCI": ci(*a3["ci95"]), "aThP": sci(a3["wilcoxon_p"]), "aThFrac": pct(a3["frac_negative"])}
    names = {"S1_I_cbb": "sOne", "S2_J_iidm": "sTwo", "S3_J_iid": "sThree", "S6_D2": "sSixA", "S6_D3": "sSixB",
             "S11_I_xdf_adaptive": "sEleven", "S8_abslogVCR": "sEightV", "S8_rho": "sEightR"}
    for k, tag in names.items():
        s = C["secondary"][k]
        dd = 0 if abs(s["median"]) > 100 else (2 if abs(s["median"]) > 0.1 else 3)
        v |= {f"{tag}Med": f(s["median"], dd), f"{tag}CI": ci(*s["ci95"], dd), f"{tag}P": sci(s["p_holm"]),
              f"{tag}Frac": pct(s["frac_negative"]), f"{tag}Sup": "yes" if s["supported"] else "no"}
    E = C["estimation"]
    for k, tag in (("S4_J_xdfjoint", "sFour"), ("S12_J_xdfadaptjoint", "sTwelve"), ("S5_nlpd_uHAC_minus_Jcbb", "sFive")):
        v |= {f"{tag}Med": f(E[k]["median"], 3), f"{tag}CI": ci(*E[k]["ci95"], 3), f"{tag}Frac": pct(E[k]["frac_negative"])}
    v["uhacRatio"] = f(E["S5_uHAC_ratio_median"])
    vc = E["S7_decision_vcr"]
    for m, tag in (("J_cbb", "Jcbb"), ("J_iid", "Jiid"), ("J_iidm", "Jiidm"), ("J_xdfjoint", "Jxdf"), ("J_xdfadaptjoint", "Jxdfa"),
                   ("I_xdf", "Ixdf"), ("I_xdf_adaptive", "Ixdfa"), ("I_cbb", "Icbb"), ("ISO", "Iso"), ("uHAC", "Uhac")):
        v[f"vcr{tag}"] = f(vc[m], 3 if vc[m] < 0.1 else 2)
    v["vcrJcbbCI"] = ci(*E["S7_J_cbb_ci95"])
    for m, tag in (("cbb", "Cbb"), ("iid", "Iid"), ("xdf", "Xdf"), ("xdf_adaptive", "Xdfa")):
        e = C["edge_level_medians"][m]
        v |= {f"edgeVcr{tag}": f(e["vcr"]), f"edgeCov{tag}": pct(e["cov95"]), f"edgeRho{tag}": f(e["rho"], 3)}
    F = C["S9_flips"]
    v |= {"flipRate": pct(F["observed_flip_rate"]), "flipK": str(int(round(F["observed_flip_rate"] * C["n"]))),
          "brierBase": f(F["J_cbb"]["uncertainty"], 3)}
    for m, tag in (("J_cbb", "Jcbb"), ("I_xdf", "Ixdf"), ("J_iid", "Jiid"), ("I_xdf_adaptive", "Ixdfa")):
        g = F[m]
        v |= {f"br{tag}": f(g["brier"], 3), f"ll{tag}": f(g["logloss"], 2), f"mp{tag}": pct(g["mean_p"], 1),
              f"auc{tag}": f(g["auc"], 2), f"slope{tag}": f(g["cal_slope"], 2), f"rel{tag}": f(g["reliability"], 3),
              f"res{tag}": f(g["resolution"], 3)}
    b = F["brier_J_cbb_minus_I_xdf"]
    v |= {"brDiff": f(b["diff"], 3), "brDiffCI": ci(*b["ci95"], 3), "brDiffP": sci(b["p_two_sided"])}
    a = F["auc_J_cbb_minus_margin"]
    v |= {"aucMarginDiff": f(a["diff"], 3), "aucMarginCI": ci(*a["ci95"], 3)}
    t = F["S10_transfer_recalibrated_J_cbb"]
    v |= {"brTransfer": f(t["brier"], 3), "mpTransfer": pct(t["mean_p"], 1), "slopeTransfer": f(t["cal_slope"], 2)}
    for pe in ("AP", "PA"):
        B = _j(f"discovery_results_{pe}.json")
        A = _j(f"discovery_xdf_{pe}.json")["edge_level"]
        p = B["primary"]
        v |= {f"b{pe}nAll": str(B["n_all"]), f"b{pe}n": str(B["n_primary"]),
              f"b{pe}HoneMed": f(p["H1"]["median_diff_abslogvcr_cbb_minus_iid"]), f"b{pe}HoneCI": ci(*p["H1"]["ci95"]),
              f"b{pe}HoneP": sci(p["H1"]["wilcoxon_p"]),
              f"b{pe}HtwoMed": f(p["H2"]["median_diff_rho_cbb_minus_iid"]), f"b{pe}HtwoCI": ci(*p["H2"]["ci95"]),
              f"b{pe}HtwoP": sci(p["H2"]["p_holm"]),
              f"b{pe}HthrMed": f(B["E3"]["H3"]["brier_diff_cbb_minus_iid"], 4), f"b{pe}HthrCI": ci(*B["E3"]["H3"]["ci95"], 4),
              f"b{pe}HthrP": sci(B["E3"]["H3"]["p_holm"]), f"b{pe}flip": pct(B["E3"]["observed_flip_rate"]),
              f"b{pe}auc": f(B["E3"]["cv_auc_scanA"])}
        for m, tag in (("iid_bootstrap", "Iid"), ("cbb", "Cbb"), ("hac_bartlett", "Hac"), ("analytic_iid", "Ana")):
            v |= {f"b{pe}Vcr{tag}": f(p[m]["median_vcr"]), f"b{pe}Cov{tag}": pct(p[m]["median_cov95"])}
            if p[m]["median_rho"] is not None:
                v[f"b{pe}Rho{tag}"] = f(p[m]["median_rho"], 3)
        v |= {f"b{pe}VcrXdf": f(A["xdf"]["median_vcr"]), f"b{pe}CovXdf": pct(A["xdf"]["median_cov95"]), f"b{pe}RhoXdf": f(A["xdf"]["median_rho"], 3)}
        M = _j(f"discovery_decisions_{pe}.json")["ridge_nested"]
        for m, tag in (("J_cbb", "Jcbb"), ("J_iid", "Jiid"), ("J_iidm", "Jiidm"), ("J_xdfjoint", "Jxdf"), ("I_xdf", "Ixdf"), ("I_cbb", "Icbb")):
            v[f"b{pe}dVcr{tag}"] = f(M["decision_vcr"][m], 4 if M["decision_vcr"][m] < 0.01 else 2)
        v |= {f"b{pe}brJcbb": f(M["forecast"]["J_cbb"]["brier"], 3), f"b{pe}brPrev": f(M["C6_crossfitted"]["prevalence_only"]["brier"], 3),
              f"b{pe}slopeJcbb": f(M["forecast"]["J_cbb"]["cal_slope"], 2)}
        D = pd.read_csv(RES / f"discovery_decisions_{pe}.csv.gz")
        D = D[D.model == "ridge_nested"]
        d = (D.lB - D.lA).values
        nl = lambda s: 0.5 * np.log(2 * np.pi * 2 * s ** 2) + d ** 2 / (4 * s ** 2)
        x = nl(D.sigma_J_cbb.values) - nl(D.sigma_I_xdf.values)
        v |= {f"xb{pe}Nlpd": f(float(np.median(x)), 0), f"xb{pe}Frac": pct(float(np.mean(x < 0))),
              f"xb{pe}SdRatio": f(float(np.median(D.sigma_J_cbb / D.sigma_I_xdf)), 1)}
    from scipy.stats import norm
    nlpd = lambda d, var: 0.5 * np.log(2 * np.pi * var) + d ** 2 / (2 * var)

    def crps(d, s):
        z = d / s
        return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

    def medci(x, B=10000, seed=1):
        r = np.random.default_rng(seed)
        bo = np.median(x[r.integers(0, len(x), (B, len(x)))], 1)
        return float(np.median(x)), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))
    fac = {}
    word = {"D1": "DOne", "D2": "DTwo", "D3": "DThree"}
    for D_ in ("D1", "D2", "D3"):
        G = P[P.D == D_]
        W_ = word[D_]
        d = (G.lB - G.lA).values
        r = (G.sA_J_cbb / G.sA_I_xdf)
        fac[D_] = float(np.mean(G.sA_J_cbb ** 2) / np.mean(G.sA_I_xdf ** 2))
        v |= {f"x{W_}Ratio": f(float(r.median())), f"x{W_}RatioIQR": ci(float(r.quantile(.25)), float(r.quantile(.75))),
              f"x{W_}Factor": f(fac[D_], 1 if fac[D_] < 100 else 0)}
        for m, tag in (("J_cbb", "Jcbb"), ("uHAC", "Uhac"), ("J_iid", "Jiid"), ("I_xdf", "Ixdf")):
            z = d / (np.sqrt(2) * G[f"sA_{m}"].values)
            v[f"x{W_}Cov{tag}"] = pct(float(np.mean(np.abs(z) <= 1.96)))
            v[f"x{W_}Absz{tag}"] = f(float(np.median(np.abs(z))))
        v[f"x{W_}UhacRatio"] = f(float(np.median(G.sA_uHAC / G.sA_J_cbb)))
    G = P[P.D == "D1"]
    d = (G.lB - G.lA).values
    sJ, sI = np.sqrt(2) * G.sA_J_cbb.values, np.sqrt(2) * G.sA_I_xdf.values
    m, lo, hi = medci(crps(d, sJ) - crps(d, sI))
    v |= {"xCrpsMed": f(m, 3), "xCrpsCI": ci(lo, hi, 3), "xCrpsFrac": pct(float(np.mean(crps(d, sJ) < crps(d, sI))))}
    k2 = fac["D1"]
    m, lo, hi = medci(nlpd(d, sJ ** 2) - nlpd(d, k2 * sI ** 2))
    v |= {"xInflSd": f(np.sqrt(k2)), "xInflMed": f(m, 3), "xInflCI": ci(lo, hi, 3),
          "xInflFrac": pct(float(np.mean(nlpd(d, sJ ** 2) < nlpd(d, k2 * sI ** 2))))}
    for D_ in ("D2", "D3"):
        Gd = P[P.D == D_]
        dd = (Gd.lB - Gd.lA).values
        vJ, vI = 2 * Gd.sA_J_cbb.values ** 2, 2 * k2 * Gd.sA_I_xdf.values ** 2
        x = nlpd(dd, vJ) - nlpd(dd, vI)
        v |= {f"xTr{word[D_]}Med": f(float(np.median(x)), 2), f"xTr{word[D_]}Frac": pct(float(np.mean(x < 0))),
              f"xTr{word[D_]}CovJ": pct(float(np.mean(np.abs(dd) <= 1.96 * np.sqrt(vJ)))),
              f"xTr{word[D_]}CovT": pct(float(np.mean(np.abs(dd) <= 1.96 * np.sqrt(vI))))}
    from scipy.stats import spearmanr
    v["xSpearJ"] = f(float(spearmanr(np.abs(d), sJ).statistic))
    v["xSpearI"] = f(float(spearmanr(np.abs(d), sI).statistic))
    from sklearn.metrics import roc_auc_score
    v["xAucA"], v["xAucB"] = f(roc_auc_score(G.y, G.lA)), f(roc_auc_score(G.y, G.lB))
    vcr = C["estimation"]["S7_decision_vcr"]
    v["kappaVcrDOne"] = f(vcr["J_cbb"] / vcr["I_cbb"], 1)
    for D_ in ("D1", "D2", "D3"):
        Gd = P[P.D == D_]
        k = (Gd.sA_J_cbb / Gd.sA_I_cbb) ** 2
        dd = 0 if k.median() > 100 else (1 if k.median() > 5 else 2)
        v[f"xKappa{word[D_]}Med"] = f(float(k.median()), dd)
        v[f"xKappa{word[D_]}IQR"] = ci(float(k.quantile(.25)), float(k.quantile(.75)), dd)
        if D_ == "D2":
            v["xKappaDTwoShare"] = f(100 * float(k.median()) / 6670, 1)
            v["xKappaDTwoRho"] = f((float(k.median()) - 1) / (6670 - 1), 2)
    T = json.loads((RES / "uhac_timing.json").read_text())
    row = {(r["T"], r["N"]): r for r in T["rows"]}
    for (t, n), tag in (((170, 116), "Small"), ((488, 434), "Large")):
        r = row[(t, n)]
        v |= {f"rt{tag}Cbb": f(1000 * r["cbb_R100_s"], 0), f"rt{tag}Uhac": f(1000 * r["uhac_s"], 1),
              f"rt{tag}Xdf": f(1000 * r["xdf_tukey_s"], 0), f"rt{tag}Speed": f(r["cbb_R100_s"] / r["uhac_s"], 0)}
    v["rtMachine"] = T["machine"]["processor_note"]
    return v
