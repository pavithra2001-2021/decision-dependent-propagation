"""Figures of the manuscript and supplement."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import build_figures as BF
from build_figures import CONST, ICBB, IID, INDEP, INK, JOINT, PURPLE, UHAC, W_FULL, panel, save, symlog

HERE = Path(__file__).resolve().parent
NA = HERE.parent / "01_analyses"
RES = HERE.parent.parent / "results"
j = BF.j
MN = BF.MN
DEC9 = [("random", "random\nweights"), ("cpm_sex", "CPM\n(sex)"), ("pos_strength", "positive\nstrength"), ("linear_svm", "linear\nSVM"),
        ("ridge_age", "ridge\n(age)"), ("ridge_logistic", "ridge\nlogistic*"), ("rbf_svm", "RBF-SVM\n(gradient)"), ("cpm_age", "CPM\n(age)"),
        ("global_mean", "global\nmean")]
ORDER9 = [d for d, _ in DEC9]
LAB9 = {"random": "random weights", "cpm_sex": "CPM (sex)", "pos_strength": "positive strength", "linear_svm": "linear SVM",
        "ridge_age": "ridge (age)", "ridge_logistic": "ridge logistic*", "rbf_svm": "RBF SVM", "cpm_age": "CPM (age)", "global_mean": "global mean"}


def declabels(ax, keys):
    ax.set_xticklabels([LAB9[k] for k in keys], rotation=45, ha="right", fontsize=7.2)


def tables():
    P = pd.read_csv(RES / "confirmation_participants.csv.gz")
    A3 = pd.read_csv(NA / "model_families/results/participants.csv.gz")
    M2 = pd.read_csv(NA / "cpm_transfer/results/participants.csv.gz")
    RE = pd.read_csv(NA / "supplementary_checks/results/participants_ridge_extended.csv.gz")
    T = {}
    for D, nm in (("D1", "ridge_logistic"), ("D2", "global_mean"), ("D3", "random")):
        g = P[P.D == D]
        T[nm] = dict(lA=g.lA.values, lB=g.lB.values, sJ=g.sA_J_cbb.values, sI=g.sA_I_xdf.values, sIc=g.sA_I_cbb.values)
    T["linear_svm"] = dict(lA=A3.M2_lA.values, lB=A3.M2_lB.values, sJ=A3.M2_s_J_cbb.values, sI=A3.M2_s_I_xdf.values, sIc=A3.M2_s_I_cbb.values)
    T["rbf_svm"] = dict(lA=A3.M4_lA.values, lB=A3.M4_lB.values, sJ=A3.M4_s_J_cbb.values, sI=A3.M4_s_I_xdf.values, sIc=A3.M4_s_I_cbb.values,
                        sJlin=A3.M4_s_J_cbb_lin.values)
    T["ridge_age"] = dict(lA=RE.lA.values, lB=RE.lB.values, sJ=RE.s_J_cbb.values, sI=RE.s_I_xdf.values, sIc=RE.s_I_cbb.values)
    for d in ("cpm_sex", "pos_strength", "cpm_age"):
        T[d] = dict(lA=M2[f"{d}_lA"].values, lB=M2[f"{d}_lB"].values, sJ=M2[f"{d}_s_J_cbb"].values, sI=M2[f"{d}_s_I_xdf"].values,
                    sIc=M2[f"{d}_s_I_cbb"].values)
    return T


def figure2_demo():
    G = j(NA / "sensor_demo/results/summary_demo.json")["decisions"]
    names = [("random", "random"), ("contrast", "contrast"), ("trained", "trained"), ("sum", "sum")]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.5), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.18, [G[n]["kappa_true"] for n, _ in names], 0.34, color=CONST, label="true $\\kappa$")
    ax.bar(x + 0.18, [G[n]["kappa_hat_median"] for n, _ in names], 0.34, color=JOINT, label="estimated $\\hat\\kappa$ (median)")
    ax.set_yscale("log"); ax.axhline(1, color=INK, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in names])
    ax.set_ylabel("amplification factor $\\kappa$")
    ax.legend(loc="upper left", borderaxespad=0.0)
    ax.grid(True, axis="y")
    panel(ax, "a", x=-0.17)
    bx = axes[1]
    tn = [n for n, _ in names if n != "trained"]
    x = np.arange(len(tn))
    bx.bar(x - 0.18, [100 * G[n]["transfer"]["cov_joint"] for n in tn], 0.34, color=JOINT, label="joint (J-CBB), transferred scale")
    bx.bar(x + 0.18, [100 * G[n]["transfer"]["cov_indep"] for n in tn], 0.34, color=INDEP, label="independent, transferred scale")
    for i, n in enumerate(tn):
        bx.text(i - 0.18, 100 * G[n]["transfer"]["cov_joint"] + 2, f"{G[n]['transfer']['vcr_joint']:.2f}", ha="center", fontsize=6.3, color=JOINT)
        bx.text(i + 0.18, 100 * G[n]["transfer"]["cov_indep"] + 2, f"{G[n]['transfer']['vcr_indep']:.2f}", ha="center", fontsize=6.3, color=INDEP)
    bx.axhline(95, color=INK, lw=0.6, ls="--")
    bx.set_ylim(0, 118); bx.set_xticks(x); bx.set_xticklabels(tn)
    bx.set_ylabel("coverage of 95% intervals (%)")
    bx.legend(loc="lower right", borderaxespad=0.0, fontsize=6.5)
    bx.grid(True, axis="y")
    panel(bx, "b", x=-0.13)
    fig.tight_layout()
    save(fig, "Figure_2")


def figure3_sim():
    S = j(RES / "decision_simulation.json")
    S3 = j(NA / "simulation_confirmation_regime/results/summary.json")
    fig = plt.figure(figsize=(W_FULL, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.32)
    meths = [("cbb_joint", "J-CBB", JOINT, "o"), ("uhac_bartlett", "uHAC", UHAC, "D"), ("iid_joint", "joint, independent-time resampling", IID, "^"),
             ("cbb_diag", "independent, bootstrap marginals", ICBB, "s"), ("xdf_diag", "I-xDF", INDEP, "v"),
             ("analytic_diag", "independent, 1/(T\u22123) edge variance", CONST, "x")]
    conds = [(0.0, 0.0), (0.5, 0.0), (0.8, 0.0), (0.0, 0.6), (0.5, 0.6), (0.8, 0.6)]
    for p, (wn, letter, ttl) in enumerate((("dense_random", "a", "low-dim., random"), ("global_mean", "b", "low-dim., global mean"))):
        ax = fig.add_subplot(gs[0, p])
        for k, (key, lab, col, mk) in enumerate(meths):
            xs = np.arange(len(conds)) + (k - 2.5) * 0.11
            ax.scatter(xs, [S[f"phi={a}|factor={b}|w={wn}"][key] for a, b in conds], s=11, marker=mk, color=col, lw=0.8 if mk == "x" else 0.3,
                       edgecolor="white" if mk != "x" else None, label=lab, zorder=3)
        ax.axhline(1, color=INK, lw=0.6); ax.axvline(2.5, color="#cfcfcf", lw=0.6)
        ax.set_yscale("log"); ax.set_ylim(0.09, 1.7)
        ax.set_yticks([0.1, 0.2, 0.5, 1.0]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1"])
        ax.set_xticks(range(6)); ax.set_xticklabels([f"{a:g}" for a, _ in conds])
        ax.set_xlabel("autocorrelation $\\phi$ (no / with shared factor)", fontsize=6.5)
        ax.set_title(ttl, loc="left", fontsize=7)
        ax.grid(True, axis="y")
        if p == 0:
            ax.set_ylabel("estimated / true decision SD")
        panel(ax, letter, x=-0.2, y=1.08)
    ax = fig.add_subplot(gs[0, 2])
    cn = [("C1_ar05_nofilter", "AR(1)\nno filter"), ("C2_white_bandpass", "white\nband-pass"), ("C3_ar05_bandpass", "AR(1)\nband-pass")]
    dn = [("random", CONST, "^"), ("trained", JOINT, "o"), ("global_mean", INDEP, "s")]
    for ci_, (c, cl) in enumerate(cn):
        for di, (d, col, mk) in enumerate(dn):
            e = S3["conditions"][c]["decisions"][d]["estimates"]
            x0 = ci_ + (di - 1) * 0.22
            ax.errorbar(x0 - 0.05, e["kappa"]["mean_ratio"], yerr=1.96 * e["kappa"]["se"], fmt=mk, ms=3.6, color=col, mfc=col, lw=0.8, capsize=0)
            ax.errorbar(x0 + 0.05, e["J_cbb"]["mean_sd_ratio"], yerr=1.96 * e["J_cbb"]["se"], fmt=mk, ms=3.6, color=col, mfc="white", lw=0.8, capsize=0)
    ax.axhline(1, color=INK, lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in cn], fontsize=6.3)
    ax.set_ylim(0.6, 1.2)
    ax.set_ylabel("estimate / truth")
    ax.set_title("E = 6670, T = 170", loc="left", fontsize=7)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=INK, marker="o", ls="none", ms=3.6, label="$\\hat\\kappa/\\kappa$"),
          Line2D([], [], color=INK, marker="o", mfc="white", ls="none", ms=3.6, label="$\\hat\\sigma_J/\\sigma_J$"),
          Line2D([], [], color=CONST, marker="^", ls="none", ms=3.6, label="random"), Line2D([], [], color=JOINT, marker="o", ls="none", ms=3.6, label="trained"),
          Line2D([], [], color=INDEP, marker="s", ls="none", ms=3.6, label="global mean")]
    ax.legend(handles=hs, loc="lower left", ncol=2, fontsize=6, handletextpad=0.1, columnspacing=0.5, borderaxespad=0.0)
    ax.grid(True, axis="y")
    panel(ax, "c", x=-0.2, y=1.08)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.36, 0.0), fontsize=6.5)
    save(fig, "Figure_3")


def figure4_primary():
    C = j(RES / "confirmation_results.json")
    P = pd.read_csv(RES / "confirmation_participants.csv.gz")
    Sm = j(NA / "supplementary_checks/results/summary.json")["nlpd_decomposition"]
    G = P[P.D == "D1"]
    x = (G.nlpd_J_cbb - G.nlpd_I_xdf).values
    fig = plt.figure(figsize=(W_FULL, 3.2))
    ax = fig.add_axes([0.04, 0.27, 0.36, 0.63])
    tr = lambda v: symlog(v, 1.0)
    rng = np.random.default_rng(0)
    yy = rng.uniform(-0.35, 0.35, len(x))
    ax.scatter(tr(x[x < 0]), yy[x < 0], s=8, color=JOINT, alpha=0.8, lw=0, label="lower NLPD with J-CBB")
    ax.scatter(tr(x[x >= 0]), yy[x >= 0], s=9, color=INDEP, alpha=0.9, lw=0, marker="^", label="lower NLPD with I-xDF")
    med, (lo, hi) = C["PRIMARY"]["median"], C["PRIMARY"]["ci95"]
    ax.plot([tr(lo), tr(hi)], [0.55, 0.55], color=INK, lw=1.4); ax.plot([tr(med)], [0.55], "o", color=INK, ms=4)
    ax.axvline(0, color=CONST, lw=0.7)
    ticks = np.array([-1000, -100, -10, 0, 10])
    ax.set_xticks(tr(ticks)); ax.set_xticklabels([MN(str(t)) for t in ticks])
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.62, 0.9)
    ax.set_xlabel("NLPD difference, J-CBB $-$ I-xDF (classifier)")
    ax.text(tr(med), 0.68, MN(f"median {med:.2f} [{lo:.2f}, {hi:.2f}]"), ha="center", color=INK)
    ax.legend(loc="lower left", handletextpad=0.1, borderaxespad=0.0, markerscale=1.3, bbox_to_anchor=(0.0, -0.02))
    panel(ax, "a", x=-0.04)
    bx = fig.add_axes([0.5, 0.27, 0.49, 0.63])
    key = {"ridge_age": "ridge_age_extended"}
    names = [d for d, _ in DEC9]
    lv = [Sm[key.get(d, d)]["median_logvar_term"] for d in names]
    rs = [Sm[key.get(d, d)]["median_residual_term"] for d in names]
    vr = [Sm[key.get(d, d)]["median_variance_ratio"] for d in names]
    xx = np.arange(len(names))
    c2 = 0.3
    bx.bar(xx, symlog(np.array(lv), c2), 0.6, color=CONST, label="log-variance term (penalty for width)")
    bx.bar(xx, symlog(np.array(rs), c2), 0.6, color=JOINT, label="standardised-residual term")
    for i, v in enumerate(vr):
        bx.text(i, symlog(max(lv[i], 0), c2) + 0.15, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", fontsize=6.2, color=INK)
    bx.axhline(0, color=INK, lw=0.6)
    yt = np.array([-1000, -100, -10, -1, 0, 1, 10])
    bx.set_yticks(symlog(yt, c2)); bx.set_yticklabels([MN(str(t)) for t in yt])
    bx.set_xticks(xx); declabels(bx, names)
    bx.set_ylabel("median NLPD component")
    bx.legend(loc="lower left", fontsize=6.3, borderaxespad=0.0)
    bx.grid(True, axis="y")
    panel(bx, "b", x=-0.1)
    save(fig, "Figure_5")


def figure5_kappa():
    T = tables()
    labs = [l.replace("RBF-SVM\n(gradient)", "RBF\nSVM") for _, l in DEC9]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.3), gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    data, cols = [], []
    for d, _ in DEC9:
        t = T[d]
        sj = t.get("sJlin", t["sJ"])
        data.append((sj / t["sIc"]) ** 2)
        cols.append(INDEP if d == "global_mean" else (CONST if d == "random" else (PURPLE if d.startswith("cpm") or d == "pos_strength" else JOINT)))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.1))
    for b, c in zip(bp["boxes"], cols):
        b.set(facecolor="white", edgecolor=c, lw=0.9)
    for k, wk in enumerate(bp["whiskers"]):
        wk.set(color=cols[k // 2], lw=0.8)
    for k, cp in enumerate(bp["caps"]):
        cp.set(color=cols[k // 2], lw=0.8)
    rng = np.random.default_rng(1)
    for i, (dd, c) in enumerate(zip(data, cols)):
        ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(dd)), dd, s=3, color=c, alpha=0.35, lw=0)
    ax.set_yscale("log"); ax.axhline(1, color=INK, lw=0.6)
    ax.set_xticks(range(1, len(DEC9) + 1)); declabels(ax, ORDER9)
    ax.set_ylabel("amplification factor $\\kappa$")
    ax.grid(True, axis="y")
    panel(ax, "a", x=-0.1)
    bx = axes[1]
    xx = np.arange(len(DEC9))
    for k, (lab, key, col, mk) in enumerate((("J-CBB", "sJ", JOINT, "o"), ("I-xDF", "sI", INDEP, "v"))):
        ys = []
        for d, _ in DEC9:
            t = T[d]
            dd = t["lB"] - t["lA"]
            ys.append(100 * np.mean(np.abs(dd) <= 1.96 * np.sqrt(2) * t[key]))
        bx.plot(xx + (k - 0.5) * 0.25, ys, mk, ms=4, color=col, mec="white", mew=0.4, ls="none", label=f"{lab}, raw")
    bx.axhline(95, color=INK, lw=0.6, ls="--")
    bx.set_ylim(0, 105)
    bx.set_xticks(xx); declabels(bx, ORDER9)
    bx.set_ylabel("coverage of 95% intervals (%)")
    bx.legend(loc="lower left", borderaxespad=0.0)
    bx.grid(True, axis="y")
    panel(bx, "b", x=-0.2)
    fig.tight_layout()
    save(fig, "Figure_4")


def figure6_transfer():
    M12 = pd.read_csv(NA / "calibration/results/metrics.csv")
    A2b = j(NA / "transfer_models/results/summary.json")["transfer"]
    M2 = j(NA / "cpm_transfer/results/summary.json")
    RE = j(NA / "supplementary_checks/results/summary.json")["ridge_extended"]["propagation"]
    DIR = j(NA / "extended_checks/results/summary_direct_power_discovery.json")["A"]
    rows = []
    for dd, key in (("D3", "random"), ("D2", "global_mean")):
        gj = M12[(M12.D == dd) & (M12.kind == "TSCALE") & (M12.method == "J_cbb")].iloc[0]
        gi = M12[(M12.D == dd) & (M12.kind == "TSCALE") & (M12.method == "I_xdf")].iloc[0]
        rows.append((key, gj.coverage95, gi.coverage95, gj.vcr, gi.vcr))
    for k2, key in (("linear_svm_sex", "linear_svm"), ("rbf_svm_sex", "rbf_svm")):
        r = A2b[k2]
        rows.append((key, r["J_cbb"]["coverage95"], r["I_xdf"]["coverage95"], r["J_cbb"]["vcr"], r["I_xdf"]["vcr"]))
    rows.append(("ridge_age", RE["transfer_coverage_J"], RE["transfer_coverage_I"], RE["transfer_vcr_J"], RE["transfer_vcr_I"]))
    for key in ("cpm_sex", "pos_strength", "cpm_age"):
        t = M2[key]["transfer"]
        rows.append((key, t["J_cbb"]["coverage95"], t["I_xdf"]["coverage95"], t["J_cbb"]["vcr"], t["I_xdf"]["vcr"]))
    rows.sort(key=lambda r: ORDER9.index(r[0]))
    dc = [DIR[r[0]]["direct_constant"] for r in rows]
    keys = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 3.1))
    xx = np.arange(len(rows))
    ax = axes[0]
    ax.plot(xx - 0.2, [100 * r[1] for r in rows], "o", ms=4.5, color=JOINT, label="joint scale (J-CBB), transferred")
    ax.plot(xx, [100 * r[2] for r in rows], "v", ms=4.5, color=INDEP, label="independent scale (I-xDF), transferred")
    ax.plot(xx + 0.2, [100 * c["coverage95"] for c in dc], "s", ms=4.0, color=CONST, label="direct calibration (constant)")
    ax.axhline(95, color=INK, lw=0.6, ls="--")
    ax.set_ylim(0, 105)
    ax.set_xticks(xx); declabels(ax, keys)
    ax.set_ylabel("coverage of 95% intervals (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), borderaxespad=0.0)
    ax.grid(True, axis="y")
    panel(ax, "a", x=-0.14)
    bx = axes[1]
    bx.plot(xx - 0.2, [r[3] for r in rows], "o", ms=4.5, color=JOINT)
    bx.plot(xx, [r[4] for r in rows], "v", ms=4.5, color=INDEP)
    bx.plot(xx + 0.2, [c["vcr"] for c in dc], "s", ms=4.0, color=CONST)
    bx.set_yscale("log"); bx.axhline(1, color=INK, lw=0.6)
    bx.set_ylim(4e-3, 30)
    bx.set_xticks(xx); declabels(bx, keys)
    bx.set_ylabel("variance capture")
    bx.grid(True, axis="y")
    panel(bx, "b", x=-0.14)
    fig.tight_layout()
    save(fig, "Figure_6")


def figure8_shortfall():
    C = j(RES / "confirmation_results.json")
    A6 = pd.read_csv(NA / "scan_length/results/summary_table.csv")
    A4n = j(NA / "split_halves/results/summary_nyu2.json")
    A4t = j(NA / "split_halves/results/summary_tcp.json")
    s2f = NA / "second_session/results/summary.json"
    S2 = j(s2f) if s2f.exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.7), gridspec_kw={"width_ratios": [1.25, 0.85, 1.3]})
    ax = axes[0]
    meths = [("J_cbb", "J-CBB"), ("J_iid", "joint, independent-time"), ("J_iidm", "joint, independent-time rescaled"),
             ("J_xdfjoint", "joint, xDF marginals"), ("I_xdf", "I-xDF"), ("I_cbb", "independent, bootstrap marginals")]
    sets = [("TCP AP", j(RES / "discovery_decisions_AP.json")["ridge_nested"]["decision_vcr"], PURPLE, "^", "white"),
            ("TCP PA", j(RES / "discovery_decisions_PA.json")["ridge_nested"]["decision_vcr"], PURPLE, "v", "white"),
            ("NYU_2", C["estimation"]["S7_decision_vcr"], JOINT, "o", JOINT)]
    yy = np.arange(len(meths))[::-1]
    for k, (lab, d, col, mk, fc) in enumerate(sets):
        ax.plot([d[m] for m, _ in meths], yy + (k - 1) * 0.2, mk, ms=4, color=col, mfc=fc, ls="none", label=lab)
    ax.set_xscale("log"); ax.set_xlim(5e-4, 2.0); ax.axvline(1, color=INK, lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels([l for _, l in meths], fontsize=6.3)
    ax.set_xlabel("variance capture (classifier)")
    ax.legend(loc="upper left", fontsize=6.3, borderaxespad=0.0)
    ax.grid(True, axis="x")
    panel(ax, "a", x=-0.3)
    bx = axes[1]
    dmark = {"D1": ("classifier", JOINT, "o"), "D2": ("global mean", INDEP, "s"), "D3": ("random", CONST, "^")}
    for coh, ls, fc in (("NYU2", "-", None), ("TCP_AP", "--", "white")):
        for d, (lab, col, mk) in dmark.items():
            g = A6[(A6.cohort == coh) & (A6.D == d)].sort_values("T")
            bx.plot(g["T"], g.vcr_J_cbb, mk + ls, ms=3.6, lw=0.9, color=col, mfc=fc if fc else col)
    bx.set_xscale("log"); bx.set_xticks([60, 100, 170, 300, 488]); bx.set_xticklabels(["60", "100", "170", "300", "488"]); bx.minorticks_off()
    bx.set_ylim(0, 0.62)
    bx.set_xlabel("scan length $T$ (both runs truncated)")
    bx.set_ylabel("variance capture, J-CBB")
    bx.text(0.98, 0.97, "solid: NYU_2; dashed: TCP AP", transform=bx.transAxes, ha="right", va="top", fontsize=6.2)
    bx.grid(True, axis="y")
    panel(bx, "b", x=-0.24)
    cx = axes[2]
    sets = [("NYU_2 halves", A4n, {"D1": "D1", "D2": "D2", "D3": "D3"}), ("TCP AP halves", A4t["AP"], {"D1": "C", "D2": "G", "D3": "Rd"})]
    x0, ticks, labs = 0, [], []
    for lab, S, kmap in sets:
        for di, d in enumerate(("D1", "D2", "D3")):
            r = S[kmap[d]]
            nm, col, mk = dmark[d]
            xw = x0 + di * 0.8
            for kind, off, fc in (("within", -0.15, col), ("across", 0.15, "white")):
                v, (lo, hi) = r[f"vcr_{kind}_J_cbb"], r[f"vcr_{kind}_J_cbb_ci95"]
                cx.plot([xw + off] * 2, [lo, hi], color=col, lw=0.8)
                cx.plot(xw + off, v, mk, ms=3.8, color=col, mfc=fc, mew=0.9)
        ticks.append(x0 + 0.8); labs.append(lab); x0 += 2.9
    if S2 is not None and "vcr" in S2:
        for di, d in enumerate(("D1", "D2", "D3")):
            nm, col, mk = dmark[d]
            r = S2["vcr"][d]
            xw = x0 + di * 0.8
            for kind, off, fc in (("within", -0.15, col), ("between", 0.15, "white")):
                v, (lo, hi) = r[kind], r[f"{kind}_ci95"]
                cx.plot([xw + off] * 2, [lo, hi], color=col, lw=0.8)
                cx.plot(xw + off, v, mk, ms=3.8, color=col, mfc=fc, mew=0.9)
        ticks.append(x0 + 0.8); labs.append("NYU_2 sessions")
    cx.set_xticks(ticks); cx.set_xticklabels([l.replace(" ", "\n", 1) for l in labs], fontsize=6.3)
    cx.set_ylim(0, 0.95)
    cx.set_ylabel("variance capture, J-CBB")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=INK, marker="o", ls="none", ms=3.8, label="within (run halves / session runs)"),
          Line2D([], [], color=INK, marker="o", mfc="white", ls="none", ms=3.8, label="across runs / sessions")]
    cx.legend(handles=hs, loc="upper right", fontsize=6.1, borderaxespad=0.0)
    cx.grid(True, axis="y")
    panel(cx, "c", x=-0.15)
    fig.tight_layout()
    save(fig, "Figure_8")


def figure_s4_tuning():
    S = j(NA / "uhac_tuning/results/summary.json")["sensitivity"]
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.3))
    labs = ["b/2", "b", "2b", "4b"]
    for ax, (D, nm) in zip(axes, (("D1", "classifier"), ("D2", "global mean"), ("D3", "random weights"))):
        ax.plot(range(4), [S[D][m]["kappa_cbb_median"] for m in labs], "o-", color=JOINT, ms=3.6, lw=0.9, label="$\\hat\\kappa$, J-CBB block length")
        ax.plot(range(4), [S[D][m]["kappa_hac_median"] for m in labs], "D--", color=UHAC, ms=3.6, lw=0.9, label="$\\hat\\kappa$, HAC bandwidth")
        xa = np.log2(S[D]["andrews_bw_median_iqr"][0] / S[D]["block_length_b_median"]) + 1
        ax.plot([xa], [S[D]["andrews"]["kappa_hac_median"]], "*", color=INK, ms=7, label="Andrews (1991) bandwidth")
        ax.set_xticks(range(4)); ax.set_xticklabels(labs)
        ax.set_title(nm, fontsize=7, loc="left")
        ax.set_xlabel("multiple of block length $b$")
        ax.grid(True, axis="y")
        lo = min(S[D][m]["kappa_cbb_median"] for m in labs + []) if True else 0
        ax.set_ylim(0, max(max(S[D][m]["kappa_cbb_median"], S[D][m]["kappa_hac_median"]) for m in labs) * 1.3)
    axes[0].set_ylabel("median $\\hat\\kappa$")
    axes[0].legend(loc="lower left", fontsize=6, borderaxespad=0.0)
    fig.tight_layout()
    save(fig, "Figure_S4")


def figure_s5_flips():
    P = pd.read_csv(RES / "confirmation_participants.csv.gz")
    FC = pd.read_csv(NA / "calibration/results/fitted_constants_by_fold.csv")
    from sklearn.model_selection import StratifiedKFold
    G = P[P.D == "D1"].reset_index(drop=True)
    fl = (np.sign(G.lA) != np.sign(G.lB)).astype(int).values
    folds = list(StratifiedKFold(10, shuffle=True, random_state=20261024).split(np.zeros((len(G), 1)), G.y.values))
    cvec = np.zeros(len(G))
    for f, (_, te) in enumerate(folds):
        cvec[te] = FC[(FC.D == "D1") & (FC.method == "J_cbb") & (FC.fold == f)].scale_c.values[0]
    fig, cx = plt.subplots(figsize=(3.4, 2.7))
    for lab, p, col, mk in (("J-CBB raw", norm.cdf(-np.abs(G.lA) / (np.sqrt(2) * G.sA_J_cbb)), JOINT, "o"),
                            ("J-CBB rescaled", norm.cdf(-np.abs(G.lA) / (np.sqrt(2 * cvec) * G.sA_J_cbb)), UHAC, "D"),
                            ("I-xDF raw", norm.cdf(-np.abs(G.lA) / (np.sqrt(2) * G.sA_I_xdf)), INDEP, "v")):
        p = np.asarray(p)
        q = pd.qcut(p, 5, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "p": p, "f": fl}).groupby("q")
        cx.plot(g.p.mean(), g.f.mean(), mk + "-", ms=3.8, lw=0.9, color=col, label=lab)
    cx.axhline(fl.mean(), color=CONST, lw=0.7, ls=":"); cx.plot([0, 0.5], [0, 0.5], color=INK, lw=0.6, ls="--")
    cx.set_xlim(-0.01, 0.5); cx.set_ylim(-0.01, 0.5)
    cx.set_xlabel("predicted flip probability"); cx.set_ylabel("observed flip frequency")
    cx.legend(loc="upper left", borderaxespad=0.0); cx.grid(True)
    fig.tight_layout()
    save(fig, "Figure_S5")


if __name__ == "__main__":
    BF.style()
    BF.figure5()
    (HERE / "Figure_5.pdf").replace(HERE / "Figure_7.pdf")
    for fn in (figure2_demo, figure3_sim, figure4_primary, figure5_kappa, figure6_transfer, figure8_shortfall, figure_s4_tuning, figure_s5_flips):
        fn()
        print("ok", fn.__name__)
    BF.figure_s1(); BF.figure_s2(); BF.figure_s3()
    print("ok supplement S1-S3")
