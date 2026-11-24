"""Figures of the manuscript and supplement (shared style and part of the figures)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = REV.parent
RES = ROOT / "results"
NA = REV / "01_analyses"
OUT = HERE
W_FULL, W_COL = 7.0, 3.4

INK = "#222222"
GRID = "#e6e6e6"
JOINT, UHAC, INDEP, ICBB, IID, CONST, PURPLE = "#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#8C8C8C", "#CC79A7"
STYLE = {"J_cbb": ("J-CBB", JOINT, "o", True), "uHAC": ("uHAC", UHAC, "D", True), "J_iid": ("J-iid", IID, "^", True),
         "I_xdf": ("I-xDF", INDEP, "v", True), "I_cbb": ("I-CBB", ICBB, "s", True), "CONST": ("constant", CONST, "x", True)}


def style():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 600,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False})


def j(p):
    return json.loads(Path(p).read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def symlog(v, c):
    return np.sign(v) * np.log10(1 + np.abs(v) / c)


MN = lambda s: s.replace("-", "−")


def figure2():
    S = j(RES / "decision_simulation.json")
    meths = [("cbb_joint", "J-CBB (joint, block bootstrap)", JOINT, "o"), ("uhac_bartlett", "uHAC (joint, analytic)", UHAC, "D"),
             ("iid_joint", "J-iid (joint, i.i.d. resampling)", IID, "^"), ("cbb_diag", "I-CBB (independent)", ICBB, "s"),
             ("xdf_diag", "I-xDF (independent)", INDEP, "v"), ("analytic_diag", "I-naive (independent)", CONST, "x")]
    conds = [(0.0, 0.0), (0.5, 0.0), (0.8, 0.0), (0.0, 0.6), (0.5, 0.6), (0.8, 0.6)]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.7), sharey=True)
    for ax, wn, letter, ttl in zip(axes, ("dense_random", "global_mean"), "ab", ("dense random decision", "global-mean decision")):
        for k, (key, lab, col, mk) in enumerate(meths):
            xs = np.arange(len(conds)) + (k - 2.5) * 0.11
            ys = [S[f"phi={p}|factor={fa}|w={wn}"][key] for p, fa in conds]
            ax.scatter(xs, ys, s=16, marker=mk, color=col, lw=0.9 if mk == "x" else 0.4, edgecolor="white" if mk != "x" else None,
                       label=lab, zorder=3)
        ax.axhline(1, color=INK, lw=0.6)
        ax.axvline(2.5, color="#cfcfcf", lw=0.6)
        ax.set_yscale("log"); ax.set_ylim(0.09, 1.7)
        ax.set_yticks([0.1, 0.2, 0.5, 1.0, 1.5]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "1.5"])
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([f"{p:g}" for p, _ in conds])
        ax.set_xlabel("autocorrelation $\\phi$")
        ax.text(1, 1.36, "no shared component", ha="center", color="#555555")
        ax.text(4, 1.36, "shared component", ha="center", color="#555555")
        ax.set_title(ttl, loc="left", pad=10)
        ax.grid(True, axis="y")
        panel(ax, letter, x=-0.1, y=1.06)
    axes[0].set_ylabel("estimated / true decision SD")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.02))
    save(fig, "Figure_2")


def figure3():
    C = j(RES / "confirmation_results.json")
    P = pd.read_csv(RES / "confirmation_participants.csv.gz")
    G = P[P.D == "D1"]
    x = (G.nlpd_J_cbb - G.nlpd_I_xdf).values
    A3 = j(NA / "model_families" / "results" / "summary.json")
    A6 = j(NA / "scan_length" / "results" / "summary.json")
    fig = plt.figure(figsize=(W_FULL, 2.9))
    ax = fig.add_axes([0.04, 0.18, 0.33, 0.72])
    tr = lambda v: symlog(v, 1.0)
    rng = np.random.default_rng(0)
    yy = rng.uniform(-0.35, 0.35, len(x))
    ax.scatter(tr(x[x < 0]), yy[x < 0], s=8, color=JOINT, alpha=0.8, lw=0, marker="o", label="lower NLPD with J-CBB")
    ax.scatter(tr(x[x >= 0]), yy[x >= 0], s=9, color=INDEP, alpha=0.9, lw=0, marker="^", label="lower NLPD with I-xDF")
    med, (lo, hi) = C["PRIMARY"]["median"], C["PRIMARY"]["ci95"]
    ax.plot([tr(lo), tr(hi)], [0.55, 0.55], color=INK, lw=1.4)
    ax.plot([tr(med)], [0.55], "o", color=INK, ms=4)
    ax.axvline(0, color=CONST, lw=0.7)
    ticks = np.array([-1000, -100, -10, 0, 10])
    ax.set_xticks(tr(ticks)); ax.set_xticklabels([MN(str(t)) for t in ticks])
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.62, 0.9)
    ax.set_xlabel("NLPD difference, J-CBB $-$ I-xDF")
    ax.text(tr(med), 0.68, MN(f"median {med:.2f} [{lo:.2f}, {hi:.2f}]"), ha="center", color=INK)
    ax.legend(loc="lower left", handletextpad=0.1, borderaxespad=0.0, markerscale=1.3, bbox_to_anchor=(0.0, -0.02))
    panel(ax, "a", x=-0.04)
    bx = fig.add_axes([0.62, 0.18, 0.37, 0.72])
    rows = [("ridge logistic, sex (prespecified)", C["PRIMARY"], JOINT, "o", True),
            ("linear SVM, sex", A3["M2_linear_svm_sex"]["nlpd_Jcbb_minus_Ixdf"], JOINT, "o", False),
            ("ridge regression, age", A3["M3_ridge_age"]["nlpd_Jcbb_minus_Ixdf"], JOINT, "o", False),
            ("RBF-SVM, sex (nonlinear)", A3["M4_rbf_svm_sex"]["nlpd_Jcbb_minus_Ixdf"], JOINT, "o", False),
            ("ridge logistic, $T$ = 100", A6["NYU2"]["D1"]["100"]["nlpd_Jcbb_minus_Ixdf"], UHAC, "D", False),
            ("ridge logistic, $T$ = 60", A6["NYU2"]["D1"]["60"]["nlpd_Jcbb_minus_Ixdf"], UHAC, "D", False)]
    c2 = 0.1
    for i, (lab, r, col, mk, filled) in enumerate(rows):
        y_ = len(rows) - 1 - i
        lo, hi = r["ci95"]
        bx.plot([symlog(lo, c2), symlog(hi, c2)], [y_, y_], color=col, lw=1.2)
        bx.plot([symlog(r["median"], c2)], [y_], mk, ms=4.3, color=col if filled else "white", mec=col, mew=1.0)
        bx.text(symlog(-20, c2) - 0.12, y_, lab, ha="right", va="center", color=INK)
    bx.axvline(0, color=CONST, lw=0.7)
    bx.axhline(len(rows) - 4.5, color="#cfcfcf", lw=0.6)
    t2 = np.array([-10, -1, -0.1, 0])
    bx.set_xticks(symlog(t2, c2)); bx.set_xticklabels([MN(str(t)) for t in ("-10", "-1", "-0.1", "0")])
    bx.set_xlim(symlog(-20, c2), symlog(0.3, c2))
    bx.set_yticks([]); bx.spines["left"].set_visible(False)
    bx.set_xlabel("median NLPD difference, J-CBB $-$ I-xDF (95% CI)")
    panel(bx, "b", x=-0.66)
    save(fig, "Figure_3")


def figure4():
    P = pd.read_csv(RES / "confirmation_participants.csv.gz")
    A3 = pd.read_csv(NA / "model_families" / "results" / "participants.csv.gz")
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.7), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    groups = [("random\nweights", (P[P.D == "D3"].sA_J_cbb / P[P.D == "D3"].sA_I_cbb) ** 2, CONST),
              ("ridge\nlogistic", (P[P.D == "D1"].sA_J_cbb / P[P.D == "D1"].sA_I_cbb) ** 2, JOINT),
              ("linear\nSVM", (A3.M2_s_J_cbb / A3.M2_s_I_cbb) ** 2, JOINT),
              ("ridge\n(age)", (A3.M3_s_J_cbb / A3.M3_s_I_cbb) ** 2, JOINT),
              ("RBF-SVM\n(gradient)", (A3.M4_s_J_cbb_lin / A3.M4_s_I_cbb) ** 2, JOINT),
              ("global\nmean", (P[P.D == "D2"].sA_J_cbb / P[P.D == "D2"].sA_I_cbb) ** 2, INDEP)]
    data = [g[1].values for g in groups]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.1))
    for b, g in zip(bp["boxes"], groups):
        b.set(facecolor="white", edgecolor=g[2], lw=0.9)
    for k, wk in enumerate(bp["whiskers"]):
        wk.set(color=groups[k // 2][2], lw=0.8)
    for k, cp in enumerate(bp["caps"]):
        cp.set(color=groups[k // 2][2], lw=0.8)
    rng = np.random.default_rng(1)
    for i, (dd, g) in enumerate(zip(data, groups)):
        ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(dd)), dd, s=3.5, color=g[2], alpha=0.35, lw=0)
    ax.set_yscale("log"); ax.axhline(1, color=INK, lw=0.6)
    ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("amplification factor $\\kappa$ (matched marginals)")
    ax.grid(True, axis="y")
    panel(ax, "a", x=-0.12)
    bx = axes[1]
    order = [("D3", "random"), ("D1", "classifier"), ("D2", "global mean")]
    for k, m in enumerate(("J_cbb", "uHAC", "J_iid", "I_xdf")):
        lab, col, mk, _ = STYLE[m]
        ys = []
        for d, _ in order:
            g = P[P.D == d]
            z = (g.lB - g.lA).values / (np.sqrt(2) * g[f"sA_{m}"].values)
            ys.append(100 * np.mean(np.abs(z) <= 1.96))
        bx.plot(np.arange(3) + (k - 1.5) * 0.13, ys, mk, ms=4.5, color=col, mec="white", mew=0.4, label=lab, ls="none")
    bx.axhline(95, color=INK, lw=0.6, ls="--")
    bx.text(2.45, 90, "nominal 95%", color=INK, ha="right", va="center")
    bx.set_ylim(0, 118); bx.set_yticks([0, 20, 40, 60, 80, 100])
    bx.set_xticks(range(3)); bx.set_xticklabels([lab for _, lab in order])
    bx.set_ylabel("coverage of 95% intervals (%)")
    bx.legend(loc="upper center", ncol=4, columnspacing=0.8, handletextpad=0.1, borderaxespad=0.0)
    bx.grid(True, axis="y")
    panel(bx, "b", x=-0.2)
    fig.tight_layout()
    save(fig, "Figure_4")


def figure5():
    P = pd.read_csv(RES / "confirmation_participants.csv.gz")
    T = j(RES / "uhac_timing.json")
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.8), gridspec_kw={"width_ratios": [1.0, 0.85]})
    ax = axes[0]
    for d, lab, col, mk in (("D3", "random", CONST, "^"), ("D1", "classifier", JOINT, "o"), ("D2", "global mean", INDEP, "s")):
        g = P[P.D == d]
        ax.scatter(g.sA_J_cbb, g.sA_uHAC, s=6, marker=mk, color=col, alpha=0.65, lw=0, label=lab)
    lim = [P.sA_J_cbb.min() * 0.7, P.sA_J_cbb.max() * 1.4]
    ax.plot(lim, lim, color=INK, lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("decision SD, J-CBB (100 replicates)")
    ax.set_ylabel("decision SD, uHAC (analytic)")
    ax.legend(loc="upper left", markerscale=1.8, handletextpad=0.2)
    ax.grid(True)
    panel(ax, "a", x=-0.17)
    bx = axes[1]
    R = pd.DataFrame(T["rows"])
    for t, ls in ((170, "-"), (488, "--")):
        S = R[R["T"] == t]
        bx.plot(S.E, 1000 * S.cbb_R100_s, "o" + ls, ms=3.5, lw=0.9, color=JOINT, label=f"J-CBB, $T$ = {t}")
        bx.plot(S.E, 1000 * S.uhac_s, "D" + ls, ms=3.5, lw=0.9, color=UHAC, label=f"uHAC, $T$ = {t}")
        bx.plot(S.E, 1000 * S.xdf_tukey_s, "v" + ls, ms=3.5, lw=0.9, color=INDEP, label=f"xDF marginals, $T$ = {t}")
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlabel("number of features (edges) $E$")
    bx.set_ylabel("time per run and decision (ms)")
    bx.grid(True)
    bx.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, handletextpad=0.3, fontsize=6.5, borderaxespad=0.0)
    panel(bx, "b", x=-0.17)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80, wspace=0.32)
    save(fig, "Figure_5")


def figure6():
    C = j(RES / "confirmation_results.json")
    M = pd.read_csv(NA / "calibration" / "results" / "metrics.csv")
    P = pd.read_csv(RES / "confirmation_participants.csv.gz")
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.75), gridspec_kw={"width_ratios": [1.2, 1.25, 0.95]})
    ax = axes[0]
    meths = [("J_cbb", "J-CBB"), ("J_xdfjoint", "J-xDF"), ("J_iidm", "J-iidm"), ("J_iid", "J-iid"), ("I_cbb", "I-CBB"), ("I_xdf", "I-xDF")]
    sets = [("TCP AP (exploratory)", j(RES / "discovery_decisions_AP.json")["ridge_nested"]["decision_vcr"], PURPLE, "^", "white"),
            ("TCP PA (exploratory)", j(RES / "discovery_decisions_PA.json")["ridge_nested"]["decision_vcr"], PURPLE, "v", "white"),
            ("NYU_2 (prespecified)", C["estimation"]["S7_decision_vcr"], JOINT, "o", JOINT)]
    for k, (lab, d, col, mk, fc) in enumerate(sets):
        ax.plot(np.arange(len(meths)) + (k - 1) * 0.17, [d[m] for m, _ in meths], mk, ms=4.2, color=col, mfc=fc, label=lab, ls="none")
    lo, hi = C["estimation"]["S7_J_cbb_ci95"]
    ax.plot([0.17, 0.17], [lo, hi], color=JOINT, lw=1.0)
    ax.set_yscale("log"); ax.set_ylim(3e-4, 2.0)
    ax.axhline(1, color=INK, lw=0.6)
    ax.set_xticks(range(len(meths))); ax.set_xticklabels([l for _, l in meths], rotation=40, ha="right")
    ax.set_ylabel("variance capture (VCR)")
    ax.legend(loc="lower left", numpoints=1, handletextpad=0.2, borderaxespad=0.0)
    ax.grid(True, axis="y")
    panel(ax, "a", x=-0.26)
    bx = axes[1]
    cats = [("raw", "RAW", "RAW"), ("scaled on\nsame decision", "SCALE", "SCALE"), ("scale transferred\nfrom classifier", "SCALE", "TSCALE")]
    decs = [("D1", "classifier"), ("D2", "global mean"), ("D3", "random")]
    xs = 0
    ticks, tlabs = [], []
    for ci_, (clab, k1, k23) in enumerate(cats):
        for di, (d, dl) in enumerate(decs):
            kind = k1 if d == "D1" else k23
            for mi, m in enumerate(("J_cbb", "I_xdf")):
                r = M[(M.D == d) & (M.kind == kind) & (M.method == m)]
                if len(r) == 0:
                    continue
                lab, col, mk, _ = STYLE[m]
                mkd = {"D1": "o", "D2": "s", "D3": "^"}[d]
                bx.plot(xs + mi * 0.28, 100 * r.coverage95.values[0], mkd, ms=4.3, color=col, mec="white", mew=0.4, ls="none")
            xs += 1
        ticks.append(xs - 2); tlabs.append(clab)
        xs += 0.6
    bx.axhline(95, color=INK, lw=0.6, ls="--")
    bx.set_xticks(ticks); bx.set_xticklabels(tlabs)
    bx.set_ylim(0, 105)
    bx.set_ylabel("coverage of 95% intervals (%)")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=JOINT, marker="o", ls="none", ms=4, label="J-CBB"), Line2D([], [], color=INDEP, marker="o", ls="none", ms=4, label="I-xDF"),
          Line2D([], [], color=CONST, marker="o", ls="none", ms=4, label="classifier"), Line2D([], [], color=CONST, marker="s", ls="none", ms=4, label="global mean"),
          Line2D([], [], color=CONST, marker="^", ls="none", ms=4, label="random")]
    bx.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.30, 0.24), ncol=2, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.0)
    bx.grid(True, axis="y")
    panel(bx, "b", x=-0.2)
    cx = axes[2]
    G = P[P.D == "D1"].reset_index(drop=True)
    fl = (np.sign(G.lA) != np.sign(G.lB)).astype(int).values
    FC = pd.read_csv(NA / "calibration" / "results" / "fitted_constants_by_fold.csv")
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(10, shuffle=True, random_state=20261024).split(np.zeros((len(G), 1)), G.y.values))
    cvec = np.zeros(len(G))
    for f, (_, te) in enumerate(folds):
        cvec[te] = FC[(FC.D == "D1") & (FC.method == "J_cbb") & (FC.fold == f)].scale_c.values[0]
    for lab, p, col, mk in (("J-CBB raw", norm.cdf(-np.abs(G.lA) / (np.sqrt(2) * G.sA_J_cbb)), JOINT, "o"),
                            ("J-CBB rescaled", norm.cdf(-np.abs(G.lA) / (np.sqrt(2 * cvec) * G.sA_J_cbb)), UHAC, "D"),
                            ("I-xDF raw", norm.cdf(-np.abs(G.lA) / (np.sqrt(2) * G.sA_I_xdf)), INDEP, "v")):
        p = np.asarray(p)
        q = pd.qcut(p, 5, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "p": p, "f": fl}).groupby("q")
        cx.plot(g.p.mean(), g.f.mean(), mk + "-", ms=3.8, lw=0.9, color=col, label=lab)
    cx.axhline(fl.mean(), color=CONST, lw=0.7, ls=":")
    cx.plot([0, 0.5], [0, 0.5], color=INK, lw=0.6, ls="--")
    cx.set_xlim(-0.01, 0.5); cx.set_ylim(-0.01, 0.5)
    cx.set_xlabel("predicted flip probability")
    cx.set_ylabel("observed flip frequency")
    cx.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.0)
    cx.grid(True)
    panel(cx, "c", x=-0.3)
    fig.tight_layout()
    save(fig, "Figure_6")


def figure7():
    A6 = pd.read_csv(NA / "scan_length" / "results" / "summary_table.csv")
    A4n = j(NA / "split_halves" / "results" / "summary_nyu2.json")
    A4t = j(NA / "split_halves" / "results" / "summary_tcp.json")
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.7), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    dmark = {"D1": ("classifier", JOINT, "o"), "D2": ("global mean", INDEP, "s"), "D3": ("random", CONST, "^")}
    for coh, ls, fc in (("NYU2", "-", None), ("TCP_AP", "--", "white")):
        for d, (lab, col, mk) in dmark.items():
            g = A6[(A6.cohort == coh) & (A6.D == d)].sort_values("T")
            ax.plot(g["T"], g.vcr_J_cbb, mk + ls, ms=4, lw=0.9, color=col, mfc=fc if fc else col,
                    label=f"{lab}, {'NYU_2' if coh == 'NYU2' else 'TCP AP'}")
    ax.set_xscale("log"); ax.set_xticks([60, 100, 150, 170, 300, 488]); ax.set_xticklabels(["60", "100", "150", "170", "300", "488"], rotation=45)
    ax.minorticks_off()
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("scan length $T$ (volumes), both runs truncated")
    ax.set_ylabel("variance capture, J-CBB")
    ax.legend(loc="upper right", ncol=1, fontsize=6.2, handletextpad=0.3, borderaxespad=0.0)
    ax.grid(True, axis="y")
    panel(ax, "a", x=-0.17)
    bx = axes[1]
    sets = [("NYU_2\n85 vol.", A4n, {"D1": "D1", "D2": "D2", "D3": "D3"}),
            ("TCP AP\n244 vol.", A4t["AP"], {"D1": "C", "D2": "G", "D3": "Rd"}),
            ("TCP PA\n244 vol.", A4t["PA"], {"D1": "C", "D2": "G", "D3": "Rd"})]
    x0 = 0
    ticks = []
    for lab, S, key in sets:
        for di, d in enumerate(("D1", "D2", "D3")):
            r = S[key[d]]
            nm, col, mk = dmark[d]
            xw = x0 + di * 0.9
            for kind, off, fc in (("within", -0.17, col), ("across", 0.17, "white")):
                v, (lo, hi) = r[f"vcr_{kind}_J_cbb"], r[f"vcr_{kind}_J_cbb_ci95"]
                bx.plot([xw + off, xw + off], [lo, hi], color=col, lw=0.9)
                bx.plot(xw + off, v, mk, ms=4.3, color=col, mfc=fc, mew=1.0)
        ticks.append(x0 + 0.9)
        x0 += 3.3
    bx.set_xticks(ticks); bx.set_xticklabels([s[0] for s in sets])
    bx.set_ylim(0, 0.9)
    bx.axhline(1, color=INK, lw=0.6)
    bx.set_ylabel("variance capture, J-CBB")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=INK, marker="o", ls="none", ms=4, label="within run (half vs half)"),
          Line2D([], [], color=INK, marker="o", mfc="white", ls="none", ms=4, label="across runs (half vs half)")]
    bx.legend(handles=hs, loc="upper right", handletextpad=0.2, borderaxespad=0.0)
    bx.grid(True, axis="y")
    panel(bx, "b", x=-0.13)
    fig.tight_layout()
    save(fig, "Figure_7")


def figure_s1():
    fig, bx = plt.subplots(figsize=(3.8, 2.6))
    E = 6670
    c = np.logspace(-3, 0, 400)
    for rho, ls in ((0.05, ":"), (0.10, "-"), (0.20, "--")):
        bx.plot(c, 1 - rho + rho * E * c ** 2, color=JOINT, lw=1.1, ls=ls, label=f"$\\bar\\rho$ = {rho:.2f}")
    bx.axhline(1, color=CONST, lw=0.6)
    bx.axvline(E ** -0.5, color=CONST, lw=0.6, ls=":")
    bx.set_yscale("log"); bx.set_xscale("log")
    bx.set_xlim(1e-3, 1); bx.set_ylim(0.5, 3e3)
    bx.set_xlabel("alignment with the shared direction, $|\\cos\\theta|$")
    bx.set_ylabel("amplification factor $\\kappa$")
    bx.legend(loc="upper left")
    bx.grid(True, which="major", axis="y")
    fig.tight_layout()
    save(fig, "Figure_S1")


def figure_s2():
    C = j(RES / "confirmation_results.json")
    fig, bx = plt.subplots(figsize=(3.6, 2.6))
    edge = []
    for pe in ("AP", "PA"):
        B = j(RES / f"discovery_results_{pe}.json")["primary"]
        A = j(RES / f"discovery_xdf_{pe}.json")["edge_level"]
        edge.append((f"TCP {pe}", {"iid": B["iid_bootstrap"]["median_cov95"], "cbb": B["cbb"]["median_cov95"], "xdf": A["xdf"]["median_cov95"]}))
    ec = C["edge_level_medians"]
    edge.append(("NYU_2", {"iid": ec["iid"]["cov95"], "cbb": ec["cbb"]["cov95"], "xdf": ec["xdf"]["cov95"]}))
    for k, (m, lab, col, mk) in enumerate((("iid", "i.i.d. bootstrap", IID, "^"), ("cbb", "block bootstrap", JOINT, "o"), ("xdf", "xDF", INDEP, "v"))):
        bx.plot(np.arange(3) + (k - 1) * 0.15, [100 * e[1][m] for e in edge], mk, ms=4.5, color=col, label=lab, ls="none")
    bx.axhline(95, color=INK, lw=0.6, ls="--")
    bx.set_ylim(30, 100)
    bx.set_xticks(range(3)); bx.set_xticklabels([e[0] for e in edge])
    bx.set_ylabel("median edge coverage of 95% intervals (%)")
    bx.legend(loc="lower right", numpoints=1, handletextpad=0.2)
    bx.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "Figure_S2")


def figure_s3():
    R = pd.read_csv(NA / "selective_prediction" / "results" / "risk_coverage.csv")
    names = [("NYU2_ridge_logistic", "NYU_2 ridge logistic"), ("NYU2_linear_svm", "NYU_2 linear SVM"), ("NYU2_rbf_svm", "NYU_2 RBF-SVM"),
             ("TCP_AP_ridge_logistic", "TCP AP (exploratory)"), ("TCP_PA_ridge_logistic", "TCP PA (exploratory)")]
    fig, axes = plt.subplots(1, 5, figsize=(W_FULL, 1.9), sharey=False)
    for ax, (k, t) in zip(axes, names):
        for sc, col, ls, lab in (("margin", CONST, "-", "margin $|l_A|$"), ("joint", JOINT, "-", "joint $P_J$"), ("independent", INDEP, "--", "independent $P_I$")):
            g = R[(R.classifier == k) & (R.score == sc)]
            ax.plot(g.retained, 100 * g.flip_rate, ls, color=col, lw=1.0, label=lab)
        ax.set_title(t, fontsize=6.5)
        ax.set_xlabel("fraction retained")
        ax.invert_xaxis()
        ax.grid(True)
    axes[0].set_ylabel("flip rate among retained (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.02))
    save(fig, "Figure_S3")


if __name__ == "__main__":
    style()
    for fn in (figure2, figure3, figure4, figure5, figure6, figure7, figure_s1, figure_s2, figure_s3):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("missing input for", fn.__name__, e)
