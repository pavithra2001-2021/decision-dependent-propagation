"""Second session: preprocessing and between- and within-session change."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DER = HERE / "derived"
META = HERE.parent / "external_metadata" / "nyu2" / "sessions"
L = json.loads((K.DERIV / "s3_listing.json").read_text())
PRE = "data/Projects/CORR/RawDataBIDS/NYU_2/"
ids = sorted({re.search(r"sub-(\d+)/", k).group(1) for k in L if "/sub-" in k})


def has(p, s, r):
    return f"{PRE}sub-{p}/ses-{s}/func/sub-{p}_ses-{s}_task-rest_run-{r}_bold.nii.gz" in L


s2 = [p for p in ids if has(p, 2, 1) and has(p, 2, 2)]
four = [p for p in s2 if has(p, 1, 1) and has(p, 1, 2)]
C = K.nyu2_cohort()
cand = [p.strip() for p in (HERE / "ses2_candidates.txt").read_text().split()]
y = (C.sex == "female").astype(int).values
ZA, ZB = K.nyu2_point_all(C)
W1, c1, _ = K.d1_weights_cached(C, ZA, y)
w2, w3 = K.fixed_weights(ZA.shape[1], K.SEED_C)
pos = {pid: i for i, pid in enumerate(C.pid)}
P = pd.read_csv(K.RESULTS / "confirmation_participants.csv.gz")


def meta(p):
    t = pd.read_csv(META / f"sub-{p}_sessions.tsv", sep="\t", dtype=str)
    r1, r2 = t[t.session_id == "ses-1"].iloc[0], t[t.session_id == "ses-2"].iloc[0]
    return float(r1.age_at_scan_1), r2.retest_duration, r2.retest_units, r2.preceding_condition


rows, flow = [], {"with_session2_runs": len(s2), "with_all_four_runs": len(four), "in_fixed_cohort": len(cand)}
processed = 0
for p in cand:
    names = [f"sub-{p}_ses2_run{r}" for r in (1, 2)]
    files = [DER / n / f"ROISignals_{n}.mat" for n in names]
    if not all(f.exists() for f in files):
        continue
    processed += 1
    Xs = [sio.loadmat(f)["ROISignals"][:, :116].astype(np.float64) for f in files]
    fds = [float(np.loadtxt(next((DER / n).glob("FD_Jenkinson_*.txt"))).mean()) for n in names]
    qc = all("P6_QC_OK" in (DER / n / "dparsf_log.txt").read_text(errors="replace") for n in names)
    ok = qc and all(X.shape[0] >= 150 and np.isfinite(X).all() and (X.std(0) > 1e-8).all() for X in Xs) and max(fds) <= 0.2
    age, dur, unit, prec = meta(p)
    rec = {"pid": p, "processed": True, "eligible": bool(ok), "qc_ok": bool(qc), "fd_s2r1": fds[0], "fd_s2r2": fds[1], "T_s2r1": Xs[0].shape[0],
           "T_s2r2": Xs[1].shape[0], "age": age, "retest_duration": dur, "retest_units": unit, "preceding": prec}
    if ok:
        k = pos[f"sub-{p}"]
        z21, z22 = K.point_z(Xs[0]), K.point_z(Xs[1])
        m21 = K.EC.run_measurement(Xs[0], np.random.default_rng([K.SEED_C, 44, k, 1]), K.R)
        for D, w, c in (("D1", W1[k], c1[k]), ("D2", w2, 0.0), ("D3", w3, 0.0)):
            arch = P[(P.pid == f"sub-{p}") & (P.D == D)].iloc[0]
            rec |= {f"{D}_l11": float(ZA[k] @ w + c), f"{D}_l12": float(ZB[k] @ w + c), f"{D}_l21": float(z21 @ w + c),
                    f"{D}_l22": float(z22 @ w + c), f"{D}_sJ11": float(arch.sA_J_cbb), f"{D}_sJ21": float(K.EC.decision_sds(m21, w)["J_cbb"])}
            assert abs(rec[f"{D}_l11"] - arch.lA) < 1e-8 * max(1, abs(arch.lA))
        del m21
    rows.append(rec)
D = pd.DataFrame(rows)
D.to_csv(OUT / "participants.csv", index=False)
E = D[D.eligible].reset_index(drop=True)
flow |= {"processed_ok": int(processed), "eligible": int(len(E))}


def boot(fn, n=10000, seed=K.SEED_C):
    rng = np.random.default_rng(seed)
    b = [fn(rng.integers(0, len(E), len(E))) for _ in range(n)]
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


res = {"flow": flow, "components": {}, "vcr": {}}
mv = {"NWith": str(flow["with_session2_runs"]), "NFour": str(flow["with_all_four_runs"]), "NCand": str(flow["in_fixed_cohort"]),
      "NProc": str(flow["processed_ok"]), "N": str(flow["eligible"])}
if len(E):
    a = E.age.values
    mins = pd.to_numeric(E.retest_duration[E.retest_units == "minutes"], errors="coerce").dropna()
    res["sample"] = {"age_median": float(np.median(a)), "age_min": float(a.min()), "age_max": float(a.max()), "n_under18": int((a < 18).sum()),
                     "n_minutes": int((E.retest_units == "minutes").sum()), "n_days": int((E.retest_units == "days").sum()),
                     "minutes_median": float(mins.median()) if len(mins) else None,
                     "minutes_range": [float(mins.min()), float(mins.max())] if len(mins) else None,
                     "n_active_task_before": int((E.preceding == "active task").sum())}
    s = res["sample"]
    mv |= {"AgeMed": f"{s['age_median']:.1f}", "AgeMin": f"{s['age_min']:.1f}", "AgeMax": f"{s['age_max']:.1f}", "Under": str(s["n_under18"]),
           "Minutes": f"{s['minutes_median']:.0f}" if s["minutes_median"] is not None else "--", "NMin": str(s["n_minutes"]), "NDays": str(s["n_days"]),
           "NTask": str(s["n_active_task_before"])}
    for Dn, t in (("D1", "Lr"), ("D2", "Gm"), ("D3", "Rw")):
        dw = np.r_[E[f"{Dn}_l12"] - E[f"{Dn}_l11"], E[f"{Dn}_l22"] - E[f"{Dn}_l21"]].reshape(2, -1)
        db = np.r_[E[f"{Dn}_l21"] - E[f"{Dn}_l11"], E[f"{Dn}_l22"] - E[f"{Dn}_l12"]].reshape(2, -1)
        pv = (2 * E[f"{Dn}_sJ11"] ** 2 + 2 * E[f"{Dn}_sJ21"] ** 2).values / 2

        def comp(i):
            mw, mb = np.mean(dw[:, i] ** 2), np.mean(db[:, i] ** 2)
            vr, vs = mw / 2, (mb - mw) / 2
            return vr, vs, vs / (vr + vs), mb / mw, np.mean(pv[i]) / mw, np.mean(pv[i]) / mb

        allidx = np.arange(len(E))
        vr, vs, share, ratio, vcw, vcb = comp(allidx)
        res["components"][Dn] = {"var_run": float(vr), "var_session": float(vs), "share_session": float(share), "ratio_between_within": float(ratio),
                                 "share_ci95": boot(lambda i: comp(i)[2]), "ratio_ci95": boot(lambda i: comp(i)[3]),
                                 "var_session_ci95": boot(lambda i: comp(i)[1])}
        res["vcr"][Dn] = {"within": float(vcw), "between": float(vcb), "within_ci95": boot(lambda i: comp(i)[4]), "between_ci95": boot(lambda i: comp(i)[5])}
        rho = spearmanr(np.abs(dw[0]), np.abs(dw[1])).statistic
        res["components"][Dn]["consistency_spearman"] = float(rho)
        res["components"][Dn]["consistency_ci95"] = boot(lambda i: spearmanr(np.abs(dw[0, i]), np.abs(dw[1, i])).statistic)
        c, v = res["components"][Dn], res["vcr"][Dn]
        mv |= {f"Share{t}": f"{100 * c['share_session']:.0f}".replace("-", "\\textminus{}"),
               f"Share{t}CI": "[" + ", ".join(f"{100 * x:.0f}".replace("-", "\\textminus{}") for x in c["share_ci95"]) + "]",
               f"Ratio{t}": f"{c['ratio_between_within']:.2f}", f"Ratio{t}CI": "[" + ", ".join(f"{x:.2f}" for x in c["ratio_ci95"]) + "]",
               f"VcrW{t}": f"{v['within']:.2f}", f"VcrW{t}CI": "[" + ", ".join(f"{x:.2f}" for x in v["within_ci95"]) + "]",
               f"VcrB{t}": f"{v['between']:.2f}", f"VcrB{t}CI": "[" + ", ".join(f"{x:.2f}" for x in v["between_ci95"]) + "]",
               f"Cons{t}": f"{c['consistency_spearman']:.2f}".replace("-", "\\textminus{}"),
               f"Cons{t}CI": "[" + ", ".join(f"{x:.2f}".replace("-", "\\textminus{}") for x in c["consistency_ci95"]) + "]"}

def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "\\textminus{}")


def fci(ci, nd=2):
    return "[" + ", ".join(fmt(x, nd) for x in ci) + "]"


if len(E):
    from scipy.stats import pearsonr, wilcoxon
    res["position_diagnostic"] = {}
    for Dn, t in (("D1", "Lr"), ("D2", "Gm"), ("D3", "Rw")):
        w1 = (E[f"{Dn}_l12"] - E[f"{Dn}_l11"]).values
        w2 = (E[f"{Dn}_l22"] - E[f"{Dn}_l21"]).values
        b1 = (E[f"{Dn}_l21"] - E[f"{Dn}_l11"]).values
        b2 = (E[f"{Dn}_l22"] - E[f"{Dn}_l12"]).values
        x1 = (E[f"{Dn}_l22"] - E[f"{Dn}_l11"]).values
        x2 = (E[f"{Dn}_l21"] - E[f"{Dn}_l12"]).values
        pv = (E[f"{Dn}_sJ11"] ** 2 + E[f"{Dn}_sJ21"] ** 2).values
        sd_ref = float(np.sqrt(np.mean(np.r_[w1, w2] ** 2) / 2))

        def dec(i):
            mw = np.mean(np.r_[w1[i], w2[i]] ** 2)
            mb = np.mean(np.r_[b1[i], b2[i]] ** 2)
            mx = np.mean(np.r_[x1[i], x2[i]] ** 2)
            su2, pi2, se2x2 = (mx - mw) / 2, mx - mb, mw + mb - mx
            return su2, pi2, se2x2, np.mean(pv[i]) / se2x2 if se2x2 > 0 else np.nan, mx / mw

        allidx = np.arange(len(E))
        su2, pi2, se2x2, vcre, xratio = dec(allidx)
        shift = (w1 + w2) / 2
        o = {"mean_shift_s1": float(w1.mean()), "mean_shift_s2": float(w2.mean()), "mean_shift_pooled": float(shift.mean()),
             "mean_shift_pooled_ci95": boot(lambda i: shift[i].mean()), "mean_shift_over_run_sd": float(shift.mean() / sd_ref),
             "wilcoxon_p_shift": float(wilcoxon(shift).pvalue),
             "frac_run2_higher_s1": float(np.mean(w1 > 0)), "frac_run2_higher_s2": float(np.mean(w2 > 0)),
             "msq_within": float(np.mean(np.r_[w1, w2] ** 2)), "msq_between_matched": float(np.mean(np.r_[b1, b2] ** 2)),
             "msq_between_cross": float(np.mean(np.r_[x1, x2] ** 2)), "ratio_cross_within": float(xratio),
             "ratio_cross_within_ci95": boot(lambda i: dec(i)[4]),
             "var_session_u": float(su2), "var_session_u_ci95": boot(lambda i: dec(i)[0]),
             "var_position_pi2": float(pi2), "var_position_pi2_ci95": boot(lambda i: dec(i)[1]),
             "two_var_e": float(se2x2), "two_var_e_ci95": boot(lambda i: dec(i)[2]),
             "share_position_of_within": float(pi2 / np.mean(np.r_[w1, w2] ** 2)),
             "share_position_of_within_ci95": boot(lambda i: dec(i)[1] / np.mean(np.r_[w1[i], w2[i]] ** 2)),
             "vcr_position_adjusted": float(vcre), "vcr_position_adjusted_ci95": boot(lambda i: dec(i)[3]),
             "stability_spearman": float(spearmanr(w1, w2).statistic), "stability_spearman_ci95": boot(lambda i: spearmanr(w1[i], w2[i]).statistic),
             "stability_pearson": float(pearsonr(w1, w2).statistic), "stability_pearson_ci95": boot(lambda i: np.corrcoef(w1[i], w2[i])[0, 1])}
        res["position_diagnostic"][Dn] = o
        mv |= {f"PosShift{t}": fmt(o["mean_shift_over_run_sd"]), f"PosShiftP{t}": f"{o['wilcoxon_p_shift']:.3f}" if o["wilcoxon_p_shift"] >= 0.001 else "<0.001",
               f"PosFracUp{t}": f"{100 * (o['frac_run2_higher_s1'] + o['frac_run2_higher_s2']) / 2:.0f}",
               f"PosXRatio{t}": fmt(o["ratio_cross_within"]), f"PosXRatio{t}CI": fci(o["ratio_cross_within_ci95"]),
               f"PosShare{t}": f"{100 * o['share_position_of_within']:.0f}".replace("-", "\\textminus{}"),
               f"PosShare{t}CI": "[" + ", ".join(f"{100 * x:.0f}".replace("-", "\\textminus{}") for x in o["share_position_of_within_ci95"]) + "]",
               f"PosVcrE{t}": fmt(o["vcr_position_adjusted"]), f"PosVcrE{t}CI": fci(o["vcr_position_adjusted_ci95"]),
               f"PosStab{t}": fmt(o["stability_spearman"]), f"PosStab{t}CI": fci(o["stability_spearman_ci95"])}
res["macro_values"] = mv
K.jdump(res, OUT / "summary.json")
print(json.dumps({k: res[k] for k in res if k != "macro_values"}, indent=1)[:4000])
