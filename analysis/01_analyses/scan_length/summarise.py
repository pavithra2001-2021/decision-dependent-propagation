"""Analyses with both runs truncated to shorter lengths."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as K

OUT = Path(__file__).resolve().parent / "results"


def long_tcp(D):
    rows = []
    for nm, Dn in (("C", "D1"), ("G", "D2"), ("Rd", "D3")):
        rows.append(pd.DataFrame({"pid": D["sub"], "T": D["T"], "D": Dn, "lA": D[f"lA_{nm}"], "lB": D[f"lB_{nm}"],
                                  "s_J_cbb": D[f"sJ_{nm}"], "s_I_cbb": D[f"sIcbb_{nm}"], "s_I_xdf": D[f"sIxdf_{nm}"],
                                  "s_uHAC": D[f"suHAC_{nm}"]}))
    return pd.concat(rows, ignore_index=True)


def fit_aT(Ts, y):
    X = np.column_stack([1.0 / np.asarray(Ts, float), np.ones(len(Ts))])
    return np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]


def summarise(D, cohort, seed):
    Ts = sorted(D["T"].unique())
    out, table = {}, []
    for Dn in ("D1", "D2", "D3"):
        g = D[D.D == Dn]
        pids = g[g["T"] == Ts[-1]].pid.values
        piv = {T: g[g["T"] == T].set_index("pid").loc[pids] for T in Ts}
        res = {}
        for T in Ts:
            x = piv[T]
            d = (x.lB - x.lA).values
            r = {"mean_d2": float(np.mean(d ** 2)), "mean_2sJ2": float(np.mean(2 * x.s_J_cbb ** 2))}
            for m in ("J_cbb", "uHAC", "I_xdf"):
                v = 2 * x[f"s_{m}"].values ** 2
                r[f"vcr_{m}"] = float(v.mean() / np.mean(d ** 2))
                r[f"cov_{m}"] = float(np.mean(np.abs(d) <= 1.96 * np.sqrt(v)))
            kap = (x.s_J_cbb / x.s_I_cbb) ** 2
            r["kappa_median"] = float(np.median(kap))
            nlj, nli = K.nlpd(np.abs(d), 2 * x.s_J_cbb.values ** 2), K.nlpd(np.abs(d), 2 * x.s_I_xdf.values ** 2)
            r["nlpd_Jcbb_minus_Ixdf"] = K.paired(nlj - nli, seed=seed)
            res[int(T)] = r
            table.append({"cohort": cohort, "D": Dn, "T": int(T)} | {k: v for k, v in r.items() if not isinstance(v, dict)}
                         | {"nlpd_median": r["nlpd_Jcbb_minus_Ixdf"]["median"], "nlpd_ci_lo": r["nlpd_Jcbb_minus_Ixdf"]["ci95"][0],
                            "nlpd_ci_hi": r["nlpd_Jcbb_minus_Ixdf"]["ci95"][1], "nlpd_frac_neg": r["nlpd_Jcbb_minus_Ixdf"]["frac_negative"]})
        d2 = np.stack([((piv[T].lB - piv[T].lA) ** 2).values for T in Ts])
        sj = np.stack([(2 * piv[T].s_J_cbb ** 2).values for T in Ts])
        a, t2 = fit_aT(Ts, d2.mean(1))
        a2, c2 = fit_aT(Ts, sj.mean(1))
        rng = np.random.default_rng(seed)
        bo = []
        for _ in range(10000):
            i = rng.integers(0, d2.shape[1], d2.shape[1])
            bo.append(list(fit_aT(Ts, d2[:, i].mean(1))) + list(fit_aT(Ts, sj[:, i].mean(1))))
        bo = np.array(bo)
        ci = lambda c: [float(v) for v in np.percentile(bo[:, c], [2.5, 97.5])]
        res["decomposition"] = {"d2_a": float(a), "d2_a_ci95": ci(0), "d2_tau2": float(t2), "d2_tau2_ci95": ci(1),
                                "tau2_share_of_d2_at_Tmax": float(t2 / d2.mean(1)[-1]),
                                "sJ_a": float(a2), "sJ_a_ci95": ci(2), "sJ_intercept": float(c2), "sJ_intercept_ci95": ci(3),
                                "n": int(d2.shape[1])}
        out[Dn] = res
    return out, table


S, tab = {}, []
f = OUT / "participants_nyu2.csv.gz"
if f.exists():
    S["NYU2"], t = summarise(pd.read_csv(f), "NYU2", K.SEED_C); tab += t
f = OUT / "participants_tcp.csv.gz"
if f.exists():
    S["TCP_AP"], t = summarise(long_tcp(pd.read_csv(f)), "TCP_AP", K.SEED_B); tab += t
K.jdump(S, OUT / "summary.json")
T = pd.DataFrame(tab)
T.to_csv(OUT / "summary_table.csv", index=False)
pd.set_option("display.width", 250)
print(T[["cohort", "D", "T", "mean_d2", "mean_2sJ2", "vcr_J_cbb", "vcr_uHAC", "vcr_I_xdf", "cov_J_cbb", "cov_I_xdf", "kappa_median",
         "nlpd_median", "nlpd_ci_lo", "nlpd_ci_hi"]].round(4).to_string())
for c, r in S.items():
    for Dn in ("D1", "D2", "D3"):
        print(c, Dn, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r[Dn]["decomposition"].items()})
