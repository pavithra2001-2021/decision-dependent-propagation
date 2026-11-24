"""Discovery study (TCP): data access, edge-level estimators and the discovery analysis."""
from __future__ import annotations


import argparse


import json


import numpy as np


import pandas as pd


from .registry import ROOT, Run


RAW = ROOT / "data" / "external" / "raw" / "ds005237"


TS = RAW / "fMRI_timeseries_clean_denoised_GSR_parcellated"


CACHE = ROOT / ".cache" / "tcp"


OUT = ROOT / "results"


METHODS = ("analytic_iid", "iid_bootstrap", "cbb", "hac_bartlett")


R = 100


T_FULL = 488


SEED = 20261023


FD_MAX = 0.2


def run_file(sub: str, pe: str, run: int):
    return TS / sub / f"task-rest{pe}_run-0{run}_bold_Atlas_hp2000_clean_GSR_parcellated.h5"


def load_run(sub: str, pe: str, run: int) -> np.ndarray:
    import h5py
    with h5py.File(run_file(sub, pe, run), "r") as f:
        return np.asarray(f["dataset"][()], np.float64).T


def mean_fd(pe: str, run: int) -> dict:
    out = {}
    for line in (RAW / "motion_FD" / f"TCP_FD_rest_{pe}_{run}.csv").read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split(",")
        if len(parts) > 1:
            v = pd.to_numeric(pd.Series(parts[1:]), errors="coerce").dropna()
            out[parts[0].strip()] = float(v.mean()) if len(v) else np.nan
    return out


def cohort(pe: str) -> tuple[pd.DataFrame, list]:
    subs = sorted(p.name for p in TS.iterdir() if p.is_dir())
    flow = [("parcellated series folders", len(subs))]
    subs = [s for s in subs if run_file(s, pe, 1).exists() and run_file(s, pe, 2).exists()]
    flow.append(("both runs present", len(subs)))
    fd1, fd2 = mean_fd(pe, 1), mean_fd(pe, 2)
    d = pd.DataFrame({"sub": subs, "fd_A": [fd1.get(s, np.nan) for s in subs], "fd_B": [fd2.get(s, np.nan) for s in subs]})
    part = pd.read_csv(RAW / "participants.tsv", sep="\t")
    part["sub"] = part.participant_id.str.replace("sub-", "", regex=False)
    key = {s.replace("_", ""): s for s in d["sub"]}
    part["sub"] = part["sub"].str.replace("_", "", regex=False).map(key)
    d = d.merge(part[["sub", "Group", "Site", "age", "sex"]], on="sub", how="left")
    return d, flow


def _std(X):
    X = X - X.mean(-2, keepdims=True)
    return X / np.maximum(X.std(-2, keepdims=True), 1e-12)


def _ztri(Z, iu):
    C = np.matmul(np.swapaxes(Z, -1, -2), Z) / Z.shape[-2]
    return np.arctanh(np.clip(C[..., iu[0], iu[1]], -0.999999, 0.999999))


def replicate_z(X: np.ndarray, idx: np.ndarray, iu, chunk: int = 20):
    for s in range(0, len(idx), chunk):
        yield _ztri(_std(X[idx[s:s + chunk]]), iu)


def hac_bartlett_z(X: np.ndarray, r: np.ndarray, iu, bw: float, chunk: int = 16000) -> np.ndarray:
    Z = _std(X)
    T = Z.shape[0]
    H = int(min(np.ceil(bw), T - 1))
    k = np.clip(1.0 - np.arange(1, H + 1) / bw, 0.0, None)
    out = np.empty(len(r))
    for s in range(0, len(r), chunk):
        i, j, rr = iu[0][s:s + chunk], iu[1][s:s + chunk], r[s:s + chunk]
        IF = Z[:, i] * Z[:, j] - 0.5 * rr * (Z[:, i] ** 2 + Z[:, j] ** 2)
        F = np.fft.rfft(IF, n=2 * T, axis=0)
        g = np.fft.irfft(np.abs(F) ** 2, n=2 * T, axis=0)[: H + 1] / T
        out[s:s + chunk] = (g[0] + 2.0 * (k[:, None] * g[1:]).sum(0)) / T
    return out / np.maximum((1.0 - r ** 2) ** 2, 1e-12)


def xdf_z(X: np.ndarray, r: np.ndarray, iu, M: int | None = None) -> np.ndarray:
    Z = _std(X)
    T = Z.shape[0]
    M = int(round(np.sqrt(T))) if M is None else M
    i, j = iu
    S1, S2, S3 = np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r))
    for k in range(1, M + 1):
        tap = 0.5 * (1.0 + np.cos(np.pi * k / M))
        Ck = (Z[:-k].T @ Z[k:]) / T * tap
        ac = np.diag(Ck)
        ax, ay, xp, xn = ac[i], ac[j], Ck[i, j], Ck[j, i]
        wk = (T - k) / T
        S1 += wk * (ax ** 2 + ay ** 2 + xp ** 2 + xn ** 2)
        S2 += wk * (ax + ay) * (xp + xn)
        S3 += wk * (ax * ay + xp * xn)
    q = (1.0 - r ** 2) ** 2
    V = np.maximum((q + r ** 2 * S1 - 2.0 * r * S2 + 2.0 * S3) / T, q / T)
    return V / np.maximum(q, 1e-12)


def xdf_z_adaptive(X: np.ndarray, r: np.ndarray, iu) -> np.ndarray:
    Z = X - X.mean(0)
    Z = Z / np.sqrt((Z ** 2).sum(0) / X.shape[0])
    T, N = Z.shape
    bnd = np.sqrt(2) * 1.3859 / np.sqrt(T)
    ac_all = np.array([(Z[:-k] * Z[k:]).sum(0) / T for k in range(1, T - 1)])
    bp = np.zeros(N, int)
    for n in range(N):
        below = np.where(np.abs(ac_all[:, n]) < bnd)[0]
        bp[n] = below[0] if below.size else 0
    ac = ac_all.copy()
    for n in range(N):
        ac[bp[n]:, n] = 0.0
    i, j = iu
    M = np.maximum(bp[i], bp[j])
    Kmax = int(M.max()) if M.size else 0
    S1, S2, S3 = np.zeros(len(r)), np.zeros(len(r)), np.zeros(len(r))
    for k in range(1, Kmax + 1):
        keep = M >= k
        Ck = (Z[:-k].T @ Z[k:]) / T
        xp, xn = Ck[i, j] * keep, Ck[j, i] * keep
        ax, ay = ac[k - 1, i], ac[k - 1, j]
        wk = T - 1 - k
        S1 += wk * (ax ** 2 + ay ** 2 + xp ** 2 + xn ** 2)
        S2 += wk * (ax + ay) * (xp + xn)
        S3 += wk * (ax * ay + xp * xn)
    q = (1.0 - r ** 2) ** 2
    V = ((T - 1) * q + r ** 2 * S1 - 2.0 * r * S2 + 2.0 * S3) / T ** 2
    V = np.maximum(V, q / T)
    return V / np.maximum(q, 1e-12)


def run_uncertainty(X: np.ndarray, rng: np.random.Generator):
    from . import uncertainty as U
    T, N = X.shape
    iu = np.triu_indices(N, 1)
    Z = _std(X)
    r = np.clip((Z.T @ Z / T)[iu], -0.999999, 0.999999)
    z = np.arctanh(r)
    b = U.fc_block_length(X, source="roi", kind="cb", n_series=256, rng=rng)
    idx = {"iid_bootstrap": U.iid_indices(T, R, rng), "cbb": U.cbb_indices(T, b, R, rng)}
    s2 = {"analytic_iid": np.full(z.shape, 1.0 / (T - 3))}
    for m, ix in idx.items():
        acc, acc2 = np.zeros(len(z)), np.zeros(len(z))
        for blk in replicate_z(X, ix, iu):
            acc += blk.sum(0); acc2 += (blk ** 2).sum(0)
        s2[m] = (acc2 - acc ** 2 / R) / (R - 1)
    s2["hac_bartlett"] = hac_bartlett_z(X, r, iu, 2.0 * b)
    return z, s2, b, idx


def _participant(k: int, sub: str, pe: str):
    from scipy.stats import spearmanr
    XA, XB = load_run(sub, pe, 1), load_run(sub, pe, 2)
    ok = all(np.isfinite(X).all() and (X.std(0) > 1e-8).all() for X in (XA, XB))
    if not ok:
        return sub, None
    zA, sA, bA, idxA = run_uncertainty(XA, np.random.default_rng(SEED + 10 * k + 1))
    zB, sB, bB, _ = run_uncertainty(XB, np.random.default_rng(SEED + 10 * k + 2))
    d = zB - zA
    md2 = float(np.mean(d ** 2))
    stats = {"sub": sub, "b_A": bA, "b_B": bB, "T_A": XA.shape[0], "T_B": XB.shape[0], "mean_d2": md2}
    for m in METHODS:
        v = sA[m] + sB[m]
        stats[f"vcr_{m}"] = float(np.mean(v) / md2)
        stats[f"cov95_{m}"] = float(np.mean(np.abs(d) <= 1.96 * np.sqrt(v)))
        stats[f"rho_{m}"] = float(spearmanr(np.sqrt(v), np.abs(d)).statistic) if m != "analytic_iid" else np.nan
    def pad(ix):
        out = np.full((R, T_FULL), -1, np.int16)
        out[:, : ix.shape[1]] = ix
        return out
    keep = {"zA": zA.astype(np.float32), "zB": zB.astype(np.float32),
            "s2A_analytic_iid": sA["analytic_iid"].astype(np.float32), "s2A_hac_bartlett": sA["hac_bartlett"].astype(np.float32),
            "idxA_iid_bootstrap": pad(idxA["iid_bootstrap"]), "idxA_cbb": pad(idxA["cbb"])}
    return sub, (stats, keep)


def features(pe: str, jobs: int):
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits
    d, flow = cohort(pe)
    elig = d[(d.fd_A <= FD_MAX) & (d.fd_B <= FD_MAX)].reset_index(drop=True)
    flow.append((f"mean FD <= {FD_MAX} in both runs", len(elig)))
    CACHE.mkdir(parents=True, exist_ok=True)

    def work(k, s):
        with threadpool_limits(3):
            return _participant(k, s, pe)
    allsub = d["sub"].tolist()
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(work)(allsub.index(s), s) for s in d["sub"])
    rows, keep = [], {}
    for sub, r in res:
        if r is None:
            continue
        rows.append(r[0]); keep[sub] = r[1]
    S = pd.DataFrame(rows).merge(d, on="sub")
    flow.append(("all parcels finite, non-zero variance (any FD)", len(S)))
    S.to_csv(CACHE / f"stats_{pe}.csv", index=False)
    subs = S["sub"].tolist()
    np.savez(CACHE / f"arrays_{pe}.npz", subs=np.array(subs), **{k: np.stack([keep[s][k] for s in subs]) for k in keep[subs[0]]})
    json.dump(flow, open(CACHE / f"flow_{pe}.json", "w"))
    return flow


def _boot_median_ci(x, n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def flips(S: pd.DataFrame, A: dict, pe: str, methods=METHODS) -> pd.DataFrame:
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    pos = {s: i for i, s in enumerate(A["subs"])}
    diag = [m for m in methods if m not in ("iid_bootstrap", "cbb")]
    S = S[S.Group.isin(["Patient", "GenPop"])].reset_index(drop=True)
    ix = np.array([pos[s] for s in S["sub"]])
    XA, XB = A["zA"][ix].astype(np.float64), A["zB"][ix].astype(np.float64)
    y = (S.Group == "Patient").astype(int).values
    strat = S.Group.astype(str) + "_" + S.Site.astype(str)
    lams = [0.01, 0.1, 1.0, 10.0]
    out = []
    iu = None
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=SEED).split(XA, strat):
        def fit(Xt, yt, lam):
            return LogisticRegression(C=1.0 / (2 * len(yt) * lam), max_iter=5000, tol=1e-6).fit(Xt, yt)
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
        ll = {}
        for lam in lams:
            v = []
            for a, b in inner.split(XA[tr], strat.values[tr]):
                p = np.clip(fit(XA[tr][a], y[tr][a], lam).predict_proba(XA[tr][b])[:, 1], 1e-6, 1 - 1e-6)
                v.append(-np.mean(y[tr][b] * np.log(p) + (1 - y[tr][b]) * np.log(1 - p)))
            ll[lam] = np.mean(v)
        lam = min(ll, key=ll.get)
        m = fit(XA[tr], y[tr], lam)
        w, c = m.coef_[0], float(m.intercept_[0])
        for i in te:
            sub = S["sub"][i]
            lA, lB = float(XA[i] @ w + c), float(XB[i] @ w + c)
            X = load_run(sub, pe, 1)
            iu = iu or np.triu_indices(X.shape[1], 1)
            row = {"sub": sub, "y": int(y[i]), "lam": lam, "lA": lA, "lB": lB, "flip": int(np.sign(lA) != np.sign(lB))}
            for mth in ("iid_bootstrap", "cbb"):
                ix = A[f"idxA_{mth}"][pos[sub]].astype(np.int64)[:, : X.shape[0]]
                L = np.concatenate([blk @ w for blk in replicate_z(X, ix, iu)])
                row[f"sl_{mth}"] = float(L.std(ddof=1))
            for mth in diag:
                row[f"sl_{mth}"] = float(np.sqrt(np.sum(w ** 2 * A[f"s2A_{mth}"][pos[sub]])))
            for mth in methods:
                row[f"P_{mth}"] = float(norm.cdf(-abs(lA) / (np.sqrt(2) * row[f"sl_{mth}"])))
            out.append(row)
    return pd.DataFrame(out)


METHODS_X = METHODS + ("xdf",)


def _xdf_participant(sub: str, pe: str):
    from scipy.stats import spearmanr
    out = []
    for run in (1, 2):
        X = load_run(sub, pe, run)
        iu = np.triu_indices(X.shape[1], 1)
        Z = _std(X)
        r = np.clip((Z.T @ Z / X.shape[0])[iu], -0.999999, 0.999999)
        out.append((np.arctanh(r), xdf_z(X, r, iu)))
    d, v = out[1][0] - out[0][0], out[0][1] + out[1][1]
    st = {"sub": sub, "vcr_xdf": float(np.mean(v) / np.mean(d ** 2)), "cov95_xdf": float(np.mean(np.abs(d) <= 1.96 * np.sqrt(v))),
          "rho_xdf": float(spearmanr(np.sqrt(v), np.abs(d)).statistic)}
    return st, out[0][1].astype(np.float32)


def xdf_stage(pe: str, jobs: int):
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits
    S = pd.read_csv(CACHE / f"stats_{pe}.csv")

    def work(s):
        with threadpool_limits(3):
            return _xdf_participant(s, pe)
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(work)(s) for s in S["sub"])
    pd.DataFrame([r[0] for r in res]).to_csv(CACHE / f"stats_xdf_{pe}.csv", index=False)
    np.savez(CACHE / f"arrays_xdf_{pe}.npz", subs=S["sub"].values.astype(str), s2A_xdf=np.stack([r[1] for r in res]))


def _paired_brier(Fl, a, b, n=10000):
    rng = np.random.default_rng(SEED)
    da = ((Fl[f"P_{a}"] - Fl.flip) ** 2 - (Fl[f"P_{b}"] - Fl.flip) ** 2).values
    bo = np.array([da[rng.integers(0, len(da), len(da))].mean() for _ in range(n)])
    return {"diff": float(da.mean()), "ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])],
            "p_two_sided": float(min(1.0, 2 * min(np.mean(bo <= 0), np.mean(bo >= 0))))}


def _e3_summary(Fl, methods):
    from sklearn.metrics import roc_auc_score
    e = {"n": len(Fl), "observed_flip_rate": float(Fl.flip.mean()), "cv_auc_scanA": float(roc_auc_score(Fl.y, Fl.lA))}
    for m in methods:
        e[m] = {"brier": float(np.mean((Fl[f"P_{m}"] - Fl.flip) ** 2)), "mean_P": float(Fl[f"P_{m}"].mean()),
                "auc": float(roc_auc_score(Fl.flip, Fl[f"P_{m}"])) if Fl.flip.nunique() == 2 else None}
    e["cbb_vs_iid_bootstrap"] = _paired_brier(Fl, "cbb", "iid_bootstrap")
    e["cbb_vs_xdf"] = _paired_brier(Fl, "cbb", "xdf")
    return e


def duplicate_columns(X: np.ndarray) -> list:
    return [j for j in range(1, X.shape[1]) if any(np.array_equal(X[:, j], X[:, i]) for i in range(j))]


def _nodup_participant(k: int, sub: str, pe: str):
    from scipy.stats import spearmanr
    XA, XB = load_run(sub, pe, 1), load_run(sub, pe, 2)
    keep = np.setdiff1d(np.arange(XA.shape[1]), duplicate_columns(XA))
    XA, XB = XA[:, keep], XB[:, keep]
    zA, sA, _, _ = run_uncertainty(XA, np.random.default_rng(SEED + 10 * k + 1))
    zB, sB, _, _ = run_uncertainty(XB, np.random.default_rng(SEED + 10 * k + 2))
    iu = np.triu_indices(XA.shape[1], 1)
    for s, X in ((sA, XA), (sB, XB)):
        Z = _std(X)
        s["xdf"] = xdf_z(X, np.clip((Z.T @ Z / X.shape[0])[iu], -0.999999, 0.999999), iu)
    d = zB - zA
    st = {"sub": sub, "n_parcels": len(keep), "max_abs_z": float(max(np.abs(zA).max(), np.abs(zB).max()))}
    for m in METHODS_X:
        v = sA[m] + sB[m]
        st[f"vcr_{m}"] = float(np.mean(v) / np.mean(d ** 2))
        st[f"cov95_{m}"] = float(np.mean(np.abs(d) <= 1.96 * np.sqrt(v)))
        st[f"rho_{m}"] = float(spearmanr(np.sqrt(v), np.abs(d)).statistic) if m != "analytic_iid" else np.nan
    return st


def nodup_sensitivity(pe: str, jobs: int) -> dict:
    from joblib import Parallel, delayed
    from scipy.stats import wilcoxon
    from threadpoolctl import threadpool_limits
    S = pd.read_csv(CACHE / f"stats_{pe}.csv")
    allsub = cohort(pe)[0]["sub"].tolist()

    def work(s):
        with threadpool_limits(2):
            return _nodup_participant(allsub.index(s), s, pe)
    N = pd.DataFrame(Parallel(n_jobs=jobs)(delayed(work)(s) for s in S["sub"])).merge(S[["sub", "fd_A", "fd_B"]], on="sub")
    P = N[(N.fd_A <= FD_MAX) & (N.fd_B <= FD_MAX)]
    r = {"n_primary": len(P), "n_parcels": sorted(P.n_parcels.unique().tolist()), "max_abs_z": float(P.max_abs_z.max())}
    r |= {m: {"median_vcr": float(P[f"vcr_{m}"].median()), "median_cov95": float(P[f"cov95_{m}"].median()),
              "median_rho": float(P[f"rho_{m}"].median()) if m != "analytic_iid" else None} for m in METHODS_X}
    dl = np.abs(np.log(P.vcr_cbb)) - np.abs(np.log(P.vcr_iid_bootstrap))
    dr = P.rho_cbb - P.rho_iid_bootstrap
    r["H1"] = {"median_diff": float(np.median(dl)), "wilcoxon_p": float(wilcoxon(dl).pvalue)}
    r["H2"] = {"median_diff": float(np.median(dr)), "wilcoxon_p": float(wilcoxon(dr).pvalue)}
    return r


def analyse(pe: str) -> dict:
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score
    from .stats import holm
    S = pd.read_csv(CACHE / f"stats_{pe}.csv")
    A = dict(np.load(CACHE / f"arrays_{pe}.npz"))
    A["subs"] = list(A["subs"])
    prim = S[(S.fd_A <= FD_MAX) & (S.fd_B <= FD_MAX)].reset_index(drop=True)
    res = {"pair": pe, "n_all": len(S), "n_primary": len(prim), "flow": json.load(open(CACHE / f"flow_{pe}.json"))}
    for name, D in (("primary", prim), ("sensitivity_no_fd_rule", S)):
        r = {m: {"median_vcr": float(D[f"vcr_{m}"].median()), "median_cov95": float(D[f"cov95_{m}"].median()),
                 "median_rho": float(D[f"rho_{m}"].median()) if m != "analytic_iid" else None} for m in METHODS}
        dl = np.abs(np.log(D.vcr_cbb)) - np.abs(np.log(D.vcr_iid_bootstrap))
        r["H1"] = {"median_diff_abslogvcr_cbb_minus_iid": float(np.median(dl)), "ci95": _boot_median_ci(dl.values),
                   "wilcoxon_p": float(wilcoxon(dl).pvalue)}
        dr = D.rho_cbb - D.rho_iid_bootstrap
        r["H2"] = {"median_diff_rho_cbb_minus_iid": float(np.median(dr)), "ci95": _boot_median_ci(dr.values),
                   "wilcoxon_p": float(wilcoxon(dr).pvalue)}
        res[name] = r
    Fl = flips(prim, A, pe)
    Fl.to_csv(OUT / f"discovery_flips_{pe}.csv", index=False)
    e3 = {"n": len(Fl), "observed_flip_rate": float(Fl.flip.mean()), "cv_auc_scanA": float(roc_auc_score(Fl.y, Fl.lA))}
    for m in METHODS:
        e3[m] = {"brier": float(np.mean((Fl[f"P_{m}"] - Fl.flip) ** 2)), "mean_P": float(Fl[f"P_{m}"].mean()),
                 "auc": float(roc_auc_score(Fl.flip, Fl[f"P_{m}"])) if Fl.flip.nunique() == 2 else None}
    rng = np.random.default_rng(SEED)
    da = (Fl.P_cbb - Fl.flip) ** 2 - (Fl.P_iid_bootstrap - Fl.flip) ** 2
    bo = np.array([da.values[rng.integers(0, len(da), len(da))].mean() for _ in range(10000)])
    e3["H3"] = {"brier_diff_cbb_minus_iid": float(da.mean()), "ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])],
                "p_two_sided": float(min(1.0, 2 * min(np.mean(bo <= 0), np.mean(bo >= 0))))}
    res["E3"] = e3
    p = res["primary"]
    h2p, h3p = holm([p["H2"]["wilcoxon_p"], e3["H3"]["p_two_sided"]])
    p["H2"]["p_holm"], e3["H3"]["p_holm"] = float(h2p), float(h3p)
    res["decisions"] = {
        "H1": bool(p["H1"]["median_diff_abslogvcr_cbb_minus_iid"] < 0 and p["H1"]["wilcoxon_p"] < 0.05),
        "H2": bool(p["H2"]["median_diff_rho_cbb_minus_iid"] > 0 and h2p < 0.05),
        "H3": bool(e3["H3"]["brier_diff_cbb_minus_iid"] < 0 and e3["H3"]["ci95"][1] < 0 and h3p < 0.05)}
    return res
