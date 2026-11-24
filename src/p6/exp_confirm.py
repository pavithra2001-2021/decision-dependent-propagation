"""Confirmation analysis (CoRR NYU_2)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import exp_retest as E
from .exp_mechanism import decision_sigma
from .registry import ROOT, Run

SEED = 20261024
R = 100
DERIV = ROOT / "data" / "external" / "derived" / "corr_nyu2"
OUT = ROOT / "results"
METHODS = ("J_cbb", "J_iid", "J_iidm", "J_xdfjoint", "J_xdfadaptjoint", "I_xdf", "I_xdf_adaptive", "I_cbb", "ISO", "uHAC")


def run_measurement(X: np.ndarray, rng: np.random.Generator, n_rep: int = R) -> dict:
    T, N = X.shape
    iu = np.triu_indices(N, 1)
    Z = E._std(X)
    r = np.clip((Z.T @ Z / T)[iu], -0.999999, 0.999999)
    z = np.arctanh(r)
    from . import uncertainty as U
    b = U.fc_block_length(X, source="roi", kind="cb", n_series=min(256, N), rng=rng)
    idx = {"iid": U.iid_indices(T, n_rep, rng), "cbb": U.cbb_indices(T, b, n_rep, rng)}
    rep = {m: np.concatenate(list(E.replicate_z(X, ix, iu))) for m, ix in idx.items()}
    dev = {m: v - v.mean(0) for m, v in rep.items()}
    s2 = {"cbb": dev["cbb"].var(0, ddof=1), "iid": dev["iid"].var(0, ddof=1),
          "xdf": E.xdf_z(X, r, iu), "xdf_adaptive": E.xdf_z_adaptive(X, r, iu)}
    return {"X": X, "z": z, "b": b, "dev": dev, "s2": s2}


def decision_sds(meas: dict, w: np.ndarray) -> dict:
    dc, di, s2 = meas["dev"]["cbb"], meas["dev"]["iid"], meas["s2"]
    sd = lambda D: float((D @ w).std(ddof=1))
    out = {"J_cbb": sd(dc), "J_iid": sd(di),
           "J_iidm": sd(di * np.sqrt(s2["cbb"] / np.maximum(s2["iid"], 1e-12))),
           "J_xdfjoint": sd(dc * np.sqrt(s2["xdf"] / np.maximum(s2["cbb"], 1e-12))),
           "J_xdfadaptjoint": sd(dc * np.sqrt(s2["xdf_adaptive"] / np.maximum(s2["cbb"], 1e-12))),
           "I_xdf": float(np.sqrt(np.sum(w ** 2 * s2["xdf"]))),
           "I_xdf_adaptive": float(np.sqrt(np.sum(w ** 2 * s2["xdf_adaptive"]))),
           "I_cbb": float(np.sqrt(np.sum(w ** 2 * s2["cbb"]))),
           "ISO": float(np.sqrt(np.mean(s2["cbb"]) * np.sum(w ** 2)))}
    t0 = time.process_time()
    out["uHAC"] = float(decision_sigma(meas["X"], w, meas["b"])["uhac_bartlett"])
    out["_t_uHAC"] = time.process_time() - t0
    return out


def nlpd(d: np.ndarray, var_diff: np.ndarray) -> np.ndarray:
    return 0.5 * np.log(2 * np.pi * var_diff) + d ** 2 / (2 * var_diff)


def paired(x: np.ndarray, n_boot: int = 10000, seed: int = SEED) -> dict:
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bo = np.median(x[rng.integers(0, len(x), (n_boot, len(x)))], axis=1)
    return {"n": int(len(x)), "median": float(np.median(x)), "ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])],
            "frac_negative": float(np.mean(x < 0)), "wilcoxon_p": float(wilcoxon(x).pvalue)}


def mean_ci(x: np.ndarray, n_boot: int = 10000, seed: int = SEED) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bo = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return {"n": int(len(x)), "mean": float(x.mean()), "ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])]}


def supported(res: dict, p_key: str = "wilcoxon_p") -> bool:
    return bool(res["median"] < 0 and res[p_key] < 0.05)


def load_cohort(fd_rule: bool = True) -> pd.DataFrame:
    import scipy.io as sio
    rows = []
    for d in sorted(p for p in DERIV.glob("sub-*_run1") if p.is_dir()):
        pid = d.name.split("_")[0]
        rec = {"pid": pid}
        ok = True
        for r in (1, 2):
            dd = DERIV / f"{pid}_run{r}"
            f = dd / f"ROISignals_{pid}_run{r}.mat"
            if not f.exists():
                ok = False
                break
            X = sio.loadmat(f)["ROISignals"][:, :116].astype(np.float64)
            fd = np.loadtxt(next(dd.glob("FD_Jenkinson_*.txt")))
            rec |= {f"T{r}": X.shape[0], f"fd{r}": float(fd.mean()),
                    f"cov{r}": bool(np.isfinite(X).all() and (X.std(0) > 1e-8).all())}
        if not ok:
            continue
        rows.append(rec)
    C = pd.DataFrame(rows)
    sex = json.loads((DERIV / "sex.json").read_text())
    C["sex"] = C.pid.map(lambda p: sex.get(p.replace("sub-", "")))
    keep = (C.T1 >= 150) & (C.T2 >= 150) & C.cov1 & C.cov2 & C.sex.isin(["male", "female"])
    if fd_rule:
        keep &= (C.fd1 <= 0.2) & (C.fd2 <= 0.2)
    return C[keep].sort_values("pid").reset_index(drop=True)


def analyse(fd_rule: bool = True, n_rep: int = R, swap: bool = False) -> tuple[dict, pd.DataFrame]:
    import scipy.io as sio
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from .stats import holm
    C = load_cohort(fd_rule)
    n = len(C)
    load = lambda pid, r: sio.loadmat(DERIV / f"{pid}_run{r}" / f"ROISignals_{pid}_run{r}.mat")["ROISignals"][:, :116].astype(np.float64)
    meas = {}
    for k, pid in enumerate(C.pid):
        A, B = (2, 1) if swap else (1, 2)
        meas[pid] = (run_measurement(load(pid, A), np.random.default_rng(SEED + 10 * k + 1), n_rep),
                     run_measurement(load(pid, B), np.random.default_rng(SEED + 10 * k + 2), n_rep))
    ZA = np.stack([meas[p][0]["z"] for p in C.pid]); ZB = np.stack([meas[p][1]["z"] for p in C.pid])
    E_ = ZA.shape[1]
    y = (C.sex == "female").astype(int).values
    W1, c1 = np.zeros((n, E_)), np.zeros(n)
    lams = [0.01, 0.1, 1.0, 10.0]
    fit = lambda X, t, lam: LogisticRegression(C=1.0 / (2 * len(t) * lam), max_iter=5000, tol=1e-6).fit(X, t)
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=SEED).split(ZA, y):
        ll = {}
        for lam in lams:
            v = []
            for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(ZA[tr], y[tr]):
                p = np.clip(fit(ZA[tr][a], y[tr][a], lam).predict_proba(ZA[tr][b])[:, 1], 1e-6, 1 - 1e-6)
                v.append(-np.mean(y[tr][b] * np.log(p) + (1 - y[tr][b]) * np.log(1 - p)))
            ll[lam] = np.mean(v)
        m = fit(ZA[tr], y[tr], min(ll, key=ll.get))
        W1[te], c1[te] = m.coef_[0], m.intercept_[0]
    w2 = np.full(E_, 1.0 / E_)
    w3 = np.random.default_rng(SEED).standard_normal(E_) / np.sqrt(E_)
    rows = []
    for i, pid in enumerate(C.pid):
        mA, mB = meas[pid]
        for D, w, c in (("D1", W1[i], c1[i]), ("D2", w2, 0.0), ("D3", w3, 0.0)):
            sa, sb = decision_sds(mA, w), decision_sds(mB, w)
            lA, lB = float(mA["z"] @ w + c), float(mB["z"] @ w + c)
            rows.append({"pid": pid, "D": D, "lA": lA, "lB": lB, "y": int(y[i])}
                        | {f"sA_{m}": sa[m] for m in METHODS} | {f"sB_{m}": sb[m] for m in METHODS} | {"t_uHAC": sa["_t_uHAC"]})
    P = pd.DataFrame(rows)
    d2 = (P.lB - P.lA) ** 2
    for m in METHODS:
        P[f"nlpd_{m}"] = nlpd(np.sqrt(d2), 2 * P[f"sA_{m}"] ** 2)
        P[f"nlpdAB_{m}"] = nlpd(np.sqrt(d2), P[f"sA_{m}"] ** 2 + P[f"sB_{m}"] ** 2)
    G = {D: P[P.D == D].reset_index(drop=True) for D in ("D1", "D2", "D3")}
    diff = lambda D, a, b, col="nlpd": (G[D][f"{col}_{a}"] - G[D][f"{col}_{b}"]).values
    res = {"n": n, "fd_rule": fd_rule, "n_rep": n_rep, "swap": swap}
    res["PRIMARY"] = paired(diff("D1", "J_cbb", "I_xdf"))
    sec = {"S1_I_cbb": paired(diff("D1", "J_cbb", "I_cbb")), "S2_J_iidm": paired(diff("D1", "J_cbb", "J_iidm")),
           "S3_J_iid": paired(diff("D1", "J_cbb", "J_iid")), "S6_D2": paired(diff("D2", "J_cbb", "I_xdf")),
           "S6_D3": paired(diff("D3", "J_cbb", "I_xdf")), "S11_I_xdf_adaptive": paired(diff("D1", "J_cbb", "I_xdf_adaptive"))}
    from scipy.stats import spearmanr
    ev = []
    for pid in C.pid:
        mA, mB = meas[pid]
        dz = mB["z"] - mA["z"]
        row = {}
        for m in ("cbb", "iid", "xdf", "xdf_adaptive"):
            v = mA["s2"][m] + mB["s2"][m]
            row[m] = (np.mean(v) / np.mean(dz ** 2), np.mean(np.abs(dz) <= 1.96 * np.sqrt(v)),
                      spearmanr(np.sqrt(v), np.abs(dz)).statistic)
        ev.append(row)
    sec["S8_abslogVCR"] = paired(np.array([abs(np.log(e["cbb"][0])) - abs(np.log(e["iid"][0])) for e in ev]))
    sec["S8_rho"] = paired(np.array([-(e["cbb"][2] - e["iid"][2]) for e in ev]))
    adj = holm([v["wilcoxon_p"] for v in sec.values()])
    for (k, v), a in zip(sec.items(), adj):
        v["p_holm"] = float(a)
        v["supported"] = supported(v, "p_holm")
    res["secondary"] = sec
    res["edge_level_medians"] = {m: {"vcr": float(np.median([e[m][0] for e in ev])), "cov95": float(np.median([e[m][1] for e in ev])),
                                     "rho": float(np.median([e[m][2] for e in ev]))} for m in ("cbb", "iid", "xdf", "xdf_adaptive")}
    D1 = G["D1"]
    res["estimation"] = {
        "S4_J_xdfjoint": paired(diff("D1", "J_cbb", "J_xdfjoint")), "S12_J_xdfadaptjoint": paired(diff("D1", "J_cbb", "J_xdfadaptjoint")),
        "S5_uHAC_ratio_median": float(np.median(D1.sA_uHAC / D1.sA_J_cbb)), "S5_nlpd_uHAC_minus_Jcbb": paired(diff("D1", "uHAC", "J_cbb")),
        "S5_cpu_s_uHAC_median": float(D1.t_uHAC.median()),
        "S7_decision_vcr": {m: float(np.mean(2 * D1[f"sA_{m}"] ** 2) / np.mean((D1.lB - D1.lA) ** 2)) for m in METHODS},
        "A3_sigmaA2_plus_sigmaB2_primary": paired(diff("D1", "J_cbb", "I_xdf", "nlpdAB")),
        "S13_mean_nlpd_diff": mean_ci(diff("D1", "J_cbb", "I_xdf"))}
    rng = np.random.default_rng(SEED)
    bo = []
    for _ in range(10000):
        k = rng.integers(0, len(D1), len(D1))
        bo.append(np.mean(2 * D1.sA_J_cbb.values[k] ** 2) / np.mean((D1.lB.values[k] - D1.lA.values[k]) ** 2))
    res["estimation"]["S7_J_cbb_ci95"] = [float(v) for v in np.percentile(bo, [2.5, 97.5])]
    res["S9_flips"] = flips(D1)
    res["decisions"] = {"PRIMARY_supported": supported(res["PRIMARY"]),
                        "robust_to_xDF_variant": supported(res["PRIMARY"]) and sec["S11_I_xdf_adaptive"]["supported"]}
    return res, P


def flips(D1: pd.DataFrame) -> dict:
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from .exp_mechanism import _metrics, _paired
    f = (np.sign(D1.lA) != np.sign(D1.lB)).astype(int).values
    out = {"observed_flip_rate": float(f.mean())}
    P = {m: norm.cdf(-np.abs(D1.lA.values) / (np.sqrt(2) * D1[f"sA_{m}"].values)) for m in ("J_cbb", "I_xdf", "J_iid", "I_xdf_adaptive")}
    out |= {m: _metrics(p, f) for m, p in P.items()}
    if 0 < f.sum() < len(f):
        out["brier_J_cbb_minus_I_xdf"] = _paired(P["J_cbb"], P["I_xdf"], f)
        rng = np.random.default_rng(SEED)
        a1, a0 = roc_auc_score(f, P["J_cbb"]), roc_auc_score(f, -np.abs(D1.lA.values))
        bo = []
        for _ in range(2000):
            k = rng.integers(0, len(f), len(f))
            if 0 < f[k].sum() < len(k):
                bo.append(roc_auc_score(f[k], P["J_cbb"][k]) - roc_auc_score(f[k], -np.abs(D1.lA.values[k])))
        out["auc_J_cbb_minus_margin"] = {"diff": float(a1 - a0), "ci95": [float(v) for v in np.percentile(bo, [2.5, 97.5])]}
    S = pd.read_csv(OUT / "discovery_decisions_AP.csv.gz"), pd.read_csv(OUT / "discovery_decisions_PA.csv.gz")
    B = pd.concat([s[s.model == "ridge_nested"] for s in S])
    zt = (np.abs(B.lA) / B.sigma_J_cbb).values[:, None]
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(zt + 1e-9), B.flip.values)
    pc = cal.predict_proba(np.log((np.abs(D1.lA) / D1.sA_J_cbb).values[:, None] + 1e-9))[:, 1]
    out["S10_transfer_recalibrated_J_cbb"] = _metrics(pc, f) if 0 < f.sum() < len(f) else None
    return out


def main():
    import argparse
    import hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", default="")
    a = ap.parse_args()
    if a.confirm != "CONFIRMATORY-ANALYSIS":
        raise SystemExit("confirmation study is analysed once, after all preprocessing: pass --confirm CONFIRMATORY-ANALYSIS")
    shas = {f: hashlib.sha256((ROOT / "configs" / f).read_bytes()).hexdigest()
            for f in ("protocol_confirm.json", "protocol_confirm_amendment1.json", "protocol_confirm_amendment2.json")}
    assert shas["protocol_confirm.json"].startswith("ad69441d") and shas["protocol_confirm_amendment1.json"].startswith("57dd1ad5") \
        and shas["protocol_confirm_amendment2.json"].startswith("d05b5ba5")
    dst = OUT / "confirmation_results.json"
    if dst.exists():
        raise SystemExit("confirmation study results already exist (one-time)")
    with Run("confirmation_analysis", shas, [str(dst.relative_to(ROOT))]):
        res, P = analyse()
        res["sensitivity"] = {"no_fd_rule": analyse(fd_rule=False)[0]["PRIMARY"], "R20": analyse(n_rep=20)[0]["PRIMARY"],
                              "swapped_roles": analyse(swap=True)[0]["PRIMARY"]}
        res["protocol_sha256"] = shas
        P.to_csv(OUT / "confirmation_participants.csv.gz", index=False, compression="gzip")
        dst.write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ("n", "PRIMARY", "decisions")}, indent=1))


if __name__ == "__main__":
    main()
