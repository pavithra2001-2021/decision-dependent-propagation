"""Edge-level uncertainty estimators and resampling indices."""
from __future__ import annotations

import numpy as np

from .fc import corr, standardize, triu_index, vec


def influence_series(X: np.ndarray, C: np.ndarray | None = None) -> np.ndarray:
    Z = standardize(X)
    if C is None:
        C = corr(X)
    i, j = triu_index(Z.shape[1])
    r = C[i, j]
    return Z[:, i] * Z[:, j] - 0.5 * r * (Z[:, i] ** 2 + Z[:, j] ** 2)


def pw_block_length(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    e = x - x.mean()
    b_max = np.ceil(min(3.0 * np.sqrt(n), n / 3.0))
    kn = max(5, int(np.log10(n)))
    m_max = int(np.ceil(np.sqrt(n))) + kn
    cv = 2.0 * np.sqrt(np.log10(n) / n)
    acv = np.array([e[k:] @ e[: n - k] / n for k in range(m_max + 1)])
    rho = np.abs(acv[1:] / acv[0]) if acv[0] > 0 else np.zeros(m_max)
    m_hat = None
    for m in range(0, m_max - kn + 1):
        if np.all(rho[m : m + kn] < cv):
            m_hat = m
            break
    M = m_max if m_hat is None else min(2 * max(m_hat, 1), m_max)
    k = np.arange(1, M + 1)
    lam = np.where(k / M <= 0.5, 1.0, 2.0 * (1.0 - k / M))
    g = np.sum(2.0 * lam * k * acv[1 : M + 1])
    lrv = acv[0] + np.sum(2.0 * lam * acv[1 : M + 1])
    if lrv <= 0 or g == 0:
        return 1.0, 1.0
    b_sb = (2.0 * g**2 / (2.0 * lrv**2)) ** (1 / 3) * n ** (1 / 3)
    b_cb = (2.0 * g**2 / ((4.0 / 3.0) * lrv**2)) ** (1 / 3) * n ** (1 / 3)
    return float(np.clip(b_sb, 1.0, b_max)), float(np.clip(b_cb, 1.0, b_max))


def fc_block_length(X: np.ndarray, source: str = "if", kind: str = "cb",
                    n_series: int = 256, rng: np.random.Generator | None = None) -> float:
    rng = np.random.default_rng(0) if rng is None else rng
    if source == "if":
        S = influence_series(X)
    elif source == "roi":
        S = standardize(X)
    else:
        raise ValueError(source)
    cols = S.shape[1]
    pick = rng.choice(cols, size=min(n_series, cols), replace=False)
    pos = 0 if kind == "sb" else 1
    return float(np.median([pw_block_length(S[:, c])[pos] for c in pick]))


def iid_indices(T: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, T, size=(B, T))


def cbb_indices(T: int, b: float, B: int, rng: np.random.Generator) -> np.ndarray:
    b = max(1, int(round(b)))
    nb = int(np.ceil(T / b))
    starts = rng.integers(0, T, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % T
    return idx.reshape(B, nb * b)[:, :T]


def sb_indices(T: int, b: float, B: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(float(b), 1.0)
    new = rng.random((B, T)) < p
    new[:, 0] = True
    t = np.arange(T)
    last = np.maximum.accumulate(np.where(new, t[None, :], 0), axis=1)
    starts = rng.integers(0, T, size=(B, T))
    s = np.take_along_axis(starts, last, axis=1)
    return (s + t[None, :] - last) % T


def resampled_fc(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return corr(np.asarray(X, dtype=np.float64)[idx])


def _circular_box(T: int, l: int) -> np.ndarray:
    k = np.zeros(T)
    k[:l] = 1.0 / np.sqrt(l)
    return k


def dwb_factor(X: np.ndarray, l: float, C: np.ndarray | None = None) -> np.ndarray:
    IF = influence_series(X, C)
    T = IF.shape[0]
    l = int(min(max(1, round(l)), T - 1))
    k = _circular_box(T, l)
    G = np.real(np.fft.ifft(np.fft.fft(k)[:, None] * np.fft.fft(IF, axis=0), axis=0))
    return G / T


def dwb_draws(G: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((B, G.shape[0])) @ G


def dwb_edge_variance(G: np.ndarray) -> np.ndarray:
    return np.einsum("te,te->e", G, G)


def lag_window_edge_variance(X: np.ndarray, bw: float, kernel: str = "flattop",
                             C: np.ndarray | None = None) -> np.ndarray:
    IF = influence_series(X, C)
    T = IF.shape[0]
    H = int(min(np.ceil(bw), T - 1))
    F = np.fft.rfft(IF, n=2 * T, axis=0)
    g = np.fft.irfft(np.abs(F) ** 2, n=2 * T, axis=0)[: H + 1] / T
    x = np.arange(1, H + 1) / bw
    if kernel == "bartlett":
        k = np.clip(1.0 - x, 0.0, None)
    elif kernel == "flattop":
        k = np.where(x <= 0.5, 1.0, np.clip(2.0 * (1.0 - x), 0.0, None))
    else:
        raise ValueError(kernel)
    return (g[0] + 2.0 * (k[:, None] * g[1:]).sum(0)) / T


def fc_replicates(X: np.ndarray, method: str, B: int, rng: np.random.Generator,
                  block: float | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    T = X.shape[0]
    if method != "iid" and block is None:
        block = fc_block_length(X, source="if", kind="sb" if method == "sb" else "cb", rng=rng)
    if method == "iid":
        return vec(resampled_fc(X, iid_indices(T, B, rng)))
    if method == "cbb":
        return vec(resampled_fc(X, cbb_indices(T, block, B, rng)))
    if method == "sb":
        return vec(resampled_fc(X, sb_indices(T, block, B, rng)))
    if method == "dwb":
        C = corr(X)
        return vec(C)[None, :] + dwb_draws(dwb_factor(X, block, C), B, rng)
    raise ValueError(method)
