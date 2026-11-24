"""Functional-connectivity utilities."""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def standardize(X: np.ndarray, axis: int = -2) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, keepdims=True)
    return (X - mu) / np.maximum(sd, _EPS)


def corr(X: np.ndarray) -> np.ndarray:
    Z = standardize(X)
    T = Z.shape[-2]
    C = np.einsum("...ti,...tj->...ij", Z, Z) / T
    idx = np.arange(C.shape[-1])
    C[..., idx, idx] = 1.0
    return np.clip(C, -1.0, 1.0)


def triu_index(N: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(N, k=1)


def vec(C: np.ndarray) -> np.ndarray:
    iu = triu_index(C.shape[-1])
    return C[..., iu[0], iu[1]]


def unvec(v: np.ndarray, N: int, diag: float = 1.0) -> np.ndarray:
    v = np.asarray(v)
    out = np.zeros(v.shape[:-1] + (N, N), dtype=v.dtype)
    iu = triu_index(N)
    out[..., iu[0], iu[1]] = v
    out[..., iu[1], iu[0]] = v
    idx = np.arange(N)
    out[..., idx, idx] = diag
    return out


def fisher_z(r: np.ndarray, clip: float = 0.999) -> np.ndarray:
    return np.arctanh(np.clip(r, -clip, clip))


def valid_rois(X: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.isfinite(X).all(axis=0) & (X.std(axis=0) > tol)
