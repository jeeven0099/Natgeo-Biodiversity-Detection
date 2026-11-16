"""LAMDA clustering: Guerrero et al. (2023), section 2.3.4.

LAMDA (Learning Algorithm for Multivariate Data Analysis) in its unsupervised form,
ported line by line from Bedoya's MATLAB `LAMDA_General.m` ('unsupervised', 'binomial',
'3pi'):

  * MAD    binomial membership of each feature: rho^x (1 - rho)^(1 - x), data in [0, 1]
  * GAD    Yager 3pi connective: prod(MAD) / (prod(MAD) + prod(1 - MAD))
  * NIC    a no-information class with prototype 0.5 everywhere

A datum whose highest GAD is the NIC's (ties go to the NIC, as MATLAB `max` returns the
first) opens a new cluster with prototype mean(NIC, datum) = (x + 0.5) / 2. Otherwise
the winner's count n_c grows by one and rho_c += (x - rho_c) / n_c. Counts persist over
passes. After the learning passes, one more pass assigns each datum to its highest-GAD
cluster with the NIC left out; NaN GADs are skipped, as MATLAB `max` skips them.

LAMDA depends on the order in which the data arrive; the paper does not give one.
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd

N_ITER = 10
NIC = 0.5


@numba.njit(cache=True)
def gad(rho: np.ndarray, x: np.ndarray) -> float:
    """3pi GAD of datum x to one prototype rho, with binomial MADs."""
    pm, pn = 1.0, 1.0
    for j in range(x.shape[0]):
        m = rho[j] ** x[j] * (1.0 - rho[j]) ** (1.0 - x[j])
        pm *= m
        pn *= 1.0 - m
    return pm / (pm + pn)


@numba.njit(cache=True)
def _learn(data: np.ndarray, n_iter: int, cap: int):
    """The learning passes: (prototypes with the NIC at row 0, counts, number of rows used).
    The row count is -1 if more than `cap` prototypes were needed."""
    n, d = data.shape
    cp = np.empty((cap, d))
    cp[0] = NIC
    elements = np.zeros(cap, np.int64)
    nc = 1
    for _ in range(n_iter):
        for i in range(n):
            x = data[i]
            best, best_g = 0, gad(cp[0], x)
            for c in range(1, nc):
                g = gad(cp[c], x)
                if g > best_g:                     # NaN never wins; ties stay with the first
                    best, best_g = c, g
            if best == 0:
                if nc == cap:
                    return cp, elements, -1
                cp[nc] = (cp[0] + x) / 2.0
                elements[nc] = 1
                nc += 1
            else:
                elements[best] += 1
                cp[best] += (x - cp[best]) / elements[best]
    return cp, elements, nc


@numba.njit(cache=True)
def _assign(data: np.ndarray, cp: np.ndarray, nc: int):
    """Final pass without cluster creation, NIC left out: (cluster 0..nc-2, its GAD)."""
    n = data.shape[0]
    idx = np.zeros(n, np.int64)
    gmax = np.full(n, np.nan)
    for i in range(n):
        for c in range(1, nc):
            g = gad(cp[c], data[i])
            if g == g and not (gmax[i] >= g):
                idx[i], gmax[i] = c - 1, g
    return idx, gmax


def lamda(data: np.ndarray, n_iter: int = N_ITER, cap: int = 4096):
    """Cluster rows of `data` (values in [0, 1]) in the order given.

    Returns (cluster index per row, its GAD, prototypes (n_clusters x d), learning counts).
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    if data.ndim != 2 or len(data) == 0:
        raise ValueError("data must be a non-empty 2-D array")
    if np.nanmin(data) < 0 or np.nanmax(data) > 1:
        raise ValueError("LAMDA's binomial MAD needs every feature scaled to [0, 1]")
    while True:
        cp, elements, nc = _learn(data, n_iter, cap)
        if nc > 0:
            break
        cap *= 4
    idx, gmax = _assign(data, cp, nc)
    return idx, gmax, cp[1:nc].copy(), elements[1:nc].copy()


def scale(raw: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Scale each column to [0, 1].

    minmax  the paper (LAMDA_General.m): (x - min) / (max - min) over all rows.
    rank    NOT in the paper: each value's mid-rank percentile, ties averaged, so every
            feature splits the rows at 0.5, the only cut a binomial MAD can make.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if method == "minmax":
        lo, hi = raw.min(0), raw.max(0)
        return (raw - lo) / np.where(hi > lo, hi - lo, 1.0)
    if method == "rank":
        return (pd.DataFrame(raw).rank(method="average").to_numpy() - 0.5) / len(raw)
    raise ValueError(f"unknown scaling {method!r}")
