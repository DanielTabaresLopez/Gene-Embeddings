#!/usr/bin/env python3
"""
pfi_metrics.py - metrics for the Pairwise Full-Intersection analysis.

Every pair is evaluated on its ENTIRE shared valid gene set. Sample sizes
therefore differ between pairs, which has one important consequence:

  - Unbiased CKA has expectation 0 under independence at any n, so its point
    estimate is already null-centred and stays roughly comparable across pairs.
  - Raw k-NN overlap does NOT. Its chance baseline is about k/n, so a pair with
    a small intersection gets a larger raw number for free. Every local result
    must therefore be read as a null-corrected effect, never raw.

That is why the permutation null is computed for every pair rather than being
optional. It is cheap: permuting gene labels relabels neighbourhoods without
recomputing any similarity matrix.

Convention assumptions to verify against the atlas (see pfi_03):
  1. "row-normalized" means each GENE vector is L2-normalized, so the linear
     kernel becomes cosine similarity.
  2. RBF bandwidth: sigma = factor * sqrt(median squared pairwise distance),
     kernel exp(-d2 / (2 sigma^2)), median from a deterministic subsample.
  3. Tie-aware overlap gives each gene total weight exactly k and combines the
     two sides with an elementwise MINIMUM ("product" is the variant; the two
     coincide when one side is tie-free).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- preprocessing


def l2_row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def preprocess(matrix: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return np.ascontiguousarray(matrix, dtype=np.float64)
    if mode == "row_normalized":
        return np.ascontiguousarray(l2_row_normalize(matrix), dtype=np.float64)
    raise ValueError(f"unknown preprocessing mode: {mode}")


# ------------------------------------------------------- unbiased linear CKA
# Song et al. (2012) unbiased HSIC, evaluated from the feature side so that no
# n x n matrix is ever formed. Cost O(n*d*e), which is what makes full
# intersections of 18,000+ genes affordable.


def _hsic1_linear(X: np.ndarray, Y: np.ndarray) -> float:
    n = X.shape[0]
    if n < 4:
        return float("nan")
    kd = np.einsum("ij,ij->i", X, X)
    ld = np.einsum("ij,ij->i", Y, Y)
    sum_kd_ld = float(kd @ ld)
    XtY = X.T @ Y
    tr_term = float((XtY**2).sum()) - sum_kd_ld
    sx, sy = X.sum(axis=0), Y.sum(axis=0)
    sum_k = float(sx @ sx) - float(kd.sum())
    sum_l = float(sy @ sy) - float(ld.sum())
    cross = (
        float(sx @ (XtY @ sy))
        - float((X @ sx) @ ld)
        - float(kd @ (Y @ sy))
        + sum_kd_ld
    )
    return (
        tr_term + sum_k * sum_l / ((n - 1) * (n - 2)) - 2.0 * cross / (n - 2)
    ) / (n * (n - 3))


def unbiased_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    hxy, hxx, hyy = _hsic1_linear(X, Y), _hsic1_linear(X, X), _hsic1_linear(Y, Y)
    if not np.isfinite(hxy) or hxx <= 0 or hyy <= 0:
        return float("nan")
    return float(hxy / np.sqrt(hxx * hyy))


# --------------------------------------------------------- unbiased RBF CKA


def median_squared_distance(X: np.ndarray, sample: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    sub = X[np.sort(rng.choice(X.shape[0], sample, replace=False))] if X.shape[0] > sample else X
    sq = np.einsum("ij,ij->i", sub, sub)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (sub @ sub.T), 0.0)
    upper = d2[np.triu_indices_from(d2, k=1)]
    positive = upper[upper > 0]
    return float(np.median(positive)) if positive.size else 1.0


def _rbf_block(X, rows, sq, gamma):
    block = sq[rows][:, None] + sq[None, :] - 2.0 * (X[rows] @ X.T)
    np.maximum(block, 0.0, out=block)
    return np.exp(-gamma * block)


def _hsic1_gram_blockwise(X, Y, gamma_x, gamma_y, block_size, want):
    n = X.shape[0]
    if n < 4:
        return float("nan")
    sqx = np.einsum("ij,ij->i", X, X)
    sqy = np.einsum("ij,ij->i", Y, Y)
    tr_term = 0.0
    rows_k, rows_l = np.zeros(n), np.zeros(n)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        rows = slice(start, stop)
        if want in ("xy", "xx"):
            kb = _rbf_block(X, rows, sqx, gamma_x)
            np.fill_diagonal(kb[:, start:stop], 0.0)
        if want in ("xy", "yy"):
            lb = _rbf_block(Y, rows, sqy, gamma_y)
            np.fill_diagonal(lb[:, start:stop], 0.0)
        first, second = (kb, lb) if want == "xy" else ((kb, kb) if want == "xx" else (lb, lb))
        tr_term += float((first * second).sum())
        rows_k[rows] = first.sum(axis=1)
        rows_l[rows] = second.sum(axis=1)
    sum_k, sum_l = float(rows_k.sum()), float(rows_l.sum())
    cross = float(rows_k @ rows_l)
    return (
        tr_term + sum_k * sum_l / ((n - 1) * (n - 2)) - 2.0 * cross / (n - 2)
    ) / (n * (n - 3))


def unbiased_rbf_cka(X, Y, bandwidth_factor=1.0, block_size=1024, seed=0) -> float:
    med_x = median_squared_distance(X, seed=seed)
    med_y = median_squared_distance(Y, seed=seed)
    gx = 1.0 / (2.0 * (bandwidth_factor**2) * med_x)
    gy = 1.0 / (2.0 * (bandwidth_factor**2) * med_y)
    hxy = _hsic1_gram_blockwise(X, Y, gx, gy, block_size, "xy")
    hxx = _hsic1_gram_blockwise(X, Y, gx, gy, block_size, "xx")
    hyy = _hsic1_gram_blockwise(X, Y, gx, gy, block_size, "yy")
    if not np.isfinite(hxy) or hxx <= 0 or hyy <= 0:
        return float("nan")
    return float(hxy / np.sqrt(hxx * hyy))


# ------------------------------------------- tie-aware local overlap + nulls


def _prepare_block(sim_block: np.ndarray, offset: int) -> np.ndarray:
    """Cast a similarity block to float64 and mask self-similarity."""
    work = sim_block.astype(np.float64, copy=True)
    rows = np.arange(work.shape[0])
    work[rows, offset + rows] = -np.inf
    return work


def _tie_aware_weights(work: np.ndarray, k: int):
    """Weights sum to exactly k per row; boundary ties share the remainder."""
    n_cols = work.shape[1]
    threshold = np.partition(work, n_cols - k, axis=1)[:, n_cols - k][:, None]
    greater = work > threshold
    equal = work == threshold
    n_greater = greater.sum(axis=1)
    n_equal = equal.sum(axis=1)
    share = ((k - n_greater) / np.where(n_equal == 0, 1, n_equal))[:, None]
    weights = greater.astype(np.float64) + equal * share
    top_k = np.argpartition(-work, k - 1, axis=1)[:, :k]
    return weights, top_k, int((n_equal > 1).sum())


def local_overlap(Xn32, Yn32, ks, block_size=512, mode="min", collect_topk=True):
    """
    Xn32 and Yn32 must be row-normalized float32 (cosine geometry).

    The cosine block for each row range is computed ONCE and reused for every
    k, which matters a lot at full-intersection sizes.

    Returns {k: {overlap, per_gene, boundary_tie_rate_x/y, topk_x, topk_y}}.
    """
    n = Xn32.shape[0]
    usable = [k for k in ks if k < n]

    per_gene = {k: np.empty(n) for k in usable}
    ties = {k: [0, 0] for k in usable}
    topk = {
        k: (
            np.empty((n, k), dtype=np.int32) if collect_topk else None,
            np.empty((n, k), dtype=np.int32) if collect_topk else None,
        )
        for k in usable
    }

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        rows = slice(start, stop)
        work_x = _prepare_block(Xn32[rows] @ Xn32.T, start)
        work_y = _prepare_block(Yn32[rows] @ Yn32.T, start)
        for k in usable:
            wx, tx, bx = _tie_aware_weights(work_x, k)
            wy, ty, by = _tie_aware_weights(work_y, k)
            ties[k][0] += bx
            ties[k][1] += by
            combined = np.minimum(wx, wy) if mode == "min" else wx * wy
            per_gene[k][rows] = combined.sum(axis=1) / k
            if collect_topk:
                topk[k][0][rows], topk[k][1][rows] = tx, ty

    out = {}
    for k in ks:
        if k not in usable:
            out[k] = {"overlap": float("nan"), "per_gene": np.full(n, np.nan),
                      "boundary_tie_rate_x": float("nan"),
                      "boundary_tie_rate_y": float("nan"),
                      "topk_x": None, "topk_y": None}
            continue
        out[k] = {"overlap": float(np.nanmean(per_gene[k])), "per_gene": per_gene[k],
                  "boundary_tie_rate_x": ties[k][0] / n,
                  "boundary_tie_rate_y": ties[k][1] / n,
                  "topk_x": topk[k][0], "topk_y": topk[k][1]}
    return out


def analytic_local_baseline(n: int, k: int) -> float:
    return k / (n - 1)


def local_permutation_null(topk_x, topk_y, n_permutations: int, seed: int):
    """
    Gene-label permutation null for hard top-k overlap. Intersections are
    counted by sorting concatenated index rows and counting adjacent equal
    entries, which is fully vectorized.
    """
    n, k = topk_x.shape
    rng = np.random.default_rng(seed)
    values = np.empty(n_permutations)
    for b in range(n_permutations):
        sigma = rng.permutation(n)
        inverse = np.empty(n, dtype=np.int64)
        inverse[sigma] = np.arange(n)
        permuted = inverse[topk_y[sigma]]
        merged = np.sort(np.concatenate([topk_x, permuted], axis=1), axis=1)
        values[b] = (merged[:, 1:] == merged[:, :-1]).sum(axis=1).mean() / k
    return float(values.mean()), float(values.std(ddof=1))


def deterministic_subsample(gene_ids: np.ndarray, n: int, seed: int) -> np.ndarray:
    ordered = np.sort(gene_ids)
    if n >= ordered.size:
        return ordered
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ordered, size=n, replace=False))
