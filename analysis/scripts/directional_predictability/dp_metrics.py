#!/usr/bin/env python3
"""
dp_metrics.py — evaluation metrics for the Directional Predictability track.
 
Design note on why R-squared and retrieval, and not raw cosine
-------------------------------------------------------------
Gene embeddings from transformer-family models are strongly anisotropic: all
gene vectors occupy a narrow cone, so the cosine between two unrelated genes
is already high. A model that ignores its input entirely and always predicts
the training mean of the target therefore scores a very high raw cosine.
 
In a controlled simulation with a realistic anisotropic target:
 
    predictor                 raw cosine   centred cosine   R^2    top-1
    train mean (learns none)     0.958      undefined       0.000  0.001
    ridge on X                   0.982      0.759           0.566  0.411
 
Raw cosine cannot separate a model that learned nothing from one that learned
a great deal. R-squared and retrieval separate them cleanly, and both have
well-defined chance baselines (0 and 1/N).
 
Every R-squared here is computed against the TRAINING mean of the target, not
the test mean. That makes the "predicts nothing" model score exactly 0 by
construction, which is the baseline a reader actually cares about. The
test-mean version (the sklearn convention) is also reported for reference.
"""
 
from __future__ import annotations
 
import numpy as np
 
 
# ----------------------------------------------------------------- R-squared
 
 
def variance_weighted_r2(
    prediction: np.ndarray, truth: np.ndarray, train_mean: np.ndarray
) -> float:
    """
    Total variance-weighted R^2 across all target components.
 
    SS_tot uses the training mean, so a model emitting `train_mean` for every
    gene scores exactly 0.0. Values can be negative: that means the model does
    worse than emitting the training mean, which is informative, not a bug.
    """
    ss_res = float(((prediction - truth) ** 2).sum())
    ss_tot = float(((truth - train_mean) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot
 
 
def r2_test_mean_baseline(prediction: np.ndarray, truth: np.ndarray) -> float:
    """R^2 against the test mean (the sklearn convention), for reference."""
    ss_res = float(((prediction - truth) ** 2).sum())
    ss_tot = float(((truth - truth.mean(axis=0)) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot
 
 
def per_component_r2(
    prediction: np.ndarray, truth: np.ndarray, train_mean: np.ndarray
) -> np.ndarray:
    """
    R^2 for each target component separately, against the training mean.
 
    In the PCA target space the components are ordered by variance, so this
    array is a spectrum: how far into the target's structure the prediction
    reaches. It is far more informative than any single scalar, and it is the
    nonlinear analogue of a CCA spectrum.
    """
    ss_res = ((prediction - truth) ** 2).sum(axis=0)
    ss_tot = ((truth - train_mean) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - ss_res / ss_tot
    out[ss_tot <= 0] = np.nan
    return out
 
 
def shared_dimension_count(spectrum: np.ndarray, threshold: float) -> int:
    """How many target components are predicted above an R^2 threshold."""
    return int(np.nansum(spectrum > threshold))
 
 
# ------------------------------------------------------------------ cosine
 
 
def centred_cosine(
    prediction: np.ndarray, truth: np.ndarray, train_mean: np.ndarray
) -> float:
    """
    Mean cosine between prediction and truth AFTER removing the training mean.
 
    Reported, never optimised. Centring removes the anisotropy that makes the
    raw version degenerate. A model emitting exactly `train_mean` produces a
    zero vector here, whose cosine is undefined — which is the honest answer.
    """
    a = prediction - train_mean
    b = truth - train_mean
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 0) & (nb > 0)
    if not ok.any():
        return float("nan")
    return float(((a[ok] * b[ok]).sum(axis=1) / (na[ok] * nb[ok])).mean())
 
 
def raw_cosine(prediction: np.ndarray, truth: np.ndarray) -> float:
    """
    Mean cosine without centring.
 
    Computed ONLY so that reports can show how misleading it is next to the
    calibrated metrics. It must never be used to rank pairs or to select a
    model.
    """
    na = np.linalg.norm(prediction, axis=1)
    nb = np.linalg.norm(truth, axis=1)
    ok = (na > 0) & (nb > 0)
    if not ok.any():
        return float("nan")
    return float(((prediction[ok] * truth[ok]).sum(axis=1) / (na[ok] * nb[ok])).mean())
 
 
# --------------------------------------------------------------- retrieval
 
 
def retrieval_accuracy(
    prediction: np.ndarray,
    truth: np.ndarray,
    ks: list[int],
    train_mean: np.ndarray | None = None,
    block_size: int = 1024,
) -> dict[int, float]:
    """
    Top-k retrieval among the held-out genes.
 
    For each test gene, rank all test genes by cosine between the predicted
    vector and the true vectors, then ask whether the correct gene appears in
    the top k. Chance is k/N, which makes this directly interpretable as
    "can the model identify which gene this is in the target space".
 
    Centring by the training mean is applied when supplied, for the same
    anisotropy reason as above.
    """
    a = prediction if train_mean is None else prediction - train_mean
    b = truth if train_mean is None else truth - train_mean
 
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    a = a / na
    b = b / nb
 
    n = a.shape[0]
    max_k = max(ks)
    if max_k >= n:
        return {k: float("nan") for k in ks}
 
    # Fractional credit is the expected accuracy under uniform random
    # tie-breaking.  It prevents a zero/null query (all similarities tied) from
    # receiving perfect retrieval merely because rows and columns share order.
    hits = {k: 0.0 for k in ks}
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        similarity = a[start:stop] @ b.T
        # Correct answer for row i (global index start+i) is column start+i.
        correct = similarity[np.arange(stop - start), np.arange(start, stop)]
        tolerance = 1e-12 + 1e-7 * np.abs(correct)
        delta = similarity - correct[:, None]
        higher = (delta > tolerance[:, None]).sum(axis=1)
        tied = (np.abs(delta) <= tolerance[:, None]).sum(axis=1)
        tied = np.maximum(tied, 1)
        for k in ks:
            credit = np.clip((k - higher) / tied, 0.0, 1.0)
            hits[k] += float(credit.sum())
 
    return {k: hits[k] / n for k in ks}
 
 
def retrieval_chance(n_test: int, ks: list[int]) -> dict[int, float]:
    return {k: k / n_test for k in ks}
 
 
# ----------------------------------------------------------- full evaluation
 
 
def evaluate(
    prediction: np.ndarray,
    truth: np.ndarray,
    train_mean: np.ndarray,
    retrieval_ks: list[int],
    spectrum_thresholds: list[float],
) -> dict:
    """All metrics for one direction of one pair. Returns a flat dict."""
    spectrum = per_component_r2(prediction, truth, train_mean)
    retrieval = retrieval_accuracy(prediction, truth, retrieval_ks, train_mean)
    chance = retrieval_chance(truth.shape[0], retrieval_ks)
 
    result = {
        "r2": variance_weighted_r2(prediction, truth, train_mean),
        "r2_test_mean_baseline": r2_test_mean_baseline(prediction, truth),
        "centred_cosine": centred_cosine(prediction, truth, train_mean),
        "raw_cosine_do_not_use": raw_cosine(prediction, truth),
        "n_test": int(truth.shape[0]),
        "n_target_components": int(truth.shape[1]),
    }
    for k in retrieval_ks:
        result[f"retrieval_top{k}"] = retrieval[k]
        result[f"retrieval_top{k}_chance"] = chance[k]
        result[f"retrieval_top{k}_lift"] = (
            retrieval[k] / chance[k] if chance[k] > 0 else float("nan")
        )
    for threshold in spectrum_thresholds:
        label = str(threshold).replace(".", "p")
        result[f"n_components_r2_above_{label}"] = shared_dimension_count(
            spectrum, threshold
        )
    result["spectrum_r2_first"] = float(spectrum[0]) if spectrum.size else float("nan")
    result["spectrum_r2_mean"] = float(np.nanmean(spectrum)) if spectrum.size else float("nan")
    return result, spectrum
