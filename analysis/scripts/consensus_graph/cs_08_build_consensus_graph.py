#!/usr/bin/env python3
"""Build and test the consensus affinity graph.

The equal-rank consensus is an affinity, not a Euclidean kernel or metric
distance. This stage therefore represents the 56 embeddings as a weighted,
symmetrized k-nearest-neighbour graph.  Its pre-specified primary graph uses
k=10.  Communities are learned without modality labels by deterministic
greedy weighted-modularity optimization at resolution 1.

The 11 manifest modalities enter only after community detection.
Their relationship to the graph is evaluated with fixed-count node-label
permutations, which preserve the dependence among edges sharing a node.
The publication run evaluates only this primary consensus configuration;
the hybrid-free modality-label sensitivity is retained separately.

This script reads only completed consensus-graph tables and summaries. It does
not load embedding arrays, fit PCA/MLP models, or use directional-predictability
outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, deque
from datetime import datetime, timezone
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_EMBEDDINGS = 56
EXPECTED_PAIRS = 1_540
EXPECTED_MODALITIES = 11
PRIMARY_GRAPH_K = 10
N_PERMUTATIONS = 9_999
SEED = 33_108
MODULARITY_RESOLUTION = 1.0
MERGE_TOLERANCE = 1e-12

PRIMARY_SCORE = "consensus_similarity"
PLATONIC_SCORE = "platonic_percentile"
ARISTOTELIAN_SCORE = "aristotelian_k100_percentile"

TEST_DESCRIPTIONS = {
    "community_modality_ari": (
        "adjusted Rand index between unsupervised communities and coarse modalities"
    ),
    "community_modality_nmi": (
        "normalized mutual information between unsupervised communities and coarse modalities"
    ),
    "modality_weighted_modularity": (
        "weighted graph modularity of the reviewed modality partition"
    ),
    "same_modality_edge_weight_fraction": (
        "fraction of primary graph edge weight joining the same modality"
    ),
    "directed_knn_modality_purity": (
        "fraction of directed k-nearest-neighbour selections in the same modality"
    ),
    "within_minus_between_mean_consensus": (
        "mean consensus similarity within modality minus between modalities"
    ),
}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--consensus-dir",
        type=Path,
        default=None,
        help="Input consensus directory (default: consensus_graph_v2/06_consensus).",
    )
    parser.add_argument(
        "--modality-dir",
        type=Path,
        default=None,
        help="Input modality directory (default: consensus_graph_v2/07_modalities).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: consensus_graph_v2/08_graph_structure).",
    )
    parser.add_argument(
        "--graph-k",
        type=int,
        default=PRIMARY_GRAPH_K,
        help=f"Primary directed neighbour count (default: {PRIMARY_GRAPH_K}).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=N_PERMUTATIONS,
        help=f"Fixed-count node-label permutations (default: {N_PERMUTATIONS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"Random seed (default: {SEED})."
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run deterministic in-memory tests."
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def numeric(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"{label} is missing {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label}/{column} contains non-finite values")
    return values


def canonicalize_pairs(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    require_columns(frame, ("embedding_x", "embedding_y"), label)
    result = frame.copy()
    left = result["embedding_x"].astype(str).to_numpy()
    right = result["embedding_y"].astype(str).to_numpy()
    if np.any(left == right):
        raise ValueError(f"{label} contains a self-pair")
    result["embedding_x"] = np.minimum(left, right)
    result["embedding_y"] = np.maximum(left, right)
    if result.duplicated(["embedding_x", "embedding_y"]).any():
        raise ValueError(f"{label} contains duplicate canonical pairs")
    return result.sort_values(["embedding_x", "embedding_y"]).reset_index(drop=True)


def validate_pair_coverage(frame: pd.DataFrame, ids: list[str]) -> None:
    observed = set(zip(frame["embedding_x"], frame["embedding_y"]))
    expected = set(combinations(sorted(ids), 2))
    if len(frame) != EXPECTED_PAIRS or observed != expected:
        raise ValueError(
            "Consensus pair coverage mismatch: "
            f"rows={len(frame)}, expected_rows={EXPECTED_PAIRS}, "
            f"missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}"
        )


def build_matrix(
    pairs: pd.DataFrame,
    ids: list[str],
    column: str,
    diagonal: float = 1.0,
) -> np.ndarray:
    values = numeric(pairs, column, "consensus_pair_scores")
    positions = {embedding_id: index for index, embedding_id in enumerate(ids)}
    matrix = np.full((len(ids), len(ids)), np.nan, dtype=float)
    np.fill_diagonal(matrix, diagonal)
    for x, y, value in zip(pairs["embedding_x"], pairs["embedding_y"], values):
        i, j = positions[str(x)], positions[str(y)]
        matrix[i, j] = value
        matrix[j, i] = value
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Matrix for {column!r} is incomplete/non-finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0):
        raise ValueError(f"Matrix for {column!r} is not exactly symmetric")
    return matrix


def build_knn_graph(
    similarity: np.ndarray,
    ids: list[str],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(ids)
    if similarity.shape != (n, n):
        raise ValueError("Similarity matrix shape does not match embedding IDs")
    if not 1 <= k < n:
        raise ValueError(f"Graph k must be between 1 and {n - 1}")
    if not np.all(np.isfinite(similarity)) or np.any(similarity < 0.0):
        raise ValueError("Graph affinities must be finite and non-negative")

    directed = np.zeros((n, n), dtype=bool)
    ranks = np.zeros((n, n), dtype=int)
    for i in range(n):
        order = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: (-float(similarity[i, j]), ids[j]),
        )
        for rank, j in enumerate(order, start=1):
            ranks[i, j] = rank
        directed[i, order[:k]] = True

    edge_mask = directed | directed.T
    np.fill_diagonal(edge_mask, False)
    adjacency = np.where(edge_mask, similarity, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    if not np.allclose(adjacency, adjacency.T, rtol=0.0, atol=0.0):
        raise AssertionError("Symmetrized kNN graph is not exactly symmetric")
    if float(adjacency.sum()) <= 0.0:
        raise ValueError("kNN graph has zero total edge weight")
    return adjacency, directed, ranks, edge_mask


def connected_components(edge_mask: np.ndarray) -> tuple[int, np.ndarray]:
    n = len(edge_mask)
    labels = np.full(n, -1, dtype=int)
    component = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        labels[start] = component
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in np.flatnonzero(edge_mask[node]):
                neighbour = int(neighbour)
                if labels[neighbour] < 0:
                    labels[neighbour] = component
                    queue.append(neighbour)
        component += 1
    return component, labels


def weighted_modularity(
    adjacency: np.ndarray,
    labels: np.ndarray,
    resolution: float = MODULARITY_RESOLUTION,
) -> float:
    labels = np.asarray(labels)
    if len(labels) != len(adjacency):
        raise ValueError("Partition length does not match graph")
    degree = adjacency.sum(axis=1)
    total_weight = float(degree.sum() / 2.0)
    if total_weight <= 0.0:
        raise ValueError("Weighted modularity requires positive graph weight")
    value = 0.0
    for community in np.unique(labels):
        mask = labels == community
        internal = float(adjacency[np.ix_(mask, mask)].sum() / 2.0)
        degree_sum = float(degree[mask].sum())
        value += internal / total_weight
        value -= resolution * (degree_sum / (2.0 * total_weight)) ** 2
    return float(value)


def canonical_partition(communities: list[set[int]], ids: list[str]) -> np.ndarray:
    ordered = sorted(communities, key=lambda members: min(ids[i] for i in members))
    labels = np.empty(len(ids), dtype=int)
    for community_id, members in enumerate(ordered, start=1):
        for node in members:
            labels[node] = community_id
    return labels


def greedy_modularity_communities(
    adjacency: np.ndarray,
    ids: list[str],
    resolution: float = MODULARITY_RESOLUTION,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Deterministic weighted greedy modularity agglomeration."""

    n = len(ids)
    degree = adjacency.sum(axis=1)
    total_weight = float(degree.sum() / 2.0)
    if total_weight <= 0.0:
        raise ValueError("Community detection requires positive graph weight")
    communities: list[set[int]] = [{i} for i in range(n)]
    initial_labels = canonical_partition(communities, ids)
    trace: list[dict[str, object]] = [
        {
            "step": 0,
            "n_communities": n,
            "weighted_modularity": weighted_modularity(
                adjacency, initial_labels, resolution
            ),
            "merge_delta": np.nan,
            "merged_group_x": "",
            "merged_group_y": "",
        }
    ]

    step = 0
    while len(communities) > 1:
        best: tuple[float, tuple[str, str], int, int] | None = None
        for i, j in combinations(range(len(communities)), 2):
            left = communities[i]
            right = communities[j]
            left_nodes = sorted(left)
            right_nodes = sorted(right)
            between = float(adjacency[np.ix_(left_nodes, right_nodes)].sum())
            left_degree = float(degree[left_nodes].sum())
            right_degree = float(degree[right_nodes].sum())
            delta = between / total_weight
            delta -= resolution * left_degree * right_degree / (
                2.0 * total_weight**2
            )
            tie_key = (
                min(ids[node] for node in left),
                min(ids[node] for node in right),
            )
            candidate = (float(delta), tie_key, i, j)
            if best is None:
                best = candidate
            elif delta > best[0] + MERGE_TOLERANCE:
                best = candidate
            elif abs(delta - best[0]) <= MERGE_TOLERANCE and tie_key < best[1]:
                best = candidate

        if best is None or best[0] <= MERGE_TOLERANCE:
            break
        delta, _, i, j = best
        left = communities[i]
        right = communities[j]
        left_name = ",".join(sorted(ids[node] for node in left))
        right_name = ",".join(sorted(ids[node] for node in right))
        merged = left | right
        communities = [
            members
            for index, members in enumerate(communities)
            if index not in (i, j)
        ]
        communities.append(merged)
        communities.sort(key=lambda members: min(ids[node] for node in members))
        labels = canonical_partition(communities, ids)
        step += 1
        trace.append(
            {
                "step": step,
                "n_communities": len(communities),
                "weighted_modularity": weighted_modularity(
                    adjacency, labels, resolution
                ),
                "merge_delta": delta,
                "merged_group_x": left_name,
                "merged_group_y": right_name,
            }
        )

    labels = canonical_partition(communities, ids)
    final_modularity = weighted_modularity(adjacency, labels, resolution)
    if not np.isclose(
        final_modularity,
        float(trace[-1]["weighted_modularity"]),
        rtol=0.0,
        atol=1e-10,
    ):
        raise AssertionError("Community trace does not reproduce final modularity")
    return labels, pd.DataFrame(trace)


def partition_agreement(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return adjusted Rand index and arithmetic normalized mutual information."""

    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("Partition-agreement inputs must be equal one-dimensional arrays")
    _, left_codes = np.unique(left, return_inverse=True)
    _, right_codes = np.unique(right, return_inverse=True)
    n_left = int(left_codes.max()) + 1
    n_right = int(right_codes.max()) + 1
    contingency = np.bincount(
        left_codes * n_right + right_codes,
        minlength=n_left * n_right,
    ).reshape(n_left, n_right)
    row = contingency.sum(axis=1).astype(float)
    column = contingency.sum(axis=0).astype(float)
    n = float(len(left))

    choose2 = lambda values: np.sum(values * (values - 1.0) / 2.0)
    cells = float(choose2(contingency.astype(float)))
    row_pairs = float(choose2(row))
    column_pairs = float(choose2(column))
    total_pairs = n * (n - 1.0) / 2.0
    expected = row_pairs * column_pairs / total_pairs
    maximum = 0.5 * (row_pairs + column_pairs)
    denominator = maximum - expected
    if abs(denominator) <= 1e-15:
        ari = 1.0 if np.array_equal(left_codes, right_codes) else 0.0
    else:
        ari = (cells - expected) / denominator

    probabilities = contingency.astype(float) / n
    row_prob = row / n
    col_prob = column / n
    expected_prob = row_prob[:, None] * col_prob[None, :]
    nonzero = probabilities > 0.0
    mutual_information = float(
        np.sum(probabilities[nonzero] * np.log(probabilities[nonzero] / expected_prob[nonzero]))
    )
    left_entropy = float(-np.sum(row_prob[row_prob > 0] * np.log(row_prob[row_prob > 0])))
    right_entropy = float(-np.sum(col_prob[col_prob > 0] * np.log(col_prob[col_prob > 0])))
    entropy_mean = 0.5 * (left_entropy + right_entropy)
    nmi = 1.0 if entropy_mean == 0.0 else mutual_information / entropy_mean
    return float(ari), float(nmi)


def association_statistics(
    similarity: np.ndarray,
    adjacency: np.ndarray,
    directed: np.ndarray,
    communities: np.ndarray,
    modalities: np.ndarray,
) -> dict[str, float]:
    n = len(similarity)
    upper_i, upper_j = np.triu_indices(n, k=1)
    same = modalities[upper_i] == modalities[upper_j]
    pair_values = similarity[upper_i, upper_j]
    adjacency_values = adjacency[upper_i, upper_j]
    total_edge_weight = float(adjacency_values.sum())
    if not np.any(same) or np.all(same) or total_edge_weight <= 0.0:
        raise ValueError("Association statistics require within and between labels")
    directed_i, directed_j = np.where(directed)
    ari, nmi = partition_agreement(communities, modalities)
    return {
        "community_modality_ari": ari,
        "community_modality_nmi": nmi,
        "modality_weighted_modularity": weighted_modularity(adjacency, modalities),
        "same_modality_edge_weight_fraction": float(
            adjacency_values[same].sum() / total_edge_weight
        ),
        "directed_knn_modality_purity": float(
            np.mean(modalities[directed_i] == modalities[directed_j])
        ),
        "within_minus_between_mean_consensus": float(
            pair_values[same].mean() - pair_values[~same].mean()
        ),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="mergesort")
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def permutation_tests(
    similarity: np.ndarray,
    adjacency: np.ndarray,
    directed: np.ndarray,
    communities: np.ndarray,
    modalities: np.ndarray,
    n_permutations: int,
    seed: int,
    scope: str,
) -> pd.DataFrame:
    if n_permutations < 99:
        raise ValueError("At least 99 permutations are required")
    _, modality_codes = np.unique(np.asarray(modalities), return_inverse=True)
    modalities = modality_codes.astype(int, copy=False)
    observed = association_statistics(
        similarity, adjacency, directed, communities, modalities
    )
    n = len(modalities)
    rng = np.random.default_rng(seed)
    permutations = np.empty((n_permutations, n), dtype=int)
    for index in range(n_permutations):
        permutations[index] = rng.permutation(modalities)

    upper_i, upper_j = np.triu_indices(n, k=1)
    same_pair = permutations[:, upper_i] == permutations[:, upper_j]
    pair_values = similarity[upper_i, upper_j]
    within_count = same_pair.sum(axis=1)
    between_count = len(pair_values) - within_count
    within_sum = same_pair @ pair_values
    total_pair_sum = float(pair_values.sum())

    adjacency_values = adjacency[upper_i, upper_j]
    total_edge_weight = float(adjacency_values.sum())
    edge_same = (same_pair @ adjacency_values) / total_edge_weight

    degree = adjacency.sum(axis=1)
    modularity_matrix = adjacency - np.outer(degree, degree) / degree.sum()
    modularity_constant = float(np.trace(modularity_matrix))
    modularity_upper = modularity_matrix[upper_i, upper_j]
    modality_modularity = (
        modularity_constant + 2.0 * (same_pair @ modularity_upper)
    ) / degree.sum()

    directed_i, directed_j = np.where(directed)
    directed_purity = np.mean(
        permutations[:, directed_i] == permutations[:, directed_j], axis=1
    )
    mean_delta = within_sum / within_count
    mean_delta -= (total_pair_sum - within_sum) / between_count

    ari_null = np.empty(n_permutations, dtype=float)
    nmi_null = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        ari_null[index], nmi_null[index] = partition_agreement(
            communities, permutations[index]
        )

    nulls = {
        "community_modality_ari": ari_null,
        "community_modality_nmi": nmi_null,
        "modality_weighted_modularity": modality_modularity,
        "same_modality_edge_weight_fraction": edge_same,
        "directed_knn_modality_purity": directed_purity,
        "within_minus_between_mean_consensus": mean_delta,
    }
    rows: list[dict[str, object]] = []
    for test_id in TEST_DESCRIPTIONS:
        null = np.asarray(nulls[test_id], dtype=float)
        value = float(observed[test_id])
        null_sd = float(null.std(ddof=1))
        rows.append(
            {
                "scope": scope,
                "test_id": test_id,
                "description": TEST_DESCRIPTIONS[test_id],
                "alternative": "greater_than_fixed_count_node_label_null",
                "observed": value,
                "null_mean": float(null.mean()),
                "null_sd": null_sd,
                "null_q025": float(np.quantile(null, 0.025)),
                "null_q975": float(np.quantile(null, 0.975)),
                "standardized_effect": (
                    (value - float(null.mean())) / null_sd if null_sd > 0.0 else np.nan
                ),
                "permutation_p_upper": float(
                    (1 + np.count_nonzero(null >= value)) / (n_permutations + 1)
                ),
                "n_permutations": n_permutations,
                "seed": seed,
            }
        )
    result = pd.DataFrame(rows)
    result["holm_p_within_scope"] = holm_adjust(
        result["permutation_p_upper"].to_numpy(dtype=float)
    )
    return result


def edge_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    upper = np.triu(np.ones_like(left, dtype=bool), k=1)
    left_edges = left & upper
    right_edges = right & upper
    union = np.count_nonzero(left_edges | right_edges)
    return float(np.count_nonzero(left_edges & right_edges) / union)


def graph_diagnostics(
    adjacency: np.ndarray,
    edge_mask: np.ndarray,
    communities: np.ndarray,
) -> dict[str, object]:
    n = len(adjacency)
    n_components, component_labels = connected_components(edge_mask)
    edge_count = int(np.count_nonzero(np.triu(edge_mask, k=1)))
    degrees = edge_mask.sum(axis=1)
    weighted_degrees = adjacency.sum(axis=1)
    return {
        "n_edges": edge_count,
        "density": float(edge_count / (n * (n - 1) / 2)),
        "n_connected_components": n_components,
        "largest_component_size": int(np.bincount(component_labels).max()),
        "minimum_degree": int(degrees.min()),
        "median_degree": float(np.median(degrees)),
        "maximum_degree": int(degrees.max()),
        "minimum_weighted_degree": float(weighted_degrees.min()),
        "median_weighted_degree": float(np.median(weighted_degrees)),
        "maximum_weighted_degree": float(weighted_degrees.max()),
        "n_communities": int(len(np.unique(communities))),
        "community_weighted_modularity": weighted_modularity(
            adjacency, communities
        ),
    }


def sensitivity_table(
    pairs: pd.DataFrame,
    ids: list[str],
    primary_communities: np.ndarray,
    primary_edges: np.ndarray,
    graph_k: int,
) -> pd.DataFrame:
    specifications: list[tuple[str, str, int, str]] = [
        ("primary", PRIMARY_SCORE, graph_k, "pre_specified_primary")
    ]

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for view_id, column, k, role in specifications:
        key = (column, k)
        if key in seen:
            continue
        seen.add(key)
        matrix = build_matrix(pairs, ids, column)
        adjacency, _, _, edges = build_knn_graph(matrix, ids, k)
        communities, _ = greedy_modularity_communities(adjacency, ids)
        diagnostics = graph_diagnostics(adjacency, edges, communities)
        ari, nmi = partition_agreement(primary_communities, communities)
        rows.append(
            {
                "view_id": view_id,
                "analysis_role": role,
                "score_column": column,
                "graph_k": k,
                **diagnostics,
                "partition_ari_vs_primary": ari,
                "partition_nmi_vs_primary": nmi,
                "edge_jaccard_vs_primary": edge_jaccard(primary_edges, edges),
            }
        )
    return pd.DataFrame(rows)


def edge_table(
    pairs: pd.DataFrame,
    ids: list[str],
    adjacency: np.ndarray,
    directed: np.ndarray,
    ranks: np.ndarray,
    modalities: np.ndarray,
    communities: np.ndarray,
    graph_k: int,
) -> pd.DataFrame:
    positions = {embedding_id: index for index, embedding_id in enumerate(ids)}
    pair_lookup = pairs.set_index(["embedding_x", "embedding_y"])
    rows: list[dict[str, object]] = []
    for i, j in combinations(range(len(ids)), 2):
        if adjacency[i, j] == 0.0:
            continue
        x, y = ids[i], ids[j]
        key = (min(x, y), max(x, y))
        pair = pair_lookup.loc[key]
        rows.append(
            {
                "embedding_x": x,
                "embedding_y": y,
                "consensus_similarity": float(adjacency[i, j]),
                "platonic_percentile": float(pair[PLATONIC_SCORE]),
                "aristotelian_k100_percentile": float(pair[ARISTOTELIAN_SCORE]),
                "component_absolute_disagreement": float(
                    pair["component_absolute_disagreement"]
                ),
                "x_selects_y": bool(directed[i, j]),
                "y_selects_x": bool(directed[j, i]),
                "reciprocal_knn": bool(directed[i, j] and directed[j, i]),
                "rank_of_y_from_x": int(ranks[i, j]),
                "rank_of_x_from_y": int(ranks[j, i]),
                "graph_k": graph_k,
                "modality_x": str(modalities[i]),
                "modality_y": str(modalities[j]),
                "same_modality": bool(modalities[i] == modalities[j]),
                "community_x": int(communities[i]),
                "community_y": int(communities[j]),
                "same_community": bool(communities[i] == communities[j]),
            }
        )
    result = pd.DataFrame(rows)
    expected_edges = int(np.count_nonzero(np.triu(adjacency != 0.0, k=1)))
    if len(result) != expected_edges:
        raise AssertionError("Edge table does not cover the weighted graph")
    if set(positions) != set(ids):
        raise AssertionError("Internal embedding-position mismatch")
    return result


def embedding_annotations(
    ids: list[str],
    labels: pd.DataFrame,
    similarity: np.ndarray,
    adjacency: np.ndarray,
    directed: np.ndarray,
    modalities: np.ndarray,
    communities: np.ndarray,
) -> pd.DataFrame:
    label_frame = labels.set_index("embedding_id")
    rows: list[dict[str, object]] = []
    for i, embedding_id in enumerate(ids):
        neighbours = np.flatnonzero(adjacency[i] != 0.0)
        directed_neighbours = np.flatnonzero(directed[i])
        top = sorted(
            (j for j in range(len(ids)) if j != i),
            key=lambda j: (-float(similarity[i, j]), ids[j]),
        )[0]
        same_graph = modalities[neighbours] == modalities[i]
        same_directed = modalities[directed_neighbours] == modalities[i]
        row = label_frame.loc[embedding_id]
        rows.append(
            {
                "embedding_id": embedding_id,
                "display_name": str(row["display_name"]),
                "coarse_modality": str(modalities[i]),
                "hybrid_or_multimodal": bool(row["hybrid_or_multimodal"]),
                "community_id": int(communities[i]),
                "graph_degree": int(len(neighbours)),
                "weighted_degree": float(adjacency[i].sum()),
                "same_modality_graph_neighbours": int(np.count_nonzero(same_graph)),
                "same_modality_graph_neighbour_fraction": float(np.mean(same_graph)),
                "same_modality_directed_knn": int(np.count_nonzero(same_directed)),
                "same_modality_directed_knn_fraction": float(np.mean(same_directed)),
                "top_consensus_neighbour": ids[top],
                "top_consensus_neighbour_display": str(
                    label_frame.loc[ids[top], "display_name"]
                ),
                "top_consensus_similarity": float(similarity[i, top]),
            }
        )
    return pd.DataFrame(rows)


def community_modality_table(
    communities: np.ndarray,
    modalities: np.ndarray,
    modality_order: list[str],
) -> pd.DataFrame:
    community_sizes = Counter(int(value) for value in communities)
    modality_sizes = Counter(str(value) for value in modalities)
    rows: list[dict[str, object]] = []
    for community in sorted(community_sizes):
        for modality in modality_order:
            count = int(
                np.count_nonzero(
                    (communities == community) & (modalities == modality)
                )
            )
            rows.append(
                {
                    "community_id": community,
                    "coarse_modality": modality,
                    "n_embeddings": count,
                    "community_size": community_sizes[community],
                    "modality_size": modality_sizes[modality],
                    "fraction_of_community": count / community_sizes[community],
                    "fraction_of_modality": count / modality_sizes[modality],
                }
            )
    return pd.DataFrame(rows)


def modality_pair_table(
    pairs: pd.DataFrame,
    label_map: dict[str, str],
    modality_order: list[str],
) -> pd.DataFrame:
    order = {modality: index for index, modality in enumerate(modality_order)}
    working = pairs.copy()
    working["modality_x_raw"] = working["embedding_x"].map(label_map)
    working["modality_y_raw"] = working["embedding_y"].map(label_map)
    if working[["modality_x_raw", "modality_y_raw"]].isna().any().any():
        raise ValueError("Some pair rows could not be mapped to coarse modalities")
    left: list[str] = []
    right: list[str] = []
    for x, y in zip(working["modality_x_raw"], working["modality_y_raw"]):
        if order[str(x)] <= order[str(y)]:
            left.append(str(x))
            right.append(str(y))
        else:
            left.append(str(y))
            right.append(str(x))
    working["modality_x"] = left
    working["modality_y"] = right

    rows: list[dict[str, object]] = []
    for x, y in combinations_with_replacement(modality_order, 2):
        subset = working.loc[
            (working["modality_x"] == x) & (working["modality_y"] == y)
        ]
        if subset.empty:
            raise ValueError(f"No pairs found for modality combination {x!r}/{y!r}")
        values = numeric(subset, PRIMARY_SCORE, "modality pair subset")
        rows.append(
            {
                "modality_x": x,
                "modality_y": y,
                "pair_scope": "within" if x == y else "between",
                "n_pairs": len(subset),
                "mean_consensus_similarity": float(np.mean(values)),
                "median_consensus_similarity": float(np.median(values)),
                "q25_consensus_similarity": float(np.quantile(values, 0.25)),
                "q75_consensus_similarity": float(np.quantile(values, 0.75)),
                "mean_platonic_percentile": float(
                    numeric(subset, PLATONIC_SCORE, "modality pair subset").mean()
                ),
                "mean_aristotelian_k100_percentile": float(
                    numeric(subset, ARISTOTELIAN_SCORE, "modality pair subset").mean()
                ),
                "mean_component_absolute_disagreement": float(
                    numeric(
                        subset,
                        "component_absolute_disagreement",
                        "modality pair subset",
                    ).mean()
                ),
            }
        )
    result = pd.DataFrame(rows)
    if int(result["n_pairs"].sum()) != EXPECTED_PAIRS:
        raise AssertionError("Modality-pair table does not cover all pairs")
    return result


def parse_boolean(series: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError(f"{label} contains invalid Boolean values")
    return parsed.to_numpy(dtype=bool)


def atomic_dataframe(
    frame: pd.DataFrame,
    destination: Path,
    index: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, sep="\t", index=index)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def self_test() -> None:
    ids = [f"e{i:02d}" for i in range(18)]
    known = np.repeat(np.arange(6), 3)
    similarity = np.zeros((18, 18), dtype=float)
    for i in range(18):
        for j in range(18):
            if i == j:
                similarity[i, j] = 1.0
            elif known[i] == known[j]:
                similarity[i, j] = 0.95 - 0.001 * abs(i - j)
            else:
                similarity[i, j] = 0.20 - 0.0001 * abs(i - j)
    adjacency, directed, _, edges = build_knn_graph(similarity, ids, 3)
    communities, trace = greedy_modularity_communities(adjacency, ids)
    if len(trace) < 2 or len(np.unique(communities)) != 6:
        raise AssertionError("Greedy-community self-test did not recover six blocks")
    ari, nmi = partition_agreement(communities, known)
    if not np.isclose(ari, 1.0) or not np.isclose(nmi, 1.0):
        raise AssertionError("Partition-agreement self-test failed")
    observed = association_statistics(
        similarity, adjacency, directed, communities, known
    )
    if observed["within_minus_between_mean_consensus"] <= 0.0:
        raise AssertionError("Association-statistic self-test failed")
    if connected_components(edges)[0] != 1:
        raise AssertionError("Graph-connectivity self-test failed")
    tiny = permutation_tests(
        similarity,
        adjacency,
        directed,
        communities,
        known,
        99,
        7,
        "self_test",
    )
    if len(tiny) != len(TEST_DESCRIPTIONS):
        raise AssertionError("Permutation-output self-test failed")
    print("cs_08 self-test: PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 1 <= args.graph_k < EXPECTED_EMBEDDINGS:
        raise ValueError(
            f"--graph-k must be between 1 and {EXPECTED_EMBEDDINGS - 1}"
        )
    if args.permutations < 99:
        raise ValueError("--permutations must be at least 99")

    root = args.root.expanduser().resolve()
    workflow_root = root / "results/consensus_graph_v2"
    consensus_dir = (
        args.consensus_dir.expanduser().resolve()
        if args.consensus_dir is not None
        else workflow_root / "06_consensus"
    )
    modality_dir = (
        args.modality_dir.expanduser().resolve()
        if args.modality_dir is not None
        else workflow_root / "07_modalities"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow_root / "08_graph_structure"
    )

    panel_path = require_file(workflow_root / "01_fixed_panel/embedding_metadata.tsv")
    pairs_path = require_file(consensus_dir / "consensus_pair_scores.tsv")
    matrix_path = require_file(consensus_dir / "consensus_similarity_matrix.tsv")
    consensus_summary_path = require_file(consensus_dir / "consensus_summary.json")
    labels_path = require_file(modality_dir / "reviewed_coarse_modalities.tsv")
    modality_summary_path = require_file(modality_dir / "modality_review_summary.json")

    consensus_summary = json.loads(consensus_summary_path.read_text())
    modality_summary = json.loads(modality_summary_path.read_text())
    if consensus_summary.get("status") != "PASS":
        raise ValueError("Upstream consensus summary does not have status PASS")
    if modality_summary.get("status") != "PASS":
        raise ValueError("Upstream modality summary does not have status PASS")
    for payload, label in (
        (consensus_summary, "consensus"),
        (modality_summary, "modality"),
    ):
        if int(payload.get("n_embeddings", -1)) != EXPECTED_EMBEDDINGS:
            raise ValueError(f"Upstream {label} embedding count changed")
        if int(payload.get("n_pairs", -1)) != EXPECTED_PAIRS:
            raise ValueError(f"Upstream {label} pair count changed")

    panel = pd.read_csv(panel_path, sep="\t")
    require_columns(panel, ("embedding_id",), "primary panel")
    if panel["embedding_id"].isna().any() or panel["embedding_id"].duplicated().any():
        raise ValueError("Primary panel has missing or duplicate embedding IDs")
    ids = sorted(panel["embedding_id"].astype(str).tolist())
    if len(ids) != EXPECTED_EMBEDDINGS:
        raise ValueError(f"Expected {EXPECTED_EMBEDDINGS} primary embeddings")
    recorded_panel_hash = (
        consensus_summary.get("sources", {}).get("primary_panel", {}).get("sha256")
    )
    if recorded_panel_hash != sha256(panel_path):
        raise ValueError("Primary-panel file changed after consensus construction")
    recorded_consensus_hash = (
        modality_summary.get("sources", {})
        .get("consensus_summary", {})
        .get("sha256")
    )
    if recorded_consensus_hash != sha256(consensus_summary_path):
        raise ValueError("Consensus summary changed after modality review")
    recorded_labels_hash = (
        modality_summary.get("outputs", {}).get("reviewed_labels", {}).get("sha256")
    )
    if recorded_labels_hash != sha256(labels_path):
        raise ValueError("Reviewed modality labels changed after they were frozen")

    construction = consensus_summary.get("construction", {})
    if construction.get("method") != "equal_weight_average_of_within_view_percentile_ranks":
        raise ValueError("Consensus construction method changed")
    if not np.isclose(float(construction.get("platonic_weight", np.nan)), 0.5):
        raise ValueError("Consensus Platonic weight changed")
    if not np.isclose(float(construction.get("aristotelian_weight", np.nan)), 0.5):
        raise ValueError("Consensus Aristotelian weight changed")
    if construction.get("platonic_view") != "platonic_linear_calibrated":
        raise ValueError("Primary Platonic view is not calibrated linear CKA")
    if construction.get("aristotelian_view") != "aristotelian_local_k100_calibrated":
        raise ValueError("Primary Aristotelian view is not calibrated local overlap k=100")

    pairs = canonicalize_pairs(pd.read_csv(pairs_path, sep="\t"), "consensus pairs")
    validate_pair_coverage(pairs, ids)
    required_pair_columns = {
        PRIMARY_SCORE,
        PLATONIC_SCORE,
        ARISTOTELIAN_SCORE,
        "component_absolute_disagreement",
    }
    require_columns(pairs, required_pair_columns, "consensus pairs")
    consensus_values = numeric(pairs, PRIMARY_SCORE, "consensus pairs")
    expected_values = 0.5 * (
        numeric(pairs, PLATONIC_SCORE, "consensus pairs")
        + numeric(pairs, ARISTOTELIAN_SCORE, "consensus pairs")
    )
    if not np.allclose(consensus_values, expected_values, rtol=0.0, atol=1e-12):
        raise ValueError("Consensus pair scores no longer reproduce equal-rank fusion")
    distribution = consensus_summary.get("consensus_distribution", {})
    for key, observed in (
        ("minimum", np.min(consensus_values)),
        ("median", np.median(consensus_values)),
        ("maximum", np.max(consensus_values)),
    ):
        if not np.isclose(float(distribution.get(key, np.nan)), observed, atol=1e-12):
            raise ValueError(f"Consensus {key} no longer matches its summary")

    similarity = build_matrix(pairs, ids, PRIMARY_SCORE)
    stored_matrix = pd.read_csv(matrix_path, sep="\t", index_col=0)
    stored_matrix.index = stored_matrix.index.astype(str)
    stored_matrix.columns = stored_matrix.columns.astype(str)
    if set(stored_matrix.index) != set(ids) or set(stored_matrix.columns) != set(ids):
        raise ValueError("Stored consensus matrix IDs do not match the primary panel")
    stored_values = stored_matrix.loc[ids, ids].to_numpy(dtype=float)
    if not np.allclose(stored_values, similarity, rtol=0.0, atol=1e-12):
        raise ValueError("Stored consensus matrix does not reproduce from pair scores")

    reviewed = pd.read_csv(labels_path, sep="\t")
    require_columns(
        reviewed,
        (
            "embedding_id",
            "display_name",
            "coarse_modality",
            "hybrid_or_multimodal",
        ),
        "reviewed modalities",
    )
    reviewed["embedding_id"] = reviewed["embedding_id"].astype(str)
    if reviewed["embedding_id"].duplicated().any() or set(reviewed["embedding_id"]) != set(ids):
        raise ValueError("Reviewed modality IDs do not match the primary panel")
    reviewed["hybrid_or_multimodal"] = parse_boolean(
        reviewed["hybrid_or_multimodal"], "hybrid_or_multimodal"
    )
    reviewed = reviewed.set_index("embedding_id").loc[ids].reset_index()
    modality_order = [str(value) for value in modality_summary.get("coarse_modalities", [])]
    if len(modality_order) != EXPECTED_MODALITIES or len(set(modality_order)) != EXPECTED_MODALITIES:
        raise ValueError(
            f"Expected {EXPECTED_MODALITIES} unique manifest modalities"
        )
    if set(reviewed["coarse_modality"].astype(str)) != set(modality_order):
        raise ValueError("Reviewed modality categories changed")
    observed_counts = reviewed["coarse_modality"].astype(str).value_counts().to_dict()
    expected_counts = {str(k): int(v) for k, v in modality_summary.get("counts", {}).items()}
    if observed_counts != expected_counts or sum(expected_counts.values()) != EXPECTED_EMBEDDINGS:
        raise ValueError("Reviewed modality counts no longer match their summary")
    modalities = reviewed["coarse_modality"].astype(str).to_numpy()
    hybrid = reviewed["hybrid_or_multimodal"].to_numpy(dtype=bool)

    adjacency, directed, ranks, edge_mask = build_knn_graph(
        similarity, ids, args.graph_k
    )
    communities, community_trace = greedy_modularity_communities(adjacency, ids)
    diagnostics = graph_diagnostics(adjacency, edge_mask, communities)

    primary_tests = permutation_tests(
        similarity,
        adjacency,
        directed,
        communities,
        modalities,
        args.permutations,
        args.seed,
        "primary_all56",
    )
    nonhybrid_indices = np.flatnonzero(~hybrid)
    if len(nonhybrid_indices) < EXPECTED_EMBEDDINGS - 10:
        raise ValueError("Too many embeddings were removed by hybrid-free sensitivity")
    nonhybrid_ids = [ids[index] for index in nonhybrid_indices]
    nonhybrid_similarity = similarity[np.ix_(nonhybrid_indices, nonhybrid_indices)]
    nonhybrid_adjacency, nonhybrid_directed, _, _ = build_knn_graph(
        nonhybrid_similarity,
        nonhybrid_ids,
        min(args.graph_k, len(nonhybrid_ids) - 1),
    )
    nonhybrid_tests = permutation_tests(
        nonhybrid_similarity,
        nonhybrid_adjacency,
        nonhybrid_directed,
        communities[nonhybrid_indices],
        modalities[nonhybrid_indices],
        args.permutations,
        args.seed + 1,
        "nonhybrid_sensitivity",
    )
    association_tests = pd.concat(
        [primary_tests, nonhybrid_tests], ignore_index=True
    )

    graph_sensitivity = sensitivity_table(
        pairs, ids, communities, edge_mask, args.graph_k
    )
    graph_edges = edge_table(
        pairs,
        ids,
        adjacency,
        directed,
        ranks,
        modalities,
        communities,
        args.graph_k,
    )
    display_map = dict(zip(reviewed["embedding_id"], reviewed["display_name"]))
    graph_edges["embedding_x_display"] = graph_edges["embedding_x"].map(display_map)
    graph_edges["embedding_y_display"] = graph_edges["embedding_y"].map(display_map)
    annotations = embedding_annotations(
        ids,
        reviewed,
        similarity,
        adjacency,
        directed,
        modalities,
        communities,
    )
    community_modalities = community_modality_table(
        communities, modalities, modality_order
    )
    modality_pairs = modality_pair_table(
        pairs,
        dict(zip(reviewed["embedding_id"], reviewed["coarse_modality"])),
        modality_order,
    )
    adjacency_frame = pd.DataFrame(adjacency, index=ids, columns=ids)
    adjacency_frame.index.name = "embedding_id"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "graph_edges": output_dir / "consensus_graph_edges.tsv",
        "embedding_annotations": output_dir / "embedding_graph_annotations.tsv",
        "community_modality_counts": output_dir / "community_modality_counts.tsv",
        "community_trace": output_dir / "community_detection_trace.tsv",
        "graph_sensitivity": output_dir / "graph_sensitivity.tsv",
        "modality_tests": output_dir / "modality_association_tests.tsv",
        "modality_pair_similarity": output_dir / "modality_pair_similarity.tsv",
        "adjacency_matrix": output_dir / "consensus_graph_adjacency_matrix.tsv",
    }
    atomic_dataframe(graph_edges, output_paths["graph_edges"])
    atomic_dataframe(annotations, output_paths["embedding_annotations"])
    atomic_dataframe(community_modalities, output_paths["community_modality_counts"])
    atomic_dataframe(community_trace, output_paths["community_trace"])
    atomic_dataframe(graph_sensitivity, output_paths["graph_sensitivity"])
    atomic_dataframe(association_tests, output_paths["modality_tests"])
    atomic_dataframe(modality_pairs, output_paths["modality_pair_similarity"])
    atomic_dataframe(adjacency_frame, output_paths["adjacency_matrix"], index=True)

    primary_test_map = primary_tests.set_index("test_id")
    key_test_ids = ("community_modality_nmi", "modality_weighted_modularity")
    modality_signal = all(
        float(primary_test_map.loc[test_id, "holm_p_within_scope"]) <= 0.05
        for test_id in key_test_ids
    )
    community_sizes = {
        str(int(key)): int(value)
        for key, value in Counter(int(value) for value in communities).items()
    }
    summary: dict[str, object] = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_pairs": EXPECTED_PAIRS,
        "graph": {
            "construction": "weighted_symmetrized_union_knn",
            "score": PRIMARY_SCORE,
            "directed_neighbours_per_embedding": args.graph_k,
            **diagnostics,
        },
        "community_detection": {
            "method": "deterministic_greedy_weighted_modularity",
            "resolution": MODULARITY_RESOLUTION,
            "merge_tolerance": MERGE_TOLERANCE,
            "modality_labels_used_for_detection": False,
            "community_sizes": community_sizes,
        },
        "modality_inference": {
            "method": "fixed_count_node_label_permutation",
            "n_permutations": args.permutations,
            "seed": args.seed,
            "ordinary_edge_or_pair_independence_p_values_used": False,
            "key_tests_holm_significant": bool(modality_signal),
            "recommendation": (
                "INTERPRET_COMMUNITIES_WITH_MODALITY_ENRICHMENT_AND_RESIDUAL_STRUCTURE"
                if modality_signal
                else "TREAT_COMMUNITIES_AS_UNSUPERVISED_WITHOUT_ESTABLISHED_MODALITY_ALIGNMENT"
            ),
        },
        "sensitivity": {
            "consensus_configurations_evaluated": ["primary"],
            "graph_k_values": [args.graph_k],
            "hybrid_embeddings_excluded": int(np.count_nonzero(hybrid)),
            "nonhybrid_embeddings": int(len(nonhybrid_indices)),
        },
        "sources": {
            "primary_panel": {"path": str(panel_path), "sha256": sha256(panel_path)},
            "consensus_pairs": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "consensus_matrix": {"path": str(matrix_path), "sha256": sha256(matrix_path)},
            "consensus_summary": {
                "path": str(consensus_summary_path),
                "sha256": sha256(consensus_summary_path),
            },
            "reviewed_modalities": {"path": str(labels_path), "sha256": sha256(labels_path)},
            "modality_summary": {
                "path": str(modality_summary_path),
                "sha256": sha256(modality_summary_path),
            },
        },
        "outputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in output_paths.items()
        },
        "interpretation_limits": {
            "affinity": (
                "The graph uses relative rank affinity; edge weights are not biological probabilities."
            ),
            "communities": (
                "Communities are exploratory graph structure, not ground-truth biological classes."
            ),
            "modalities": (
                "Coarse modalities follow dominant provenance and do not erase hybrid content."
            ),
            "quality": (
                "Graph centrality or community membership is not an embedding-quality score."
            ),
        },
    }
    atomic_json(summary, output_dir / "graph_structure_summary.json")

    print("===== CONSENSUS GRAPH STRUCTURE =====")
    print("status: PASS")
    print(f"embeddings: {EXPECTED_EMBEDDINGS}")
    print(f"pairs: {EXPECTED_PAIRS}")
    print("primary graph:")
    print(f"  construction: weighted symmetrized union kNN (k={args.graph_k})")
    print(f"  edges: {diagnostics['n_edges']}")
    print(f"  connected components: {diagnostics['n_connected_components']}")
    print("community structure:")
    print(f"  communities: {diagnostics['n_communities']}")
    print(f"  weighted modularity: {diagnostics['community_weighted_modularity']:.6f}")
    print("modality association (fixed-count node-label permutations):")
    for test_id in (
        "community_modality_nmi",
        "modality_weighted_modularity",
        "directed_knn_modality_purity",
        "within_minus_between_mean_consensus",
    ):
        row = primary_test_map.loc[test_id]
        print(
            f"  {test_id}: {row['observed']:.6f} "
            f"(p={row['permutation_p_upper']:.6g}, "
            f"Holm p={row['holm_p_within_scope']:.6g})"
        )
    print("sensitivity:")
    print("  consensus configurations: primary only")
    print(f"  graph k values: {args.graph_k}")
    print(f"  hybrid-free embeddings: {len(nonhybrid_indices)}")
    print(f"recommendation: {summary['modality_inference']['recommendation']}")
    print(f"output: {output_dir}")
    print(
        "NOTE: communities are unsupervised affinity-graph structure; modality labels "
        "are tested afterward and are not treated as ground truth."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print("===== CONSENSUS GRAPH STRUCTURE =====", file=sys.stderr)
        print("status: FAIL", file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
