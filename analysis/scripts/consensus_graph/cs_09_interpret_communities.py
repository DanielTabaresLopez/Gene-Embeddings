#!/usr/bin/env python3
"""Interpret consensus-graph communities and residual relationships.

The primary communities were learned without biological labels in cs_08.  This
stage keeps that partition fixed and uses fixed-count node-label permutations to
test every community-by-modality cell.  It also reports graph bridge statistics
and pair affinities adjusted for the mean of their exact modality-pair block.

Community enrichment is inferential because modality labels were not used to
learn the partition.  Bridge rankings and modality-adjusted pair residuals are
descriptive: they reuse the same consensus affinities that produced the graph.
No ordinary edge- or pair-independence p-values are calculated.

This script reads only completed consensus-graph tables and summaries. It does not
load embedding arrays, fit PCA/MLP models, or use directional-predictability
outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations, combinations_with_replacement
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_EMBEDDINGS = 56
EXPECTED_PAIRS = 1_540
EXPECTED_MODALITIES = 11
N_PERMUTATIONS = 9_999
SEED = 33_109
TOP_BRIDGES = 10
TOP_CROSS_MODALITY_PAIRS = 25
PRIMARY_SCORE = "consensus_similarity"
DISAGREEMENT = "component_absolute_disagreement"


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
        "--graph-dir",
        type=Path,
        default=None,
        help="Input graph directory (default: consensus_graph_v2/08_graph_structure).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: consensus_graph_v2/09_community_interpretation).",
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
        "--top-bridges",
        type=int,
        default=TOP_BRIDGES,
        help=f"Bridge embeddings retained (default: {TOP_BRIDGES}).",
    )
    parser.add_argument(
        "--top-cross-modality-pairs",
        type=int,
        default=TOP_CROSS_MODALITY_PAIRS,
        help=(
            "Modality-adjusted cross-modality pairs retained "
            f"(default: {TOP_CROSS_MODALITY_PAIRS})."
        ),
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


def parse_boolean(series: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = series.astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError(f"{label} contains invalid Boolean values")
    return parsed.to_numpy(dtype=bool)


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


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Holm adjustment requires one finite p-value vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def enrichment_table(
    communities: np.ndarray,
    modalities: np.ndarray,
    modality_order: list[str],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    if n_permutations < 99:
        raise ValueError("At least 99 label permutations are required")
    community_order = sorted(int(value) for value in np.unique(communities))
    community_index = {value: index for index, value in enumerate(community_order)}
    modality_index = {value: index for index, value in enumerate(modality_order)}
    if set(str(value) for value in modalities) != set(modality_order):
        raise ValueError("Observed modalities do not match the frozen modality order")
    codes = np.array([modality_index[str(value)] for value in modalities], dtype=int)
    community_codes = np.array(
        [community_index[int(value)] for value in communities], dtype=int
    )
    n_communities = len(community_order)
    n_modalities = len(modality_order)
    observed = np.zeros((n_communities, n_modalities), dtype=int)
    for community_code in range(n_communities):
        observed[community_code] = np.bincount(
            codes[community_codes == community_code], minlength=n_modalities
        )
    community_sizes = observed.sum(axis=1)
    modality_sizes = observed.sum(axis=0)
    expected = np.outer(community_sizes, modality_sizes) / len(modalities)

    rng = np.random.default_rng(seed)
    null = np.empty(
        (n_permutations, n_communities, n_modalities), dtype=np.int16
    )
    for permutation_index in range(n_permutations):
        shuffled = rng.permutation(codes)
        for community_code in range(n_communities):
            null[permutation_index, community_code] = np.bincount(
                shuffled[community_codes == community_code],
                minlength=n_modalities,
            )

    rows: list[dict[str, object]] = []
    for community_code, community_id in enumerate(community_order):
        for modality_code, modality in enumerate(modality_order):
            value = int(observed[community_code, modality_code])
            expected_value = float(expected[community_code, modality_code])
            null_values = null[:, community_code, modality_code].astype(float)
            upper = (1 + int(np.count_nonzero(null_values >= value))) / (
                n_permutations + 1
            )
            lower = (1 + int(np.count_nonzero(null_values <= value))) / (
                n_permutations + 1
            )
            deviation = abs(value - expected_value)
            two_sided = (
                1
                + int(
                    np.count_nonzero(
                        np.abs(null_values - expected_value) >= deviation - 1e-12
                    )
                )
            ) / (n_permutations + 1)
            community_size = int(community_sizes[community_code])
            modality_size = int(modality_sizes[modality_code])
            fold = (value / community_size) / (modality_size / len(modalities))
            variance = (
                modality_size
                * (len(modalities) - modality_size)
                * community_size
                * (len(modalities) - community_size)
                / (len(modalities) ** 2 * (len(modalities) - 1))
            )
            standardized = (
                (value - expected_value) / np.sqrt(variance)
                if variance > 0.0
                else 0.0
            )
            rows.append(
                {
                    "community_id": community_id,
                    "coarse_modality": modality,
                    "observed_count": value,
                    "community_size": community_size,
                    "modality_size": modality_size,
                    "expected_count_under_fixed_margins": expected_value,
                    "fold_enrichment": fold,
                    "standardized_count_residual": standardized,
                    "permutation_p_upper_enrichment": upper,
                    "permutation_p_lower_depletion": lower,
                    "permutation_p_two_sided": two_sided,
                }
            )
    result = pd.DataFrame(rows)
    result["holm_p_two_sided_across_cells"] = holm_adjust(
        result["permutation_p_two_sided"].to_numpy(dtype=float)
    )
    result["holm_significant_two_sided"] = (
        result["holm_p_two_sided_across_cells"] <= 0.05
    )
    result["cell_interpretation"] = "not_significant_after_holm"
    significant = result["holm_significant_two_sided"]
    enriched = result["observed_count"] > result["expected_count_under_fixed_margins"]
    depleted = result["observed_count"] < result["expected_count_under_fixed_margins"]
    result.loc[significant & enriched, "cell_interpretation"] = "enriched"
    result.loc[significant & depleted, "cell_interpretation"] = "depleted"
    return result


def make_community_profiles(
    annotations: pd.DataFrame,
    enrichment: pd.DataFrame,
    modality_order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for community_id, subset in annotations.groupby("community_id", sort=True):
        counts = subset["coarse_modality"].value_counts()
        dominant_count = int(counts.max())
        dominant_modalities = sorted(
            str(value) for value in counts.index[counts == dominant_count]
        )
        probabilities = np.array(
            [counts.get(modality, 0) / len(subset) for modality in modality_order],
            dtype=float,
        )
        positive = probabilities[probabilities > 0.0]
        entropy = float(-np.sum(positive * np.log2(positive)))
        normalized_entropy = entropy / np.log2(len(modality_order))
        cells = enrichment.loc[enrichment["community_id"] == int(community_id)]
        enriched = sorted(
            cells.loc[cells["cell_interpretation"] == "enriched", "coarse_modality"]
            .astype(str)
            .tolist()
        )
        depleted = sorted(
            cells.loc[cells["cell_interpretation"] == "depleted", "coarse_modality"]
            .astype(str)
            .tolist()
        )
        rows.append(
            {
                "community_id": int(community_id),
                "n_embeddings": int(len(subset)),
                "dominant_modality": "|".join(dominant_modalities),
                "dominant_modality_count": dominant_count,
                "dominant_modality_fraction": dominant_count / len(subset),
                "modalities_present": int(np.count_nonzero(probabilities)),
                "modality_entropy_bits": entropy,
                "normalized_modality_entropy": normalized_entropy,
                "holm_significant_enriched_modalities": "|".join(enriched),
                "holm_significant_depleted_modalities": "|".join(depleted),
                "embedding_ids": "|".join(sorted(subset["embedding_id"].astype(str))),
            }
        )
    return pd.DataFrame(rows)


def modality_adjusted_pair_table(
    pairs: pd.DataFrame,
    annotation_map: pd.DataFrame,
    modality_order: list[str],
) -> pd.DataFrame:
    label_map = annotation_map.set_index("embedding_id")["coarse_modality"].to_dict()
    community_map = annotation_map.set_index("embedding_id")["community_id"].to_dict()
    order = {value: index for index, value in enumerate(modality_order)}
    result = pairs.copy()
    raw_x = result["embedding_x"].map(label_map)
    raw_y = result["embedding_y"].map(label_map)
    if raw_x.isna().any() or raw_y.isna().any():
        raise ValueError("Some pair endpoints could not be mapped to modalities")
    modality_x: list[str] = []
    modality_y: list[str] = []
    for left, right in zip(raw_x.astype(str), raw_y.astype(str)):
        if order[left] <= order[right]:
            modality_x.append(left)
            modality_y.append(right)
        else:
            modality_x.append(right)
            modality_y.append(left)
    result["embedding_x_modality"] = raw_x.astype(str)
    result["embedding_y_modality"] = raw_y.astype(str)
    result["modality_block_x"] = modality_x
    result["modality_block_y"] = modality_y
    result["same_modality"] = raw_x.astype(str).to_numpy() == raw_y.astype(str).to_numpy()
    result["community_x"] = result["embedding_x"].map(community_map).astype(int)
    result["community_y"] = result["embedding_y"].map(community_map).astype(int)
    result["same_community"] = result["community_x"] == result["community_y"]
    result["modality_block"] = (
        result["modality_block_x"] + "__" + result["modality_block_y"]
    )

    scores = numeric(result, PRIMARY_SCORE, "consensus pairs")
    result["modality_block_n_pairs"] = result.groupby("modality_block")[
        PRIMARY_SCORE
    ].transform("size")
    result["modality_block_mean_consensus"] = result.groupby("modality_block")[
        PRIMARY_SCORE
    ].transform("mean")
    result["modality_block_median_consensus"] = result.groupby("modality_block")[
        PRIMARY_SCORE
    ].transform("median")
    result["modality_block_sd_consensus"] = result.groupby("modality_block")[
        PRIMARY_SCORE
    ].transform("std").fillna(0.0)
    counts = result["modality_block_n_pairs"].to_numpy(dtype=float)
    means = result["modality_block_mean_consensus"].to_numpy(dtype=float)
    # Functional has two embeddings in the frozen panel, so its within-modality
    # block contains one pair. A leave-one-pair-out baseline is undefined for
    # that singleton. Use the inclusive block mean for singleton blocks, which
    # assigns the sole pair a neutral residual of zero and a neutral percentile.
    baseline = means.copy()
    non_singleton = counts > 1.0
    baseline[non_singleton] = (
        counts[non_singleton] * means[non_singleton] - scores[non_singleton]
    ) / (counts[non_singleton] - 1.0)
    result["leave_one_pair_out_modality_block_mean"] = baseline
    result["modality_block_baseline_method"] = np.where(
        non_singleton, "leave_one_pair_out", "inclusive_singleton"
    )
    result["modality_adjusted_consensus_residual"] = scores - result[
        "leave_one_pair_out_modality_block_mean"
    ].to_numpy(dtype=float)
    sd = result["modality_block_sd_consensus"].to_numpy(dtype=float)
    result["modality_block_z_score"] = np.divide(
        scores - means,
        sd,
        out=np.zeros_like(scores),
        where=sd > 0.0,
    )
    result["within_modality_block_percentile"] = result.groupby("modality_block")[
        PRIMARY_SCORE
    ].rank(method="average", pct=False)
    ranks = result["within_modality_block_percentile"].to_numpy(dtype=float) - 1.0
    denominators = counts - 1.0
    result["within_modality_block_percentile"] = np.divide(
        ranks,
        denominators,
        out=np.full_like(ranks, 0.5),
        where=denominators > 0.0,
    )

    residual_sums = result.groupby("modality_block")[
        "modality_adjusted_consensus_residual"
    ].sum()
    if not np.allclose(residual_sums.to_numpy(dtype=float), 0.0, atol=1e-10):
        raise AssertionError("Leave-one-pair-out residuals do not center within block")
    return result


def embedding_role_table(
    annotations: pd.DataFrame,
    edges: pd.DataFrame,
    pairs: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    ids = sorted(annotations["embedding_id"].astype(str))
    index = {embedding_id: position for position, embedding_id in enumerate(ids)}
    n = len(ids)
    weights = np.zeros((n, n), dtype=float)
    for x, y, weight in zip(
        edges["embedding_x"], edges["embedding_y"], edges[PRIMARY_SCORE]
    ):
        i, j = index[str(x)], index[str(y)]
        weights[i, j] = float(weight)
        weights[j, i] = float(weight)
    base = annotations.set_index("embedding_id").loc[ids].copy()
    communities = base["community_id"].to_numpy(dtype=int)
    modalities = base["coarse_modality"].astype(str).to_numpy()
    profile_map = profiles.set_index("community_id")["dominant_modality"].to_dict()
    pair_means = {embedding_id: [] for embedding_id in ids}
    disagreement_means = {embedding_id: [] for embedding_id in ids}
    for row in pairs.itertuples(index=False):
        x = str(row.embedding_x)
        y = str(row.embedding_y)
        score = float(getattr(row, PRIMARY_SCORE))
        disagreement = float(getattr(row, DISAGREEMENT))
        pair_means[x].append(score)
        pair_means[y].append(score)
        disagreement_means[x].append(disagreement)
        disagreement_means[y].append(disagreement)

    rows: list[dict[str, object]] = []
    for i, embedding_id in enumerate(ids):
        total = float(weights[i].sum())
        if total <= 0.0:
            raise ValueError(f"Embedding {embedding_id!r} has zero graph weight")
        community_weights = np.array(
            [weights[i, communities == community].sum() for community in np.unique(communities)],
            dtype=float,
        )
        participation = float(1.0 - np.sum((community_weights / total) ** 2))
        within_community = float(weights[i, communities == communities[i]].sum())
        cross_community = total - within_community
        same_modality = float(weights[i, modalities == modalities[i]].sum())
        cross_modality = total - same_modality
        rows.append(
            {
                "embedding_id": embedding_id,
                "coarse_modality": str(modalities[i]),
                "hybrid_or_multimodal": bool(base.iloc[i]["hybrid_or_multimodal"]),
                "community_id": int(communities[i]),
                "community_dominant_modality": str(profile_map[int(communities[i])]),
                "matches_single_dominant_modality": bool(
                    str(profile_map[int(communities[i])]) == str(modalities[i])
                ),
                "graph_degree": int(np.count_nonzero(weights[i])),
                "weighted_degree": total,
                "within_community_weighted_degree": within_community,
                "cross_community_weighted_degree": cross_community,
                "cross_community_weight_fraction": cross_community / total,
                "weighted_participation_coefficient": participation,
                "same_modality_weighted_degree": same_modality,
                "cross_modality_weighted_degree": cross_modality,
                "cross_modality_weight_fraction": cross_modality / total,
                "mean_consensus_similarity_all_pairs": float(np.mean(pair_means[embedding_id])),
                "mean_component_absolute_disagreement": float(
                    np.mean(disagreement_means[embedding_id])
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["bridge_rank"] = result[
        "weighted_participation_coefficient"
    ].rank(method="min", ascending=False).astype(int)
    result["within_community_hub_rank"] = result.groupby("community_id")[
        "within_community_weighted_degree"
    ].rank(method="min", ascending=False).astype(int)
    return result.sort_values("embedding_id").reset_index(drop=True)


def residual_summaries(
    residuals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_rows: list[dict[str, object]] = []
    for (same_modality, same_community), subset in residuals.groupby(
        ["same_modality", "same_community"], sort=True
    ):
        values = numeric(subset, "modality_adjusted_consensus_residual", "residual scope")
        scope_rows.append(
            {
                "same_modality": bool(same_modality),
                "same_community": bool(same_community),
                "n_pairs": int(len(subset)),
                "mean_consensus_similarity": float(subset[PRIMARY_SCORE].mean()),
                "mean_modality_adjusted_residual": float(np.mean(values)),
                "median_modality_adjusted_residual": float(np.median(values)),
                "fraction_positive_residual": float(np.mean(values > 0.0)),
            }
        )
    community_rows: list[dict[str, object]] = []
    community_order = sorted(
        set(residuals["community_x"].astype(int))
        | set(residuals["community_y"].astype(int))
    )
    for left, right in combinations_with_replacement(community_order, 2):
        mask = (
            ((residuals["community_x"] == left) & (residuals["community_y"] == right))
            | ((residuals["community_x"] == right) & (residuals["community_y"] == left))
        )
        subset = residuals.loc[mask]
        if subset.empty:
            continue
        values = numeric(subset, "modality_adjusted_consensus_residual", "community pair")
        community_rows.append(
            {
                "community_x": left,
                "community_y": right,
                "same_community": left == right,
                "n_pairs": int(len(subset)),
                "mean_consensus_similarity": float(subset[PRIMARY_SCORE].mean()),
                "mean_modality_adjusted_residual": float(np.mean(values)),
                "median_modality_adjusted_residual": float(np.median(values)),
                "fraction_positive_residual": float(np.mean(values > 0.0)),
            }
        )
    return pd.DataFrame(scope_rows), pd.DataFrame(community_rows)


def atomic_dataframe(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, sep="\t", index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_recorded_hash(payload: dict[str, object], section: str, key: str, path: Path) -> None:
    recorded = payload.get(section, {}).get(key, {}).get("sha256")
    if recorded != sha256(path):
        raise ValueError(f"Upstream {key!r} changed after graph construction")


def run_analysis(args: argparse.Namespace, quiet: bool = False) -> dict[str, object]:
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
    graph_dir = (
        args.graph_dir.expanduser().resolve()
        if args.graph_dir is not None
        else workflow_root / "08_graph_structure"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow_root / "09_community_interpretation"
    )
    if args.permutations < 99:
        raise ValueError("At least 99 permutations are required")
    if args.top_bridges < 1 or args.top_cross_modality_pairs < 1:
        raise ValueError("Top-table sizes must be positive")

    graph_summary_path = require_file(graph_dir / "graph_structure_summary.json")
    annotations_path = require_file(graph_dir / "embedding_graph_annotations.tsv")
    edges_path = require_file(graph_dir / "consensus_graph_edges.tsv")
    counts_path = require_file(graph_dir / "community_modality_counts.tsv")
    pairs_path = require_file(consensus_dir / "consensus_pair_scores.tsv")
    labels_path = require_file(modality_dir / "reviewed_coarse_modalities.tsv")

    graph_summary = json.loads(graph_summary_path.read_text())
    if graph_summary.get("status") != "PASS":
        raise ValueError("Upstream graph summary does not have status PASS")
    if int(graph_summary.get("n_embeddings", -1)) != EXPECTED_EMBEDDINGS:
        raise ValueError("Upstream graph embedding count changed")
    if int(graph_summary.get("n_pairs", -1)) != EXPECTED_PAIRS:
        raise ValueError("Upstream graph pair count changed")
    graph_recommendation = graph_summary.get("modality_inference", {}).get(
        "recommendation"
    )
    accepted_recommendations = {
        "INTERPRET_COMMUNITIES_WITH_MODALITY_ENRICHMENT_AND_RESIDUAL_STRUCTURE",
        "TREAT_COMMUNITIES_AS_UNSUPERVISED_WITHOUT_ESTABLISHED_MODALITY_ALIGNMENT",
    }
    if graph_recommendation not in accepted_recommendations:
        raise ValueError(f"Unexpected graph recommendation: {graph_recommendation!r}")
    verify_recorded_hash(graph_summary, "outputs", "embedding_annotations", annotations_path)
    verify_recorded_hash(graph_summary, "outputs", "graph_edges", edges_path)
    verify_recorded_hash(graph_summary, "outputs", "community_modality_counts", counts_path)
    verify_recorded_hash(graph_summary, "sources", "consensus_pairs", pairs_path)
    verify_recorded_hash(graph_summary, "sources", "reviewed_modalities", labels_path)

    annotations = pd.read_csv(annotations_path, sep="\t")
    require_columns(
        annotations,
        (
            "embedding_id",
            "coarse_modality",
            "hybrid_or_multimodal",
            "community_id",
            "graph_degree",
            "weighted_degree",
        ),
        "graph annotations",
    )
    annotations["embedding_id"] = annotations["embedding_id"].astype(str)
    if annotations["embedding_id"].duplicated().any() or len(annotations) != EXPECTED_EMBEDDINGS:
        raise ValueError(
            f"Graph annotations must contain {EXPECTED_EMBEDDINGS} unique embeddings"
        )
    annotations["coarse_modality"] = annotations["coarse_modality"].astype(str)
    annotations["hybrid_or_multimodal"] = parse_boolean(
        annotations["hybrid_or_multimodal"], "graph hybrid flags"
    )
    annotations["community_id"] = pd.to_numeric(
        annotations["community_id"], errors="raise"
    ).astype(int)
    ids = sorted(annotations["embedding_id"].tolist())
    community_sizes = Counter(annotations["community_id"].astype(int))
    recorded_sizes = {
        int(key): int(value)
        for key, value in graph_summary.get("community_detection", {})
        .get("community_sizes", {})
        .items()
    }
    if dict(community_sizes) != recorded_sizes:
        raise ValueError("Community assignments no longer match the graph summary")

    labels = pd.read_csv(labels_path, sep="\t")
    require_columns(labels, ("embedding_id", "coarse_modality"), "reviewed modalities")
    labels["embedding_id"] = labels["embedding_id"].astype(str)
    if labels["embedding_id"].duplicated().any() or set(labels["embedding_id"]) != set(ids):
        raise ValueError("Reviewed modality IDs do not match graph annotations")
    expected_labels = labels.set_index("embedding_id").loc[ids, "coarse_modality"].astype(str)
    observed_labels = annotations.set_index("embedding_id").loc[ids, "coarse_modality"]
    if not np.array_equal(expected_labels.to_numpy(), observed_labels.to_numpy()):
        raise ValueError("Graph annotation modalities differ from reviewed labels")
    modality_order = list(dict.fromkeys(labels["coarse_modality"].astype(str).tolist()))
    if len(modality_order) != EXPECTED_MODALITIES:
        raise ValueError(f"Expected {EXPECTED_MODALITIES} manifest modalities")

    counts = pd.read_csv(counts_path, sep="\t")
    require_columns(
        counts,
        ("community_id", "coarse_modality", "n_embeddings"),
        "community modality counts",
    )
    recomputed_counts = (
        annotations.groupby(["community_id", "coarse_modality"])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [sorted(community_sizes), modality_order],
                names=["community_id", "coarse_modality"],
            ),
            fill_value=0,
        )
        .rename("n_embeddings")
        .reset_index()
    )
    stored_counts = counts[["community_id", "coarse_modality", "n_embeddings"]].copy()
    stored_counts["community_id"] = stored_counts["community_id"].astype(int)
    stored_counts["coarse_modality"] = stored_counts["coarse_modality"].astype(str)
    stored_counts["n_embeddings"] = stored_counts["n_embeddings"].astype(int)
    stored_counts = stored_counts.sort_values(["community_id", "coarse_modality"]).reset_index(drop=True)
    recomputed_counts = recomputed_counts.sort_values(["community_id", "coarse_modality"]).reset_index(drop=True)
    if not stored_counts.equals(recomputed_counts):
        raise ValueError("Stored community-modality counts do not reproduce")

    pairs = canonicalize_pairs(pd.read_csv(pairs_path, sep="\t"), "consensus pairs")
    validate_pair_coverage(pairs, ids)
    require_columns(pairs, (PRIMARY_SCORE, DISAGREEMENT), "consensus pairs")
    scores = numeric(pairs, PRIMARY_SCORE, "consensus pairs")
    numeric(pairs, DISAGREEMENT, "consensus pairs")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Consensus scores must lie in [0, 1]")

    edges = canonicalize_pairs(pd.read_csv(edges_path, sep="\t"), "graph edges")
    require_columns(edges, (PRIMARY_SCORE,), "graph edges")
    edge_scores = numeric(edges, PRIMARY_SCORE, "graph edges")
    if np.any(edge_scores <= 0.0):
        raise ValueError("Graph edge weights must be positive")
    if not set(edges["embedding_x"]).union(edges["embedding_y"]).issubset(set(ids)):
        raise ValueError("Graph edges contain unexpected embedding IDs")
    if len(edges) != int(graph_summary.get("graph", {}).get("n_edges", -1)):
        raise ValueError("Graph edge count changed")

    ordered_annotations = annotations.set_index("embedding_id").loc[ids].reset_index()
    communities = ordered_annotations["community_id"].to_numpy(dtype=int)
    modalities = ordered_annotations["coarse_modality"].astype(str).to_numpy()
    enrichment = enrichment_table(
        communities, modalities, modality_order, args.permutations, args.seed
    )
    profiles = make_community_profiles(ordered_annotations, enrichment, modality_order)
    residuals = modality_adjusted_pair_table(pairs, ordered_annotations, modality_order)
    scope_summary, community_pair_summary = residual_summaries(residuals)
    roles = embedding_role_table(ordered_annotations, edges, pairs, profiles)
    top_bridges = roles.sort_values(
        ["weighted_participation_coefficient", "cross_community_weight_fraction", "embedding_id"],
        ascending=[False, False, True],
    ).head(min(args.top_bridges, len(roles)))
    top_cross = residuals.loc[~residuals["same_modality"]].sort_values(
        ["modality_adjusted_consensus_residual", PRIMARY_SCORE, "embedding_x", "embedding_y"],
        ascending=[False, False, True, True],
    ).head(args.top_cross_modality_pairs)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "community_modality_enrichment": output_dir / "community_modality_enrichment.tsv",
        "community_profiles": output_dir / "community_profiles.tsv",
        "embedding_roles": output_dir / "embedding_graph_roles.tsv",
        "top_bridge_embeddings": output_dir / "top_bridge_embeddings.tsv",
        "pair_residuals": output_dir / "modality_adjusted_pair_residuals.tsv",
        "top_cross_modality_pairs": output_dir / "top_cross_modality_residual_pairs.tsv",
        "residual_scope_summary": output_dir / "residual_scope_summary.tsv",
        "community_pair_residual_summary": output_dir / "community_pair_residual_summary.tsv",
    }
    atomic_dataframe(enrichment, output_paths["community_modality_enrichment"])
    atomic_dataframe(profiles, output_paths["community_profiles"])
    atomic_dataframe(roles, output_paths["embedding_roles"])
    atomic_dataframe(top_bridges, output_paths["top_bridge_embeddings"])
    atomic_dataframe(residuals, output_paths["pair_residuals"])
    atomic_dataframe(top_cross, output_paths["top_cross_modality_pairs"])
    atomic_dataframe(scope_summary, output_paths["residual_scope_summary"])
    atomic_dataframe(community_pair_summary, output_paths["community_pair_residual_summary"])

    enriched_cells = int(np.count_nonzero(enrichment["cell_interpretation"] == "enriched"))
    depleted_cells = int(np.count_nonzero(enrichment["cell_interpretation"] == "depleted"))
    same_community_residual = float(
        residuals.loc[residuals["same_community"], "modality_adjusted_consensus_residual"].mean()
    )
    between_community_residual = float(
        residuals.loc[~residuals["same_community"], "modality_adjusted_consensus_residual"].mean()
    )
    recommendation = (
        "REPORT_MODALITY_ENRICHED_COMMUNITIES_WITH_RESIDUAL_CROSS_MODALITY_STRUCTURE"
        if enriched_cells > 0
        else "REPORT_UNSUPERVISED_COMMUNITIES_WITHOUT_CELL_LEVEL_ENRICHMENT"
    )
    summary: dict[str, object] = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_pairs": EXPECTED_PAIRS,
        "n_communities": len(community_sizes),
        "n_modalities": EXPECTED_MODALITIES,
        "enrichment_inference": {
            "method": "fixed_count_node_label_permutation",
            "n_permutations": int(args.permutations),
            "seed": int(args.seed),
            "cells_tested": int(len(enrichment)),
            "multiple_testing": "Holm across all community-by-modality cells",
            "significant_enriched_cells": enriched_cells,
            "significant_depleted_cells": depleted_cells,
            "ordinary_node_or_pair_independence_p_values_used": False,
        },
        "residual_structure": {
            "adjustment": (
                "leave-one-pair-out mean of the exact coarse-modality-pair block; "
                "the single Functional--Functional pair receives a neutral zero "
                "residual against its inclusive singleton mean"
            ),
            "inferential_status": "descriptive_only_because_graph_uses_same_affinities",
            "same_community_mean_modality_adjusted_residual": same_community_residual,
            "between_community_mean_modality_adjusted_residual": between_community_residual,
        },
        "bridge_analysis": {
            "measure": "weighted participation coefficient across graph communities",
            "inferential_status": "descriptive_only",
            "top_n": int(len(top_bridges)),
        },
        "recommendation": recommendation,
        "upstream_graph_recommendation": graph_recommendation,
        "sources": {
            "graph_summary": {"path": str(graph_summary_path), "sha256": sha256(graph_summary_path)},
            "graph_annotations": {"path": str(annotations_path), "sha256": sha256(annotations_path)},
            "graph_edges": {"path": str(edges_path), "sha256": sha256(edges_path)},
            "community_modality_counts": {"path": str(counts_path), "sha256": sha256(counts_path)},
            "consensus_pairs": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "reviewed_modalities": {"path": str(labels_path), "sha256": sha256(labels_path)},
        },
        "outputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in output_paths.items()
        },
        "interpretation_limits": {
            "community": "Communities are unsupervised affinity structure, not ground-truth classes.",
            "enrichment": "Enrichment describes association with dominant-provenance strata, not exclusive biological content.",
            "residuals": "Pair residuals rank unusual affinities within a modality combination and are not independent-test statistics.",
            "bridges": "Bridge or hub status is not an embedding-quality score.",
        },
    }
    summary_path = output_dir / "community_interpretation_summary.json"
    atomic_json(summary, summary_path)

    if not quiet:
        print("===== CONSENSUS GRAPH COMMUNITY INTERPRETATION =====")
        print("status: PASS")
        print(f"embeddings: {EXPECTED_EMBEDDINGS}")
        print(f"pairs: {EXPECTED_PAIRS}")
        print(f"communities: {len(community_sizes)}")
        print("community profiles:")
        for row in profiles.itertuples(index=False):
            enriched_text = row.holm_significant_enriched_modalities or "none"
            print(
                f"  community {row.community_id}: n={row.n_embeddings}; "
                f"dominant={row.dominant_modality} "
                f"({row.dominant_modality_fraction:.3f}); "
                f"Holm-enriched={enriched_text}"
            )
        print("cell-wise modality inference:")
        print(f"  permutations: {args.permutations}")
        print(f"  cells tested: {len(enrichment)}")
        print(f"  Holm-significant enriched cells: {enriched_cells}")
        print(f"  Holm-significant depleted cells: {depleted_cells}")
        print("modality-adjusted residual structure (descriptive):")
        print(f"  same-community mean residual: {same_community_residual:.6f}")
        print(f"  between-community mean residual: {between_community_residual:.6f}")
        print("top bridge embeddings (descriptive):")
        for row in top_bridges.head(5).itertuples(index=False):
            print(
                f"  {row.embedding_id}: participation="
                f"{row.weighted_participation_coefficient:.6f}; "
                f"cross-community weight fraction={row.cross_community_weight_fraction:.6f}"
            )
        print("top cross-modality residual pairs (descriptive):")
        for row in top_cross.head(5).itertuples(index=False):
            print(
                f"  {row.embedding_x} -- {row.embedding_y}: "
                f"residual={row.modality_adjusted_consensus_residual:.6f}; "
                f"consensus={row.consensus_similarity:.6f}"
            )
        print(f"recommendation: {recommendation}")
        print(f"output: {output_dir}")
        print(
            "NOTE: enrichment uses node-label permutations; bridge rankings and "
            "pair residuals are descriptive and are not embedding-quality scores."
        )
    return summary


def build_self_test_fixture(base: Path) -> SimpleNamespace:
    workflow = base / "results/consensus_graph_v2"
    consensus_dir = workflow / "06_consensus"
    modality_dir = workflow / "07_modalities"
    graph_dir = workflow / "08_graph_structure"
    output_dir = workflow / "09_community_interpretation"
    consensus_dir.mkdir(parents=True)
    modality_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)

    ids = [f"embedding_{index:02d}" for index in range(EXPECTED_EMBEDDINGS)]
    modality_order = [
        "Protein sequence",
        "Biomedical text",
        "Single-cell",
        "Regulatory genomics",
        "Networks",
        "Multimodal",
        "Knowledge graph",
        "DNA sequence",
        "RNA sequence",
        "Expression",
        "Functional",
    ]
    modality_counts = [17, 8, 5, 5, 4, 3, 3, 3, 3, 3, 2]
    modalities: list[str] = []
    for modality, count in zip(modality_order, modality_counts):
        modalities.extend([modality] * count)
    communities = np.array(
        [
            1
            if modality in {"DNA sequence", "RNA sequence", "Regulatory genomics"}
            else 2
            if modality in {"Protein sequence", "Functional"}
            else 3
            for modality in modalities
        ],
        dtype=int,
    )
    labels = pd.DataFrame(
        {
            "embedding_id": ids,
            "coarse_modality": modalities,
            "hybrid_or_multimodal": [
                modality == "Multimodal" for modality in modalities
            ],
        }
    )
    labels_path = modality_dir / "reviewed_coarse_modalities.tsv"
    labels.to_csv(labels_path, sep="\t", index=False)

    rows: list[dict[str, object]] = []
    for i, j in combinations(range(EXPECTED_EMBEDDINGS), 2):
        jitter = ((i * 19 + j * 7) % 101) / 1000.0
        score = 0.72 + jitter if communities[i] == communities[j] else 0.18 + jitter
        rows.append(
            {
                "embedding_x": ids[i],
                "embedding_y": ids[j],
                PRIMARY_SCORE: score,
                DISAGREEMENT: abs(((i * 13 + j * 11) % 37) / 100.0),
            }
        )
    pairs = pd.DataFrame(rows)
    pairs_path = consensus_dir / "consensus_pair_scores.tsv"
    pairs.to_csv(pairs_path, sep="\t", index=False)

    edge_rows: list[dict[str, object]] = []
    for i, j in combinations(range(EXPECTED_EMBEDDINGS), 2):
        if communities[i] == communities[j] or (j == i + 1):
            score = float(
                pairs.loc[
                    (pairs["embedding_x"] == ids[i]) & (pairs["embedding_y"] == ids[j]),
                    PRIMARY_SCORE,
                ].iloc[0]
            )
            edge_rows.append(
                {"embedding_x": ids[i], "embedding_y": ids[j], PRIMARY_SCORE: score}
            )
    edges = pd.DataFrame(edge_rows)
    edges_path = graph_dir / "consensus_graph_edges.tsv"
    edges.to_csv(edges_path, sep="\t", index=False)

    weighted_degree = {embedding_id: 0.0 for embedding_id in ids}
    degree = {embedding_id: 0 for embedding_id in ids}
    for row in edges.itertuples(index=False):
        weight = float(getattr(row, PRIMARY_SCORE))
        weighted_degree[row.embedding_x] += weight
        weighted_degree[row.embedding_y] += weight
        degree[row.embedding_x] += 1
        degree[row.embedding_y] += 1
    annotations = labels.copy()
    annotations["community_id"] = communities
    annotations["graph_degree"] = annotations["embedding_id"].map(degree)
    annotations["weighted_degree"] = annotations["embedding_id"].map(weighted_degree)
    annotations_path = graph_dir / "embedding_graph_annotations.tsv"
    annotations.to_csv(annotations_path, sep="\t", index=False)

    count_rows: list[dict[str, object]] = []
    for community_id in sorted(set(communities)):
        for modality in modality_order:
            count_rows.append(
                {
                    "community_id": int(community_id),
                    "coarse_modality": modality,
                    "n_embeddings": int(
                        np.count_nonzero(
                            (communities == community_id) & (np.array(modalities) == modality)
                        )
                    ),
                }
            )
    counts_path = graph_dir / "community_modality_counts.tsv"
    pd.DataFrame(count_rows).to_csv(counts_path, sep="\t", index=False)

    summary = {
        "status": "PASS",
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_pairs": EXPECTED_PAIRS,
        "graph": {"n_edges": len(edges)},
        "community_detection": {
            "community_sizes": {
                str(key): int(value) for key, value in Counter(communities).items()
            }
        },
        "modality_inference": {
            "recommendation": "INTERPRET_COMMUNITIES_WITH_MODALITY_ENRICHMENT_AND_RESIDUAL_STRUCTURE"
        },
        "sources": {
            "consensus_pairs": {"sha256": sha256(pairs_path)},
            "reviewed_modalities": {"sha256": sha256(labels_path)},
        },
        "outputs": {
            "embedding_annotations": {"sha256": sha256(annotations_path)},
            "graph_edges": {"sha256": sha256(edges_path)},
            "community_modality_counts": {"sha256": sha256(counts_path)},
        },
    }
    (graph_dir / "graph_structure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return SimpleNamespace(
        root=base,
        consensus_dir=consensus_dir,
        modality_dir=modality_dir,
        graph_dir=graph_dir,
        output_dir=output_dir,
        permutations=999,
        seed=SEED,
        top_bridges=5,
        top_cross_modality_pairs=7,
    )


def run_self_test() -> None:
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    if not np.allclose(adjusted, np.array([0.03, 0.06, 0.06])):
        raise AssertionError("Holm-adjustment self-test failed")
    with tempfile.TemporaryDirectory(prefix="cs_09_self_test_") as directory:
        fixture_args = build_self_test_fixture(Path(directory))
        summary = run_analysis(fixture_args, quiet=True)
        if summary.get("status") != "PASS":
            raise AssertionError("End-to-end self-test did not PASS")
        if int(summary["enrichment_inference"]["cells_tested"]) != 33:
            raise AssertionError("Expected 33 enrichment cells in self-test")
        if int(summary["enrichment_inference"]["significant_enriched_cells"]) < 1:
            raise AssertionError("Aligned fixture did not yield expected enrichment")
        output_dir = fixture_args.output_dir
        expected_outputs = {
            "community_modality_enrichment.tsv",
            "community_profiles.tsv",
            "embedding_graph_roles.tsv",
            "top_bridge_embeddings.tsv",
            "modality_adjusted_pair_residuals.tsv",
            "top_cross_modality_residual_pairs.tsv",
            "residual_scope_summary.tsv",
            "community_pair_residual_summary.tsv",
            "community_interpretation_summary.json",
        }
        if {path.name for path in output_dir.iterdir()} != expected_outputs:
            raise AssertionError("End-to-end self-test output contract changed")
    print("SELF-TEST: PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_analysis(args)
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
        print("===== CONSENSUS GRAPH COMMUNITY INTERPRETATION =====", file=sys.stderr)
        print("status: FAIL", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
