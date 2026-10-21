#!/usr/bin/env python3
"""Build the guarded two-view equal-rank consensus for Workflow 33.

The upstream concordance stage pre-specifies the fusion rule:

1. rank the 1,540 calibrated fixed-panel Platonic linear-CKA scores;
2. rank the 1,540 calibrated fixed-panel Aristotelian local-overlap scores at
   k=100;
3. convert each tie-aware rank to a percentile in [0, 1]; and
4. average the two percentile scores with weights 0.5 and 0.5.

This script implements only that rule.  It retains both component views,
constructs complete symmetric similarity and dissimilarity matrices, and
creates nearest-neighbour and per-embedding summaries. Geometry diagnostics
explicitly test whether the rank-derived
similarity is positive semidefinite and whether 1 - similarity is a metric;
neither property is assumed.

The stage reads only consensus-graph tables and JSON summaries. It does not load
embedding arrays, fit PCA/MLP models, or use directional-predictability output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


EXPECTED_EMBEDDINGS = 56
EXPECTED_PAIRS = 1_540
PRIMARY_PLATONIC = "platonic_linear_calibrated"
PRIMARY_ARISTOTELIAN = "aristotelian_local_k100_calibrated"
EXPECTED_RECOMMENDATION = "BUILD_TWO_VIEW_EQUAL_RANK_CONSENSUS"
EXPECTED_PRIMARY_COMPARISON = (
    "primary_platonic_vs_aristotelian_k100_calibrated"
)
PLATONIC_WEIGHT = 0.5
ARISTOTELIAN_WEIGHT = 0.5
TOP_NEIGHBOURS = 10
PSD_TOLERANCE = 1e-10
TRIANGLE_TOLERANCE = 1e-12


VARIANTS: dict[str, tuple[str, str, str]] = {
    "primary": (
        "platonic_linear_calibrated",
        "aristotelian_local_k100_calibrated",
        "pre_specified_primary",
    ),
}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--views-dir",
        type=Path,
        default=None,
        help="Input view directory (default: consensus_graph_v2/04_views).",
    )
    parser.add_argument(
        "--concordance-dir",
        type=Path,
        default=None,
        help="Concordance directory (default: consensus_graph_v2/05_view_concordance).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: consensus_graph_v2/06_consensus).",
    )
    parser.add_argument(
        "--top-neighbours",
        type=int,
        default=TOP_NEIGHBOURS,
        help=f"Neighbours retained per embedding (default: {TOP_NEIGHBOURS}).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic in-memory tests and exit.",
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
    if np.ptp(values) == 0.0:
        raise ValueError(f"{label}/{column} is constant")
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
            "Primary pair coverage mismatch: "
            f"rows={len(frame)}, expected_rows={EXPECTED_PAIRS}, "
            f"missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}"
        )


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Percentile ranking requires a one-dimensional vector")
    if not np.all(np.isfinite(values)) or np.ptp(values) == 0.0:
        raise ValueError("Percentile ranking requires finite non-constant values")
    ranks = rankdata(values, method="average")
    result = (ranks - 1.0) / (len(values) - 1.0)
    if np.any((result < 0.0) | (result > 1.0)):
        raise AssertionError("Percentile ranks escaped [0, 1]")
    return result


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    norm = np.linalg.norm(centered)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Cannot standardize a constant/non-finite vector")
    return centered / norm


def spearman_rho(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 3:
        raise ValueError("Spearman inputs must have equal length >= 3")
    left_rank = standardize(rankdata(left, method="average"))
    right_rank = standardize(rankdata(right, method="average"))
    return float(np.dot(left_rank, right_rank))


def equal_rank_variant(
    pairs: pd.DataFrame,
    platonic_column: str,
    aristotelian_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    platonic = percentile_ranks(
        numeric(pairs, platonic_column, "primary_pair_views")
    )
    aristotelian = percentile_ranks(
        numeric(pairs, aristotelian_column, "primary_pair_views")
    )
    consensus = (
        PLATONIC_WEIGHT * platonic + ARISTOTELIAN_WEIGHT * aristotelian
    )
    if not np.allclose(
        consensus,
        0.5 * (platonic + aristotelian),
        rtol=0.0,
        atol=1e-15,
    ):
        raise AssertionError("Consensus weights are not exactly equal")
    return platonic, aristotelian, consensus


def build_matrix(
    pairs: pd.DataFrame,
    ids: list[str],
    values: np.ndarray,
    diagonal: float,
    label: str,
) -> pd.DataFrame:
    if len(values) != len(pairs) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label}: invalid pair-value vector")
    ordered = sorted(ids)
    positions = {embedding_id: index for index, embedding_id in enumerate(ordered)}
    matrix = np.full((len(ordered), len(ordered)), np.nan, dtype=float)
    np.fill_diagonal(matrix, diagonal)
    for x, y, value in zip(
        pairs["embedding_x"].astype(str),
        pairs["embedding_y"].astype(str),
        values,
    ):
        i, j = positions[x], positions[y]
        matrix[i, j] = value
        matrix[j, i] = value
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label}: matrix is incomplete/non-finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0):
        raise ValueError(f"{label}: matrix is not exactly symmetric")
    result = pd.DataFrame(matrix, index=ordered, columns=ordered)
    result.index.name = "embedding_id"
    return result


def geometry_diagnostics(similarity: np.ndarray) -> dict[str, object]:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Geometry diagnostics require a square matrix")
    if not np.allclose(similarity, similarity.T, rtol=0.0, atol=0.0):
        raise ValueError("Geometry diagnostics require an exactly symmetric matrix")
    eigenvalues = np.linalg.eigvalsh(similarity)
    negative = eigenvalues < -PSD_TOLERANCE

    dissimilarity = 1.0 - similarity
    n = len(similarity)
    tested = 0
    violations = 0
    maximum_excess = 0.0
    for i, j, k in combinations(range(n), 3):
        sides = (
            (dissimilarity[i, j], dissimilarity[i, k], dissimilarity[k, j]),
            (dissimilarity[i, k], dissimilarity[i, j], dissimilarity[j, k]),
            (dissimilarity[j, k], dissimilarity[j, i], dissimilarity[i, k]),
        )
        for direct, first, second in sides:
            tested += 1
            excess = float(direct - first - second)
            if excess > TRIANGLE_TOLERANCE:
                violations += 1
                maximum_excess = max(maximum_excess, excess)

    return {
        "similarity_min_eigenvalue": float(eigenvalues[0]),
        "similarity_max_eigenvalue": float(eigenvalues[-1]),
        "similarity_negative_eigenvalues": int(np.count_nonzero(negative)),
        "similarity_is_psd_at_tolerance": bool(not np.any(negative)),
        "psd_tolerance": PSD_TOLERANCE,
        "triangle_inequalities_tested": tested,
        "triangle_violations": violations,
        "triangle_violation_fraction": float(violations / tested),
        "maximum_triangle_excess": maximum_excess,
        "dissimilarity_is_metric_at_tolerance": bool(violations == 0),
        "triangle_tolerance": TRIANGLE_TOLERANCE,
    }


def make_neighbour_table(
    pairs: pd.DataFrame,
    ids: list[str],
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required = (
        "consensus_similarity",
        "platonic_percentile",
        "aristotelian_k100_percentile",
        PRIMARY_PLATONIC,
        PRIMARY_ARISTOTELIAN,
    )
    require_columns(pairs, required, "consensus pair table")
    for embedding_id in sorted(ids):
        mask = (pairs["embedding_x"] == embedding_id) | (
            pairs["embedding_y"] == embedding_id
        )
        local = pairs.loc[mask].copy()
        local["neighbour_embedding"] = np.where(
            local["embedding_x"] == embedding_id,
            local["embedding_y"],
            local["embedding_x"],
        )
        local = local.sort_values(
            ["consensus_similarity", "neighbour_embedding"],
            ascending=[False, True],
        ).head(top_n)
        for neighbour_rank, (_, row) in enumerate(local.iterrows(), start=1):
            rows.append(
                {
                    "embedding_id": embedding_id,
                    "neighbour_rank": neighbour_rank,
                    "neighbour_embedding": row["neighbour_embedding"],
                    "consensus_similarity": float(row["consensus_similarity"]),
                    "platonic_percentile": float(row["platonic_percentile"]),
                    "aristotelian_k100_percentile": float(
                        row["aristotelian_k100_percentile"]
                    ),
                    "component_absolute_disagreement": float(
                        row["component_absolute_disagreement"]
                    ),
                    PRIMARY_PLATONIC: float(row[PRIMARY_PLATONIC]),
                    PRIMARY_ARISTOTELIAN: float(row[PRIMARY_ARISTOTELIAN]),
                }
            )
    return pd.DataFrame(rows)


def embedding_summary(
    pairs: pd.DataFrame,
    ids: list[str],
    neighbours: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for embedding_id in sorted(ids):
        mask = (pairs["embedding_x"] == embedding_id) | (
            pairs["embedding_y"] == embedding_id
        )
        values = pairs.loc[mask, "consensus_similarity"].to_numpy(dtype=float)
        disagreements = pairs.loc[
            mask, "component_absolute_disagreement"
        ].to_numpy(dtype=float)
        top = neighbours.loc[
            (neighbours["embedding_id"] == embedding_id)
            & (neighbours["neighbour_rank"] == 1)
        ].iloc[0]
        rows.append(
            {
                "embedding_id": embedding_id,
                "n_other_embeddings": len(values),
                "mean_consensus_similarity": float(np.mean(values)),
                "median_consensus_similarity": float(np.median(values)),
                "q90_consensus_similarity": float(np.quantile(values, 0.90)),
                "maximum_consensus_similarity": float(np.max(values)),
                "mean_component_absolute_disagreement": float(
                    np.mean(disagreements)
                ),
                "top_neighbour": str(top["neighbour_embedding"]),
                "top_neighbour_consensus_similarity": float(
                    top["consensus_similarity"]
                ),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_table(
    variants: dict[str, np.ndarray],
) -> pd.DataFrame:
    primary = variants["primary"]
    rows: list[dict[str, object]] = []
    for variant_id, values in variants.items():
        platonic_column, aristotelian_column, role = VARIANTS[variant_id]
        delta = values - primary
        rows.append(
            {
                "variant_id": variant_id,
                "analysis_role": role,
                "platonic_view": platonic_column,
                "aristotelian_view": aristotelian_column,
                "platonic_weight": PLATONIC_WEIGHT,
                "aristotelian_weight": ARISTOTELIAN_WEIGHT,
                "spearman_vs_primary": spearman_rho(primary, values),
                "median_absolute_delta_vs_primary": float(np.median(np.abs(delta))),
                "maximum_absolute_delta_vs_primary": float(np.max(np.abs(delta))),
            }
        )
    return pd.DataFrame(rows)


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
    values = np.array([4.0, 1.0, 1.0, 3.0, 2.0])
    observed = percentile_ranks(values)
    expected = np.array([1.0, 0.125, 0.125, 0.75, 0.5])
    if not np.allclose(observed, expected, rtol=0.0, atol=1e-15):
        raise AssertionError("Tie-aware percentile-rank self-test failed")

    ids = [f"e{i}" for i in range(5)]
    pair_rows = [
        {"embedding_x": x, "embedding_y": y}
        for x, y in combinations(ids, 2)
    ]
    pairs = pd.DataFrame(pair_rows)
    scores = np.linspace(0.0, 1.0, len(pairs))
    matrix = build_matrix(pairs, ids, scores, 1.0, "self-test")
    if matrix.shape != (5, 5) or not np.allclose(np.diag(matrix), 1.0):
        raise AssertionError("Matrix-construction self-test failed")
    diagnostics = geometry_diagnostics(matrix.to_numpy(dtype=float))
    if diagnostics["triangle_inequalities_tested"] != 30:
        raise AssertionError("Triangle-audit self-test failed")
    print("cs_06 self-test: PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 1 <= args.top_neighbours < EXPECTED_EMBEDDINGS:
        raise ValueError(
            f"--top-neighbours must be between 1 and {EXPECTED_EMBEDDINGS - 1}"
        )

    root = args.root.expanduser().resolve()
    workflow_root = root / "results/consensus_graph_v2"
    views_dir = (
        args.views_dir.expanduser().resolve()
        if args.views_dir is not None
        else workflow_root / "04_views"
    )
    concordance_dir = (
        args.concordance_dir.expanduser().resolve()
        if args.concordance_dir is not None
        else workflow_root / "05_view_concordance"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow_root / "06_consensus"
    )

    ids_path = require_file(workflow_root / "01_fixed_panel/embedding_metadata.tsv")
    pairs_path = require_file(views_dir / "primary_pair_views.tsv")
    concordance_path = require_file(concordance_dir / "concordance_summary.json")

    id_frame = pd.read_csv(ids_path, sep="\t")
    require_columns(
        id_frame, ("embedding_id", "display_name", "modality"), "primary panel"
    )
    if id_frame["embedding_id"].isna().any():
        raise ValueError("Primary panel contains missing embedding IDs")
    ids = id_frame["embedding_id"].astype(str).tolist()
    if len(ids) != EXPECTED_EMBEDDINGS or len(set(ids)) != EXPECTED_EMBEDDINGS:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDINGS} unique primary embedding IDs"
        )

    concordance = json.loads(concordance_path.read_text())
    if concordance.get("status") != "PASS":
        raise ValueError("Upstream concordance summary does not have status PASS")
    if int(concordance.get("n_embeddings", -1)) != EXPECTED_EMBEDDINGS:
        raise ValueError("Upstream concordance embedding count changed")
    if int(concordance.get("n_pairs", -1)) != EXPECTED_PAIRS:
        raise ValueError("Upstream concordance pair count changed")
    decision = concordance.get("decision", {})
    if decision.get("consensus_supported") is not True:
        raise ValueError("Upstream concordance did not support consensus construction")
    if decision.get("recommendation") != EXPECTED_RECOMMENDATION:
        raise ValueError("Upstream concordance recommendation changed")
    if decision.get("primary_comparison_id") != EXPECTED_PRIMARY_COMPARISON:
        raise ValueError(
            "Upstream primary concordance comparison is not calibrated local "
            "overlap k=100"
        )
    criteria = decision.get("criteria", {})
    if not criteria or not all(value is True for value in criteria.values()):
        raise ValueError("Not every upstream consensus criterion is True")

    upstream_sources = concordance.get("sources", {})
    expected_ids_hash = upstream_sources.get("primary_panel", {}).get("sha256")
    expected_pairs_hash = upstream_sources.get("pair_views", {}).get("sha256")
    if expected_ids_hash != sha256(ids_path):
        raise ValueError("Primary-panel file changed after concordance testing")
    if expected_pairs_hash != sha256(pairs_path):
        raise ValueError("Pair-view file changed after concordance testing")

    pairs = canonicalize_pairs(
        pd.read_csv(pairs_path, sep="\t"), "primary_pair_views"
    )
    validate_pair_coverage(pairs, ids)
    required_columns = {
        view
        for platonic, aristotelian, _ in VARIANTS.values()
        for view in (platonic, aristotelian)
    }
    require_columns(pairs, required_columns, "primary_pair_views")

    observed_primary_rho = spearman_rho(
        numeric(pairs, PRIMARY_PLATONIC, "primary_pair_views"),
        numeric(pairs, PRIMARY_ARISTOTELIAN, "primary_pair_views"),
    )
    recorded_primary_rho = float(decision["primary_observed_spearman_rho"])
    if not np.isclose(
        observed_primary_rho, recorded_primary_rho, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "Primary concordance coefficient does not reproduce from pair views"
        )

    variant_scores: dict[str, np.ndarray] = {}
    primary_platonic_percentile: np.ndarray | None = None
    primary_aristotelian_percentile: np.ndarray | None = None
    for variant_id, (platonic_column, aristotelian_column, _) in VARIANTS.items():
        platonic, aristotelian, consensus = equal_rank_variant(
            pairs, platonic_column, aristotelian_column
        )
        variant_scores[variant_id] = consensus
        pairs[f"consensus_{variant_id}"] = consensus
        if variant_id == "primary":
            primary_platonic_percentile = platonic
            primary_aristotelian_percentile = aristotelian

    if primary_platonic_percentile is None or primary_aristotelian_percentile is None:
        raise AssertionError("Primary component percentiles were not constructed")
    pairs["platonic_percentile"] = primary_platonic_percentile
    pairs["aristotelian_k100_percentile"] = primary_aristotelian_percentile
    pairs["consensus_similarity"] = variant_scores["primary"]
    pairs["consensus_dissimilarity"] = 1.0 - pairs["consensus_similarity"]
    pairs["component_signed_difference_platonic_minus_aristotelian"] = (
        pairs["platonic_percentile"] - pairs["aristotelian_k100_percentile"]
    )
    pairs["component_absolute_disagreement"] = np.abs(
        pairs["component_signed_difference_platonic_minus_aristotelian"]
    )

    consensus_similarity = build_matrix(
        pairs,
        ids,
        pairs["consensus_similarity"].to_numpy(dtype=float),
        1.0,
        "consensus similarity",
    )
    consensus_dissimilarity = build_matrix(
        pairs,
        ids,
        pairs["consensus_dissimilarity"].to_numpy(dtype=float),
        0.0,
        "consensus dissimilarity",
    )
    platonic_percentile_matrix = build_matrix(
        pairs,
        ids,
        pairs["platonic_percentile"].to_numpy(dtype=float),
        1.0,
        "Platonic percentile similarity",
    )
    aristotelian_percentile_matrix = build_matrix(
        pairs,
        ids,
        pairs["aristotelian_k100_percentile"].to_numpy(dtype=float),
        1.0,
        "Aristotelian percentile similarity",
    )

    diagnostics = geometry_diagnostics(
        consensus_similarity.to_numpy(dtype=float)
    )
    sensitivity = sensitivity_table(variant_scores)
    neighbours = make_neighbour_table(pairs, ids, args.top_neighbours)
    per_embedding = embedding_summary(pairs, ids, neighbours)
    display_map = dict(zip(id_frame["embedding_id"], id_frame["display_name"]))
    modality_map = dict(zip(id_frame["embedding_id"], id_frame["modality"]))
    neighbours["embedding_display"] = neighbours["embedding_id"].map(display_map)
    neighbours["neighbour_display"] = neighbours["neighbour_embedding"].map(display_map)
    per_embedding["display_name"] = per_embedding["embedding_id"].map(display_map)
    per_embedding["modality"] = per_embedding["embedding_id"].map(modality_map)
    per_embedding["top_neighbour_display"] = per_embedding["top_neighbour"].map(display_map)
    if neighbours[["embedding_display", "neighbour_display"]].isna().any().any():
        raise ValueError("Manifest display-name mapping is incomplete")

    primary = pairs["consensus_similarity"].to_numpy(dtype=float)
    component_rhos = {
        "platonic": spearman_rho(primary, primary_platonic_percentile),
        "aristotelian": spearman_rho(primary, primary_aristotelian_percentile),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_dataframe(pairs, output_dir / "consensus_pair_scores.tsv")
    atomic_dataframe(
        consensus_similarity,
        output_dir / "consensus_similarity_matrix.tsv",
        index=True,
    )
    atomic_dataframe(
        consensus_dissimilarity,
        output_dir / "consensus_dissimilarity_matrix.tsv",
        index=True,
    )
    atomic_dataframe(
        platonic_percentile_matrix,
        output_dir / "platonic_percentile_similarity_matrix.tsv",
        index=True,
    )
    atomic_dataframe(
        aristotelian_percentile_matrix,
        output_dir / "aristotelian_k100_percentile_similarity_matrix.tsv",
        index=True,
    )
    atomic_dataframe(sensitivity, output_dir / "consensus_sensitivity.tsv")
    atomic_dataframe(neighbours, output_dir / "consensus_top_neighbours.tsv")
    atomic_dataframe(per_embedding, output_dir / "embedding_consensus_summary.tsv")

    summary: dict[str, object] = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_pairs": EXPECTED_PAIRS,
        "construction": {
            "method": "equal_weight_average_of_within_view_percentile_ranks",
            "tie_method": "average",
            "percentile_formula": "(average_rank - 1) / (n_pairs - 1)",
            "platonic_view": PRIMARY_PLATONIC,
            "aristotelian_view": PRIMARY_ARISTOTELIAN,
            "platonic_weight": PLATONIC_WEIGHT,
            "aristotelian_weight": ARISTOTELIAN_WEIGHT,
            "diagonal_similarity": 1.0,
            "dissimilarity_definition": "1 - consensus_similarity",
        },
        "consensus_distribution": {
            "minimum": float(np.min(primary)),
            "median": float(np.median(primary)),
            "maximum": float(np.max(primary)),
            "unique_scores": int(len(np.unique(primary))),
        },
        "consensus_to_component_spearman_rho": component_rhos,
        "validated_upstream_concordance": {
            "comparison_id": EXPECTED_PRIMARY_COMPARISON,
            "spearman_rho": recorded_primary_rho,
            "node_permutation_p_two_sided": float(
                decision["primary_node_permutation_p_two_sided"]
            ),
            "leave_one_embedding_out_minimum_rho": float(
                decision["primary_loo_min_rho"]
            ),
        },
        "evaluated_consensus_configurations": ["primary"],
        "geometry_diagnostics": diagnostics,
        "sources": {
            "primary_panel": {"path": str(ids_path), "sha256": sha256(ids_path)},
            "pair_views": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "concordance_summary": {
                "path": str(concordance_path),
                "sha256": sha256(concordance_path),
            },
        },
        "interpretation_limits": {
            "rank_consensus": (
                "The fused value is a relative pair ranking, not an absolute "
                "biological similarity probability."
            ),
            "embedding_summary": (
                "Mean consensus similarity is network centrality, not embedding quality."
            ),
            "geometry": (
                "Downstream methods must respect the reported PSD and triangle-"
                "inequality diagnostics; Euclidean geometry is not assumed."
            ),
            "rbf": (
                "RBF CKA remains a nonlinear sensitivity view and is not a third "
                "consensus component."
            ),
        },
    }
    atomic_json(summary, output_dir / "consensus_summary.json")

    print("===== CONSENSUS GRAPH EQUAL-RANK CONSENSUS =====")
    print("status: PASS")
    print(f"embeddings: {EXPECTED_EMBEDDINGS}")
    print(f"pairs: {EXPECTED_PAIRS}")
    print("construction:")
    print(f"  Platonic weight: {PLATONIC_WEIGHT:.1f}")
    print(f"  Aristotelian k=100 weight: {ARISTOTELIAN_WEIGHT:.1f}")
    print("  within-view transform: tie-aware percentile rank")
    print(
        "consensus similarity range: "
        f"[{np.min(primary):.6f}, {np.max(primary):.6f}]"
    )
    print("consensus-to-component Spearman rho:")
    print(f"  Platonic: {component_rhos['platonic']:.6f}")
    print(f"  Aristotelian k=100: {component_rhos['aristotelian']:.6f}")
    print("validated primary-view concordance:")
    print(f"  Spearman rho: {recorded_primary_rho:.6f}")
    print(
        "  node-permutation p (two-sided): "
        f"{float(decision['primary_node_permutation_p_two_sided']):.6g}"
    )
    print("consensus configurations evaluated: primary only")
    print("geometry diagnostics:")
    print(
        "  consensus similarity PSD: "
        f"{diagnostics['similarity_is_psd_at_tolerance']} "
        f"(negative eigenvalues: {diagnostics['similarity_negative_eigenvalues']})"
    )
    print(
        "  1-similarity is metric: "
        f"{diagnostics['dissimilarity_is_metric_at_tolerance']} "
        f"(triangle violations: {diagnostics['triangle_violations']}/"
        f"{diagnostics['triangle_inequalities_tested']})"
    )
    print(f"output: {output_dir}")
    print(
        "NOTE: the consensus is a relative rank affinity; RBF CKA is retained "
        "as sensitivity only."
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
        print("===== CONSENSUS GRAPH EQUAL-RANK CONSENSUS =====", file=sys.stderr)
        print("status: FAIL", file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
