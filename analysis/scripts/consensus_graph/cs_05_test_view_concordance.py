#!/usr/bin/env python3
"""Test dependence-aware concordance among consensus-graph similarity views.

The 1,540 embedding pairs are not independent because every embedding occurs in
55 pairs. This script therefore compares complete 56 x 56 similarity matrices
with a Mantel-style node-label permutation: one matrix has its rows and columns
permuted together, and Spearman correlation is recomputed on the upper triangle.

This stage reads only consensus-graph view tables. It does not load embeddings,
PCA caches, or directional-predictability/MLP outputs, and it does not construct
the consensus itself.
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
DEFAULT_PERMUTATIONS = 9_999
DEFAULT_SEED = 33_105
ALPHA = 0.05

PRIMARY_LEFT = "platonic_linear_calibrated"
PRIMARY_RIGHT = "aristotelian_local_k100_calibrated"


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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: consensus_graph_v2/05_view_concordance).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help=f"Node-label permutations (default: {DEFAULT_PERMUTATIONS:,}).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a small deterministic in-memory test and exit.",
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


def expected_pair_set(ids: Iterable[str]) -> set[tuple[str, str]]:
    return set(combinations(sorted(ids), 2))


def validate_pair_coverage(frame: pd.DataFrame, ids: list[str]) -> None:
    observed = set(zip(frame["embedding_x"], frame["embedding_y"]))
    expected = expected_pair_set(ids)
    if len(frame) != EXPECTED_PAIRS or observed != expected:
        raise ValueError(
            "Primary pair coverage mismatch: "
            f"rows={len(frame)}, expected_rows={EXPECTED_PAIRS}, "
            f"missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}"
        )


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


def score_matrix(
    pairs: pd.DataFrame,
    ids: list[str],
    column: str,
) -> np.ndarray:
    ordered = sorted(ids)
    positions = {embedding_id: index for index, embedding_id in enumerate(ordered)}
    matrix = np.full((len(ordered), len(ordered)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    values = numeric(pairs, column, "primary_pair_views")
    for x, y, value in zip(
        pairs["embedding_x"].astype(str),
        pairs["embedding_y"].astype(str),
        values,
    ):
        if x not in positions or y not in positions:
            raise ValueError(f"{column} contains an ID outside the primary panel")
        i, j = positions[x], positions[y]
        matrix[i, j] = value
        matrix[j, i] = value
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{column} matrix is incomplete/non-finite")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0):
        raise ValueError(f"{column} matrix is not exactly symmetric")
    return matrix


def rank_matrix(matrix: np.ndarray, upper: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    ranks = rankdata(matrix[upper], method="average")
    result = np.zeros_like(matrix, dtype=float)
    result[upper] = ranks
    result[(upper[1], upper[0])] = ranks
    return result


def make_permutations(n: int, count: int, seed: int) -> np.ndarray:
    if count < 99:
        raise ValueError("At least 99 permutations are required")
    rng = np.random.default_rng(seed)
    return np.vstack([rng.permutation(n) for _ in range(count)])


def node_permutation_test(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    permutations: np.ndarray,
) -> dict[str, float | int]:
    if left_matrix.shape != right_matrix.shape or left_matrix.ndim != 2:
        raise ValueError("Permutation-test matrices must have the same square shape")
    n = left_matrix.shape[0]
    upper = np.triu_indices(n, k=1)
    left_rank = standardize(rankdata(left_matrix[upper], method="average"))
    right_rank_matrix = rank_matrix(right_matrix, upper)
    right_ranks = right_rank_matrix[upper]
    right_rank_mean = float(right_ranks.mean())
    right_rank_norm = float(np.linalg.norm(right_ranks - right_rank_mean))
    if not np.isfinite(right_rank_norm) or right_rank_norm == 0.0:
        raise ValueError("Right-hand rank matrix is constant/non-finite")
    right_rank_vector = (right_ranks - right_rank_mean) / right_rank_norm
    observed = float(np.dot(left_rank, right_rank_vector))

    null = np.empty(len(permutations), dtype=float)
    for index, permutation in enumerate(permutations):
        permuted = right_rank_matrix[np.ix_(permutation, permutation)][upper]
        # The permutation preserves the rank multiset, so the original mean and
        # norm apply exactly.
        null[index] = float(
            np.dot(left_rank, (permuted - right_rank_mean) / right_rank_norm)
        )

    greater_count = int(np.count_nonzero(null >= observed))
    two_sided_count = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return {
        "observed_rho": observed,
        "n_permutations": int(len(permutations)),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null, ddof=1)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "greater_exceedances": greater_count,
        "p_greater": float((greater_count + 1) / (len(null) + 1)),
        "two_sided_exceedances": two_sided_count,
        "p_two_sided": float((two_sided_count + 1) / (len(null) + 1)),
    }


def leave_one_out(
    pairs: pd.DataFrame,
    ids: list[str],
    left: str,
    right: str,
    comparison_id: str,
) -> pd.DataFrame:
    left_values = numeric(pairs, left, "primary_pair_views")
    right_values = numeric(pairs, right, "primary_pair_views")
    x = pairs["embedding_x"].astype(str).to_numpy()
    y = pairs["embedding_y"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for omitted in sorted(ids):
        keep = (x != omitted) & (y != omitted)
        rows.append(
            {
                "comparison_id": comparison_id,
                "omitted_embedding": omitted,
                "n_remaining_embeddings": len(ids) - 1,
                "n_remaining_pairs": int(np.count_nonzero(keep)),
                "spearman_rho": spearman_rho(left_values[keep], right_values[keep]),
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def comparison_definitions() -> list[dict[str, str]]:
    return [
        {
            "comparison_id": "primary_platonic_vs_aristotelian_k100_calibrated",
            "left_view": PRIMARY_LEFT,
            "right_view": PRIMARY_RIGHT,
            "analysis_role": "primary",
        },
        {
            "comparison_id": "raw_platonic_vs_aristotelian_k100",
            "left_view": "platonic_linear_raw",
            "right_view": "aristotelian_local_k100_raw",
            "analysis_role": "calibration_sensitivity",
        },
        {
            "comparison_id": "platonic_vs_aristotelian_k10_calibrated",
            "left_view": PRIMARY_LEFT,
            "right_view": "aristotelian_local_k10_calibrated",
            "analysis_role": "k_sensitivity",
        },
        {
            "comparison_id": "platonic_vs_aristotelian_k25_calibrated",
            "left_view": PRIMARY_LEFT,
            "right_view": "aristotelian_local_k25_calibrated",
            "analysis_role": "k_sensitivity",
        },
        {
            "comparison_id": "platonic_vs_aristotelian_k50_calibrated",
            "left_view": PRIMARY_LEFT,
            "right_view": "aristotelian_local_k50_calibrated",
            "analysis_role": "k_sensitivity",
        },
        {
            "comparison_id": "platonic_vs_aristotelian_multi_k_calibrated",
            "left_view": PRIMARY_LEFT,
            "right_view": "aristotelian_local_mean_calibrated",
            "analysis_role": "multi_k_sensitivity",
        },
        {
            "comparison_id": "linear_vs_rbf_pairwise_intersection",
            "left_view": "platonic_linear_raw",
            "right_view": "pairwise_primary_rbf_raw",
            "analysis_role": "nonlinear_sensitivity",
        },
        {
            "comparison_id": "aristotelian_k100_vs_rbf_pairwise_intersection",
            "left_view": PRIMARY_RIGHT,
            "right_view": "pairwise_primary_rbf_raw",
            "analysis_role": "nonlinear_sensitivity",
        },
    ]


def analyse(
    pairs: pd.DataFrame,
    ids: list[str],
    permutations: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    definitions = comparison_definitions()
    required = {
        view
        for definition in definitions
        for view in (definition["left_view"], definition["right_view"])
    }
    require_columns(pairs, required, "primary_pair_views")

    matrices = {
        view: score_matrix(pairs, ids, view)
        for view in sorted(required)
    }
    test_rows: list[dict[str, object]] = []
    loo_frames: list[pd.DataFrame] = []

    for definition in definitions:
        comparison_id = definition["comparison_id"]
        left = definition["left_view"]
        right = definition["right_view"]
        permutation = node_permutation_test(
            matrices[left], matrices[right], permutations
        )
        loo = leave_one_out(pairs, ids, left, right, comparison_id)
        loo_frames.append(loo)
        minimum = loo.loc[loo["spearman_rho"].idxmin()]
        maximum = loo.loc[loo["spearman_rho"].idxmax()]
        test_rows.append(
            {
                **definition,
                "n_embeddings": len(ids),
                "n_pairs": len(pairs),
                **permutation,
                "loo_min_rho": float(minimum["spearman_rho"]),
                "loo_min_embedding": str(minimum["omitted_embedding"]),
                "loo_median_rho": float(loo["spearman_rho"].median()),
                "loo_max_rho": float(maximum["spearman_rho"]),
                "loo_max_embedding": str(maximum["omitted_embedding"]),
                "loo_all_positive": bool((loo["spearman_rho"] > 0.0).all()),
            }
        )

    tests = pd.DataFrame(test_rows)
    tests["p_two_sided_holm_all_comparisons"] = holm_adjust(
        tests["p_two_sided"].to_numpy(dtype=float)
    )
    k_comparison_ids = [
        "platonic_vs_aristotelian_k10_calibrated",
        "platonic_vs_aristotelian_k25_calibrated",
        "platonic_vs_aristotelian_k50_calibrated",
        "primary_platonic_vs_aristotelian_k100_calibrated",
    ]
    k_mask = tests["comparison_id"].isin(k_comparison_ids)
    tests["p_two_sided_holm_k_family"] = np.nan
    tests.loc[k_mask, "p_two_sided_holm_k_family"] = holm_adjust(
        tests.loc[k_mask, "p_two_sided"].to_numpy(dtype=float)
    )
    leave_one_out_table = pd.concat(loo_frames, ignore_index=True)

    primary = tests.loc[tests["analysis_role"] == "primary"]
    if len(primary) != 1:
        raise ValueError("Expected exactly one primary comparison")
    primary_row = primary.iloc[0]
    k_rows = tests.loc[tests["comparison_id"].isin(k_comparison_ids)]
    if len(k_rows) != 4:
        raise ValueError("Expected exactly four k-specific comparisons")

    criteria = {
        "primary_positive": bool(primary_row["observed_rho"] > 0.0),
        "primary_permutation_significant": bool(
            primary_row["p_two_sided"] <= ALPHA
        ),
        "primary_leave_one_out_all_positive": bool(
            primary_row["loo_all_positive"]
        ),
        "all_k_positive": bool((k_rows["observed_rho"] > 0.0).all()),
        "all_k_permutation_significant_after_holm": bool(
            (k_rows["p_two_sided_holm_k_family"] <= ALPHA).all()
        ),
        "all_k_leave_one_out_all_positive": bool(
            k_rows["loo_all_positive"].all()
        ),
        # Below 0.95 means the two primary rank geometries are not effectively
        # duplicates under the workflow's explicit redundancy threshold.
        "primary_views_not_rank_redundant": bool(
            primary_row["observed_rho"] < 0.95
        ),
    }
    supported = all(criteria.values())
    decision: dict[str, object] = {
        "consensus_supported": supported,
        "recommendation": (
            "BUILD_TWO_VIEW_EQUAL_RANK_CONSENSUS"
            if supported
            else "KEEP_PRIMARY_VIEWS_SEPARATE_PENDING_REVIEW"
        ),
        "criteria_alpha": ALPHA,
        "rank_redundancy_threshold": 0.95,
        "criteria": criteria,
        "primary_comparison_id": str(primary_row["comparison_id"]),
        "primary_observed_spearman_rho": float(primary_row["observed_rho"]),
        "primary_node_permutation_p_two_sided": float(
            primary_row["p_two_sided"]
        ),
        "primary_loo_min_rho": float(primary_row["loo_min_rho"]),
        "k_specific_rho_range": [
            float(k_rows["observed_rho"].min()),
            float(k_rows["observed_rho"].max()),
        ],
        "methodological_note": (
            "The decision uses node-label permutation and leave-one-node-out "
            "stability. Ordinary pair-level correlation p-values are not used."
        ),
        "next_stage_if_supported": (
            "Construct an equal-weight average of within-view percentile ranks "
            "from calibrated Platonic linear CKA and calibrated Aristotelian "
            "local overlap at k=100; retain each component view separately."
        ),
    }
    return tests, leave_one_out_table, decision


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
    ids = [f"e{index}" for index in range(8)]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    latent = rng.normal(size=len(ids))
    for i, j in combinations(range(len(ids)), 2):
        base = -abs(latent[i] - latent[j])
        rows.append(
            {
                "embedding_x": ids[i],
                "embedding_y": ids[j],
                "left": base + rng.normal(scale=0.02),
                "right": base + rng.normal(scale=0.05),
            }
        )
    frame = pd.DataFrame(rows)
    left = score_matrix(frame, ids, "left")
    right = score_matrix(frame, ids, "right")
    permutations = make_permutations(len(ids), 199, 17)
    result = node_permutation_test(left, right, permutations)
    if result["observed_rho"] < 0.9:
        raise AssertionError("Self-test failed to recover strong matrix alignment")
    loo = leave_one_out(frame, ids, "left", "right", "self_test")
    if len(loo) != len(ids) or not (loo["spearman_rho"] > 0.0).all():
        raise AssertionError("Leave-one-out self-test failed")
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    if np.any(adjusted < np.array([0.01, 0.04, 0.03])):
        raise AssertionError("Holm self-test failed")
    print("cs_05 self-test: PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.n_permutations < 99:
        raise ValueError("--n-permutations must be at least 99")

    root = args.root.expanduser().resolve()
    workflow_root = root / "results/consensus_graph_v2"
    views_dir = (
        args.views_dir.expanduser().resolve()
        if args.views_dir is not None
        else workflow_root / "04_views"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow_root / "05_view_concordance"
    )

    ids_path = require_file(workflow_root / "01_fixed_panel/embedding_metadata.tsv")
    pairs_path = require_file(views_dir / "primary_pair_views.tsv")
    view_summary_path = require_file(views_dir / "view_summary.json")

    id_frame = pd.read_csv(ids_path, sep="\t")
    require_columns(id_frame, ("embedding_id",), "primary panel")
    if id_frame["embedding_id"].isna().any():
        raise ValueError("Primary panel contains missing embedding IDs")
    ids = id_frame["embedding_id"].astype(str).tolist()
    if len(ids) != EXPECTED_EMBEDDINGS or len(ids) != len(set(ids)):
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDINGS} unique primary IDs; found "
            f"{len(ids)} rows and {len(set(ids))} unique IDs"
        )

    summary = json.loads(view_summary_path.read_text())
    if summary.get("status") != "PASS":
        raise ValueError("Upstream view_summary.json does not have status PASS")
    primary_summary = summary.get("primary_panel", {})
    if int(primary_summary.get("embeddings", -1)) != EXPECTED_EMBEDDINGS:
        raise ValueError("Upstream summary primary embedding count changed")
    if int(primary_summary.get("pairs", -1)) != EXPECTED_PAIRS:
        raise ValueError("Upstream summary primary pair count changed")

    pairs = canonicalize_pairs(
        pd.read_csv(pairs_path, sep="\t"), "primary_pair_views"
    )
    validate_pair_coverage(pairs, ids)
    permutations = make_permutations(
        EXPECTED_EMBEDDINGS, args.n_permutations, args.seed
    )
    tests, loo, decision = analyse(pairs, ids, permutations)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_dataframe(tests, output_dir / "view_concordance_tests.tsv")
    atomic_dataframe(loo, output_dir / "leave_one_embedding_out.tsv")
    summary_payload: dict[str, object] = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_pairs": EXPECTED_PAIRS,
        "n_node_label_permutations": args.n_permutations,
        "random_seed": args.seed,
        "decision": decision,
        "sources": {
            "primary_panel": {"path": str(ids_path), "sha256": sha256(ids_path)},
            "pair_views": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "view_summary": {
                "path": str(view_summary_path),
                "sha256": sha256(view_summary_path),
            },
        },
        "interpretation_limits": {
            "permutation_resolution": 1.0 / (args.n_permutations + 1),
            "ordinary_pair_p_values": (
                "Not used because the 1,540 embedding pairs are dependent."
            ),
            "rbf": (
                "Sensitivity only: RBF CKA used 6,000 deterministic "
                "pair-specific genes rather than the fixed 14,039-gene panel."
            ),
            "consensus": (
                "This stage recommends or rejects construction; it does not "
                "write a fused similarity matrix."
            ),
        },
    }
    atomic_json(summary_payload, output_dir / "concordance_summary.json")

    primary = tests.loc[tests["analysis_role"] == "primary"].iloc[0]
    k_rows = tests.loc[
        tests["comparison_id"].isin(
            [
                "platonic_vs_aristotelian_k10_calibrated",
                "platonic_vs_aristotelian_k25_calibrated",
                "platonic_vs_aristotelian_k50_calibrated",
                "primary_platonic_vs_aristotelian_k100_calibrated",
            ]
        )
    ]
    nonlinear = tests.loc[
        tests["comparison_id"] == "linear_vs_rbf_pairwise_intersection"
    ].iloc[0]

    print("===== CONSENSUS GRAPH VIEW CONCORDANCE =====")
    print("status: PASS")
    print(f"embeddings: {EXPECTED_EMBEDDINGS}")
    print(f"pairs: {EXPECTED_PAIRS}")
    print(f"node-label permutations: {args.n_permutations}")
    print(f"seed: {args.seed}")
    print("\nprimary Platonic vs Aristotelian k=100:")
    print(f"  Spearman rho: {primary['observed_rho']:.6f}")
    print(f"  node-permutation p (two-sided): {primary['p_two_sided']:.6g}")
    print(
        "  permutation null 95% interval: "
        f"[{primary['null_q025']:.6f}, {primary['null_q975']:.6f}]"
    )
    print(
        "  leave-one-embedding-out rho range: "
        f"[{primary['loo_min_rho']:.6f}, {primary['loo_max_rho']:.6f}]"
    )
    print(
        "k=10/25/50/100 Platonic-Aristotelian rho range: "
        f"[{k_rows['observed_rho'].min():.6f}, "
        f"{k_rows['observed_rho'].max():.6f}]"
    )
    print("\nnonlinear sensitivity (linear vs RBF):")
    print(f"  Spearman rho: {nonlinear['observed_rho']:.6f}")
    print(f"  node-permutation p (two-sided): {nonlinear['p_two_sided']:.6g}")
    print("\nconsensus decision:")
    print(f"  supported: {decision['consensus_supported']}")
    print(f"  recommendation: {decision['recommendation']}")
    for criterion, passed in decision["criteria"].items():
        print(f"  {criterion}: {passed}")
    print(f"\noutput: {output_dir}")
    print(
        "NOTE: ordinary p-values over the 1,540 dependent pair rows are not used."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        print("===== CONSENSUS GRAPH VIEW CONCORDANCE =====", file=sys.stderr)
        print("status: FAIL", file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
