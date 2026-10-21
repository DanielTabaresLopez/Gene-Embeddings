#!/usr/bin/env python3
"""Build validated Platonic, Aristotelian, and sensitivity similarity views.

Workflow 33 has two evidence sources:

* a primary fixed-common atlas (56 embeddings, the same 14,039 genes); and
* PFI pair-specific-intersection evidence (56 aligned embeddings and all 67
  embeddings), including the only available RBF CKA results.

This script converts those long pair tables into explicit symmetric matrices,
computes normalized permutation-calibrated fixed-panel scores

    (observed - null_mean) / (1 - null_mean),

and quantifies agreement between fixed-common and pair-specific estimates.  It
does not load embedding arrays, fit models, or recompute any pairwise metric.
The five-shuffle fixed-panel null remains descriptive rather than a formal
pairwise significance test.
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
from scipy.stats import pearsonr, spearmanr


PRIMARY_PANEL = "primary_fixed_56"
SENSITIVITY_PANEL = "sensitivity_manifest_67"
EXPECTED_PRIMARY_EMBEDDINGS = 56
EXPECTED_SENSITIVITY_EMBEDDINGS = 67
EXPECTED_PRIMARY_PAIRS = 1_540
EXPECTED_SENSITIVITY_PAIRS = 2_211
EXPECTED_FIXED_GENES = 14_039
EXPECTED_NULL_SHUFFLES = 5
EXPECTED_RBF_GENES = 6_000
EXPECTED_LOCAL_GENES = 8_000
K_VALUES = (10, 25, 50, 100)


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=None,
        help="Frozen panel directory (default: consensus_graph_v2/01_fixed_panel).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Imported evidence directory (default: consensus_graph_v2/03_pairwise_evidence).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: consensus_graph_v2/04_views).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a small in-memory test and exit.",
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


def read_ids(path: Path, column: str = "embedding_id") -> list[str]:
    frame = pd.read_csv(require_file(path), sep="\t")
    if column not in frame.columns:
        raise ValueError(f"{path} has no {column!r} column")
    if frame[column].isna().any():
        raise ValueError(f"{path} has missing {column} values")
    values = frame[column].astype(str).tolist()
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate embedding IDs")
    return values


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


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


def observed_pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["embedding_x"], frame["embedding_y"]))


def require_pair_coverage(
    frame: pd.DataFrame,
    ids: Iterable[str],
    expected_count: int,
    label: str,
) -> None:
    expected = expected_pair_set(ids)
    observed = observed_pair_set(frame)
    if len(frame) != expected_count or observed != expected:
        raise ValueError(
            f"{label} pair coverage mismatch: rows={len(frame)}, "
            f"expected_rows={expected_count}, missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}"
        )


def numeric(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"{label} is missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label}/{column} contains non-finite values")
    return values


def require_constant(
    frame: pd.DataFrame,
    column: str,
    expected: int,
    label: str,
) -> None:
    values = numeric(frame, column, label)
    if not np.all(values == expected):
        unique = sorted(np.unique(values).tolist())
        raise ValueError(f"{label}/{column}: expected only {expected}, found {unique}")


def calibrated_score(
    observed: np.ndarray,
    null_mean: np.ndarray,
    label: str,
) -> np.ndarray:
    denominator = 1.0 - null_mean
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(null_mean)):
        raise ValueError(f"{label}: observed/null values must be finite")
    if np.any(denominator <= 0.0):
        raise ValueError(f"{label}: null mean must be below the metric maximum of 1")
    result = (observed - null_mean) / denominator
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: calibrated values are non-finite")
    return result


def build_matrix(
    pairs: pd.DataFrame,
    ids: list[str],
    value_column: str,
    label: str,
) -> pd.DataFrame:
    values = numeric(pairs, value_column, label)
    ordered_ids = sorted(ids)
    positions = {embedding_id: index for index, embedding_id in enumerate(ordered_ids)}
    matrix = np.full((len(ordered_ids), len(ordered_ids)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)

    for x, y, value in zip(
        pairs["embedding_x"].astype(str),
        pairs["embedding_y"].astype(str),
        values,
    ):
        if x not in positions or y not in positions:
            raise ValueError(f"{label}: pair contains an ID outside its panel")
        i, j = positions[x], positions[y]
        matrix[i, j] = value
        matrix[j, i] = value

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label}: incomplete/non-finite similarity matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0):
        raise ValueError(f"{label}: matrix is not exactly symmetric")
    if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=0.0):
        raise ValueError(f"{label}: matrix diagonal is not one")

    result = pd.DataFrame(matrix, index=ordered_ids, columns=ordered_ids)
    result.index.name = "embedding_id"
    return result


def atomic_dataframe(frame: pd.DataFrame, destination: Path, index: bool = False) -> None:
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


def correlation_row(
    metric: str,
    fixed: np.ndarray,
    pairwise: np.ndarray,
) -> dict[str, object]:
    spearman = spearmanr(fixed, pairwise)
    pearson = pearsonr(fixed, pairwise)
    delta = pairwise - fixed
    return {
        "metric": metric,
        "n_pairs": int(len(fixed)),
        "spearman_rho": float(spearman[0]),
        "spearman_p_value": float(spearman[1]),
        "pearson_r": float(pearson[0]),
        "pearson_p_value": float(pearson[1]),
        "median_fixed_common": float(np.median(fixed)),
        "median_pairwise_intersection": float(np.median(pairwise)),
        "median_pairwise_minus_fixed": float(np.median(delta)),
        "median_absolute_delta": float(np.median(np.abs(delta))),
    }


def pairwise_view_correlations(
    frame: pd.DataFrame,
    view_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in combinations(view_columns, 2):
        x = numeric(frame, left, "primary pair views")
        y = numeric(frame, right, "primary pair views")
        result = spearmanr(x, y)
        rows.append(
            {
                "view_x": left,
                "view_y": right,
                "n_pairs": len(frame),
                "spearman_rho": float(result[0]),
                "spearman_p_value": float(result[1]),
            }
        )
    return pd.DataFrame(rows)


def add_fixed_views(atlas: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "embedding_x",
        "embedding_y",
        "gene_count",
        "null_shuffles",
        "unbiased_linear_cka_row_normalized",
        "linear_observed",
        "linear_null_mean",
    ]
    for k in K_VALUES:
        required.extend((f"local_k{k}_observed", f"local_k{k}_null_mean"))
    require_columns(atlas, required, "fixed-common atlas")

    linear_raw = numeric(
        atlas, "unbiased_linear_cka_row_normalized", "fixed-common atlas"
    )
    linear_observed = numeric(atlas, "linear_observed", "fixed-common atlas")
    if not np.allclose(linear_raw, linear_observed, rtol=1e-12, atol=1e-12):
        raise ValueError(
            "fixed-common atlas linear_observed differs from row-normalized linear CKA"
        )

    result = atlas[["embedding_x", "embedding_y", "gene_count", "null_shuffles"]].copy()
    result = result.rename(columns={"gene_count": "fixed_common_n_genes"})
    result["platonic_linear_raw"] = linear_raw
    result["platonic_linear_calibrated"] = calibrated_score(
        linear_observed,
        numeric(atlas, "linear_null_mean", "fixed-common atlas"),
        "fixed-common linear CKA",
    )

    raw_columns: list[str] = []
    calibrated_columns: list[str] = []
    for k in K_VALUES:
        raw_name = f"aristotelian_local_k{k}_raw"
        calibrated_name = f"aristotelian_local_k{k}_calibrated"
        observed = numeric(atlas, f"local_k{k}_observed", "fixed-common atlas")
        null_mean = numeric(atlas, f"local_k{k}_null_mean", "fixed-common atlas")
        result[raw_name] = observed
        result[calibrated_name] = calibrated_score(
            observed, null_mean, f"fixed-common local k={k}"
        )
        raw_columns.append(raw_name)
        calibrated_columns.append(calibrated_name)

    result["aristotelian_local_mean_raw"] = result[raw_columns].mean(axis=1)
    result["aristotelian_local_mean_calibrated"] = result[
        calibrated_columns
    ].mean(axis=1)

    view_columns = [
        "platonic_linear_raw",
        "platonic_linear_calibrated",
        *raw_columns,
        *calibrated_columns,
        "aristotelian_local_mean_raw",
        "aristotelian_local_mean_calibrated",
    ]
    return result, view_columns


def add_pairwise_views(
    pfi: pd.DataFrame,
    prefix: str,
) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "embedding_x",
        "embedding_y",
        "n_genes_full",
        "rbf_n_genes",
        "local_n_genes",
        "unbiased_linear_cka_row_normalized",
        "unbiased_rbf_cka_row_normalized_bw_1p0",
    ]
    for k in K_VALUES:
        required.extend(
            (f"local_overlap_raw_k{k}", f"local_normalized_effect_k{k}")
        )
    require_columns(pfi, required, f"{prefix} PFI table")

    result = pfi[["embedding_x", "embedding_y", "n_genes_full"]].copy()
    result[f"{prefix}_linear_raw"] = numeric(
        pfi, "unbiased_linear_cka_row_normalized", f"{prefix} PFI table"
    )
    result[f"{prefix}_rbf_raw"] = numeric(
        pfi,
        "unbiased_rbf_cka_row_normalized_bw_1p0",
        f"{prefix} PFI table",
    )

    calibrated_columns: list[str] = []
    view_columns = [f"{prefix}_linear_raw", f"{prefix}_rbf_raw"]
    for k in K_VALUES:
        raw_name = f"{prefix}_local_k{k}_raw"
        calibrated_name = f"{prefix}_local_k{k}_calibrated"
        result[raw_name] = numeric(
            pfi, f"local_overlap_raw_k{k}", f"{prefix} PFI table"
        )
        result[calibrated_name] = numeric(
            pfi,
            f"local_normalized_effect_k{k}",
            f"{prefix} PFI table",
        )
        calibrated_columns.append(calibrated_name)
        view_columns.extend((raw_name, calibrated_name))

    result[f"{prefix}_local_mean_calibrated"] = result[calibrated_columns].mean(axis=1)
    view_columns.append(f"{prefix}_local_mean_calibrated")
    return result, view_columns


def registry_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "view_id": "platonic_linear_raw",
            "panel": PRIMARY_PANEL,
            "source": "fixed_common_atlas",
            "concept": "Platonic_global",
            "metric": "row-normalized unbiased linear CKA",
            "calibration": "raw",
            "gene_basis": "same 14,039 genes for every pair",
            "analysis_role": "primary_descriptive",
        },
        {
            "view_id": "platonic_linear_calibrated",
            "panel": PRIMARY_PANEL,
            "source": "fixed_common_atlas",
            "concept": "Platonic_global",
            "metric": "row-normalized unbiased linear CKA",
            "calibration": "(observed-null_mean)/(1-null_mean); five shuffles",
            "gene_basis": "same 14,039 genes for every pair",
            "analysis_role": "primary_null_corrected",
        },
    ]
    for k in K_VALUES:
        for suffix, calibration, role in (
            ("raw", "raw", "primary_descriptive" if k == 100 else "k_sensitivity"),
            (
                "calibrated",
                "(observed-null_mean)/(1-null_mean); five shuffles",
                "primary_null_corrected" if k == 100 else "k_sensitivity",
            ),
        ):
            rows.append(
                {
                    "view_id": f"aristotelian_local_k{k}_{suffix}",
                    "panel": PRIMARY_PANEL,
                    "source": "fixed_common_atlas",
                    "concept": "Aristotelian_local",
                    "metric": f"tie-aware mutual-neighbour overlap, k={k}",
                    "calibration": calibration,
                    "gene_basis": "same 14,039 genes for every pair",
                    "analysis_role": role,
                }
            )
    rows.extend(
        [
            {
                "view_id": "aristotelian_local_mean_raw",
                "panel": PRIMARY_PANEL,
                "source": "fixed_common_atlas",
                "concept": "Aristotelian_local",
                "metric": "mean local overlap across k=10,25,50,100",
                "calibration": "raw",
                "gene_basis": "same 14,039 genes for every pair",
                "analysis_role": "multi_k_summary",
            },
            {
                "view_id": "aristotelian_local_mean_calibrated",
                "panel": PRIMARY_PANEL,
                "source": "fixed_common_atlas",
                "concept": "Aristotelian_local",
                "metric": "mean calibrated local overlap across k=10,25,50,100",
                "calibration": "per-k normalized calibration; five shuffles",
                "gene_basis": "same 14,039 genes for every pair",
                "analysis_role": "multi_k_summary",
            },
        ]
    )

    for prefix, panel, role_prefix in (
        ("pairwise_primary", PRIMARY_PANEL, "direct_comparison"),
        ("sensitivity_manifest", SENSITIVITY_PANEL, "manifest_sensitivity"),
    ):
        rows.extend(
            [
                {
                    "view_id": f"{prefix}_linear_raw",
                    "panel": panel,
                    "source": "workflow20_pairwise_intersection",
                    "concept": "Platonic_global_sensitivity",
                    "metric": "row-normalized unbiased linear CKA",
                    "calibration": "raw",
                    "gene_basis": "full pair-specific intersection",
                    "analysis_role": role_prefix,
                },
                {
                    "view_id": f"{prefix}_rbf_raw",
                    "panel": panel,
                    "source": "workflow20_pairwise_intersection",
                    "concept": "Platonic_nonlinear_sensitivity",
                    "metric": "row-normalized unbiased RBF CKA, bandwidth factor 1.0",
                    "calibration": "raw",
                    "gene_basis": "6,000 deterministic pair-specific genes",
                    "analysis_role": f"{role_prefix}_nonlinear",
                },
            ]
        )
        for k in K_VALUES:
            rows.extend(
                [
                    {
                        "view_id": f"{prefix}_local_k{k}_raw",
                        "panel": panel,
                        "source": "workflow20_pairwise_intersection",
                        "concept": "Aristotelian_local_sensitivity",
                        "metric": f"tie-aware mutual-neighbour overlap, k={k}",
                        "calibration": "raw",
                        "gene_basis": "8,000 deterministic pair-specific genes",
                        "analysis_role": role_prefix,
                    },
                    {
                        "view_id": f"{prefix}_local_k{k}_calibrated",
                        "panel": panel,
                        "source": "workflow20_pairwise_intersection",
                        "concept": "Aristotelian_local_sensitivity",
                        "metric": f"tie-aware mutual-neighbour overlap, k={k}",
                        "calibration": "(observed-null_mean)/(1-null_mean); 199 permutations",
                        "gene_basis": "8,000 deterministic pair-specific genes",
                        "analysis_role": role_prefix,
                    },
                ]
            )
        rows.append(
            {
                "view_id": f"{prefix}_local_mean_calibrated",
                "panel": panel,
                "source": "workflow20_pairwise_intersection",
                "concept": "Aristotelian_local_sensitivity",
                "metric": "mean calibrated local overlap across k=10,25,50,100",
                "calibration": "mean of per-k normalized calibration; 199 permutations",
                "gene_basis": "8,000 deterministic pair-specific genes",
                "analysis_role": role_prefix,
            }
        )
    return rows


def self_test() -> None:
    ids = ["a", "b", "c", "d"]
    rows = []
    value = 0.1
    for x, y in combinations(ids, 2):
        rows.append({"embedding_x": y, "embedding_y": x, "score": value})
        value += 0.1
    frame = canonicalize_pairs(pd.DataFrame(rows), "self-test")
    require_pair_coverage(frame, ids, 6, "self-test")
    matrix = build_matrix(frame, ids, "score", "self-test")
    if matrix.shape != (4, 4) or not np.allclose(np.diag(matrix), 1.0):
        raise AssertionError("matrix self-test failed")
    calibrated = calibrated_score(
        np.array([0.4, 0.7]), np.array([0.1, 0.2]), "self-test"
    )
    expected = np.array([1.0 / 3.0, 0.625])
    if not np.allclose(calibrated, expected):
        raise AssertionError("calibration self-test failed")
    correlations = correlation_row(
        "self-test",
        np.array([0.1, 0.2, 0.4, 0.8]),
        np.array([0.11, 0.19, 0.39, 0.81]),
    )
    if not np.isfinite(correlations["spearman_rho"]):
        raise AssertionError("correlation self-test failed")
    print("cs_04 self-test: PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    root = args.root.expanduser().resolve()
    workflow_root = root / "results/consensus_graph_v2"
    panel_dir = (
        args.panel_dir.expanduser().resolve()
        if args.panel_dir is not None
        else workflow_root / "01_fixed_panel"
    )
    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else workflow_root / "03_pairwise_evidence"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow_root / "04_views"
    )

    atlas_path = require_file(input_dir / "primary_fixed_common_atlas.tsv")
    primary_pfi_path = require_file(input_dir / "primary_pairwise_intersection.tsv")
    sensitivity_pfi_path = require_file(
        input_dir / "sensitivity_manifest_pairwise_intersection.tsv"
    )
    import_summary_path = require_file(input_dir / "import_summary.json")

    primary_ids = read_ids(panel_dir / "embedding_metadata.tsv")
    if len(primary_ids) != EXPECTED_PRIMARY_EMBEDDINGS:
        raise ValueError(
            f"Expected {EXPECTED_PRIMARY_EMBEDDINGS} primary IDs, found {len(primary_ids)}"
        )
    atlas = canonicalize_pairs(pd.read_csv(atlas_path, sep="\t"), "fixed-common atlas")
    primary_pfi = canonicalize_pairs(
        pd.read_csv(primary_pfi_path, sep="\t"), "PFI primary panel"
    )
    sensitivity_pfi = canonicalize_pairs(
        pd.read_csv(sensitivity_pfi_path, sep="\t"), "PFI full manifest"
    )
    sensitivity_ids = sorted(
        set(sensitivity_pfi["embedding_x"].astype(str))
        | set(sensitivity_pfi["embedding_y"].astype(str))
    )
    if len(sensitivity_ids) != EXPECTED_SENSITIVITY_EMBEDDINGS:
        raise ValueError(
            f"Expected {EXPECTED_SENSITIVITY_EMBEDDINGS} sensitivity IDs, "
            f"found {len(sensitivity_ids)}"
        )
    if not set(primary_ids).issubset(sensitivity_ids):
        raise ValueError("Primary panel is not a subset of the sensitivity panel")
    require_pair_coverage(atlas, primary_ids, EXPECTED_PRIMARY_PAIRS, "fixed-common atlas")
    require_pair_coverage(
        primary_pfi, primary_ids, EXPECTED_PRIMARY_PAIRS, "PFI primary panel"
    )
    require_pair_coverage(
        sensitivity_pfi,
        sensitivity_ids,
        EXPECTED_SENSITIVITY_PAIRS,
        "PFI full manifest",
    )
    require_constant(atlas, "gene_count", EXPECTED_FIXED_GENES, "fixed-common atlas")
    require_constant(atlas, "null_shuffles", EXPECTED_NULL_SHUFFLES, "fixed-common atlas")
    require_constant(primary_pfi, "rbf_n_genes", EXPECTED_RBF_GENES, "PFI primary panel")
    require_constant(primary_pfi, "local_n_genes", EXPECTED_LOCAL_GENES, "PFI primary panel")
    require_constant(
        sensitivity_pfi, "rbf_n_genes", EXPECTED_RBF_GENES, "PFI full manifest"
    )
    require_constant(
        sensitivity_pfi, "local_n_genes", EXPECTED_LOCAL_GENES, "PFI full manifest"
    )

    fixed_views, fixed_columns = add_fixed_views(atlas)
    primary_pairwise_views, primary_pairwise_columns = add_pairwise_views(
        primary_pfi, "pairwise_primary"
    )
    sensitivity_views, sensitivity_columns = add_pairwise_views(
        sensitivity_pfi, "sensitivity_manifest"
    )

    primary_views = fixed_views.merge(
        primary_pairwise_views,
        on=["embedding_x", "embedding_y"],
        how="inner",
        validate="one_to_one",
    )
    if len(primary_views) != EXPECTED_PRIMARY_PAIRS:
        raise ValueError("Primary fixed/pairwise merge is incomplete")
    panel_metadata = pd.read_csv(panel_dir / "embedding_metadata.tsv", sep="\t")
    require_columns(
        panel_metadata,
        ("embedding_id", "display_name", "modality"),
        "frozen panel metadata",
    )
    display_map = dict(zip(panel_metadata.embedding_id, panel_metadata.display_name))
    modality_map = dict(zip(panel_metadata.embedding_id, panel_metadata.modality))
    primary_views["embedding_x_display"] = primary_views["embedding_x"].map(display_map)
    primary_views["embedding_y_display"] = primary_views["embedding_y"].map(display_map)
    primary_views["modality_x"] = primary_views["embedding_x"].map(modality_map)
    primary_views["modality_y"] = primary_views["embedding_y"].map(modality_map)
    if primary_views[
        ["embedding_x_display", "embedding_y_display", "modality_x", "modality_y"]
    ].isna().any().any():
        raise ValueError("Some primary pair rows could not be labeled from the manifest")

    robustness_rows = [
        correlation_row(
            "linear_raw",
            numeric(primary_views, "platonic_linear_raw", "primary views"),
            numeric(primary_views, "pairwise_primary_linear_raw", "primary views"),
        )
    ]
    for k in K_VALUES:
        robustness_rows.append(
            correlation_row(
                f"local_k{k}_calibrated",
                numeric(
                    primary_views,
                    f"aristotelian_local_k{k}_calibrated",
                    "primary views",
                ),
                numeric(
                    primary_views,
                    f"pairwise_primary_local_k{k}_calibrated",
                    "primary views",
                ),
            )
        )
    robustness_rows.append(
        correlation_row(
            "local_mean_calibrated",
            numeric(
                primary_views,
                "aristotelian_local_mean_calibrated",
                "primary views",
            ),
            numeric(
                primary_views,
                "pairwise_primary_local_mean_calibrated",
                "primary views",
            ),
        )
    )
    robustness = pd.DataFrame(robustness_rows)

    primary_correlation_columns = [
        "platonic_linear_raw",
        "platonic_linear_calibrated",
        "aristotelian_local_k100_raw",
        "aristotelian_local_k100_calibrated",
        "aristotelian_local_mean_calibrated",
        "pairwise_primary_rbf_raw",
    ]
    view_correlations = pairwise_view_correlations(
        primary_views, primary_correlation_columns
    )

    matrices: dict[str, dict[str, pd.DataFrame]] = {
        "primary_fixed": {
            view: build_matrix(primary_views, primary_ids, view, f"primary_fixed/{view}")
            for view in fixed_columns
        },
        "pairwise_primary": {
            view: build_matrix(primary_views, primary_ids, view, f"pairwise_primary/{view}")
            for view in primary_pairwise_columns
        },
        "sensitivity_manifest": {
            view: build_matrix(
                sensitivity_views, sensitivity_ids, view, f"sensitivity_manifest/{view}"
            )
            for view in sensitivity_columns
        },
    }

    registry = pd.DataFrame(registry_rows())
    if registry["view_id"].duplicated().any():
        raise ValueError("View registry contains duplicate view IDs")
    expected_registry = set(
        fixed_columns + primary_pairwise_columns + sensitivity_columns
    )
    observed_registry = set(registry["view_id"].astype(str))
    if observed_registry != expected_registry:
        raise ValueError(
            "View registry/matrix mismatch: "
            f"missing={sorted(expected_registry - observed_registry)}, "
            f"unexpected={sorted(observed_registry - expected_registry)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_dataframe(primary_views, output_dir / "primary_pair_views.tsv")
    atomic_dataframe(
        sensitivity_views, output_dir / "sensitivity_manifest_pair_views.tsv"
    )
    atomic_dataframe(robustness, output_dir / "source_robustness.tsv")
    atomic_dataframe(view_correlations, output_dir / "view_correlations.tsv")
    atomic_dataframe(registry, output_dir / "view_registry.tsv")

    matrix_files: dict[str, list[str]] = {}
    for group, group_matrices in matrices.items():
        matrix_files[group] = []
        matrix_dir = output_dir / f"matrices_{group}"
        for view, matrix in group_matrices.items():
            destination = matrix_dir / f"{view}.tsv"
            atomic_dataframe(matrix, destination, index=True)
            matrix_files[group].append(str(destination.relative_to(output_dir)))

    summary: dict[str, object] = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_panel": {
            "embeddings": len(primary_ids),
            "pairs": len(primary_views),
            "fixed_common_genes": EXPECTED_FIXED_GENES,
            "fixed_null_shuffles": EXPECTED_NULL_SHUFFLES,
        },
        "sensitivity_panel": {
            "embeddings": len(sensitivity_ids),
            "pairs": len(sensitivity_views),
            "rbf_genes_per_pair": EXPECTED_RBF_GENES,
            "local_genes_per_pair": EXPECTED_LOCAL_GENES,
        },
        "registered_primary_views": {
            "platonic_descriptive": "platonic_linear_raw",
            "platonic_null_corrected": "platonic_linear_calibrated",
            "aristotelian_descriptive": "aristotelian_local_k100_raw",
            "aristotelian_null_corrected": "aristotelian_local_k100_calibrated",
        },
        "interpretation_limits": {
            "fixed_panel_empirical_p_values": (
                "Descriptive only: five null shuffles cannot support "
                "high-resolution pairwise significance claims."
            ),
            "rbf": (
                "Nonlinear sensitivity view only: computed on 6,000 "
                "deterministic pair-specific genes and not permutation-calibrated."
            ),
            "source_robustness_p_values": (
                "Descriptive association diagnostics; pairs are not independent."
            ),
        },
        "matrix_files": matrix_files,
        "sources": {
            "import_summary": {
                "path": str(import_summary_path),
                "sha256": sha256(import_summary_path),
            },
            "fixed_common_atlas": {"path": str(atlas_path), "sha256": sha256(atlas_path)},
            "primary_pairwise": {
                "path": str(primary_pfi_path),
                "sha256": sha256(primary_pfi_path),
            },
            "sensitivity_pairwise": {
                "path": str(sensitivity_pfi_path),
                "sha256": sha256(sensitivity_pfi_path),
            },
        },
    }
    atomic_json(summary, output_dir / "view_summary.json")

    print("===== CONSENSUS GRAPH SIMILARITY VIEWS =====")
    print("status: PASS")
    print("primary fixed-common views:")
    print(f"  embeddings: {len(primary_ids)}")
    print(f"  genes: {EXPECTED_FIXED_GENES}")
    print(f"  pairs: {len(primary_views)}")
    print(f"  matrices: {len(matrices['primary_fixed'])}")
    print("PFI primary-panel direct-comparison views:")
    print(f"  pairs: {len(primary_views)}")
    print(f"  matrices: {len(matrices['pairwise_primary'])}")
    print("PFI full-manifest sensitivity views:")
    print(f"  embeddings: {len(sensitivity_ids)}")
    print(f"  pairs: {len(sensitivity_views)}")
    print(f"  matrices: {len(matrices['sensitivity_manifest'])}")
    print("primary Platonic view: row-normalized unbiased linear CKA")
    print("primary Aristotelian view: tie-aware local overlap at k=100")
    print("RBF CKA role: nonlinear pair-specific sensitivity only")
    print("\nsource robustness (fixed common vs pair-specific):")
    print(
        robustness[["metric", "spearman_rho", "median_absolute_delta"]]
        .to_string(index=False)
    )
    print(f"\noutput: {output_dir}")
    print(
        "NOTE: fixed-panel calibrated scores use five-shuffle null means; "
        "pairwise significance is descriptive only."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError, KeyError) as error:
        print("===== CONSENSUS GRAPH SIMILARITY VIEWS =====", file=sys.stderr)
        print("status: FAIL", file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
