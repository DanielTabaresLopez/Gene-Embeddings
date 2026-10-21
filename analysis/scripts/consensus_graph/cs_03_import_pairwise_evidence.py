#!/usr/bin/env python3
"""Import validated pairwise evidence into the consensus-graph workflow.

The primary evidence is the recomputed atlas in which all 56 retained
embeddings are compared on the same 14,039 genes. Pairwise full-intersection
(PFI) results for all 67 manifest embeddings are imported as sensitivity
evidence, with a 56-embedding subset written for direct alignment.

The script validates panel identity, gene identity, pair completeness, metric
finiteness, and expected sampling settings before writing derived artifacts.
It never modifies either source result, the PCA cache, or MLP outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PRIMARY_PANEL = "primary_fixed_56"
SENSITIVITY_PANEL = "sensitivity_manifest_67"
EXPECTED_PRIMARY_EMBEDDINGS = 56
EXPECTED_SENSITIVITY_EMBEDDINGS = 67
EXPECTED_PRIMARY_GENES = 14_039
EXPECTED_PRIMARY_PAIRS = 1_540
EXPECTED_SENSITIVITY_PAIRS = 2_211
EXPECTED_NULL_SHUFFLES = 5
EXPECTED_RBF_N_GENES = 6_000
EXPECTED_LOCAL_N_GENES = 8_000

ATLAS_METRICS = (
    "unbiased_linear_cka_row_normalized",
    "linear_observed_minus_null",
    "local_k10_observed_minus_null",
    "local_k25_observed_minus_null",
    "local_k50_observed_minus_null",
    "local_k100_observed_minus_null",
)

PFI_METRICS = (
    "unbiased_linear_cka_row_normalized",
    "unbiased_rbf_cka_row_normalized_bw_1p0",
    "local_effect_k10",
    "local_effect_k25",
    "local_effect_k50",
    "local_effect_k100",
    "local_normalized_effect_k10",
    "local_normalized_effect_k25",
    "local_normalized_effect_k50",
    "local_normalized_effect_k100",
)


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=None,
        help="Frozen panel directory (default: results/consensus_graph_v2/01_fixed_panel).",
    )
    parser.add_argument(
        "--atlas",
        type=Path,
        default=None,
        help="Recomputed fixed-common atlas pair table.",
    )
    parser.add_argument(
        "--pfi",
        type=Path,
        default=None,
        help="Current 67-embedding PFI pair_results.tsv (required).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination (default: consensus_graph_v2/03_pairwise_evidence).",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_one_column(path: Path, expected_column: str | None = None) -> list[str]:
    path = require_file(path)
    if path.suffix.lower() == ".txt":
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        frame = pd.read_csv(path, sep="\t")
        if expected_column is not None and expected_column not in frame.columns:
            raise ValueError(f"{path} has no {expected_column!r} column")
        if frame.shape[1] < 1:
            raise ValueError(f"{path} has no columns")
        column = expected_column if expected_column is not None else frame.columns[0]
        if frame[column].isna().any():
            raise ValueError(f"{path} has missing values in {column}")
        values = frame[column].astype(str).tolist()

    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate identifiers")
    return values


def canonical_pair(x: object, y: object) -> tuple[str, str]:
    left, right = str(x), str(y)
    if left == right:
        raise ValueError(f"Self-pair encountered: {left}")
    return (left, right) if left < right else (right, left)


def pair_keys(frame: pd.DataFrame, label: str) -> set[tuple[str, str]]:
    required = {"embedding_x", "embedding_y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing pair columns: {sorted(missing)}")

    keys = {
        canonical_pair(x, y)
        for x, y in zip(frame["embedding_x"], frame["embedding_y"])
    }
    if len(keys) != len(frame):
        raise ValueError(f"{label} has {len(frame) - len(keys)} duplicate pair rows")
    return keys


def expected_pairs(embedding_ids: Iterable[str]) -> set[tuple[str, str]]:
    return set(combinations(sorted(embedding_ids), 2))


def check_pair_coverage(
    observed: set[tuple[str, str]],
    expected: set[tuple[str, str]],
    label: str,
) -> None:
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise ValueError(
            f"{label} pair coverage mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )


def require_finite(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    absent = set(columns) - set(frame.columns)
    if absent:
        raise ValueError(f"{label} is missing metrics: {sorted(absent)}")

    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        finite = int(np.isfinite(values).sum())
        if finite != len(frame):
            raise ValueError(
                f"{label}/{column}: expected {len(frame)} finite values, found {finite}"
            )


def require_single_numeric_value(
    frame: pd.DataFrame,
    column: str,
    expected: int,
    label: str,
) -> None:
    if column not in frame.columns:
        raise ValueError(f"{label} is missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    observed = sorted(values.dropna().unique().tolist())
    if values.isna().any() or observed != [expected]:
        raise ValueError(f"{label}/{column}: expected only {expected}, found {observed}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


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


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    workflow_root = root / "results/consensus_graph_v2"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow_root / "03_pairwise_evidence"
    )
    panel_dir = (
        args.panel_dir.expanduser().resolve()
        if args.panel_dir is not None
        else workflow_root / "01_fixed_panel"
    )
    atlas_path = require_file(
        args.atlas.expanduser().resolve()
        if args.atlas is not None
        else workflow_root / "02_common_panel_atlas/atlas_pair_scores.tsv"
    )
    if args.pfi is None:
        raise ValueError("Pass --pfi with the current 67-embedding PFI pair_results.tsv")
    pfi_path = require_file(args.pfi)

    current_primary_ids = read_one_column(panel_dir / "retained_embeddings.txt")
    current_primary_genes = read_one_column(panel_dir / "common_genes.txt")

    if len(current_primary_ids) != EXPECTED_PRIMARY_EMBEDDINGS:
        raise ValueError(
            f"Expected {EXPECTED_PRIMARY_EMBEDDINGS} primary embeddings, "
            f"found {len(current_primary_ids)}"
        )
    if len(current_primary_genes) != EXPECTED_PRIMARY_GENES:
        raise ValueError(
            f"Expected {EXPECTED_PRIMARY_GENES} primary genes, "
            f"found {len(current_primary_genes)}"
        )
    primary_expected = expected_pairs(current_primary_ids)
    if len(primary_expected) != EXPECTED_PRIMARY_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PRIMARY_PAIRS} primary pairs")
    atlas = pd.read_csv(atlas_path, sep="\t")
    atlas_keys = pair_keys(atlas, "fixed-common atlas")
    check_pair_coverage(atlas_keys, primary_expected, "fixed-common atlas")
    require_single_numeric_value(
        atlas, "gene_count", EXPECTED_PRIMARY_GENES, "fixed-common atlas"
    )
    require_single_numeric_value(
        atlas, "null_shuffles", EXPECTED_NULL_SHUFFLES, "fixed-common atlas"
    )
    require_finite(atlas, ATLAS_METRICS, "fixed-common atlas")

    pfi = pd.read_csv(pfi_path, sep="\t")
    current_sensitivity_ids = sorted(
        set(pfi["embedding_x"].astype(str)) | set(pfi["embedding_y"].astype(str))
    )
    if len(current_sensitivity_ids) != EXPECTED_SENSITIVITY_EMBEDDINGS:
        raise ValueError(
            f"Expected {EXPECTED_SENSITIVITY_EMBEDDINGS} PFI embeddings, "
            f"found {len(current_sensitivity_ids)}"
        )
    if not set(current_primary_ids).issubset(current_sensitivity_ids):
        raise ValueError("Fixed panel is not a subset of the PFI panel")
    sensitivity_expected = expected_pairs(current_sensitivity_ids)
    if len(sensitivity_expected) != EXPECTED_SENSITIVITY_PAIRS:
        raise ValueError(f"Expected {EXPECTED_SENSITIVITY_PAIRS} sensitivity pairs")
    pfi_keys = pair_keys(pfi, "PFI table")
    check_pair_coverage(pfi_keys, sensitivity_expected, "PFI table")
    if "status" not in pfi.columns:
        raise ValueError("PFI table is missing status")
    statuses = pfi["status"].astype(str).value_counts(dropna=False).to_dict()
    if statuses != {"PASS": EXPECTED_SENSITIVITY_PAIRS}:
        raise ValueError(f"Unexpected PFI statuses: {statuses}")
    require_single_numeric_value(
        pfi, "rbf_n_genes", EXPECTED_RBF_N_GENES, "PFI table"
    )
    require_single_numeric_value(
        pfi, "local_n_genes", EXPECTED_LOCAL_N_GENES, "PFI table"
    )
    require_finite(pfi, PFI_METRICS, "PFI table")

    primary_mask = [
        canonical_pair(x, y) in primary_expected
        for x, y in zip(pfi["embedding_x"], pfi["embedding_y"])
    ]
    primary_pfi = pfi.loc[primary_mask].copy()
    primary_pfi_keys = pair_keys(primary_pfi, "PFI primary subset")
    check_pair_coverage(
        primary_pfi_keys, primary_expected, "PFI primary subset"
    )

    atlas_output = output_dir / "primary_fixed_common_atlas.tsv"
    primary_pfi_output = output_dir / "primary_pairwise_intersection.tsv"
    sensitivity_pfi_output = output_dir / "sensitivity_manifest_pairwise_intersection.tsv"
    summary_output = output_dir / "import_summary.json"

    # Copy the complete source tables byte-for-byte; write only the fixed-panel
    # PFI subset as a derived table.
    atomic_copy(atlas_path, atlas_output)
    atomic_dataframe(primary_pfi, primary_pfi_output)
    atomic_copy(pfi_path, sensitivity_pfi_output)

    summary: dict[str, object] = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_roles": {
            "primary": (
                "fixed common panel: 56 embeddings compared on the same "
                "14,039 genes"
            ),
            "direct_comparison": (
                "PFI rows for the same 56 embeddings, using each "
                "pair's intersection"
            ),
            "sensitivity": (
                "PFI complete 67-embedding table, using each pair's "
                "intersection"
            ),
        },
        "verified_identity": {
            "primary_embedding_sets_identical": True,
            "primary_gene_sets_identical": True,
        },
        "sources": {
            "fixed_panel": str(panel_dir),
            "fixed_common_atlas": {
                "path": str(atlas_path),
                "sha256": sha256(atlas_path),
            },
            "pfi_pair_results": {
                "path": str(pfi_path),
                "sha256": sha256(pfi_path),
            },
        },
        "outputs": {
            "primary_fixed_common_atlas": {
                "path": str(atlas_output),
                "rows": len(atlas),
                "embeddings": len(current_primary_ids),
                "genes": EXPECTED_PRIMARY_GENES,
                "null_shuffles": EXPECTED_NULL_SHUFFLES,
                "sha256": sha256(atlas_output),
            },
            "primary_pairwise_intersection": {
                "path": str(primary_pfi_output),
                "rows": len(primary_pfi),
                "embeddings": len(current_primary_ids),
                "rbf_n_genes": EXPECTED_RBF_N_GENES,
                "local_n_genes": EXPECTED_LOCAL_N_GENES,
                "sha256": sha256(primary_pfi_output),
            },
            "sensitivity_pairwise_intersection": {
                "path": str(sensitivity_pfi_output),
                "rows": len(pfi),
                "embeddings": len(current_sensitivity_ids),
                "rbf_n_genes": EXPECTED_RBF_N_GENES,
                "local_n_genes": EXPECTED_LOCAL_N_GENES,
                "sha256": sha256(sensitivity_pfi_output),
            },
        },
        "metric_interpretation": {
            "primary_linear_and_local": (
                "Use primary_fixed_common_atlas.tsv; every pair uses the "
                "same 14,039 genes."
            ),
            "rbf": (
                "Available only in the PFI pair-specific-intersection "
                "tables; do not label it fixed-common-panel RBF."
            ),
            "atlas_empirical_p_values": (
                "Descriptive only because the atlas used five null shuffles."
            ),
        },
    }
    atomic_json(summary, summary_output)

    print("===== CONSENSUS GRAPH PAIRWISE EVIDENCE IMPORT =====")
    print("status: PASS")
    print("primary fixed-common atlas:")
    print(f"  embeddings: {len(current_primary_ids)}")
    print(f"  genes: {len(current_primary_genes)}")
    print(f"  pairs: {len(atlas)}")
    print(f"  null shuffles: {EXPECTED_NULL_SHUFFLES}")
    print("PFI primary subset:")
    print(f"  pairs: {len(primary_pfi)}")
    print(f"  RBF genes per pair: {EXPECTED_RBF_N_GENES}")
    print(f"  local genes per pair: {EXPECTED_LOCAL_N_GENES}")
    print("PFI all-67 sensitivity:")
    print(f"  pairs: {len(pfi)}")
    print(f"output: {output_dir}")
    print(
        "NOTE: atlas empirical p-values are descriptive only (five null shuffles)."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, OSError) as error:
        print("===== CONSENSUS GRAPH PAIRWISE EVIDENCE IMPORT =====", file=sys.stderr)
        print("status: FAIL", file=sys.stderr)
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
