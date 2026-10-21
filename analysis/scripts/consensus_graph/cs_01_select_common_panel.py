#!/usr/bin/env python3
"""Select the largest manifest panel whose strict intersection has >= target genes.

Every manifest embedding is audited against its ``genes.tsv`` and
``embeddings.npz``. A binary mixed-integer program first minimizes the number
of excluded embeddings and then, at that minimum, maximizes the number of
common valid genes. Equally minimal alternatives are emitted for review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csr_array

from cs_manifest import load_manifest, manifest_maps, panel_sha256, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--library-root",
        type=Path,
        default=None,
        help="Override library_root in the YAML. May be the repository root or data/embeddings.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--previous-panel-dir",
        type=Path,
        default=None,
        help="Optional directory containing retained_embeddings.txt and common_genes.txt.",
    )
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--alternatives", type=int, default=None)
    parser.add_argument("--min-retained", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument(
        "--skip-matrix-audit",
        action="store_true",
        help="Do not verify finite, non-zero matrix rows. Never use for a paper run.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def require_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")
    return path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    text = "".join(f"{value}\n" for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in require_file(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_embeddings_root(
    library_root: Path, embeddings_subdir: str | Path, manifest_ids: list[str]
) -> tuple[Path, list[tuple[str, int]]]:
    """Accept either a dataset checkout root or its data/embeddings directory."""
    base = require_dir(library_root)
    candidates = [base, base / embeddings_subdir, base / "data" / "embeddings"]
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in unique and resolved.is_dir():
            unique.append(resolved)
    scored = [
        (candidate, sum((candidate / name).is_dir() for name in manifest_ids))
        for candidate in unique
    ]
    if not scored:
        raise FileNotFoundError(f"No embeddings directory found below {base}")
    selected, count = max(scored, key=lambda item: item[1])
    audit = [(str(path), found) for path, found in scored]
    if count == 0:
        raise FileNotFoundError(
            "None of the manifest packages was found. Checked: "
            + ", ".join(str(path) for path, _ in scored)
        )
    return selected, audit


def load_gene_ids(path: Path, gene_column: str) -> list[str]:
    table = pd.read_csv(path, sep="\t", dtype=str)
    if gene_column not in table.columns:
        raise ValueError(f"{path} lacks {gene_column!r}; columns={list(table.columns)}")
    values = table[gene_column]
    if values.isna().any():
        raise ValueError(f"{path} contains missing values in {gene_column!r}")
    genes = values.astype(str).str.strip().tolist()
    if any(not gene for gene in genes):
        raise ValueError(f"{path} contains blank gene identifiers")
    duplicated = pd.Series(genes).duplicated(keep=False)
    if duplicated.any():
        examples = sorted(set(pd.Series(genes)[duplicated]))[:10]
        raise ValueError(f"{path} contains duplicate gene IDs; examples={examples}")
    return genes


def audit_embedding(
    embedding_id: str,
    embeddings_root: Path,
    gene_column: str,
    matrix_key: str,
    skip_matrix_audit: bool,
) -> tuple[set[str], dict[str, object]]:
    package = embeddings_root / embedding_id
    genes_path = require_file(package / "genes.tsv")
    matrix_path = require_file(package / "embeddings.npz")
    genes = load_gene_ids(genes_path, gene_column)

    zero_count = 0
    nonfinite_count = 0
    dimensions: int | str = ""
    if skip_matrix_audit:
        valid_mask = np.ones(len(genes), dtype=bool)
    else:
        with np.load(matrix_path, allow_pickle=False) as archive:
            if matrix_key not in archive.files:
                raise KeyError(f"{matrix_path} lacks {matrix_key!r}; keys={archive.files}")
            matrix = archive[matrix_key]
            if matrix.ndim != 2:
                raise ValueError(f"{matrix_path}:{matrix_key} is not two-dimensional")
            if matrix.shape[0] != len(genes):
                raise ValueError(
                    f"Row mismatch for {embedding_id}: matrix={matrix.shape[0]}, genes={len(genes)}"
                )
            dimensions = int(matrix.shape[1])
            finite_mask = np.isfinite(matrix).all(axis=1)
            nonzero_mask = np.any(matrix != 0, axis=1)
            nonfinite_count = int((~finite_mask).sum())
            zero_count = int((finite_mask & ~nonzero_mask).sum())
            valid_mask = finite_mask & nonzero_mask

    valid_genes = {
        gene for gene, valid in zip(genes, valid_mask, strict=True) if bool(valid)
    }
    if not valid_genes:
        raise ValueError(f"{embedding_id} has no valid non-zero gene rows")
    return valid_genes, {
        "embedding_id": embedding_id,
        "gene_count": len(genes),
        "valid_nonzero_gene_count": len(valid_genes),
        "zero_row_count": zero_count,
        "nonfinite_row_count": nonfinite_count,
        "dimensions": dimensions,
        "genes_tsv": str(genes_path),
        "embeddings_npz": str(matrix_path),
        "genes_tsv_sha256": sha256_file(genes_path),
        "embeddings_npz_sha256": sha256_file(matrix_path),
        "valid_gene_set_sha256": sha256_lines(sorted(valid_genes)),
    }


def intersection_for(retained: Iterable[str], gene_sets: dict[str, set[str]]) -> set[str]:
    names = list(retained)
    return set.intersection(*(gene_sets[name] for name in names)) if names else set()


def make_milp_constraints(
    embeddings: list[str],
    universe: list[str],
    gene_sets: dict[str, set[str]],
    target: int,
    min_retained: int,
) -> tuple[list[LinearConstraint], int]:
    embedding_index = {name: i for i, name in enumerate(embeddings)}
    gene_index = {gene: i for i, gene in enumerate(universe)}
    n_embeddings = len(embeddings)
    n_variables = n_embeddings + len(universe)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row = 0
    universe_set = set(universe)

    for embedding in embeddings:
        for gene in universe_set.difference(gene_sets[embedding]):
            # y_gene <= z_embedding; z=1 means that embedding is excluded.
            rows.extend((row, row))
            cols.extend((n_embeddings + gene_index[gene], embedding_index[embedding]))
            data.extend((1.0, -1.0))
            row += 1

    eligibility = LinearConstraint(
        coo_array(
            (data, (rows, cols)), shape=(row, n_variables), dtype=float
        ).tocsr(),
        lb=np.full(row, -np.inf),
        ub=np.zeros(row),
    )
    target_row = csr_array(
        (
            np.ones(len(universe)),
            (
                np.zeros(len(universe), dtype=int),
                np.arange(n_embeddings, n_variables),
            ),
        ),
        shape=(1, n_variables),
    )
    target_constraint = LinearConstraint(target_row, lb=[float(target)], ub=[np.inf])
    exclusion_row = csr_array(
        (
            np.ones(n_embeddings),
            (np.zeros(n_embeddings, dtype=int), np.arange(n_embeddings)),
        ),
        shape=(1, n_variables),
    )
    retained_constraint = LinearConstraint(
        exclusion_row, lb=[-np.inf], ub=[float(n_embeddings - min_retained)]
    )
    return [eligibility, target_constraint, retained_constraint], n_variables


def exact_solutions_for_target(
    embeddings: list[str],
    gene_sets: dict[str, set[str]],
    target: int,
    min_retained: int,
    alternatives: int,
    time_limit: float,
) -> tuple[int, list[dict[str, object]], str]:
    universe = sorted(set.union(*(gene_sets[name] for name in embeddings)))
    constraints, n_variables = make_milp_constraints(
        embeddings, universe, gene_sets, target, min_retained
    )
    n_embeddings = len(embeddings)
    integrality = np.ones(n_variables, dtype=np.uint8)
    bounds = Bounds(np.zeros(n_variables), np.ones(n_variables))
    options = {
        "time_limit": float(time_limit),
        "mip_rel_gap": 0.0,
        "presolve": True,
    }

    objective = np.zeros(n_variables)
    objective[:n_embeddings] = 1.0
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Target {target}: minimum-exclusion MILP failed: {result.message}"
        )
    minimum_removed = int(np.rint(result.x[:n_embeddings]).sum())

    exclusion_count_row = csr_array(
        (
            np.ones(n_embeddings),
            (np.zeros(n_embeddings, dtype=int), np.arange(n_embeddings)),
        ),
        shape=(1, n_variables),
    )
    fixed_count = LinearConstraint(
        exclusion_count_row, lb=[minimum_removed], ub=[minimum_removed]
    )
    maximize_genes = np.zeros(n_variables)
    maximize_genes[n_embeddings:] = -1.0
    no_good: list[LinearConstraint] = []
    solutions: list[dict[str, object]] = []
    last_message = str(result.message)

    for rank in range(1, alternatives + 1):
        result = milp(
            maximize_genes,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints + [fixed_count] + no_good,
            options=options,
        )
        last_message = str(result.message)
        if not result.success or result.x is None:
            break
        excluded = [
            name
            for name, value in zip(
                embeddings, result.x[:n_embeddings], strict=True
            )
            if value >= 0.5
        ]
        excluded_set = set(excluded)
        retained = [name for name in embeddings if name not in excluded_set]
        common = intersection_for(retained, gene_sets)
        if len(common) < target:
            raise AssertionError(
                f"Candidate has {len(common)} genes, below target {target}"
            )
        solutions.append(
            {
                "alternative_rank": rank,
                "excluded": excluded,
                "retained": retained,
                "common": common,
            }
        )
        if minimum_removed == 0:
            break
        columns = np.asarray(
            [embeddings.index(name) for name in excluded], dtype=int
        )
        no_good.append(
            LinearConstraint(
                csr_array(
                    (
                        np.ones(minimum_removed),
                        (np.zeros(minimum_removed, dtype=int), columns),
                    ),
                    shape=(1, n_variables),
                ),
                lb=[-np.inf],
                ub=[minimum_removed - 1],
            )
        )
    if not solutions:
        raise RuntimeError(
            f"Target {target}: no optimal alternative was enumerated"
        )
    return minimum_removed, solutions, last_message


def greedy_path(
    embeddings: list[str],
    gene_sets: dict[str, set[str]],
    metadata: dict[str, dict[str, str]],
) -> pd.DataFrame:
    retained = list(embeddings)
    rows: list[dict[str, object]] = []
    step = 0
    while len(retained) >= 2:
        common = intersection_for(retained, gene_sets)
        rows.append(
            {
                "step": step,
                "removed_at_this_step": "",
                "retained_embedding_count": len(retained),
                "common_gene_count": len(common),
                "marginal_gain": 0 if step == 0 else "",
                "modality_count": len(
                    {metadata[name]["modality"] for name in retained}
                ),
            }
        )
        candidates = []
        for name in retained:
            candidate = [value for value in retained if value != name]
            candidates.append(
                (len(intersection_for(candidate, gene_sets)), name)
            )
        best_count, removed = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        retained.remove(removed)
        step += 1
        rows.append(
            {
                "step": f"{step}:removal",
                "removed_at_this_step": removed,
                "retained_embedding_count": len(retained),
                "common_gene_count": best_count,
                "marginal_gain": best_count - len(common),
                "modality_count": len(
                    {metadata[name]["modality"] for name in retained}
                ),
            }
        )
        if len(retained) == 2:
            break
    return pd.DataFrame(rows)


def find_previous_file(directory: Path, filename: str) -> Path:
    direct = directory / filename
    if direct.is_file():
        return direct
    matches = sorted(directory.glob(f"candidate_panels/*/{filename}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} below {directory}")
    raise ValueError(
        f"Multiple {filename} files found below {directory}; "
        "pass one candidate directory"
    )


def previous_panel_data(
    path: Path | None,
) -> tuple[set[str] | None, set[str] | None, dict[str, object]]:
    if path is None:
        return None, None, {"provided": False}
    directory = require_dir(path)
    retained_path = find_previous_file(directory, "retained_embeddings.txt")
    genes_path = find_previous_file(directory, "common_genes.txt")
    retained = set(read_lines(retained_path))
    genes = set(read_lines(genes_path))
    if len(retained) < 2 or not genes:
        raise ValueError("Previous panel files are empty or invalid")
    return retained, genes, {
        "provided": True,
        "directory": str(directory),
        "retained_embeddings_file": str(retained_path),
        "retained_embeddings_sha256": sha256_file(retained_path),
        "common_genes_file": str(genes_path),
        "common_genes_sha256": sha256_lines(sorted(genes)),
        "retained_embedding_count": len(retained),
        "common_gene_count": len(genes),
    }


def write_checksums(output: Path) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    (output / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.relative_to(output)}" for path in files
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    config_path = require_file(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    panel, manifest_path = load_manifest(config, config_path)
    display_map, modality_map = manifest_maps(panel)
    embeddings = panel.embedding_id.tolist()
    metadata = {
        name: {
            "display_name": display_map[name],
            "modality": modality_map[name],
        }
        for name in embeddings
    }

    library_value = (
        args.library_root
        or os.environ.get("CS_LIBRARY_ROOT")
        or config.get("library_root")
    )
    if not library_value:
        raise ValueError(
            "Set library_root in the YAML, CS_LIBRARY_ROOT, or --library-root"
        )
    library_root = resolve_path(library_value, config_path)
    embeddings_root, root_audit = resolve_embeddings_root(
        library_root,
        config.get("embeddings_subdir", "data/embeddings"),
        embeddings,
    )

    output_value = args.output or resolve_path(
        config["selection_output"], config_path
    )
    output = Path(output_value).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output is not empty: {output}. Use a new versioned directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    manifest_dir = output / "manifests" / "valid_nonzero_genes"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    target = int(
        args.target
        if args.target is not None
        else config.get("target_common_genes", 14_000)
    )
    alternatives = int(
        args.alternatives
        if args.alternatives is not None
        else config.get("alternatives", 10)
    )
    min_retained = int(
        args.min_retained
        if args.min_retained is not None
        else config.get("min_retained_embeddings", 2)
    )
    time_limit = float(
        args.time_limit
        if args.time_limit is not None
        else config.get("milp_time_limit_seconds", 900)
    )
    if target <= 0 or alternatives <= 0 or not (
        2 <= min_retained <= len(embeddings)
    ):
        raise ValueError("Invalid target, alternatives, or min-retained setting")

    local_ids = {
        path.name for path in embeddings_root.iterdir() if path.is_dir()
    }
    missing = sorted(set(embeddings) - local_ids)
    ignored = sorted(local_ids - set(embeddings))
    if missing:
        raise FileNotFoundError(
            f"Manifest packages missing below {embeddings_root}: {missing}"
        )

    print("=" * 78)
    print("CONSENSUS GRAPH STEP 1: >=14K COMMON-GENE PANEL")
    print("=" * 78)
    print(f"Manifest: {manifest_path}")
    print(f"Panel: {len(embeddings)} embeddings")
    print(f"Embedding packages: {embeddings_root}")
    print(f"Target: {target:,} common valid genes\n")

    gene_sets: dict[str, set[str]] = {}
    inventory_rows: list[dict[str, object]] = []
    for index, embedding in enumerate(embeddings, start=1):
        print(
            f"[{index:02d}/{len(embeddings):02d}] auditing {embedding}",
            flush=True,
        )
        valid_genes, record = audit_embedding(
            embedding,
            embeddings_root,
            str(config.get("id_column", "ensembl_gene_id")),
            str(config.get("array_key", "embeddings")),
            args.skip_matrix_audit,
        )
        gene_sets[embedding] = valid_genes
        record.update(metadata[embedding])
        inventory_rows.append(record)
        write_lines(manifest_dir / f"{embedding}.txt", sorted(valid_genes))

    starting_common = intersection_for(embeddings, gene_sets)
    write_lines(output / "starting_panel_embeddings.txt", embeddings)
    write_lines(
        output / "starting_panel_common_genes.txt", sorted(starting_common)
    )
    panel.to_csv(output / "starting_panel.tsv", sep="\t", index=False)
    pd.DataFrame(inventory_rows).sort_values(
        ["valid_nonzero_gene_count", "embedding_id"]
    ).to_csv(output / "embedding_gene_coverage.tsv", sep="\t", index=False)

    leave_one_out = []
    for embedding in embeddings:
        common = intersection_for(
            [name for name in embeddings if name != embedding], gene_sets
        )
        leave_one_out.append(
            {
                "embedding_removed": embedding,
                "embedding_removed_display": display_map[embedding],
                "modality": modality_map[embedding],
                "remaining_embedding_count": len(embeddings) - 1,
                "common_gene_count_after_removal": len(common),
                "gain_over_starting_panel": len(common) - len(starting_common),
            }
        )
    pd.DataFrame(leave_one_out).sort_values(
        ["gain_over_starting_panel", "embedding_removed"],
        ascending=[False, True],
    ).to_csv(output / "leave_one_out_effects.tsv", sep="\t", index=False)
    greedy_path(embeddings, gene_sets, metadata).to_csv(
        output / "greedy_removal_path.tsv", sep="\t", index=False
    )

    previous_override = args.previous_panel_dir or os.environ.get(
        "CS_PREVIOUS_PANEL_DIR"
    )
    previous_config = config.get("previous_panel_dir")
    previous_path = previous_override or (
        resolve_path(previous_config, config_path) if previous_config else None
    )
    previous_ids, previous_genes, previous_audit = previous_panel_data(
        Path(previous_path).expanduser().resolve() if previous_path else None
    )

    print(f"\nSolving exact selection for >= {target:,} genes", flush=True)
    minimum_removed, solutions, solver_message = exact_solutions_for_target(
        embeddings,
        gene_sets,
        target,
        min_retained,
        alternatives,
        time_limit,
    )
    summary_rows: list[dict[str, object]] = []
    for solution in solutions:
        retained = list(solution["retained"])
        excluded = list(solution["excluded"])
        common = set(solution["common"])
        rank = int(solution["alternative_rank"])
        panel_dir = (
            output
            / "candidate_panels"
            / f"target_{target}__alternative_{rank:02d}"
        )
        write_lines(panel_dir / "retained_embeddings.txt", retained)
        write_lines(panel_dir / "excluded_embeddings.txt", excluded)
        write_lines(panel_dir / "common_genes.txt", sorted(common))
        panel.loc[panel.embedding_id.isin(retained)].to_csv(
            panel_dir / "retained_embeddings.tsv", sep="\t", index=False
        )
        panel.loc[panel.embedding_id.isin(excluded)].to_csv(
            panel_dir / "excluded_embeddings.tsv", sep="\t", index=False
        )

        same_genes = previous_genes is not None and common == previous_genes
        shared_ids = (
            len(set(retained) & previous_ids)
            if previous_ids is not None
            else 0
        )
        compatible_pairs = (
            math.comb(shared_ids, 2) if same_genes and shared_ids >= 2 else 0
        )
        summary_rows.append(
            {
                "target_gene_count": target,
                "minimum_exclusion_count": minimum_removed,
                "alternative_rank": rank,
                "retained_embedding_count": len(retained),
                "common_gene_count": len(common),
                "modality_count": len(
                    {modality_map[name] for name in retained}
                ),
                "excluded_embedding_ids": ";".join(excluded),
                "excluded_embedding_display_names": ";".join(
                    display_map[name] for name in excluded
                ),
                "retained_embeddings_sha256": sha256_lines(retained),
                "common_genes_sha256": sha256_lines(sorted(common)),
                "same_common_gene_set_as_previous": bool(same_genes),
                "shared_retained_embeddings_with_previous": shared_ids,
                "gene_set_compatible_previous_pair_count": compatible_pairs,
                "full_metric_recomputation_required_by_gene_set": (
                    previous_genes is not None and not same_genes
                ),
                "candidate_panel_directory": str(
                    panel_dir.relative_to(output)
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("alternative_rank")
    summary.to_csv(
        output / "candidate_panel_frontier.tsv", sep="\t", index=False
    )
    previous_audit["candidate_comparisons"] = summary[
        [
            "alternative_rank",
            "same_common_gene_set_as_previous",
            "shared_retained_embeddings_with_previous",
            "gene_set_compatible_previous_pair_count",
            "full_metric_recomputation_required_by_gene_set",
        ]
    ].to_dict("records")
    (output / "previous_panel_comparison.json").write_text(
        json.dumps(previous_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata_payload = {
        "analysis": "consensus_graph_common_panel_selection",
        "version": "v2_stage1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "panel_sha256": panel_sha256(panel),
        "hf_repo_id": config.get("hf_repo_id"),
        "hf_revision": config.get("hf_revision"),
        "embeddings_root": str(embeddings_root),
        "embeddings_root_candidates": root_audit,
        "starting_embedding_count": len(embeddings),
        "starting_common_gene_count": len(starting_common),
        "ignored_local_package_count": len(ignored),
        "ignored_local_packages": ignored,
        "target_common_genes": target,
        "alternatives_requested": alternatives,
        "alternatives_written": len(summary),
        "minimum_exclusion_count": minimum_removed,
        "min_retained_embeddings": min_retained,
        "matrix_audit_performed": not args.skip_matrix_audit,
        "solver": "scipy.optimize.milp (HiGHS)",
        "solver_message": solver_message,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best = summary.iloc[0]
    report = [
        "# Consensus-graph common-panel selection",
        "",
        f"- Manifest embeddings: **{len(embeddings)}**",
        f"- Strict intersection before exclusions: **{len(starting_common):,} genes**",
        f"- Required threshold: **at least {target:,} genes**",
        f"- Minimum exclusions: **{minimum_removed}**",
        f"- Rank-1 retained panel: **{int(best['retained_embedding_count'])} embeddings**",
        f"- Rank-1 strict intersection: **{int(best['common_gene_count']):,} genes**",
        "",
        "## Reuse decision",
        "",
    ]
    if previous_genes is None:
        report.append(
            "No previous panel was supplied, so reuse was not assessed."
        )
    elif bool(best["same_common_gene_set_as_previous"]):
        report.extend(
            [
                "The rank-1 candidate has the same common-gene set as the supplied previous panel.",
                "Old pair scores are reusable only for shared embeddings if preprocessing and metric settings also match exactly.",
            ]
        )
    else:
        report.append(
            "The rank-1 common-gene set differs from the previous one; "
            "recompute all fixed-panel CKA and kNN pair scores."
        )
    report.extend(
        [
            "",
            "Review all rows of `candidate_panel_frontier.tsv` before freezing one candidate.",
            "Stable IDs are used for computation; Excel display names and modalities are exported alongside them.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    write_checksums(output)

    print("\n" + "=" * 78)
    print("PANEL SELECTION COMPLETE")
    print("=" * 78)
    print(
        summary[
            [
                "alternative_rank",
                "minimum_exclusion_count",
                "retained_embedding_count",
                "common_gene_count",
                "modality_count",
                "same_common_gene_set_as_previous",
                "full_metric_recomputation_required_by_gene_set",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutput: {output}")
    print(
        "Review candidate_panel_frontier.tsv and report.md before running CKA/kNN."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
