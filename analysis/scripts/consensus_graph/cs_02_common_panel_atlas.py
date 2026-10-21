#!/usr/bin/env python3
"""Build the frozen 56-embedding/14,039-gene common-panel atlas.

Subcommands
-----------
freeze
    Freeze the selected panel, validate it, and create its pair manifest.
prepare
    Extract one embedding on the frozen genes and cache exact linear-CKA terms.
neighbors
    Build exact tie-aware cosine-neighborhood graphs for one embedding.
pair
    Compute one pair's observed metrics and five gene-identity shuffled controls.
aggregate
    Assemble matrices, heatmaps, PCoAs, validation summaries, and checksums.

The primary global endpoint is row-L2-normalized unbiased linear CKA. The raw
version is retained as a sensitivity endpoint. The local endpoints are exact
tie-aware neighborhood overlaps at k=10,25,50,100.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


PANEL_NAME = "consensus_56_all14039_common_v2"
EXPECTED_EMBEDDINGS = 56
EXPECTED_GENES = 14_039
EXPECTED_PAIRS = 1_540
EXPECTED_RETAINED_FILE_SHA256 = "8c60d7a0bca2b6e10916b0d750f4e304a5a93e72e758e0cc3b0d167ed9f966c7"
EXPECTED_GENE_FILE_SHA256 = "caa45ecca3ce0526d2dbb53e9dbf008016a1d6ba513488682a22ca69a04ee238"
K_VALUES = (10, 25, 50, 100)
SEED = 20260725
# Square-root color scaling: raw 0.20 is displayed at sqrt(0.20) = 0.447
# on viridis, which is approximately blue-green rather than dark blue.
# The same mapping and registered reference order are used in every heatmap.
HEATMAP_GAMMA = 0.50
HEATMAP_RAW_TICKS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)
# Stable repository IDs are retained for computation. Plot order follows the
# frozen manifest, and paper-facing axes use the manifest display names.


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def require_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def resolve_embeddings_root(base: Path, embedding_ids: Iterable[str]) -> Path:
    """Accept a dataset checkout or its embedding-package directory."""
    base = require_dir(base)
    ids = list(embedding_ids)
    candidates = [base, base / "data" / "embeddings", base / "embeddings"]
    scored = [
        (sum((candidate / name).is_dir() for name in ids), candidate)
        for candidate in candidates
        if candidate.is_dir()
    ]
    if not scored:
        raise FileNotFoundError(f"No embedding-package directory found below {base}")
    score, selected = max(scored, key=lambda item: item[0])
    if score != len(ids):
        missing = sorted(name for name in ids if not (selected / name).is_dir())
        raise FileNotFoundError(
            f"Embedding root {selected} is missing {len(missing)} retained packages; "
            f"first missing={missing[:5]}"
        )
    return selected


def read_lines(path: Path) -> list[str]:
    values = [x.strip() for x in require_file(path).read_text().splitlines()]
    if not values or any(not x for x in values):
        raise ValueError(f"Empty or blank entry in {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate entry in {path}")
    return values


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{x}\n" for x in values), encoding="utf-8")


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(block):
            digest.update(data)
    return digest.hexdigest()


def stable_seed(token: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{token}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def task_index(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if value is None:
        raise ValueError("Pass --task-index or run as a SLURM array.")
    return int(value)


def freeze(args: argparse.Namespace) -> None:
    source = require_dir(args.source_panel)
    output = args.panel_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.allow_existing:
        raise FileExistsError(f"Non-empty frozen panel exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    retained = read_lines(source / "retained_embeddings.txt")
    excluded = read_lines(source / "excluded_embeddings.txt")
    genes = read_lines(source / "common_genes.txt")
    if len(retained) != EXPECTED_EMBEDDINGS:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDINGS} retained embeddings, found {len(retained)}"
        )
    if len(genes) != EXPECTED_GENES:
        raise ValueError(f"Expected {EXPECTED_GENES:,} common genes, found {len(genes)}")
    if set(retained) & set(excluded):
        raise ValueError("Retained and excluded lists overlap.")
    if sha256_file(source / "retained_embeddings.txt") != EXPECTED_RETAINED_FILE_SHA256:
        raise ValueError("Frozen retained-embedding order/hash changed")
    if sha256_file(source / "common_genes.txt") != EXPECTED_GENE_FILE_SHA256:
        raise ValueError("Frozen common-gene set/hash changed")

    write_lines(output / "retained_embeddings.txt", retained)
    write_lines(output / "excluded_embeddings.txt", excluded)
    write_lines(output / "common_genes.txt", genes)

    metadata_path = (
        require_file(args.metadata)
        if args.metadata is not None
        else require_file(source / "retained_embeddings.tsv")
    )
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    required_metadata = {"embedding_id", "display_name", "modality"}
    if not required_metadata.issubset(metadata.columns):
        raise ValueError(
            f"{metadata_path} lacks columns "
            f"{sorted(required_metadata - set(metadata.columns))}"
        )
    metadata = metadata.set_index("embedding_id", drop=False)
    missing = sorted(set(retained) - set(metadata.index.astype(str)))
    if missing:
        raise ValueError(f"Panel metadata is missing embeddings: {missing}")
    metadata = metadata.loc[retained].reset_index(drop=True)
    if metadata["display_name"].duplicated().any():
        raise ValueError("Panel metadata contains duplicate display names")
    atomic_dataframe(output / "embedding_metadata.tsv", metadata)

    rows: list[dict[str, object]] = []
    for index, (x, y) in enumerate(itertools.combinations(retained, 2)):
        rows.append(
            {
                "pair_index": index,
                "pair_id": f"{x}__VS__{y}",
                "embedding_x": x,
                "embedding_y": y,
            }
        )
    pairs = pd.DataFrame(rows)
    if len(pairs) != EXPECTED_PAIRS:
        raise AssertionError(len(pairs))
    atomic_dataframe(output / "pair_manifest.tsv", pairs)

    metadata_payload = {
        "panel_name": PANEL_NAME,
        "created_utc": utc_now(),
        "source_panel": str(source),
        "embedding_count": len(retained),
        "gene_count": len(genes),
        "pair_count": len(pairs),
        "retained_sha256": sha256_file(output / "retained_embeddings.txt"),
        "excluded_sha256": sha256_file(output / "excluded_embeddings.txt"),
        "genes_sha256": sha256_file(output / "common_genes.txt"),
        "metadata_sha256": sha256_file(output / "embedding_metadata.tsv"),
    }
    atomic_json(output / "panel_metadata.json", metadata_payload)
    print(json.dumps(metadata_payload, indent=2))


def read_embedding_matrix(
    embeddings_root: Path,
    embedding_id: str,
    common_genes: list[str],
    gene_column: str,
    matrix_key: str,
) -> np.ndarray:
    package = require_dir(embeddings_root / embedding_id)
    genes = pd.read_csv(
        require_file(package / "genes.tsv"), sep="\t", dtype=str
    )
    if gene_column not in genes:
        raise ValueError(f"{package}/genes.tsv lacks {gene_column}")
    ids = genes[gene_column].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate genes in {embedding_id}")
    positions = {gene: index for index, gene in enumerate(ids)}
    missing = [gene for gene in common_genes if gene not in positions]
    if missing:
        raise ValueError(f"{embedding_id} lacks {len(missing)} frozen genes")
    with np.load(require_file(package / "embeddings.npz"), allow_pickle=False) as z:
        if matrix_key not in z.files:
            raise KeyError(f"{embedding_id}: keys={z.files}")
        source = z[matrix_key]
        if source.ndim != 2 or source.shape[0] != len(ids):
            raise ValueError(f"Invalid matrix shape for {embedding_id}: {source.shape}")
        matrix = np.asarray(
            source[np.asarray([positions[g] for g in common_genes])],
            dtype=np.float32,
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Nonfinite frozen rows in {embedding_id}")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise ValueError(f"Zero frozen rows in {embedding_id}")
    return matrix


def frobenius_cross_product_sq(
    x: np.ndarray, y: np.ndarray, feature_block: int
) -> float:
    total = 0.0
    xt = x.T
    for start in range(0, y.shape[1], feature_block):
        stop = min(start + feature_block, y.shape[1])
        cross = xt @ y[:, start:stop]
        total += float(np.einsum("ij,ij->", cross, cross, dtype=np.float64))
    return total


def self_terms(x: np.ndarray, feature_block: int) -> dict[str, np.ndarray | float]:
    q = np.einsum("ij,ij->i", x, x, dtype=np.float64)
    total_vector = x.sum(axis=0, dtype=np.float64)
    row = x @ total_vector - q
    total = float(row.sum(dtype=np.float64))
    gram_sq_offdiag = (
        frobenius_cross_product_sq(x, x, feature_block)
        - float(np.dot(q, q))
    )
    n = x.shape[0]
    hsic = (
        gram_sq_offdiag
        + total * total / ((n - 1) * (n - 2))
        - 2.0 * float(np.dot(row, row)) / (n - 2)
    ) / (n * (n - 3))
    if not np.isfinite(hsic) or hsic <= 0:
        raise ValueError(f"Invalid unbiased self-HSIC: {hsic}")
    return {"q": q, "row": row, "total": total, "self_hsic": hsic}


def prepare(args: argparse.Namespace) -> None:
    panel = require_dir(args.panel_dir)
    embeddings = read_lines(panel / "retained_embeddings.txt")
    genes = read_lines(panel / "common_genes.txt")
    index = task_index(args.task_index)
    if not 0 <= index < len(embeddings):
        raise IndexError(index)
    embedding = embeddings[index]
    output = args.cache_dir.expanduser().resolve() / "embeddings" / embedding
    output.mkdir(parents=True, exist_ok=True)
    expected = [
        output / "raw.npy",
        output / "row_normalized.npy",
        output / "raw_stats.npz",
        output / "row_normalized_stats.npz",
        output / "metadata.json",
    ]
    if not args.force and all(path.is_file() for path in expected):
        print(f"SKIP PREPARED {index}: {embedding}")
        return

    matrix = read_embedding_matrix(
        resolve_embeddings_root(args.embeddings_root, embeddings),
        embedding,
        genes,
        args.gene_column,
        args.matrix_key,
    )
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    normalized = (matrix / norms[:, None]).astype(np.float32)

    for name, values in (("raw", matrix), ("row_normalized", normalized)):
        matrix_path = output / f"{name}.npy"
        temporary = matrix_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(matrix_path)
        terms = self_terms(values.astype(np.float64), args.feature_block)
        stats_path = output / f"{name}_stats.npz"
        temporary_stats = stats_path.with_suffix(".npz.tmp")
        with temporary_stats.open("wb") as handle:
            np.savez_compressed(handle, **terms)
        temporary_stats.replace(stats_path)

    atomic_json(
        output / "metadata.json",
        {
            "embedding_id": embedding,
            "embedding_index": index,
            "genes": len(genes),
            "dimensions": int(matrix.shape[1]),
            "created_utc": utc_now(),
        },
    )
    print(f"PREPARED {index}: {embedding} {matrix.shape}")


def build_tie_aware_graphs(
    x: np.ndarray,
    k_values: tuple[int, ...],
    row_block: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[int, sparse.csr_matrix]:
    n = x.shape[0]
    if max(k_values) >= n:
        raise ValueError("k must be smaller than n")
    rows: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    cols: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    data: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    xt = x.T

    for start in range(0, n, row_block):
        stop = min(start + row_block, n)
        similarity = x[start:stop] @ xt
        similarity[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        for local_row, global_row in enumerate(range(start, stop)):
            values = similarity[local_row]
            for k in k_values:
                threshold = float(np.partition(values, -k)[-k])
                tolerance = absolute_tolerance + relative_tolerance * abs(threshold)
                strict = np.flatnonzero(values > threshold + tolerance)
                ties = np.flatnonzero(np.abs(values - threshold) <= tolerance)
                remaining = k - len(strict)
                if remaining < 0 or not len(ties):
                    raise RuntimeError(
                        f"Invalid boundary at row={global_row}, k={k}, "
                        f"strict={len(strict)}, ties={len(ties)}"
                    )
                indices = np.concatenate((strict, ties))
                weights = np.concatenate(
                    (
                        np.ones(len(strict), dtype=np.float32),
                        np.full(
                            len(ties),
                            remaining / len(ties),
                            dtype=np.float32,
                        ),
                    )
                )
                keep = weights > 0
                indices = indices[keep]
                weights = weights[keep]
                rows[k].append(
                    np.full(len(indices), global_row, dtype=np.int32)
                )
                cols[k].append(indices.astype(np.int32, copy=False))
                data[k].append(weights)
        print(f"  neighborhood rows {start:,}:{stop:,}/{n:,}", flush=True)

    result: dict[int, sparse.csr_matrix] = {}
    for k in k_values:
        row = np.concatenate(rows[k])
        col = np.concatenate(cols[k])
        values = np.concatenate(data[k])
        graph = sparse.csr_matrix((values, (row, col)), shape=(n, n))
        row_sums = np.asarray(graph.sum(axis=1)).ravel()
        if not np.allclose(row_sums, k, atol=2e-5, rtol=2e-6):
            raise RuntimeError(f"Tie-aware graph at k={k} does not have mass k")
        result[k] = graph
    return result


def neighbors(args: argparse.Namespace) -> None:
    panel = require_dir(args.panel_dir)
    embeddings = read_lines(panel / "retained_embeddings.txt")
    index = task_index(args.task_index)
    embedding = embeddings[index]
    cache = args.cache_dir.expanduser().resolve()
    output = cache / "neighbors" / embedding
    expected = [output / f"k{k}.npz" for k in args.k_values]
    expected.append(output / "metadata.json")
    if not args.force and all(path.is_file() for path in expected):
        print(f"SKIP NEIGHBORS {index}: {embedding}")
        return
    matrix = np.load(
        require_file(cache / "embeddings" / embedding / "row_normalized.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    graphs = build_tie_aware_graphs(
        np.asarray(matrix),
        tuple(args.k_values),
        args.row_block,
        args.tie_absolute_tolerance,
        args.tie_relative_tolerance,
    )
    output.mkdir(parents=True, exist_ok=True)
    for k, graph in graphs.items():
        sparse.save_npz(output / f"k{k}.npz", graph, compressed=True)
    atomic_json(
        output / "metadata.json",
        {
            "embedding_id": embedding,
            "embedding_index": index,
            "gene_count": int(matrix.shape[0]),
            "k_values": list(args.k_values),
            "created_utc": utc_now(),
        },
    )
    print(f"NEIGHBORS {index}: {embedding}")


def load_stats(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(require_file(path), allow_pickle=False) as z:
        return {
            "q": z["q"],
            "row": z["row"],
            "total": float(z["total"]),
            "self_hsic": float(z["self_hsic"]),
        }


def unbiased_linear_cka(
    x: np.ndarray,
    y: np.ndarray,
    sx: dict[str, np.ndarray | float],
    sy: dict[str, np.ndarray | float],
    feature_block: int,
    permutation: np.ndarray | None = None,
) -> float:
    if permutation is None:
        yp = y
        qy = np.asarray(sy["q"])
        rowy = np.asarray(sy["row"])
    else:
        yp = y[permutation]
        qy = np.asarray(sy["q"])[permutation]
        rowy = np.asarray(sy["row"])[permutation]
    qx = np.asarray(sx["q"])
    rowx = np.asarray(sx["row"])
    cross_offdiag = (
        frobenius_cross_product_sq(x, yp, feature_block)
        - float(np.dot(qx, qy))
    )
    n = x.shape[0]
    hsic = (
        cross_offdiag
        + float(sx["total"]) * float(sy["total"]) / ((n - 1) * (n - 2))
        - 2.0 * float(np.dot(rowx, rowy)) / (n - 2)
    ) / (n * (n - 3))
    return hsic / math.sqrt(float(sx["self_hsic"]) * float(sy["self_hsic"]))


def local_overlap(
    gx: sparse.csr_matrix,
    gy: sparse.csr_matrix,
    k: int,
    permutation: np.ndarray | None = None,
) -> float:
    if permutation is not None:
        gy = gy[permutation][:, permutation]
    return float(gx.minimum(gy).sum(dtype=np.float64) / (gx.shape[0] * k))


def pair(args: argparse.Namespace) -> None:
    panel = require_dir(args.panel_dir)
    manifest = pd.read_csv(require_file(panel / "pair_manifest.tsv"), sep="\t")
    index = task_index(args.task_index)
    if not 0 <= index < len(manifest):
        raise IndexError(index)
    record = manifest.iloc[index]
    x_id, y_id = str(record.embedding_x), str(record.embedding_y)
    cache = args.cache_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() / "pairs"
    result_path = output / f"{index:04d}.tsv"
    if result_path.is_file() and not args.force:
        print(f"SKIP PAIR {index}: {x_id} vs {y_id}")
        return

    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    stats: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for mode in ("raw", "row_normalized"):
        x = np.load(
            require_file(cache / "embeddings" / x_id / f"{mode}.npy"),
            mmap_mode="r",
            allow_pickle=False,
        )
        y = np.load(
            require_file(cache / "embeddings" / y_id / f"{mode}.npy"),
            mmap_mode="r",
            allow_pickle=False,
        )
        matrices[mode] = (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        stats[mode] = (
            load_stats(cache / "embeddings" / x_id / f"{mode}_stats.npz"),
            load_stats(cache / "embeddings" / y_id / f"{mode}_stats.npz"),
        )

    raw = unbiased_linear_cka(
        *matrices["raw"], *stats["raw"], args.feature_block
    )
    normalized = unbiased_linear_cka(
        *matrices["row_normalized"],
        *stats["row_normalized"],
        args.feature_block,
    )

    graphs: dict[int, tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
    local_observed: dict[int, float] = {}
    for k in args.k_values:
        gx = sparse.load_npz(require_file(cache / "neighbors" / x_id / f"k{k}.npz"))
        gy = sparse.load_npz(require_file(cache / "neighbors" / y_id / f"k{k}.npz"))
        graphs[k] = (gx, gy)
        local_observed[k] = local_overlap(gx, gy, k)

    rng = np.random.default_rng(stable_seed(str(record.pair_id)))
    null_linear: list[float] = []
    null_local: dict[int, list[float]] = {k: [] for k in args.k_values}
    n = matrices["row_normalized"][0].shape[0]
    for repetition in range(args.null_shuffles):
        permutation = rng.permutation(n)
        null_linear.append(
            unbiased_linear_cka(
                *matrices["row_normalized"],
                *stats["row_normalized"],
                args.feature_block,
                permutation=permutation,
            )
        )
        for k in args.k_values:
            null_local[k].append(
                local_overlap(*graphs[k], k, permutation=permutation)
            )
        print(
            f"  pair {index} null {repetition + 1}/{args.null_shuffles}",
            flush=True,
        )

    result: dict[str, object] = {
        **record.to_dict(),
        "gene_count": n,
        "null_shuffles": args.null_shuffles,
        "unbiased_linear_cka_raw": raw,
        "unbiased_linear_cka_row_normalized": normalized,
    }
    for prefix, observed, null in [
        ("linear", normalized, null_linear),
        *[
            (f"local_k{k}", local_observed[k], null_local[k])
            for k in args.k_values
        ],
    ]:
        values = np.asarray(null, dtype=float)
        mean = float(values.mean()) if len(values) else np.nan
        sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        result[f"{prefix}_observed"] = observed
        result[f"{prefix}_null_mean"] = mean
        result[f"{prefix}_null_sd"] = sd
        result[f"{prefix}_observed_minus_null"] = observed - mean
        result[f"{prefix}_observed_gt_all_null"] = bool(
            len(values) and observed > float(values.max())
        )
        result[f"{prefix}_descriptive_empirical_p"] = (
            (1 + int(np.sum(values >= observed))) / (len(values) + 1)
            if len(values)
            else np.nan
        )
        result[f"{prefix}_null_values"] = ";".join(f"{x:.12g}" for x in values)

    atomic_dataframe(result_path, pd.DataFrame([result]))
    print(f"PAIR {index}/{len(manifest) - 1}: {x_id} vs {y_id}")


def matrix_from_pairs(
    frame: pd.DataFrame, embeddings: list[str], column: str
) -> pd.DataFrame:
    values = np.eye(len(embeddings), dtype=float)
    positions = {name: i for i, name in enumerate(embeddings)}
    for row in frame.itertuples(index=False):
        i, j = positions[row.embedding_x], positions[row.embedding_y]
        values[i, j] = values[j, i] = float(getattr(row, column))
    return pd.DataFrame(values, index=embeddings, columns=embeddings)


def gower_eigendecomposition(
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(distance)
    center = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * center @ (distance**2) @ center
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def triangle_violation_fraction(distance: np.ndarray) -> float:
    violations = 0
    total = 0
    for i, j, k in itertools.combinations(range(len(distance)), 3):
        for direct, side1, side2 in (
            (distance[i, j], distance[i, k], distance[k, j]),
            (distance[i, k], distance[i, j], distance[j, k]),
            (distance[j, k], distance[j, i], distance[i, k]),
        ):
            total += 1
            violations += int(direct > side1 + side2 + 1e-10)
    return violations / total if total else 0.0


def pcoa(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    raw_eigenvalues, _ = gower_eigendecomposition(distance)
    raw_scale = max(1.0, float(np.max(np.abs(raw_eigenvalues))))
    apply_lingoes = float(raw_eigenvalues.min()) < -1e-10 * raw_scale
    lingoes_constant = max(0.0, -float(raw_eigenvalues.min()))
    if apply_lingoes:
        corrected_squared = distance**2 + 2.0 * lingoes_constant
        np.fill_diagonal(corrected_squared, 0.0)
        used_distance = np.sqrt(np.maximum(corrected_squared, 0.0))
    else:
        used_distance = distance
        lingoes_constant = 0.0
    eigenvalues, eigenvectors = gower_eigendecomposition(used_distance)
    positive = eigenvalues > max(1.0, np.max(np.abs(eigenvalues))) * 1e-10
    negative_mass = float(np.abs(eigenvalues[eigenvalues < 0]).sum())
    positive_mass = float(eigenvalues[eigenvalues > 0].sum())
    coordinates = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    explained = eigenvalues[positive] / eigenvalues[positive].sum()
    return coordinates, explained, {
        "raw_minimum_eigenvalue": float(raw_eigenvalues.min()),
        "raw_triangle_violation_fraction": triangle_violation_fraction(distance),
        "lingoes_applied": bool(apply_lingoes),
        "lingoes_constant": lingoes_constant,
        "used_minimum_eigenvalue": float(eigenvalues.min()),
        "negative_to_positive_mass_ratio": (
            negative_mass / positive_mass if positive_mass else np.nan
        ),
    }


def resolve_heatmap_order(embedding_ids: Iterable[str]) -> list[str]:
    """Use the frozen manifest order, validating uniqueness."""
    order = list(embedding_ids)
    if len(order) != len(set(order)):
        raise ValueError("Heatmap order contains duplicate embedding IDs")
    return order


def render_heatmap(
    similarity: pd.DataFrame,
    name: str,
    kind: str,
    heatmap_order: list[str],
    display_map: dict[str, str],
    plot_dir: Path,
    dpi: int,
) -> None:
    """Render one heatmap using the registered common order and color scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ordered = similarity.loc[heatmap_order, heatmap_order]
    if kind == "global":
        title = (
            "Absolute global representation alignment\n"
            + (
                "Row-normalized unbiased linear CKA"
                if name == "linear_cka"
                else "Raw unbiased linear CKA sensitivity analysis"
            )
        )
        colorbar_label = "Unbiased linear CKA (raw value)"
    else:
        local_label = (
            "Mean tie-aware overlap across k=10, 25, 50 and 100"
            if name == "local_mean"
            else f"Tie-aware neighbourhood overlap at k={name.removeprefix('local_k')}"
        )
        title = f"Absolute local gene-neighbourhood alignment\n{local_label}"
        colorbar_label = "Tie-aware overlap (raw proportion)"

    # Apply the non-linear mapping explicitly. This avoids backend/version
    # differences in how Seaborn forwards matplotlib normalization objects.
    raw_values = np.clip(ordered.to_numpy(dtype=float), 0.0, 1.0)
    display_values = np.power(raw_values, HEATMAP_GAMMA)
    display_matrix = pd.DataFrame(
        display_values,
        index=ordered.index,
        columns=ordered.columns,
    )
    display_tick_positions = np.power(
        np.asarray(HEATMAP_RAW_TICKS, dtype=float),
        HEATMAP_GAMMA,
    )

    size = max(14, 0.28 * len(heatmap_order))
    fig, ax = plt.subplots(figsize=(size, size))
    heatmap = sns.heatmap(
        display_matrix,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        square=True,
        xticklabels=[display_map[value] for value in heatmap_order],
        yticklabels=[display_map[value] for value in heatmap_order],
        cbar_kws={
            "label": (
                f"{colorbar_label}; displayed color = sqrt(raw value)"
            ),
            "ticks": display_tick_positions,
        },
        ax=ax,
    )
    ax.set_title(
        (
            f"{title} · shared registered reference order · "
            "nonlinear square-root color scale"
        ),
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Embedding")
    ax.set_ylabel("Embedding")
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
        va="top",
        fontsize=6.5,
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6.5)
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticklabels([f"{value:g}" for value in HEATMAP_RAW_TICKS])
    colorbar.ax.tick_params(labelsize=8)
    fig.text(
        0.5,
        0.01,
        (
            "Colors equal sqrt(raw value): raw 0.20 maps to color position "
            "0.447 (blue-green). Saved matrices and colorbar labels are raw."
        ),
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(plot_dir / f"heatmap__{name}.png", dpi=dpi)
    # Publication PDFs are deliberately cleaner and use slightly larger text.
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    colorbar.set_label(colorbar_label, fontsize=10)
    fig.texts.clear()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7.5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(plot_dir / f"heatmap__{name}.pdf")
    plt.close(fig)


def write_heatmap_audit(plot_dir: Path, heatmap_order: list[str]) -> None:
    """Write machine-readable proof of the shared order and nonlinear mapping."""
    write_lines(
        plot_dir / "heatmap_embedding_order.txt",
        heatmap_order,
    )
    mapping = pd.DataFrame(
        {
            "raw_value": HEATMAP_RAW_TICKS,
            "display_color_position": np.power(
                np.asarray(HEATMAP_RAW_TICKS, dtype=float),
                HEATMAP_GAMMA,
            ),
        }
    )
    atomic_dataframe(plot_dir / "heatmap_color_mapping.tsv", mapping)
    (plot_dir / "heatmap_settings.json").write_text(
        json.dumps(
            {
                "embedding_order": "frozen manifest order; identical in every heatmap",
                "color_mapping": "display_color_value = sqrt(clipped_raw_value)",
                "gamma": HEATMAP_GAMMA,
                "raw_0.2_display_position": float(0.2 ** HEATMAP_GAMMA),
                "colormap": "viridis",
                "raw_values_modified": False,
                "heatmap_files_overwritten_in_place": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def replot_heatmaps(args: argparse.Namespace) -> None:
    """Overwrite the seven heatmaps directly from the saved matrix TSVs."""
    output_dir = require_dir(args.output_dir)
    matrix_dir = require_dir(output_dir / "matrices")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    definitions = {
        "linear_cka": "global",
        "linear_cka_raw": "global",
        **{f"local_k{k}": "local" for k in K_VALUES},
        "local_mean": "local",
    }
    matrices: dict[str, pd.DataFrame] = {}
    reference_set: set[str] | None = None
    for name in definitions:
        matrix = pd.read_csv(
            require_file(matrix_dir / f"{name}.tsv"),
            sep="\t",
            index_col=0,
        )
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name}: matrix is not square: {matrix.shape}")
        if set(matrix.index) != set(matrix.columns):
            raise ValueError(f"{name}: row and column identifiers differ")
        current_set = set(matrix.index)
        if reference_set is None:
            reference_set = current_set
        elif current_set != reference_set:
            raise ValueError(f"{name}: embedding set differs from other matrices")
        values = matrix.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: non-finite values")
        if not np.allclose(values, values.T, rtol=0.0, atol=1e-8):
            raise ValueError(f"{name}: matrix is not symmetric")
        matrices[name] = matrix

    if reference_set is None:
        raise ValueError("No heatmap matrices found")
    panel = require_dir(args.panel_dir)
    metadata = pd.read_csv(require_file(panel / "embedding_metadata.tsv"), sep="\t")
    heatmap_order = resolve_heatmap_order(metadata["embedding_id"].astype(str).tolist())
    if set(heatmap_order) != reference_set:
        raise ValueError("Saved matrix IDs differ from frozen panel metadata")
    display_map = dict(
        zip(metadata["embedding_id"].astype(str), metadata["display_name"].astype(str))
    )
    for name, kind in definitions.items():
        render_heatmap(
            similarity=matrices[name],
            name=name,
            kind=kind,
            heatmap_order=heatmap_order,
            display_map=display_map,
            plot_dir=plot_dir,
            dpi=args.dpi,
        )
        print(f"OVERWROTE {plot_dir / f'heatmap__{name}.png'}", flush=True)
    write_heatmap_audit(plot_dir, heatmap_order)
    checksums = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(
                f"{sha256_file(path)}  {path.relative_to(output_dir)}"
            )
    (output_dir / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    print(
        "HEATMAP REPLOT COMPLETE: 7 files; shared registered reference order; "
        "display=sqrt(raw); raw 0.20 -> color position "
        f"{0.2 ** HEATMAP_GAMMA:.3f}"
    )


def aggregate(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel = require_dir(args.panel_dir)
    embeddings = read_lines(panel / "retained_embeddings.txt")
    if len(embeddings) != EXPECTED_EMBEDDINGS:
        raise ValueError(f"Expected {EXPECTED_EMBEDDINGS} embeddings, found {len(embeddings)}")
    manifest = pd.read_csv(require_file(panel / "pair_manifest.tsv"), sep="\t")
    pair_dir = require_dir(args.output_dir / "pairs")
    paths = [pair_dir / f"{i:04d}.tsv" for i in range(len(manifest))]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} pair outputs missing; first={missing[0]}"
        )
    pairs = pd.concat([pd.read_csv(path, sep="\t") for path in paths], ignore_index=True)
    expected_pairs = len(embeddings) * (len(embeddings) - 1) // 2
    if expected_pairs != EXPECTED_PAIRS:
        raise ValueError(f"Frozen panel implies {expected_pairs} rather than {EXPECTED_PAIRS} pairs")
    if len(pairs) != expected_pairs or pairs["pair_id"].duplicated().any():
        raise ValueError("Invalid aggregated pair results.")

    metadata = pd.read_csv(panel / "embedding_metadata.tsv", sep="\t")
    metadata = metadata.set_index("embedding_id").loc[embeddings].reset_index()
    display_map = dict(
        zip(metadata["embedding_id"].astype(str), metadata["display_name"].astype(str))
    )
    for side in ("x", "y"):
        renamed = metadata.rename(
            columns={
                column: f"{column}_{side}"
                for column in metadata.columns
                if column != "embedding_id"
            }
        ).rename(columns={"embedding_id": f"embedding_{side}"})
        pairs = pairs.merge(renamed, on=f"embedding_{side}", how="left", validate="many_to_one")
    atomic_dataframe(args.output_dir / "atlas_pair_scores.tsv", pairs)

    matrix_definitions = {
        "linear_cka": ("unbiased_linear_cka_row_normalized", "global"),
        "linear_cka_raw": ("unbiased_linear_cka_raw", "global"),
        **{f"local_k{k}": (f"local_k{k}_observed", "local") for k in K_VALUES},
    }
    pairs["local_mean_observed"] = pairs[
        [f"local_k{k}_observed" for k in K_VALUES]
    ].mean(axis=1)
    matrix_definitions["local_mean"] = ("local_mean_observed", "local")
    matrix_dir = args.output_dir / "matrices"
    plot_dir = args.output_dir / "plots"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    diagnostics: list[dict[str, object]] = []
    heatmap_order = resolve_heatmap_order(embeddings)
    for name, (column, kind) in matrix_definitions.items():
        similarity = matrix_from_pairs(pairs, embeddings, column)
        similarity.rename_axis("embedding_id").to_csv(
            matrix_dir / f"{name}.tsv", sep="\t"
        )
        similarity_values = similarity.to_numpy(dtype=float, copy=True)
        if kind == "global":
            distance = np.sqrt(
                np.maximum(
                    0.0,
                    2.0 * (1.0 - np.clip(similarity_values, -1.0, 1.0)),
                )
            )
        else:
            distance = 1.0 - np.clip(similarity_values, 0.0, 1.0)
        distance = np.array(distance, dtype=float, copy=True)
        np.fill_diagonal(distance, 0.0)

        render_heatmap(
            similarity=similarity,
            name=name,
            kind=kind,
            heatmap_order=heatmap_order,
            display_map=display_map,
            plot_dir=plot_dir,
            dpi=args.dpi,
        )

        coordinates, explained, diag = pcoa(distance)
        coordinate_frame = metadata.copy()
        for axis in range(coordinates.shape[1]):
            coordinate_frame[f"PCoA{axis + 1}"] = coordinates[:, axis]
        coordinate_frame.to_csv(
            args.output_dir / f"pcoa_coordinates__{name}.tsv",
            sep="\t",
            index=False,
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        categories = coordinate_frame.get(
            "modality",
            pd.Series(["unknown"] * len(coordinate_frame)),
        ).astype(str)
        codes, uniques = pd.factorize(categories)
        scatter = ax.scatter(
            coordinate_frame["PCoA1"],
            coordinate_frame["PCoA2"],
            c=codes,
            cmap="tab20",
            s=52,
            alpha=0.85,
        )
        ax.set_xlabel(f"PCoA1 ({100 * explained[0]:.1f}%)")
        ax.set_ylabel(
            f"PCoA2 ({100 * explained[1]:.1f}%)"
            if len(explained) > 1
            else "PCoA2"
        )
        ax.set_title(f"{PANEL_NAME}: {name}")
        handles = [
            plt.Line2D(
                [], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)),
                label=label,
            )
            for i, label in enumerate(uniques)
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / f"pcoa__{name}.png", dpi=args.dpi)
        ax.set_title("")
        fig.savefig(plot_dir / f"pcoa__{name}.pdf")
        plt.close(fig)
        diagnostics.append(
            {
                "metric": name,
                "axis1_percent": 100 * explained[0],
                "axis2_percent": 100 * explained[1] if len(explained) > 1 else np.nan,
                **diag,
            }
        )
    write_heatmap_audit(plot_dir, heatmap_order)

    validation_rows: list[dict[str, object]] = []
    for prefix in ["linear", *[f"local_k{k}" for k in K_VALUES]]:
        observed = pairs[f"{prefix}_observed"].to_numpy(float)
        null = pairs[f"{prefix}_null_mean"].to_numpy(float)
        delta = observed - null
        validation_rows.append(
            {
                "metric": prefix,
                "pair_count": len(pairs),
                "median_observed": float(np.median(observed)),
                "median_null_mean": float(np.median(null)),
                "median_observed_minus_null": float(np.median(delta)),
                "fraction_observed_above_null_mean": float(np.mean(delta > 0)),
                "fraction_observed_above_all_five_nulls": float(
                    pairs[f"{prefix}_observed_gt_all_null"].mean()
                ),
                "warning": (
                    "Five shuffles are an effect-size negative control, not "
                    "fine-resolution pairwise significance testing."
                ),
            }
        )
    atomic_dataframe(
        args.output_dir / "negative_control_summary.tsv",
        pd.DataFrame(validation_rows),
    )

    family_x = pairs.get("family_x", pd.Series(["unknown"] * len(pairs))).fillna("unknown")
    family_y = pairs.get("family_y", pd.Series(["unknown"] * len(pairs))).fillna("unknown")
    prov_x = pairs.get("provenance_group_x", pd.Series(["unknown"] * len(pairs))).fillna("unknown")
    prov_y = pairs.get("provenance_group_y", pd.Series(["unknown"] * len(pairs))).fillna("unknown")
    pairs["control_tier"] = np.where(
        family_x.eq(family_y) & family_x.ne("unknown"),
        "same_family",
        np.where(
            prov_x.eq(prov_y) & prov_x.ne("unknown"),
            "same_provenance",
            "independent_source",
        ),
    )
    control = (
        pairs.groupby("control_tier", dropna=False)
        .agg(
            pair_count=("pair_id", "size"),
            median_linear=("linear_observed", "median"),
            median_linear_minus_null=("linear_observed_minus_null", "median"),
            median_local_k50=("local_k50_observed", "median"),
            median_local_k50_minus_null=("local_k50_observed_minus_null", "median"),
        )
        .reset_index()
    )
    atomic_dataframe(args.output_dir / "positive_control_summary.tsv", control)
    atomic_dataframe(
        args.output_dir / "pcoa_diagnostics.tsv", pd.DataFrame(diagnostics)
    )

    checksums = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(
                f"{sha256_file(path)}  {path.relative_to(args.output_dir)}"
            )
    (args.output_dir / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n"
    )
    report = [
        f"# {PANEL_NAME}",
        "",
        f"- Embeddings: **{len(embeddings)}**",
        f"- Common genes: **{EXPECTED_GENES:,}**",
        f"- Pairs: **{len(pairs):,}**",
        "- Primary global metric: row-normalized unbiased linear CKA.",
        "- Primary local metric: exact tie-aware overlap at k=50.",
        f"- Negative controls: **{int(pairs['null_shuffles'].iloc[0])}** "
        "deterministic gene-identity shuffles per pair.",
        "",
        "The five-shuffle analysis tests effect separation from shuffled identity "
        "at the atlas level. It must not be presented as high-resolution "
        "pairwise hypothesis testing; selected independent pairs require the "
        "registered repeated-sample, 999-permutation confirmation stage.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(report))
    print(pd.DataFrame(validation_rows).to_string(index=False))
    print(f"AGGREGATED: {args.output_dir.resolve()}")


def self_test(_: argparse.Namespace) -> None:
    rng = np.random.default_rng(7)
    for n, dx, dy in ((20, 5, 7), (31, 12, 3)):
        x = rng.normal(size=(n, dx))
        y = rng.normal(size=(n, dy))
        sx = self_terms(x, 4)
        sy = self_terms(y, 4)
        observed = unbiased_linear_cka(x, y, sx, sy, 4)
        kx, ky = x @ x.T, y @ y.T
        np.fill_diagonal(kx, 0.0)
        np.fill_diagonal(ky, 0.0)

        def dense_hsic(a: np.ndarray, b: np.ndarray) -> float:
            return (
                float(np.sum(a * b))
                + float(a.sum()) * float(b.sum()) / ((n - 1) * (n - 2))
                - 2.0 * float(np.dot(a.sum(axis=1), b.sum(axis=1))) / (n - 2)
            ) / (n * (n - 3))

        expected = dense_hsic(kx, ky) / math.sqrt(
            dense_hsic(kx, kx) * dense_hsic(ky, ky)
        )
        if not np.allclose(observed, expected, rtol=1e-10, atol=1e-12):
            raise AssertionError((observed, expected))

    x = rng.normal(size=(50, 8))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    graphs = build_tie_aware_graphs(x, (5, 10), 16, 1e-12, 1e-10)
    for k, graph in graphs.items():
        if not np.isclose(local_overlap(graph, graph, k), 1.0):
            raise AssertionError(f"Self-overlap failed for k={k}")
    print("SELF-TEST PASS: exact unbiased CKA and tie-aware local overlap")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("self-test")
    p.set_defaults(function=self_test)

    p = sub.add_parser("freeze")
    p.add_argument("--source-panel", type=Path, required=True)
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Retained-panel TSV with embedding_id, display_name and modality.",
    )
    p.add_argument("--panel-dir", type=Path, required=True)
    p.add_argument("--allow-existing", action="store_true")
    p.set_defaults(function=freeze)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--panel-dir", type=Path, required=True)
    common.add_argument("--cache-dir", type=Path, required=True)
    common.add_argument("--task-index", type=int)

    p = sub.add_parser("prepare", parents=[common])
    p.add_argument("--embeddings-root", type=Path, required=True)
    p.add_argument("--gene-column", default="ensembl_gene_id")
    p.add_argument("--matrix-key", default="embeddings")
    p.add_argument("--feature-block", type=int, default=256)
    p.add_argument("--force", action="store_true")
    p.set_defaults(function=prepare)

    p = sub.add_parser("neighbors", parents=[common])
    p.add_argument("--k-values", type=int, nargs="+", default=list(K_VALUES))
    p.add_argument("--row-block", type=int, default=128)
    p.add_argument("--tie-absolute-tolerance", type=float, default=1e-12)
    p.add_argument("--tie-relative-tolerance", type=float, default=1e-10)
    p.add_argument("--force", action="store_true")
    p.set_defaults(function=neighbors)

    p = sub.add_parser("pair", parents=[common])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--null-shuffles", type=int, default=5)
    p.add_argument("--k-values", type=int, nargs="+", default=list(K_VALUES))
    p.add_argument("--feature-block", type=int, default=256)
    p.add_argument("--force", action="store_true")
    p.set_defaults(function=pair)

    p = sub.add_parser("aggregate")
    p.add_argument("--panel-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.set_defaults(function=aggregate)

    p = sub.add_parser(
        "replot-heatmaps",
        help=(
            "Overwrite the seven heatmap PNG/PDF pairs from existing matrices using "
            "one registered reference order and a square-root color scale."
        ),
    )
    p.add_argument("--panel-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.set_defaults(function=replot_heatmaps)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
