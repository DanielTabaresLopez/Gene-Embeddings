#!/usr/bin/env python3
"""Download and locally process the official DRKG TransE entity embedding.

Required packages:
    python -m pip install numpy pandas

Only numeric ``Gene::<Entrez ID>`` entities are mapped. This script does not
authenticate to Hugging Face and does not upload anything.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

EMBEDDING_NAME = "drkg_transe"
DISPLAY_NAME = "DRKG TransE"
DIMENSION = 400

SOURCE_ARCHIVE_URL = (
    "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"
)
SOURCE_ARCHIVE_SIZE = 216_650_245
SOURCE_ARCHIVE_SHA256 = (
    "00a8154a35bda496d0fdfe0d76b058bf4bd3dd11f69d0ec00c3559523608b930"
)
SOURCE_MATRIX_MEMBER = "embed/DRKG_TransE_l2_entity.npy"
SOURCE_MATRIX_SHA256 = (
    "2b6ed2683b960391c08c27b46a35c641918ad224e5f0c85e7a8b9d14d3f34c17"
)
SOURCE_ENTITIES_MEMBER = "embed/entities.tsv"
SOURCE_ENTITIES_SHA256 = (
    "0464735a7f1013d5c0beb02ed88bbf5e6702e3ba73919c7cff3e48f10f542132"
)
SOURCE_ENTITY_COUNT = 97_238
SOURCE_GENE_COUNT = 39_220
SOURCE_REPOSITORY = "https://github.com/gnn4dr/DRKG"
SOURCE_REPOSITORY_REVISION = "d4bb1974312013c4bd79a13e42c1d9492033f8c7"

HOME = Path.home()
RAW_DIR = HOME / "data/raw_embeddings" / EMBEDDING_NAME
MASTER_PATH = HOME / "metadata/master_gene_table_v1_1_enriched.csv"
OUT_DIR = HOME / "data/processed_embeddings" / EMBEDDING_NAME
REPORT_DIR = HOME / "reports/mapping_reports"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(
    url: str, target: Path, expected_sha256: str, expected_size: int
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        actual = sha256_file(target)
        if actual != expected_sha256 or target.stat().st_size != expected_size:
            raise ValueError(
                f"Existing source failed verification: {target}\n"
                f"Expected size/SHA-256: {expected_size}/{expected_sha256}\n"
                f"Observed size/SHA-256: {target.stat().st_size}/{actual}"
            )
        print(f"Reusing verified archive: {target}")
        return target

    partial = target.with_name(target.name + ".part")
    if partial.exists():
        raise ValueError(
            f"Incomplete download exists: {partial}; move it aside and retry"
        )
    request = urllib.request.Request(url, headers={"User-Agent": "gene-embeddings/1.0"})
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request) as response, partial.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception:
        print(f"Download interrupted; partial file retained at {partial}")
        raise
    actual = sha256_file(partial)
    if partial.stat().st_size != expected_size or actual != expected_sha256:
        raise ValueError(
            f"Downloaded archive failed verification: size={partial.stat().st_size}, sha256={actual}"
        )
    partial.replace(target)
    return target


def copy_stream(source: BinaryIO, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.exists():
        raise ValueError(f"Incomplete extracted file exists: {partial}")
    with partial.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    partial.replace(target)


def extract_verified_member(
    archive: Path,
    member_name: str,
    target: Path,
    expected_sha256: str,
) -> Path:
    if target.is_file():
        actual = sha256_file(target)
        if actual != expected_sha256:
            raise ValueError(
                f"Existing extracted file has wrong SHA-256: {target}: {actual}"
            )
        print(f"Reusing verified extracted source: {target}")
        return target

    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(member_name)
        if not member.isfile() or member.name != member_name:
            raise ValueError(f"Unsafe or unexpected archive member: {member_name}")
        source = tar.extractfile(member)
        if source is None:
            raise ValueError(f"Could not read archive member: {member_name}")
        with source:
            copy_stream(source, target)
    actual = sha256_file(target)
    if actual != expected_sha256:
        raise ValueError(
            f"Extracted checksum mismatch for {member_name}: expected {expected_sha256}, found {actual}"
        )
    return target


def load_master() -> tuple[pd.DataFrame, str]:
    if not MASTER_PATH.is_file():
        raise FileNotFoundError(f"Missing master table: {MASTER_PATH}")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    required = {
        "ensembl_gene_id",
        "gene_symbol",
        "gene_type",
        "entrez_id",
        "uniprot_id",
    }
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"Master table is missing columns: {sorted(missing)}")
    ids = master["ensembl_gene_id"].str.strip()
    if (ids == "").any() or ids.duplicated().any():
        raise ValueError("Master Ensembl IDs must be non-empty and unique")
    master["ensembl_gene_id"] = ids
    mapping_column = (
        "map_entrez_ids_all" if "map_entrez_ids_all" in master.columns else "entrez_id"
    )
    return master, mapping_column


def build_entrez_lookup(
    master: pd.DataFrame, mapping_column: str
) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for row in master[["ensembl_gene_id", mapping_column]].itertuples(
        index=False, name=None
    ):
        ensembl_id, identifiers = str(row[0]).strip(), str(row[1])
        for identifier in identifiers.split("|"):
            identifier = identifier.strip()
            if identifier.isdigit():
                lookup[str(int(identifier))].add(ensembl_id)
    if not lookup:
        raise ValueError(
            f"No numeric Entrez identifiers found in master column {mapping_column}"
        )
    return dict(lookup)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    out_npz = OUT_DIR / f"{EMBEDDING_NAME}_embeddings.npz"
    out_genes = OUT_DIR / f"{EMBEDDING_NAME}_genes.tsv"
    out_metadata = OUT_DIR / f"{EMBEDDING_NAME}_metadata.json"
    out_row_report = REPORT_DIR / f"{EMBEDDING_NAME}_row_mapping_report.tsv"
    out_report = REPORT_DIR / f"{EMBEDDING_NAME}_mapping_report.txt"

    archive = download_verified(
        SOURCE_ARCHIVE_URL,
        RAW_DIR / "drkg.tar.gz",
        SOURCE_ARCHIVE_SHA256,
        SOURCE_ARCHIVE_SIZE,
    )
    source_matrix_path = extract_verified_member(
        archive,
        SOURCE_MATRIX_MEMBER,
        RAW_DIR / "DRKG_TransE_l2_entity.npy",
        SOURCE_MATRIX_SHA256,
    )
    entities_path = extract_verified_member(
        archive,
        SOURCE_ENTITIES_MEMBER,
        RAW_DIR / "entities.tsv",
        SOURCE_ENTITIES_SHA256,
    )

    master, mapping_column = load_master()
    lookup = build_entrez_lookup(master, mapping_column)

    source_matrix = np.load(source_matrix_path, mmap_mode="r", allow_pickle=False)
    if (
        source_matrix.shape != (SOURCE_ENTITY_COUNT, DIMENSION)
        or source_matrix.dtype != np.float32
    ):
        raise ValueError(
            f"Unexpected DRKG matrix: shape={source_matrix.shape}, dtype={source_matrix.dtype}"
        )
    if not np.isfinite(source_matrix).all():
        raise ValueError("DRKG source matrix contains NaN or infinite values")

    entities = pd.read_csv(
        entities_path,
        sep="\t",
        header=None,
        names=["source_entity_name", "source_entity_index"],
        dtype={"source_entity_name": str, "source_entity_index": np.int64},
    )
    if len(entities) != SOURCE_ENTITY_COUNT:
        raise ValueError(f"Unexpected entities.tsv row count: {len(entities)}")
    if entities["source_entity_name"].duplicated().any():
        raise ValueError("entities.tsv contains duplicate entity names")
    indices = entities["source_entity_index"].to_numpy()
    if not np.array_equal(np.sort(indices), np.arange(SOURCE_ENTITY_COUNT)):
        raise ValueError("entities.tsv indices are not a permutation of matrix rows")

    gene_entities = entities[
        entities["source_entity_name"].str.startswith("Gene::", na=False)
    ].copy()
    if len(gene_entities) != SOURCE_GENE_COUNT:
        raise ValueError(f"Unexpected DRKG Gene entity count: {len(gene_entities)}")
    gene_entities["source_entrez_id"] = gene_entities["source_entity_name"].str.extract(
        r"^Gene::([0-9]+)$", expand=False
    )

    grouped: dict[str, list[int]] = defaultdict(list)
    source_info: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    report_rows: list[dict[str, object]] = []
    for row in gene_entities.itertuples(index=False):
        source_entrez = row.source_entrez_id
        mapped_ensembl = ""
        if pd.isna(source_entrez):
            status = "non_numeric_gene_identifier"
            candidates: set[str] = set()
        else:
            source_entrez = str(int(str(source_entrez)))
            candidates = lookup.get(source_entrez, set())
            if not candidates:
                status = "unmapped_entrez_id"
            elif len(candidates) > 1:
                status = "ambiguous_entrez_id"
            else:
                mapped_ensembl = next(iter(candidates))
                status = "mapped_unique_entrez_id"
                grouped[mapped_ensembl].append(int(row.source_entity_index))
                source_info[mapped_ensembl].append(
                    (
                        source_entrez,
                        str(row.source_entity_name),
                        int(row.source_entity_index),
                    )
                )
        report_rows.append(
            {
                "source_entity_index": int(row.source_entity_index),
                "source_entity_name": row.source_entity_name,
                "source_entrez_id": "" if pd.isna(source_entrez) else source_entrez,
                "status": status,
                "mapped_ensembl_gene_id": mapped_ensembl,
                "candidate_ensembl_gene_ids": "|".join(sorted(candidates)),
            }
        )

    final_ids = list(grouped)
    if not final_ids:
        raise ValueError(
            "No numeric DRKG Gene entities mapped uniquely to the master table"
        )
    final_vectors: list[np.ndarray] = []
    for ensembl_id in final_ids:
        source_indices = grouped[ensembl_id]
        vector = (
            np.asarray(source_matrix[source_indices[0]], dtype=np.float32)
            if len(source_indices) == 1
            else np.asarray(
                source_matrix[source_indices].mean(axis=0, dtype=np.float64),
                dtype=np.float32,
            )
        )
        final_vectors.append(vector)
    embeddings = np.ascontiguousarray(final_vectors, dtype=np.float32)
    if embeddings.shape != (len(final_ids), DIMENSION):
        raise ValueError(f"Unexpected mapped matrix shape: {embeddings.shape}")

    master_indexed = master.set_index("ensembl_gene_id", drop=False)
    genes = (
        master_indexed.loc[final_ids]
        .reset_index(drop=True)[
            ["ensembl_gene_id", "gene_symbol", "gene_type", "entrez_id", "uniprot_id"]
        ]
        .copy()
    )
    genes.insert(0, "embedding_row", np.arange(len(genes), dtype=int))
    genes["source_entrez_ids"] = [
        "|".join(item[0] for item in source_info[gene_id]) for gene_id in final_ids
    ]
    genes["source_entity_names"] = [
        "|".join(item[1] for item in source_info[gene_id]) for gene_id in final_ids
    ]
    genes["source_entity_indices"] = [
        "|".join(str(item[2]) for item in source_info[gene_id]) for gene_id in final_ids
    ]
    genes["n_source_rows"] = [len(source_info[gene_id]) for gene_id in final_ids]
    genes["mapping_status"] = np.where(
        genes["n_source_rows"] == 1,
        "mapped_unique_entrez_id",
        "mapped_multiple_source_rows_averaged",
    )

    embedding_row_lookup = {gene_id: row for row, gene_id in enumerate(final_ids)}
    row_report = pd.DataFrame(report_rows)
    row_report["embedding_row"] = (
        row_report["mapped_ensembl_gene_id"].map(embedding_row_lookup).fillna("")
    )

    status_counts = row_report["status"].value_counts().to_dict()
    unique_source_rows = int(status_counts.get("mapped_unique_entrez_id", 0))
    multi_source_final_genes = int((genes["n_source_rows"] > 1).sum())
    collapsed_source_rows = int(genes["n_source_rows"].sum() - len(genes))

    norms = np.linalg.norm(embeddings, axis=1)
    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    zero_vectors = int(np.count_nonzero(norms == 0))
    duplicate_vectors = int(len(embeddings) - len(np.unique(embeddings, axis=0)))
    constant_columns = int(np.count_nonzero(np.ptp(embeddings, axis=0) == 0))
    duplicated_ensembl = int(genes["ensembl_gene_id"].duplicated().sum())
    status = "processed_qc_passed"
    if (
        embeddings.shape != (len(genes), DIMENSION)
        or nan_count
        or inf_count
        or zero_vectors
        or duplicate_vectors
        or constant_columns
        or duplicated_ensembl
    ):
        status = "processed_qc_check_needed"

    np.savez_compressed(out_npz, embeddings=embeddings)
    genes.to_csv(out_genes, sep="\t", index=False)
    row_report.to_csv(out_row_report, sep="\t", index=False)
    embeddings_sha256 = sha256_file(out_npz)
    genes_sha256 = sha256_file(out_genes)

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "display_name": DISPLAY_NAME,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": "Network",
        "modality": "biomedical knowledge graph",
        "species": "human genes extracted from a multi-species biomedical graph",
        "source_archive_url": SOURCE_ARCHIVE_URL,
        "source_archive_local_file": str(archive),
        "source_archive_size_bytes": SOURCE_ARCHIVE_SIZE,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_matrix_member": SOURCE_MATRIX_MEMBER,
        "source_matrix_sha256": SOURCE_MATRIX_SHA256,
        "source_entities_member": SOURCE_ENTITIES_MEMBER,
        "source_entities_sha256": SOURCE_ENTITIES_SHA256,
        "source_repository": SOURCE_REPOSITORY,
        "source_repository_revision": SOURCE_REPOSITORY_REVISION,
        "source_license_note": (
            "DRKG code is Apache-2.0; the official DRKG source-license table includes "
            "mixed and non-commercial upstream terms. Review those terms before redistribution."
        ),
        "original_method": "TransE_l2 knowledge-graph entity embeddings",
        "master_table": str(MASTER_PATH),
        "master_mapping_column": mapping_column,
        "original_identifier_type": "DRKG Gene::<NCBI Entrez Gene ID>",
        "identifier_mapping": (
            f"numeric DRKG Entrez IDs mapped uniquely through master {mapping_column}; "
            "unmapped and ambiguous rows excluded; multiple source rows per Ensembl gene averaged"
        ),
        "preprocessing": "No transformation of single-source vectors; multi-source rows averaged in float64 then cast to float32.",
        "row_order": "first occurrence of each uniquely mapped Ensembl gene in DRKG Gene entity order",
        "source_shape": [SOURCE_ENTITY_COUNT, DIMENSION],
        "matrix_shape": list(embeddings.shape),
        "embedding_dimension": DIMENSION,
        "dtype": "float32",
        "npz_key": "embeddings",
        "embeddings_sha256": embeddings_sha256,
        "genes_sha256": genes_sha256,
        "n_source_entities": SOURCE_ENTITY_COUNT,
        "n_source_gene_entities": SOURCE_GENE_COUNT,
        "n_non_numeric_gene_entities": int(
            status_counts.get("non_numeric_gene_identifier", 0)
        ),
        "n_unique_mapped_source_rows": unique_source_rows,
        "n_unmapped_source_rows": int(status_counts.get("unmapped_entrez_id", 0)),
        "n_ambiguous_source_rows": int(status_counts.get("ambiguous_entrez_id", 0)),
        "n_multi_source_final_genes": multi_source_final_genes,
        "n_collapsed_source_rows": collapsed_source_rows,
        "n_processed_genes": len(genes),
        "n_master_genes": len(master),
        "coverage_of_master_table_percent": round(100 * len(genes) / len(master), 4),
        "qc": {
            "rows_match_genes_tsv": bool(len(genes) == embeddings.shape[0]),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_vectors": zero_vectors,
            "duplicate_vectors": duplicate_vectors,
            "constant_columns": constant_columns,
            "duplicated_ensembl_ids": duplicated_ensembl,
            "min_norm": float(norms.min()) if len(norms) else None,
            "median_norm": float(np.median(norms)) if len(norms) else None,
            "max_norm": float(norms.max()) if len(norms) else None,
        },
        "mapping_status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "output_files": {
            "embeddings": str(out_npz),
            "genes": str(out_genes),
            "metadata": str(out_metadata),
            "row_mapping_report": str(out_row_report),
            "mapping_report": str(out_report),
        },
    }
    out_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    status_lines = "\n".join(
        f"  {key}: {value}" for key, value in status_counts.items()
    )
    report = f"""Mapping/QC report for {DISPLAY_NAME}
{"=" * (22 + len(DISPLAY_NAME))}

Status: {status}
Archive: {archive}
Master table: {MASTER_PATH}
Master mapping column: {mapping_column}

Source entity matrix: {SOURCE_ENTITY_COUNT} x {DIMENSION}
Source Gene entities: {SOURCE_GENE_COUNT}
Processed shape: {embeddings.shape}
Multi-source final genes: {multi_source_final_genes}
Collapsed source rows: {collapsed_source_rows}
Master coverage: {metadata["coverage_of_master_table_percent"]}%

Mapping status counts:
{status_lines}

NaN count: {nan_count}
Inf count: {inf_count}
Zero vectors: {zero_vectors}
Exact duplicate vectors: {duplicate_vectors}
Constant columns: {constant_columns}
Duplicated Ensembl IDs: {duplicated_ensembl}

Embeddings: {out_npz}
Genes: {out_genes}
Metadata: {out_metadata}
Row mapping report: {out_row_report}
"""
    out_report.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
