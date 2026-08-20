#!/usr/bin/env python3
"""Download and locally process the STRING WaveGC gene embedding.

Required packages:
    python -m pip install numpy pandas scipy anndata huggingface_hub

This script does not authenticate to Hugging Face and does not upload anything.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy import sparse

EMBEDDING_NAME = "string_wavegc"
DISPLAY_NAME = "STRING WaveGC"
DIMENSION = 128

SOURCE_DATASET = "genbio-ai/foundation-models-perturbation"
SOURCE_REVISION = "d67ba21019814bb55dbdb448b3b84d18bbc693b8"
SOURCE_REPO_PATH = "gene_embeddings/KG_WaveGC_G_StringDB_combined_0.0_(D=128).h5ad"
SOURCE_SHA256 = "c1076da1a2118c71f7f46c42fd7011622bb5ab9c5aae6292879f7ed2f4ba5d16"
SOURCE_PUBLICATION = "https://doi.org/10.64898/2026.02.18.706454"
SOURCE_CODE = "https://github.com/genbio-ai/foundation-models-perturbation"

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


def load_master() -> pd.DataFrame:
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
    return master


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    out_npz = OUT_DIR / f"{EMBEDDING_NAME}_embeddings.npz"
    out_genes = OUT_DIR / f"{EMBEDDING_NAME}_genes.tsv"
    out_metadata = OUT_DIR / f"{EMBEDDING_NAME}_metadata.json"
    out_row_report = REPORT_DIR / f"{EMBEDDING_NAME}_row_mapping_report.tsv"
    out_report = REPORT_DIR / f"{EMBEDDING_NAME}_mapping_report.txt"

    print(f"Embedding: {DISPLAY_NAME}")
    print(f"Source: {SOURCE_DATASET}@{SOURCE_REVISION}:{SOURCE_REPO_PATH}")
    print(f"Master: {MASTER_PATH}")

    source_path = Path(
        hf_hub_download(
            repo_id=SOURCE_DATASET,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            filename=SOURCE_REPO_PATH,
            local_dir=str(RAW_DIR),
        )
    )
    source_sha256 = sha256_file(source_path)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {SOURCE_SHA256}, found {source_sha256}"
        )

    master = load_master()
    master_ids = set(master["ensembl_gene_id"])

    adata = ad.read_h5ad(source_path)
    source_matrix = (
        adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    )
    source_matrix = np.asarray(source_matrix, dtype=np.float32)
    source_ids = np.asarray(adata.obs_names.astype(str).tolist(), dtype=str)

    if source_matrix.shape != (len(source_ids), DIMENSION):
        raise ValueError(f"Unexpected source shape: {source_matrix.shape}")
    if len(source_ids) != len(set(source_ids.tolist())):
        raise ValueError("Source h5ad contains duplicate obs_names")
    if not pd.Series(source_ids).str.fullmatch(r"ENSG[0-9]{11}").all():
        raise ValueError("Source obs_names are not unversioned Ensembl gene IDs")
    if not np.isfinite(source_matrix).all():
        raise ValueError("Source embedding contains NaN or infinite values")

    keep = np.fromiter((gene_id in master_ids for gene_id in source_ids), dtype=bool)
    final_ids = source_ids[keep]
    if not len(final_ids):
        raise ValueError("No source genes matched the master gene table")
    embeddings = np.ascontiguousarray(source_matrix[keep], dtype=np.float32)

    master_indexed = master.set_index("ensembl_gene_id", drop=False)
    genes = (
        master_indexed.loc[final_ids]
        .reset_index(drop=True)[
            ["ensembl_gene_id", "gene_symbol", "gene_type", "entrez_id", "uniprot_id"]
        ]
        .copy()
    )
    genes.insert(0, "embedding_row", np.arange(len(genes), dtype=int))
    genes["source_obs_name"] = final_ids
    genes["mapping_status"] = "mapped_exact_ensembl_gene_id"

    embedding_row_lookup = {gene_id: row for row, gene_id in enumerate(final_ids)}
    row_report = pd.DataFrame(
        {
            "source_row": np.arange(len(source_ids), dtype=int),
            "source_obs_name": source_ids,
            "status": np.where(
                keep, "mapped_exact_ensembl_gene_id", "not_in_master_table"
            ),
            "embedding_row": [
                embedding_row_lookup.get(gene_id, "") for gene_id in source_ids
            ],
        }
    )

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
        "modality": "STRINGdb protein-protein association network",
        "species": "Homo sapiens",
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "source_repo_path": SOURCE_REPO_PATH,
        "source_local_file": str(source_path),
        "source_sha256": source_sha256,
        "source_publication": SOURCE_PUBLICATION,
        "source_code": SOURCE_CODE,
        "source_license": "GenBio AI Community License Agreement (non-commercial terms)",
        "original_method": "WaveGC graph spectral wavelet convolution with link prediction",
        "master_table": str(MASTER_PATH),
        "original_identifier_type": "unversioned Ensembl gene ID in h5ad obs_names",
        "identifier_mapping": "exact match of source obs_names to master ensembl_gene_id",
        "preprocessing": "No numeric transformation; retained h5ad X rows converted to float32.",
        "row_order": "retained source h5ad row order",
        "source_shape": list(source_matrix.shape),
        "matrix_shape": list(embeddings.shape),
        "embedding_dimension": DIMENSION,
        "dtype": "float32",
        "npz_key": "embeddings",
        "embeddings_sha256": embeddings_sha256,
        "genes_sha256": genes_sha256,
        "n_source_genes": len(source_ids),
        "n_processed_genes": len(genes),
        "n_source_genes_not_in_master": int((~keep).sum()),
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
        "output_files": {
            "embeddings": str(out_npz),
            "genes": str(out_genes),
            "metadata": str(out_metadata),
            "row_mapping_report": str(out_row_report),
            "mapping_report": str(out_report),
        },
    }
    out_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report = f"""Mapping/QC report for {DISPLAY_NAME}
{"=" * (22 + len(DISPLAY_NAME))}

Status: {status}
Source file: {source_path}
Source SHA-256: {source_sha256}
Master table: {MASTER_PATH}

Source shape: {source_matrix.shape}
Processed shape: {embeddings.shape}
Source genes not in master: {int((~keep).sum())}
Master coverage: {metadata["coverage_of_master_table_percent"]}%

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
