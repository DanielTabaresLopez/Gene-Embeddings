#!/usr/bin/env python3
"""Build the local GO Biological Process TF-IDF/SVD-256 gene embedding.

Required packages:
    python -m pip install numpy pandas scipy scikit-learn

The script downloads a date-pinned human UniProt GAF, constructs binary direct
GO-BP profiles, applies TF-IDF and deterministic TruncatedSVD, maps proteins to
the local master gene table, and writes local files only. It uploads nothing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

EMBEDDING_NAME = "go_bp_uniprot_tfidf_svd256"
DISPLAY_NAME = "GO Biological Process TF-IDF SVD-256"
DIMENSION = 256
RANDOM_STATE = 42
SVD_ITERATIONS = 10
SVD_OVERSAMPLES = 10

GO_RELEASE = "2026-08-05"
GO_RELEASE_DOI = "10.5281/zenodo.21844811"
SOURCE_URL = (
    f"https://release.geneontology.org/{GO_RELEASE}/annotations/gaf/"
    "HUMAN-uniprot.gaf.gz"
)
SOURCE_FILENAME = "HUMAN-uniprot.gaf.gz"
GO_LICENSE = "CC BY 4.0"
GO_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
GO_CITATION_POLICY = "https://geneontology.org/docs/go-citation-policy/"

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


def download_source(target: Path) -> tuple[Path, str]:
    """Download the immutable dated GO release and return its observed SHA-256."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        observed_sha256 = sha256_file(target)
        print(f"Reusing dated GO source: {target}")
        print(f"Observed source SHA-256: {observed_sha256}")
        return target, observed_sha256

    partial = target.with_name(target.name + ".part")
    if partial.exists():
        raise ValueError(
            f"Incomplete download exists: {partial}; move it aside and retry"
        )
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "gene-embeddings-go-bp-builder/1.0"},
    )
    print(f"Downloading {SOURCE_URL}")
    try:
        with urllib.request.urlopen(request) as response, partial.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception:
        print(f"Download interrupted; partial file retained at {partial}")
        raise
    if partial.stat().st_size < 1_000_000:
        raise ValueError(
            f"Downloaded GO file is unexpectedly small: {partial.stat().st_size} bytes"
        )
    partial.replace(target)
    observed_sha256 = sha256_file(target)
    print(f"Observed source SHA-256: {observed_sha256}")
    return target, observed_sha256


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
        "map_uniprot_ids_all"
        if "map_uniprot_ids_all" in master.columns
        else "uniprot_id"
    )
    return master, mapping_column


def build_uniprot_lookup(
    master: pd.DataFrame, mapping_column: str
) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = defaultdict(set)
    for row in master[["ensembl_gene_id", mapping_column]].itertuples(
        index=False, name=None
    ):
        ensembl_id, identifiers = str(row[0]).strip(), str(row[1])
        for identifier in identifiers.split("|"):
            identifier = identifier.strip()
            if identifier:
                lookup[identifier].add(ensembl_id)
    if not lookup:
        raise ValueError(
            f"No UniProt accessions found in master column {mapping_column}"
        )
    return dict(lookup)


def parse_go_bp_profiles(
    source_path: Path,
) -> tuple[list[str], list[str], csr_matrix, dict[str, Any]]:
    profiles: dict[str, set[str]] = defaultdict(set)
    headers: dict[str, str] = {}
    data_rows = 0
    bp_rows = 0
    negated_bp_rows = 0
    non_bp_rows = 0
    non_uniprot_rows = 0

    try:
        with gzip.open(source_path, "rt", encoding="utf-8", newline="") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith("!"):
                    body = line[1:].strip()
                    if ":" in body:
                        key, value = body.split(":", 1)
                        headers[key.strip().casefold()] = value.strip()
                    continue

                fields = line.split("\t")
                if len(fields) != 17:
                    raise ValueError(
                        f"GAF line {line_number} has {len(fields)} fields; expected 17"
                    )
                data_rows += 1
                database = fields[0].strip()
                accession = fields[1].strip()
                relation = fields[3].strip()
                go_id = fields[4].strip()
                aspect = fields[8].strip()

                if aspect != "P":
                    non_bp_rows += 1
                    continue
                if "NOT" in {item.strip() for item in relation.split("|")}:
                    negated_bp_rows += 1
                    continue
                if database != "UniProtKB":
                    non_uniprot_rows += 1
                    continue
                if not accession or not go_id.startswith("GO:"):
                    raise ValueError(
                        f"Malformed BP association on GAF line {line_number}"
                    )
                bp_rows += 1
                profiles[accession].add(go_id)
    except (OSError, EOFError) as exc:
        raise ValueError(f"Invalid or truncated gzip source: {source_path}") from exc

    gaf_version = headers.get("gaf-version", "")
    if gaf_version not in {"2.1", "2.2"}:
        raise ValueError(f"Unsupported or missing GAF version: {gaf_version!r}")
    if not profiles:
        raise ValueError("No non-negated human Biological Process annotations found")

    source_ids = sorted(profiles)
    go_terms = sorted({term for terms in profiles.values() for term in terms})
    if len(source_ids) <= DIMENSION or len(go_terms) <= DIMENSION:
        raise ValueError(
            f"Source matrix is too small for {DIMENSION} dimensions: "
            f"{len(source_ids)} proteins x {len(go_terms)} terms"
        )

    term_index = {term: index for index, term in enumerate(go_terms)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row_index, accession in enumerate(source_ids):
        for term in sorted(profiles[accession]):
            row_indices.append(row_index)
            column_indices.append(term_index[term])
    values = np.ones(len(row_indices), dtype=np.float32)
    matrix = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(source_ids), len(go_terms)),
        dtype=np.float32,
    )
    matrix.sort_indices()
    if matrix.nnz != len(row_indices):
        raise ValueError("Sparse GO association matrix contains duplicate coordinates")
    if data_rows != non_bp_rows + negated_bp_rows + non_uniprot_rows + bp_rows:
        raise ValueError("GAF row accounting is inconsistent")

    stats = {
        "gaf_version": gaf_version,
        "gaf_date_generated": headers.get("date-generated"),
        "go_version": headers.get("go-version"),
        "source_data_rows": data_rows,
        "non_bp_rows_excluded": non_bp_rows,
        "negated_bp_rows_excluded": negated_bp_rows,
        "non_uniprot_bp_rows_excluded": non_uniprot_rows,
        "nonnegated_bp_annotation_rows": bp_rows,
        "unique_bp_associations": int(matrix.nnz),
        "duplicate_bp_annotation_rows": bp_rows - int(matrix.nnz),
        "source_protein_count": len(source_ids),
        "go_bp_term_count": len(go_terms),
    }
    return source_ids, go_terms, matrix, stats


def embed_go_profiles(matrix: csr_matrix) -> tuple[np.ndarray, dict[str, Any]]:
    transformer = TfidfTransformer(
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    tfidf = transformer.fit_transform(matrix)
    svd = TruncatedSVD(
        n_components=DIMENSION,
        algorithm="randomized",
        n_iter=SVD_ITERATIONS,
        n_oversamples=SVD_OVERSAMPLES,
        power_iteration_normalizer="LU",
        random_state=RANDOM_STATE,
    )
    vectors = svd.fit_transform(tfidf)

    # SVD component signs are arbitrary. Fix their orientation deterministically.
    components = np.asarray(svd.components_)
    maxima = np.abs(components).argmax(axis=1)
    signs = np.sign(components[np.arange(DIMENSION), maxima])
    signs[signs == 0] = 1
    vectors = np.asarray(vectors * signs[np.newaxis, :], dtype=np.float32)

    if vectors.shape != (matrix.shape[0], DIMENSION):
        raise ValueError(f"Unexpected source embedding shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Source embedding contains NaN or infinite values")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Source embedding contains zero vectors")

    stats = {
        "input_shape": list(matrix.shape),
        "input_nonzero_associations": int(matrix.nnz),
        "tfidf_norm": "l2",
        "tfidf_use_idf": True,
        "tfidf_smooth_idf": True,
        "tfidf_sublinear_tf": False,
        "svd_algorithm": "randomized",
        "svd_components": DIMENSION,
        "svd_iterations": SVD_ITERATIONS,
        "svd_oversamples": SVD_OVERSAMPLES,
        "svd_power_iteration_normalizer": "LU",
        "random_state": RANDOM_STATE,
        "component_sign_rule": "largest-absolute loading is positive",
        "explained_variance_ratio_sum": float(
            np.sum(svd.explained_variance_ratio_, dtype=np.float64)
        ),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    return vectors, stats


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    out_npz = OUT_DIR / f"{EMBEDDING_NAME}_embeddings.npz"
    out_genes = OUT_DIR / f"{EMBEDDING_NAME}_genes.tsv"
    out_metadata = OUT_DIR / f"{EMBEDDING_NAME}_metadata.json"
    out_row_report = REPORT_DIR / f"{EMBEDDING_NAME}_row_mapping_report.tsv"
    out_report = REPORT_DIR / f"{EMBEDDING_NAME}_mapping_report.txt"

    source_path, source_sha256 = download_source(RAW_DIR / SOURCE_FILENAME)
    master, mapping_column = load_master()
    lookup = build_uniprot_lookup(master, mapping_column)

    source_ids, go_terms, profile_matrix, source_stats = parse_go_bp_profiles(
        source_path
    )
    source_vectors, transform_stats = embed_go_profiles(profile_matrix)

    grouped: dict[str, list[int]] = defaultdict(list)
    source_info: dict[str, list[str]] = defaultdict(list)
    report_rows: list[dict[str, object]] = []
    for source_index, accession in enumerate(source_ids):
        candidates = lookup.get(accession, set())
        mapped_ensembl = ""
        if not candidates:
            status = "unmapped_uniprot_accession"
        elif len(candidates) > 1:
            status = "ambiguous_uniprot_accession"
        else:
            mapped_ensembl = next(iter(candidates))
            status = "mapped_unique_uniprot_accession"
            grouped[mapped_ensembl].append(source_index)
            source_info[mapped_ensembl].append(accession)
        report_rows.append(
            {
                "source_row": source_index,
                "source_uniprot_accession": accession,
                "status": status,
                "mapped_ensembl_gene_id": mapped_ensembl,
                "candidate_ensembl_gene_ids": "|".join(sorted(candidates)),
            }
        )

    final_ids = sorted(grouped)
    if not final_ids:
        raise ValueError("No GO source proteins mapped uniquely to the master table")
    final_vectors: list[np.ndarray] = []
    for ensembl_id in final_ids:
        source_indices = grouped[ensembl_id]
        vector = (
            source_vectors[source_indices[0]]
            if len(source_indices) == 1
            else source_vectors[source_indices].mean(axis=0, dtype=np.float64)
        )
        final_vectors.append(np.asarray(vector, dtype=np.float32))
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
    genes["source_uniprot_accessions"] = [
        "|".join(source_info[gene_id]) for gene_id in final_ids
    ]
    genes["n_source_rows"] = [len(source_info[gene_id]) for gene_id in final_ids]
    genes["mapping_status"] = np.where(
        genes["n_source_rows"] == 1,
        "mapped_unique_uniprot_accession",
        "mapped_multiple_source_rows_averaged",
    )

    embedding_row_lookup = {gene_id: row for row, gene_id in enumerate(final_ids)}
    row_report = pd.DataFrame(report_rows)
    row_report["embedding_row"] = (
        row_report["mapped_ensembl_gene_id"].map(embedding_row_lookup).fillna("")
    )
    status_counts = row_report["status"].value_counts().to_dict()
    unique_source_rows = int(status_counts.get("mapped_unique_uniprot_accession", 0))
    multi_source_final_genes = int((genes["n_source_rows"] > 1).sum())
    collapsed_source_rows = int(genes["n_source_rows"].sum() - len(genes))

    norms = np.linalg.norm(embeddings, axis=1)
    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    zero_vectors = int(np.count_nonzero(norms == 0))
    # Exact duplicates can be legitimate when two genes have identical direct GO profiles.
    duplicate_vectors = int(len(embeddings) - len(np.unique(embeddings, axis=0)))
    constant_columns = int(np.count_nonzero(np.ptp(embeddings, axis=0) == 0))
    duplicated_ensembl = int(genes["ensembl_gene_id"].duplicated().sum())
    status = "processed_qc_passed"
    if (
        embeddings.shape != (len(genes), DIMENSION)
        or nan_count
        or inf_count
        or zero_vectors
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
        "category": "Functional Genomics",
        "modality": "Gene Ontology Biological Process annotation profiles",
        "species": "Homo sapiens",
        "source_url": SOURCE_URL,
        "source_local_file": str(source_path),
        "source_release": GO_RELEASE,
        "source_release_doi": GO_RELEASE_DOI,
        "source_sha256_observed": source_sha256,
        "source_license": GO_LICENSE,
        "source_license_url": GO_LICENSE_URL,
        "source_citation_policy": GO_CITATION_POLICY,
        "original_method": "binary GO Biological Process profiles + TF-IDF + TruncatedSVD-256",
        "master_table": str(MASTER_PATH),
        "master_mapping_column": mapping_column,
        "original_identifier_type": "UniProtKB accession",
        "identifier_mapping": (
            f"UniProt accessions mapped uniquely through master {mapping_column}; "
            "unmapped and ambiguous rows excluded; multiple source rows per Ensembl gene averaged"
        ),
        "preprocessing": (
            "Direct non-negated GAF aspect=P UniProtKB annotations were deduplicated into a "
            "binary protein-by-GO-term matrix, TF-IDF weighted, reduced with deterministic "
            "randomized TruncatedSVD, sign-canonicalized, then mapped to Ensembl genes."
        ),
        "row_order": "ascending final Ensembl gene ID",
        "source_profile_shape": list(profile_matrix.shape),
        "source_embedding_shape": list(source_vectors.shape),
        "matrix_shape": list(embeddings.shape),
        "embedding_dimension": DIMENSION,
        "dtype": "float32",
        "npz_key": "embeddings",
        "embeddings_sha256": embeddings_sha256,
        "genes_sha256": genes_sha256,
        "n_go_bp_terms": len(go_terms),
        "n_unique_mapped_source_rows": unique_source_rows,
        "n_unmapped_source_rows": int(
            status_counts.get("unmapped_uniprot_accession", 0)
        ),
        "n_ambiguous_source_rows": int(
            status_counts.get("ambiguous_uniprot_accession", 0)
        ),
        "n_multi_source_final_genes": multi_source_final_genes,
        "n_collapsed_source_rows": collapsed_source_rows,
        "n_processed_genes": len(genes),
        "n_master_genes": len(master),
        "coverage_of_master_table_percent": round(100 * len(genes) / len(master), 4),
        "source_counts": source_stats,
        "transformation": transform_stats,
        "qc": {
            "rows_match_genes_tsv": bool(len(genes) == embeddings.shape[0]),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_vectors": zero_vectors,
            "duplicate_vectors": duplicate_vectors,
            "duplicate_vectors_note": "recorded but allowed because identical GO profiles can be biological",
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
Source file: {source_path}
Observed source SHA-256: {source_sha256}
GO release: {GO_RELEASE} ({GO_RELEASE_DOI})
Master table: {MASTER_PATH}
Master mapping column: {mapping_column}

Source profile matrix: {profile_matrix.shape[0]} proteins x {profile_matrix.shape[1]} GO-BP terms
Unique source associations: {profile_matrix.nnz}
Source embedding shape: {source_vectors.shape}
Processed shape: {embeddings.shape}
Multi-source final genes: {multi_source_final_genes}
Collapsed source rows: {collapsed_source_rows}
Master coverage: {metadata["coverage_of_master_table_percent"]}%

Mapping status counts:
{status_lines}

NaN count: {nan_count}
Inf count: {inf_count}
Zero vectors: {zero_vectors}
Exact duplicate vectors (allowed and recorded): {duplicate_vectors}
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
