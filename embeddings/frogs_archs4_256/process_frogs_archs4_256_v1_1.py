#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import subprocess
import time

import numpy as np
import pandas as pd


EMBEDDING_NAME = "frogs_archs4_256_v1_1"

HOME = Path.home()
MASTER_PATH = HOME / "metadata/master_gene_table_v1_1_enriched.csv"
RAW_REPO = HOME / "data/raw_embeddings/frogs_archs4_256_v1_1/official_repo"
SOURCE_CSV = RAW_REPO / "data/gene_vec_archs4_256.csv"

OUT_DIR = HOME / "data/processed_embeddings" / EMBEDDING_NAME
REPORT_PATH = HOME / "reports/mapping_reports" / f"{EMBEDDING_NAME}_mapping_report.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def extract_entrez_ids(value):
    if pd.isna(value):
        return []
    return re.findall(r"\d+", str(value))


def build_entrez_to_master(master: pd.DataFrame):
    entrez_cols = [
        c for c in [
            "entrez_id",
            "map_entrez_ids_all",
            "entrez_id_hgnc",
            "entrez_id_biomart",
            "hgnc_entrez_id",
        ]
        if c in master.columns
    ]

    pairs = []
    for _, row in master.iterrows():
        ensg = row["ensembl_gene_id"]
        for col in entrez_cols:
            for eid in extract_entrez_ids(row[col]):
                pairs.append((eid, ensg))

    tmp = pd.DataFrame(pairs, columns=["entrez_id", "ensembl_gene_id"]).drop_duplicates()

    counts = tmp.groupby("entrez_id")["ensembl_gene_id"].nunique()
    ambiguous = set(counts[counts > 1].index)
    unambig = tmp[~tmp["entrez_id"].isin(ambiguous)].copy()

    mapping = dict(zip(unambig["entrez_id"], unambig["ensembl_gene_id"]))
    return mapping, ambiguous, entrez_cols


def collapse_strings(values):
    vals = sorted({str(v) for v in values if str(v) and str(v) != "nan"})
    return ";".join(vals)


def main():
    start = time.time()

    print("Embedding:", EMBEDDING_NAME)
    print("Source CSV:", SOURCE_CSV)
    print("Master:", MASTER_PATH)

    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    entrez_to_ensg, ambiguous_entrez, entrez_cols = build_entrez_to_master(master)

    raw = pd.read_csv(SOURCE_CSV, header=None)
    print("Raw FRoGS shape:", raw.shape)

    raw = raw.rename(columns={0: "source_entrez_id"})
    raw["source_entrez_id"] = raw["source_entrez_id"].astype(str).str.extract(r"(\d+)")[0]

    emb_cols = [c for c in raw.columns if c != "source_entrez_id"]
    raw[emb_cols] = raw[emb_cols].apply(pd.to_numeric, errors="coerce")

    raw = raw.dropna(subset=["source_entrez_id"])
    raw = raw.dropna(subset=emb_cols)

    duplicate_entrez_rows_before_averaging = int(raw["source_entrez_id"].duplicated().sum())

    # First collapse duplicate Entrez IDs, if any.
    raw_avg = raw.groupby("source_entrez_id", as_index=False)[emb_cols].mean()

    raw_avg["ensembl_gene_id"] = raw_avg["source_entrez_id"].map(entrez_to_ensg)
    mapped = raw_avg.dropna(subset=["ensembl_gene_id"]).copy()
    mapped = mapped[~mapped["source_entrez_id"].isin(ambiguous_entrez)].copy()

    duplicated_ensembl_before_collapse = int(mapped["ensembl_gene_id"].duplicated().sum())

    # Important fix:
    # multiple source Entrez IDs can map to the same master Ensembl gene.
    # Collapse those by averaging vectors at the final Ensembl level.
    vector_by_ensg = mapped.groupby("ensembl_gene_id", as_index=False)[emb_cols].mean()

    source_ids_by_ensg = (
        mapped.groupby("ensembl_gene_id")["source_entrez_id"]
        .apply(collapse_strings)
        .reset_index()
    )

    final = vector_by_ensg.merge(source_ids_by_ensg, on="ensembl_gene_id", how="left")

    master_small = master[
        ["ensembl_gene_id", "gene_symbol", "entrez_id", "uniprot_id"]
    ].drop_duplicates("ensembl_gene_id")

    final = final.merge(master_small, on="ensembl_gene_id", how="left")
    final = final.sort_values("ensembl_gene_id").reset_index(drop=True)

    matrix = final[emb_cols].to_numpy(dtype=np.float32)

    genes_df = final[
        ["ensembl_gene_id", "gene_symbol", "source_entrez_id", "entrez_id", "uniprot_id"]
    ].copy()
    genes_df = genes_df.rename(columns={"entrez_id": "master_entrez_id"})

    rows_match = matrix.shape[0] == len(genes_df)
    nan_count = int(np.isnan(matrix).sum())
    inf_count = int(np.isinf(matrix).sum())
    zero_vectors = int((np.linalg.norm(matrix, axis=1) == 0).sum())
    duplicated_genes = int(genes_df["ensembl_gene_id"].duplicated().sum())

    status = "processed_qc_passed"
    if not rows_match or nan_count or inf_count or zero_vectors or duplicated_genes:
        status = "processed_qc_failed"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    npz_path = OUT_DIR / f"{EMBEDDING_NAME}_embeddings.npz"
    genes_path = OUT_DIR / f"{EMBEDDING_NAME}_genes.tsv"
    meta_path = OUT_DIR / f"{EMBEDDING_NAME}_metadata.json"

    np.savez_compressed(npz_path, embeddings=matrix)
    genes_df.to_csv(genes_path, sep="\t", index=False)

    runtime = time.time() - start

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "FRoGS ARCHS4 gene embedding",
        "model_source": "chenhcs/FRoGS data/gene_vec_archs4_256.csv",
        "official_repo": str(RAW_REPO),
        "official_repo_commit": git_commit(RAW_REPO),
        "source_csv": str(SOURCE_CSV),
        "source_csv_sha256": sha256_file(SOURCE_CSV),
        "original_identifier_type": "Entrez Gene ID",
        "mapping_policy": (
            "FRoGS source Entrez IDs mapped to master Ensembl gene IDs. "
            "Entrez IDs mapping to multiple master Ensembl genes were excluded. "
            "Duplicate source Entrez IDs were averaged before mapping. "
            "Multiple source Entrez IDs mapping to the same final Ensembl gene were averaged after mapping."
        ),
        "master_gene_table": str(MASTER_PATH),
        "entrez_columns_used": entrez_cols,
        "embedding_dimension": int(matrix.shape[1]),
        "matrix_shape": list(matrix.shape),
        "raw_shape": list(raw.shape),
        "duplicate_entrez_rows_before_averaging": duplicate_entrez_rows_before_averaging,
        "duplicated_ensembl_rows_before_final_collapse": duplicated_ensembl_before_collapse,
        "ambiguous_entrez_ids_excluded_from_master_mapping": len(ambiguous_entrez),
        "runtime_seconds": round(runtime, 2),
    }

    meta_path.write_text(json.dumps(metadata, indent=2))

    report = []
    report.append(f"Mapping/QC report for {EMBEDDING_NAME}")
    report.append("=" * (22 + len(EMBEDDING_NAME)))
    report.append("")
    report.append(f"Status: {status}")
    report.append(f"Output directory: {OUT_DIR}")
    report.append(f"Embeddings: {npz_path}")
    report.append(f"Genes: {genes_path}")
    report.append(f"Metadata: {meta_path}")
    report.append("")
    report.append(f"Source CSV: {SOURCE_CSV}")
    report.append(f"Source CSV SHA256: {metadata['source_csv_sha256']}")
    report.append(f"Official repo commit: {metadata['official_repo_commit']}")
    report.append("")
    report.append(f"Raw FRoGS rows after numeric cleanup: {raw.shape[0]}")
    report.append(f"Duplicate Entrez rows before averaging: {duplicate_entrez_rows_before_averaging}")
    report.append(f"Ambiguous Entrez IDs excluded from master mapping: {len(ambiguous_entrez)}")
    report.append(f"Duplicated Ensembl rows before final collapse: {duplicated_ensembl_before_collapse}")
    report.append("")
    report.append(f"Matrix shape: {matrix.shape}")
    report.append(f"Rows match genes.tsv: {rows_match}")
    report.append(f"NaN count: {nan_count}")
    report.append(f"Inf count: {inf_count}")
    report.append(f"Zero vectors: {zero_vectors}")
    report.append(f"Duplicated Ensembl IDs: {duplicated_genes}")
    report.append(f"Processed genes: {len(genes_df)}")
    report.append(f"Coverage of master table: {100 * len(genes_df) / len(master):.4f}%")
    report.append(f"Runtime seconds: {runtime:.2f}")

    report_text = "\n".join(report)
    REPORT_PATH.write_text(report_text + "\n")

    print("\nDone")
    print("----")
    print(report_text)


if __name__ == "__main__":
    main()
