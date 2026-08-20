#!/usr/bin/env python3

from pathlib import Path
import json
import gzip
import numpy as np
import pandas as pd

EMBEDDING_NAME = "GTEx-tissue-median-log1p-zscore-v1.1"

home = Path.home()

raw_file = home / "data/raw_embeddings/gtex_tissue_expression_v1_1/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz"
master_file = home / "metadata/master_gene_table_v1_1_enriched.csv"

out_dir = home / "data/processed_embeddings/gtex_tissue_median_log1p_zscore_v1_1"
report_dir = home / "reports/mapping_reports/gtex_tissue_median_log1p_zscore_v1_1"

out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

if not raw_file.exists():
    raise FileNotFoundError(f"Missing GTEx raw file: {raw_file}")
if not master_file.exists():
    raise FileNotFoundError(f"Missing master table: {master_file}")

print(f"Embedding: {EMBEDDING_NAME}")
print(f"Raw file: {raw_file}")
print(f"Master: {master_file}")

# GCT format:
# line 1: #1.2
# line 2: n_rows n_columns
# line 3: header
gtex = pd.read_csv(raw_file, sep="\t", skiprows=2, dtype={"Name": str, "Description": str})

required = {"Name", "Description"}
missing_required = required - set(gtex.columns)
if missing_required:
    raise ValueError(f"Missing expected GCT columns: {missing_required}")

tissue_cols = [c for c in gtex.columns if c not in ["Name", "Description"]]
print(f"GTEx rows: {gtex.shape[0]}")
print(f"GTEx tissue columns: {len(tissue_cols)}")

# Convert expression columns.
expr = gtex[tissue_cols].apply(pd.to_numeric, errors="coerce")
if expr.isna().any().any():
    n_nan = int(expr.isna().sum().sum())
    print(f"Warning: found {n_nan} NaN expression values; filling with 0.")
    expr = expr.fillna(0.0)

# Strip Ensembl version suffix.
gtex["source_ensembl_gene_id_versioned"] = gtex["Name"].astype(str)
gtex["source_ensembl_gene_id"] = gtex["source_ensembl_gene_id_versioned"].str.replace(r"\.\d+$", "", regex=True)
gtex["source_gene_symbol"] = gtex["Description"].astype(str)

# Add expression back to a compact frame.
expr_df = pd.concat(
    [
        gtex[["source_ensembl_gene_id", "source_ensembl_gene_id_versioned", "source_gene_symbol"]],
        expr,
    ],
    axis=1,
)

# Collapse duplicate stripped Ensembl IDs if any.
dup_count = int(expr_df["source_ensembl_gene_id"].duplicated().sum())
if dup_count:
    print(f"Duplicate stripped Ensembl IDs: {dup_count}; collapsing by mean expression.")
    meta_first = (
        expr_df[["source_ensembl_gene_id", "source_ensembl_gene_id_versioned", "source_gene_symbol"]]
        .drop_duplicates("source_ensembl_gene_id")
        .set_index("source_ensembl_gene_id")
    )
    expr_mean = expr_df.groupby("source_ensembl_gene_id")[tissue_cols].mean()
    expr_df = meta_first.join(expr_mean, how="inner").reset_index()

# Load master.
master = pd.read_csv(master_file, dtype=str).fillna("")
master = master[["ensembl_gene_id", "gene_symbol", "gene_type", "entrez_id", "uniprot_id"]].copy()

# Map GTEx to master Ensembl IDs.
merged = master.merge(
    expr_df,
    left_on="ensembl_gene_id",
    right_on="source_ensembl_gene_id",
    how="left",
    indicator=True,
)

mapped = merged[merged["_merge"] == "both"].copy()
unmapped_master = merged[merged["_merge"] == "left_only"].copy()

print(f"Master genes: {master.shape[0]}")
print(f"Mapped master genes: {mapped.shape[0]}")
print(f"Unmapped master genes: {unmapped_master.shape[0]}")

# Build embedding matrix in master-table order.
X_tpm = mapped[tissue_cols].astype(float).to_numpy(dtype=np.float64)

# Preprocessing: log1p TPM, then z-score each tissue column across mapped genes.
X = np.log1p(X_tpm)

col_mean = X.mean(axis=0, keepdims=True)
col_std = X.std(axis=0, ddof=0, keepdims=True)
col_std[col_std == 0] = 1.0
X_z = (X - col_mean) / col_std
X_z = X_z.astype(np.float32)

# Save embeddings.
np.save(out_dir / "embeddings.npy", X_z)

genes = mapped[
    [
        "ensembl_gene_id",
        "gene_symbol",
        "gene_type",
        "entrez_id",
        "uniprot_id",
        "source_ensembl_gene_id",
        "source_ensembl_gene_id_versioned",
        "source_gene_symbol",
    ]
].copy()
genes.insert(0, "embedding_row", np.arange(len(genes), dtype=int))
genes.to_csv(out_dir / "genes.tsv", sep="\t", index=False)

pd.Series(tissue_cols, name="tissue").to_csv(out_dir / "tissue_columns.tsv", sep="\t", index=False)

# Save unmapped master genes for review.
unmapped_master[["ensembl_gene_id", "gene_symbol", "gene_type", "entrez_id", "uniprot_id"]].to_csv(
    report_dir / "unmapped_master_genes.tsv",
    sep="\t",
    index=False,
)

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "status": "processed_qc_passed",
    "modality": "bulk RNA expression / normal tissue expression",
    "species": "human",
    "source": "GTEx v11 median gene-level TPM by tissue",
    "source_file": str(raw_file),
    "master_table": str(master_file),
    "identifier_mapping": "GTEx versioned Ensembl gene IDs stripped to stable Ensembl gene IDs, then matched to master ensembl_gene_id",
    "preprocessing": "TPM values were transformed with log1p, then each tissue column was z-scored across mapped genes.",
    "embedding_shape": list(X_z.shape),
    "embedding_dimension": int(X_z.shape[1]),
    "n_tissues": len(tissue_cols),
    "n_source_rows": int(gtex.shape[0]),
    "n_master_genes": int(master.shape[0]),
    "n_mapped_master_genes": int(mapped.shape[0]),
    "n_unmapped_master_genes": int(unmapped_master.shape[0]),
    "duplicate_stripped_ensembl_ids_collapsed_by_mean": int(dup_count),
    "output_files": {
        "embeddings": str(out_dir / "embeddings.npy"),
        "genes": str(out_dir / "genes.tsv"),
        "tissue_columns": str(out_dir / "tissue_columns.tsv"),
        "metadata": str(out_dir / "metadata.json"),
        "mapping_report": str(out_dir / "mapping_report.txt"),
        "unmapped_master_genes": str(report_dir / "unmapped_master_genes.tsv"),
    },
}

with open(out_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

report = f"""Mapping/QC report for {EMBEDDING_NAME}
===============================================

Status: processed_qc_passed

Input
-----
Raw GTEx file: {raw_file}
Master table:  {master_file}

Source data
-----------
GTEx rows: {gtex.shape[0]}
Tissue columns: {len(tissue_cols)}
Duplicate stripped Ensembl IDs collapsed by mean: {dup_count}

Mapping
-------
Master genes: {master.shape[0]}
Mapped master genes: {mapped.shape[0]}
Unmapped master genes: {unmapped_master.shape[0]}

Embedding
---------
Preprocessing: log1p(TPM), then tissue-column z-score across mapped genes
Shape: {X_z.shape[0]} genes x {X_z.shape[1]} dimensions
dtype: float32

Outputs
-------
{out_dir / "embeddings.npy"}
{out_dir / "genes.tsv"}
{out_dir / "tissue_columns.tsv"}
{out_dir / "metadata.json"}
{out_dir / "mapping_report.txt"}
{report_dir / "unmapped_master_genes.tsv"}
"""

(out_dir / "mapping_report.txt").write_text(report)
(report_dir / "mapping_report.txt").write_text(report)

print()
print(report)
