#!/usr/bin/env python3

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

EMBEDDING_NAME = "DepMap-CRISPRGeneEffect-PCA256-v1.1"

home = Path.home()

raw_file = home / "data/raw_embeddings/depmap_crispr_gene_effect_v1_1/CRISPRGeneEffect.csv"
model_file = home / "data/raw_embeddings/depmap_crispr_gene_effect_v1_1/Model.csv"
master_file = home / "metadata/master_gene_table_v1_1_enriched.csv"

out_dir = home / "data/processed_embeddings/depmap_crispr_gene_effect_pca256_v1_1"
report_dir = home / "reports/mapping_reports/depmap_crispr_gene_effect_pca256_v1_1"

out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

if not raw_file.exists():
    raise FileNotFoundError(f"Missing DepMap raw file: {raw_file}")
if not master_file.exists():
    raise FileNotFoundError(f"Missing master table: {master_file}")

print(f"Embedding: {EMBEDDING_NAME}")
print(f"Raw file: {raw_file}")
print(f"Master: {master_file}")

# Load DepMap gene effect matrix.
# Rows are models/cell lines, columns are genes formatted like SYMBOL (Entrez).
dep = pd.read_csv(raw_file, index_col=0)
dep.index.name = "ModelID"

print(f"DepMap raw shape: {dep.shape[0]} models x {dep.shape[1]} gene columns")

# Parse gene columns.
pattern = re.compile(r"^(?P<symbol>.+) \((?P<entrez>\d+)\)$")

parsed = []
unparsed_cols = []

for col in dep.columns:
    m = pattern.match(str(col))
    if m:
        parsed.append(
            {
                "source_column": col,
                "source_gene_symbol": m.group("symbol"),
                "source_entrez_id": m.group("entrez"),
            }
        )
    else:
        unparsed_cols.append(col)

source_meta = pd.DataFrame(parsed)

if source_meta.empty:
    raise ValueError("No DepMap gene columns could be parsed as SYMBOL (Entrez).")

if unparsed_cols:
    print(f"Warning: {len(unparsed_cols)} unparsed columns will be excluded.")

dep = dep[source_meta["source_column"].tolist()].copy()

# Convert to numeric and handle missing values.
dep = dep.apply(pd.to_numeric, errors="coerce")
n_nan = int(dep.isna().sum().sum())
if n_nan:
    print(f"Found {n_nan} missing values; filling each gene column with its column mean.")
    dep = dep.fillna(dep.mean(axis=0))
    dep = dep.fillna(0.0)

# Transpose to gene x model matrix.
X_source = dep.T
X_source.index = source_meta["source_entrez_id"].values

# Collapse duplicate Entrez IDs, if present.
duplicate_source_entrez = int(pd.Index(X_source.index).duplicated().sum())
if duplicate_source_entrez:
    print(f"Duplicate source Entrez IDs: {duplicate_source_entrez}; collapsing by mean.")
    X_source = X_source.groupby(level=0).mean()

# Keep one metadata row per Entrez.
source_meta_unique = (
    source_meta.drop_duplicates("source_entrez_id")
    .set_index("source_entrez_id")
    .loc[X_source.index]
    .reset_index()
)

# Load master.
master = pd.read_csv(master_file, dtype=str).fillna("")
needed_cols = ["ensembl_gene_id", "gene_symbol", "gene_type", "entrez_id", "uniprot_id"]
missing_master_cols = [c for c in needed_cols if c not in master.columns]
if missing_master_cols:
    raise ValueError(f"Missing master columns: {missing_master_cols}")

master = master[needed_cols].copy()

# Map by Entrez ID.
source_embed = X_source.copy()
source_embed["source_entrez_id"] = source_embed.index.astype(str)

merged = master.merge(
    source_embed,
    left_on="entrez_id",
    right_on="source_entrez_id",
    how="left",
    indicator=True,
)

mapped = merged[merged["_merge"] == "both"].copy()
unmapped_master = merged[merged["_merge"] == "left_only"].copy()

print(f"Master genes: {master.shape[0]}")
print(f"Mapped master genes: {mapped.shape[0]}")
print(f"Unmapped master genes: {unmapped_master.shape[0]}")

model_ids = dep.index.astype(str).tolist()

# Build matrix in mapped master order.
X = mapped[model_ids].astype(float).to_numpy(dtype=np.float64)

# Standardize each model/cell-line feature across genes.
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)

n_components = min(256, X_scaled.shape[0], X_scaled.shape[1])
if n_components < 256:
    print(f"Warning: using only {n_components} PCA components because of matrix shape.")

pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
X_pca = pca.fit_transform(X_scaled).astype(np.float32)

print(f"Final embedding shape: {X_pca.shape}")

# Save embeddings.
np.save(out_dir / "embeddings.npy", X_pca)

# Add source metadata to genes table.
source_meta_unique = source_meta_unique.rename(columns={"index": "source_entrez_id"})
source_meta_unique["source_entrez_id"] = source_meta_unique["source_entrez_id"].astype(str)

genes = mapped[needed_cols + ["source_entrez_id"]].copy()
genes = genes.merge(
    source_meta_unique[["source_entrez_id", "source_gene_symbol", "source_column"]],
    on="source_entrez_id",
    how="left",
)
genes.insert(0, "embedding_row", np.arange(len(genes), dtype=int))
genes.to_csv(out_dir / "genes.tsv", sep="\t", index=False)

# Save model/cell-line IDs.
models_df = pd.DataFrame({"ModelID": model_ids})
if model_file.exists():
    try:
        model_meta = pd.read_csv(model_file, dtype=str).fillna("")
        if "ModelID" in model_meta.columns:
            models_df = models_df.merge(model_meta, on="ModelID", how="left")
    except Exception as e:
        print(f"Warning: could not merge Model.csv metadata: {e}")

models_df.to_csv(out_dir / "model_columns.tsv", sep="\t", index=False)

# Save PCA variance.
pca_df = pd.DataFrame(
    {
        "component": np.arange(1, n_components + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance": pca.explained_variance_,
    }
)
pca_df["cumulative_explained_variance_ratio"] = pca_df["explained_variance_ratio"].cumsum()
pca_df.to_csv(out_dir / "pca_explained_variance.tsv", sep="\t", index=False)

# Reports.
unmapped_master[needed_cols].to_csv(
    report_dir / "unmapped_master_genes.tsv",
    sep="\t",
    index=False,
)

if unparsed_cols:
    pd.Series(unparsed_cols, name="unparsed_source_column").to_csv(
        report_dir / "unparsed_source_columns.tsv",
        sep="\t",
        index=False,
    )

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "status": "processed_qc_passed",
    "modality": "functional genomics / CRISPR dependency",
    "species": "human",
    "source": "DepMap CRISPRGeneEffect.csv",
    "source_file": str(raw_file),
    "model_metadata_file": str(model_file) if model_file.exists() else None,
    "master_table": str(master_file),
    "identifier_mapping": "DepMap columns parsed as SYMBOL (Entrez); mapped to master entrez_id.",
    "preprocessing": "Gene-effect matrix transposed to genes x models; missing values filled by source gene mean; model/cell-line features standardized across mapped genes; PCA with randomized SVD.",
    "pca_n_components": int(n_components),
    "embedding_shape": list(X_pca.shape),
    "embedding_dimension": int(X_pca.shape[1]),
    "n_depmap_models": int(dep.shape[0]),
    "n_source_gene_columns_parsed": int(source_meta.shape[0]),
    "n_source_gene_columns_unparsed": int(len(unparsed_cols)),
    "n_unique_source_entrez_after_collapse": int(X_source.shape[0]),
    "duplicate_source_entrez_ids_collapsed_by_mean": int(duplicate_source_entrez),
    "n_master_genes": int(master.shape[0]),
    "n_mapped_master_genes": int(mapped.shape[0]),
    "n_unmapped_master_genes": int(unmapped_master.shape[0]),
    "pca_total_explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
    "output_files": {
        "embeddings": str(out_dir / "embeddings.npy"),
        "genes": str(out_dir / "genes.tsv"),
        "model_columns": str(out_dir / "model_columns.tsv"),
        "pca_explained_variance": str(out_dir / "pca_explained_variance.tsv"),
        "metadata": str(out_dir / "metadata.json"),
        "mapping_report": str(out_dir / "mapping_report.txt"),
        "unmapped_master_genes": str(report_dir / "unmapped_master_genes.tsv"),
    },
}

with open(out_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

report = f"""Mapping/QC report for {EMBEDDING_NAME}
================================================

Status: processed_qc_passed

Input
-----
Raw DepMap file: {raw_file}
Model metadata:  {model_file if model_file.exists() else "not found"}
Master table:    {master_file}

Source data
-----------
DepMap raw shape: {dep.shape[0]} models x {dep.shape[1]} parsed gene columns
Parsed source gene columns: {source_meta.shape[0]}
Unparsed source columns excluded: {len(unparsed_cols)}
Duplicate source Entrez IDs collapsed by mean: {duplicate_source_entrez}

Mapping
-------
Master genes: {master.shape[0]}
Mapped master genes: {mapped.shape[0]}
Unmapped master genes: {unmapped_master.shape[0]}

Embedding
---------
Preprocessing: transpose to genes x models, standardize model features across genes, PCA
Shape: {X_pca.shape[0]} genes x {X_pca.shape[1]} dimensions
dtype: float32
PCA total explained variance ratio: {pca.explained_variance_ratio_.sum():.6f}

Outputs
-------
{out_dir / "embeddings.npy"}
{out_dir / "genes.tsv"}
{out_dir / "model_columns.tsv"}
{out_dir / "pca_explained_variance.tsv"}
{out_dir / "metadata.json"}
{out_dir / "mapping_report.txt"}
{report_dir / "unmapped_master_genes.tsv"}
"""

(out_dir / "mapping_report.txt").write_text(report)
(report_dir / "mapping_report.txt").write_text(report)

print()
print(report)
