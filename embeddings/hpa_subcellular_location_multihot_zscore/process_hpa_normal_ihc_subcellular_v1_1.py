#!/usr/bin/env python3

from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd

home = Path.home()

raw_dir = home / "data/raw_embeddings/hpa_v1_1"
normal_zip = raw_dir / "normal_ihc_data.tsv.zip"
subcell_zip = raw_dir / "subcellular_location.tsv.zip"
master_file = home / "metadata/master_gene_table_v1_1_enriched.csv"

normal_out = home / "data/processed_embeddings/hpa_normal_ihc_tissue_level_zscore_v1_1"
subcell_out = home / "data/processed_embeddings/hpa_subcellular_location_multihot_zscore_v1_1"

normal_report_dir = home / "reports/mapping_reports/hpa_normal_ihc_tissue_level_zscore_v1_1"
subcell_report_dir = home / "reports/mapping_reports/hpa_subcellular_location_multihot_zscore_v1_1"

for d in [normal_out, subcell_out, normal_report_dir, subcell_report_dir]:
    d.mkdir(parents=True, exist_ok=True)

if not normal_zip.exists():
    raise FileNotFoundError(f"Missing HPA normal IHC file: {normal_zip}")
if not subcell_zip.exists():
    raise FileNotFoundError(f"Missing HPA subcellular file: {subcell_zip}")
if not master_file.exists():
    raise FileNotFoundError(f"Missing master table: {master_file}")

def read_single_tsv_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected one file inside {path}, found {names}")
        with z.open(names[0]) as f:
            return pd.read_csv(f, sep="\t", dtype=str).fillna("")

def zscore_columns(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    return ((X - mean) / std).astype(np.float32)

master = pd.read_csv(master_file, dtype=str).fillna("")
needed_master_cols = ["ensembl_gene_id", "gene_symbol", "gene_type", "entrez_id", "uniprot_id"]
missing = [c for c in needed_master_cols if c not in master.columns]
if missing:
    raise ValueError(f"Missing master columns: {missing}")
master = master[needed_master_cols].copy()

print("Master genes:", master.shape[0])

# ---------------------------------------------------------------------
# 1) HPA normal IHC tissue protein expression
# ---------------------------------------------------------------------

NORMAL_EMBEDDING_NAME = "HPA-normal-IHC-tissue-level-zscore-v1.1"

print("\n" + "=" * 80)
print("Processing:", NORMAL_EMBEDDING_NAME)

normal = read_single_tsv_zip(normal_zip)

required_normal = ["Gene", "Gene name", "Tissue", "Cell type", "Level", "Reliability"]
missing_normal = [c for c in required_normal if c not in normal.columns]
if missing_normal:
    raise ValueError(f"Missing HPA normal IHC columns: {missing_normal}")

level_map = {
    "Not detected": 0.0,
    "Low": 1.0,
    "Medium": 2.0,
    "High": 3.0,
}

# HPA also contains non-expression / non-ordinal values such as:
# "", "Ascending", "Descending", and "Not representative".
# These are not directly comparable to protein-expression intensity levels,
# so they are excluded from the tissue-level expression embedding.
valid_level_mask = normal["Level"].isin(level_map)
excluded_level_counts = normal.loc[~valid_level_mask, "Level"].value_counts(dropna=False)

print("Rows excluded because Level is not an ordinal expression value:")
print(excluded_level_counts.to_string() if len(excluded_level_counts) else "None")

normal_expr = normal.loc[valid_level_mask].copy()
normal_expr["level_numeric"] = normal_expr["Level"].map(level_map).astype(float)

print("Normal IHC rows:", normal.shape[0])
print("Normal IHC unique genes:", normal["Gene"].nunique())
print("Normal IHC unique tissues:", normal["Tissue"].nunique())
print("Normal IHC Level counts:")
print(normal["Level"].value_counts().to_string())
print("Normal IHC Reliability counts:")
print(normal["Reliability"].value_counts().to_string())

# Aggregate cell types within each tissue by max protein-expression level.
# This captures whether a protein is detected in any annotated cell type of that tissue.
tissue_matrix = (
    normal_expr.groupby(["Gene", "Tissue"])["level_numeric"]
    .max()
    .unstack(fill_value=0.0)
)

tissue_cols = sorted(tissue_matrix.columns.tolist())
tissue_matrix = tissue_matrix[tissue_cols]

gene_meta = (
    normal_expr[["Gene", "Gene name"]]
    .drop_duplicates("Gene")
    .rename(columns={"Gene": "hpa_gene", "Gene name": "hpa_gene_name"})
)

source = tissue_matrix.copy()
source["hpa_gene"] = source.index.astype(str)

merged = master.merge(
    source,
    left_on="ensembl_gene_id",
    right_on="hpa_gene",
    how="left",
    indicator=True,
)

mapped = merged[merged["_merge"] == "both"].copy()
unmapped_master = merged[merged["_merge"] == "left_only"].copy()

X_raw = mapped[tissue_cols].astype(float).to_numpy()
X_z = zscore_columns(X_raw)

np.save(normal_out / "embeddings.npy", X_z)

genes = mapped[needed_master_cols + ["hpa_gene"]].copy()
genes = genes.merge(gene_meta, on="hpa_gene", how="left")
genes.insert(0, "embedding_row", np.arange(len(genes), dtype=int))
genes.to_csv(normal_out / "genes.tsv", sep="\t", index=False)

pd.Series(tissue_cols, name="tissue").to_csv(normal_out / "tissue_columns.tsv", sep="\t", index=False)

unmapped_master[needed_master_cols].to_csv(
    normal_report_dir / "unmapped_master_genes.tsv",
    sep="\t",
    index=False,
)

normal_metadata = {
    "embedding_name": NORMAL_EMBEDDING_NAME,
    "status": "processed_qc_passed",
    "modality": "protein expression / normal tissue immunohistochemistry",
    "species": "human",
    "source": "Human Protein Atlas normal_ihc_data.tsv.zip",
    "source_file": str(normal_zip),
    "master_table": str(master_file),
    "identifier_mapping": "HPA Ensembl gene IDs matched to master ensembl_gene_id.",
    "preprocessing": "HPA Level encoded as Not detected=0, Low=1, Medium=2, High=3. For each gene and tissue, maximum level across annotated cell types was used. Tissue columns were z-scored across mapped genes.",
    "aggregation": "max across HPA cell types within each tissue",
    "embedding_shape": list(X_z.shape),
    "embedding_dimension": int(X_z.shape[1]),
    "n_source_rows": int(normal.shape[0]),
    "n_source_genes": int(normal["Gene"].nunique()),
    "n_valid_expression_rows": int(normal_expr.shape[0]),
    "n_valid_expression_genes": int(normal_expr["Gene"].nunique()),
    "excluded_level_counts": {str(k): int(v) for k, v in excluded_level_counts.items()},
    "n_tissues": int(len(tissue_cols)),
    "n_master_genes": int(master.shape[0]),
    "n_mapped_master_genes": int(mapped.shape[0]),
    "n_unmapped_master_genes": int(unmapped_master.shape[0]),
    "level_counts": {str(k): int(v) for k, v in normal["Level"].value_counts().items()},
    "reliability_counts": {str(k): int(v) for k, v in normal["Reliability"].value_counts().items()},
    "output_files": {
        "embeddings": str(normal_out / "embeddings.npy"),
        "genes": str(normal_out / "genes.tsv"),
        "tissue_columns": str(normal_out / "tissue_columns.tsv"),
        "metadata": str(normal_out / "metadata.json"),
        "mapping_report": str(normal_out / "mapping_report.txt"),
        "unmapped_master_genes": str(normal_report_dir / "unmapped_master_genes.tsv"),
    },
}

with open(normal_out / "metadata.json", "w") as f:
    json.dump(normal_metadata, f, indent=2)

normal_report = f"""Mapping/QC report for {NORMAL_EMBEDDING_NAME}
=======================================================

Status: processed_qc_passed

Input
-----
Raw HPA file: {normal_zip}
Master table: {master_file}

Source data
-----------
Rows: {normal.shape[0]}
Unique source genes: {normal["Gene"].nunique()}
Tissues: {len(tissue_cols)}
Level encoding: Not detected=0, Low=1, Medium=2, High=3
Aggregation: max level across cell types within each tissue

Mapping
-------
Master genes: {master.shape[0]}
Mapped master genes: {mapped.shape[0]}
Unmapped master genes: {unmapped_master.shape[0]}

Embedding
---------
Preprocessing: ordinal HPA levels, then tissue-column z-score across mapped genes
Shape: {X_z.shape[0]} genes x {X_z.shape[1]} dimensions
dtype: float32

Outputs
-------
{normal_out / "embeddings.npy"}
{normal_out / "genes.tsv"}
{normal_out / "tissue_columns.tsv"}
{normal_out / "metadata.json"}
{normal_out / "mapping_report.txt"}
{normal_report_dir / "unmapped_master_genes.tsv"}
"""

(normal_out / "mapping_report.txt").write_text(normal_report)
(normal_report_dir / "mapping_report.txt").write_text(normal_report)

print(normal_report)

# ---------------------------------------------------------------------
# 2) HPA subcellular localization
# ---------------------------------------------------------------------

SUBCELL_EMBEDDING_NAME = "HPA-subcellular-location-multihot-zscore-v1.1"

print("\n" + "=" * 80)
print("Processing:", SUBCELL_EMBEDDING_NAME)

sub = read_single_tsv_zip(subcell_zip)

required_sub = ["Gene", "Gene name", "Reliability", "Main location", "Additional location", "Extracellular location"]
missing_sub = [c for c in required_sub if c not in sub.columns]
if missing_sub:
    raise ValueError(f"Missing HPA subcellular columns: {missing_sub}")

print("Subcellular rows:", sub.shape[0])
print("Subcellular unique genes:", sub["Gene"].nunique())
print("Subcellular Reliability counts:")
print(sub["Reliability"].value_counts().to_string())

location_cols = ["Main location", "Additional location", "Extracellular location"]

def split_locations(x):
    x = str(x).strip()
    if not x or x.lower() == "nan":
        return []
    return [p.strip() for p in x.split(";") if p.strip()]

all_locations = sorted(
    {
        loc
        for col in location_cols
        for val in sub[col].tolist()
        for loc in split_locations(val)
    }
)

if not all_locations:
    raise ValueError("No subcellular locations parsed.")

print("Parsed subcellular locations:", len(all_locations))

rows = []
for _, row in sub.iterrows():
    locs = set()
    for col in location_cols:
        locs.update(split_locations(row[col]))

    out = {
        "Gene": row["Gene"],
        "Gene name": row["Gene name"],
        "Reliability": row["Reliability"],
    }
    for loc in all_locations:
        out[loc] = 1.0 if loc in locs else 0.0
    rows.append(out)

loc_df = pd.DataFrame(rows)

# Collapse duplicate genes by max location presence.
loc_matrix = loc_df.groupby("Gene")[all_locations].max()
loc_matrix = loc_matrix[all_locations]

sub_gene_meta = (
    sub[["Gene", "Gene name", "Reliability"]]
    .drop_duplicates("Gene")
    .rename(columns={"Gene": "hpa_gene", "Gene name": "hpa_gene_name", "Reliability": "hpa_reliability"})
)

source = loc_matrix.copy()
source["hpa_gene"] = source.index.astype(str)

merged = master.merge(
    source,
    left_on="ensembl_gene_id",
    right_on="hpa_gene",
    how="left",
    indicator=True,
)

mapped = merged[merged["_merge"] == "both"].copy()
unmapped_master = merged[merged["_merge"] == "left_only"].copy()

X_raw = mapped[all_locations].astype(float).to_numpy()
X_z = zscore_columns(X_raw)

np.save(subcell_out / "embeddings.npy", X_z)

genes = mapped[needed_master_cols + ["hpa_gene"]].copy()
genes = genes.merge(sub_gene_meta, on="hpa_gene", how="left")
genes.insert(0, "embedding_row", np.arange(len(genes), dtype=int))
genes.to_csv(subcell_out / "genes.tsv", sep="\t", index=False)

pd.Series(all_locations, name="subcellular_location").to_csv(
    subcell_out / "location_columns.tsv",
    sep="\t",
    index=False,
)

unmapped_master[needed_master_cols].to_csv(
    subcell_report_dir / "unmapped_master_genes.tsv",
    sep="\t",
    index=False,
)

sub_metadata = {
    "embedding_name": SUBCELL_EMBEDDING_NAME,
    "status": "processed_qc_passed",
    "modality": "protein subcellular localization",
    "species": "human",
    "source": "Human Protein Atlas subcellular_location.tsv.zip",
    "source_file": str(subcell_zip),
    "master_table": str(master_file),
    "identifier_mapping": "HPA Ensembl gene IDs matched to master ensembl_gene_id.",
    "preprocessing": "Main location, Additional location, and Extracellular location were parsed as semicolon-separated labels. A binary multi-hot location vector was created per gene, duplicate genes were collapsed by max, then location columns were z-scored across mapped genes.",
    "embedding_shape": list(X_z.shape),
    "embedding_dimension": int(X_z.shape[1]),
    "n_source_rows": int(sub.shape[0]),
    "n_source_genes": int(sub["Gene"].nunique()),
    "n_locations": int(len(all_locations)),
    "n_master_genes": int(master.shape[0]),
    "n_mapped_master_genes": int(mapped.shape[0]),
    "n_unmapped_master_genes": int(unmapped_master.shape[0]),
    "reliability_counts": {str(k): int(v) for k, v in sub["Reliability"].value_counts().items()},
    "output_files": {
        "embeddings": str(subcell_out / "embeddings.npy"),
        "genes": str(subcell_out / "genes.tsv"),
        "location_columns": str(subcell_out / "location_columns.tsv"),
        "metadata": str(subcell_out / "metadata.json"),
        "mapping_report": str(subcell_out / "mapping_report.txt"),
        "unmapped_master_genes": str(subcell_report_dir / "unmapped_master_genes.tsv"),
    },
}

with open(subcell_out / "metadata.json", "w") as f:
    json.dump(sub_metadata, f, indent=2)

sub_report = f"""Mapping/QC report for {SUBCELL_EMBEDDING_NAME}
==========================================================

Status: processed_qc_passed

Input
-----
Raw HPA file: {subcell_zip}
Master table: {master_file}

Source data
-----------
Rows: {sub.shape[0]}
Unique source genes: {sub["Gene"].nunique()}
Parsed subcellular locations: {len(all_locations)}

Mapping
-------
Master genes: {master.shape[0]}
Mapped master genes: {mapped.shape[0]}
Unmapped master genes: {unmapped_master.shape[0]}

Embedding
---------
Preprocessing: binary multi-hot location vector, then location-column z-score across mapped genes
Shape: {X_z.shape[0]} genes x {X_z.shape[1]} dimensions
dtype: float32

Outputs
-------
{subcell_out / "embeddings.npy"}
{subcell_out / "genes.tsv"}
{subcell_out / "location_columns.tsv"}
{subcell_out / "metadata.json"}
{subcell_out / "mapping_report.txt"}
{subcell_report_dir / "unmapped_master_genes.tsv"}
"""

(subcell_out / "mapping_report.txt").write_text(sub_report)
(subcell_report_dir / "mapping_report.txt").write_text(sub_report)

print(sub_report)
