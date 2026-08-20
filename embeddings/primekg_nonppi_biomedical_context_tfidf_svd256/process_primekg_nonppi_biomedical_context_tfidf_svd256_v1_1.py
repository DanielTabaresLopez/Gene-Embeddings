#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

EMBEDDING_NAME = "PrimeKG-nonPPI-biomedical-context-TFIDF-SVD256-v1.1"

home = Path.home()

raw_file = home / "data/raw_embeddings/primekg_v1_1/kg.csv"
master_file = home / "metadata/master_gene_table_v1_1_enriched.csv"

out_dir = home / "data/processed_embeddings/primekg_nonppi_biomedical_context_tfidf_svd256_v1_1"
report_dir = home / "reports/mapping_reports/primekg_nonppi_biomedical_context_tfidf_svd256_v1_1"

out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

if not raw_file.exists():
    raise FileNotFoundError(f"Missing PrimeKG file: {raw_file}")
if not master_file.exists():
    raise FileNotFoundError(f"Missing master table: {master_file}")

print(f"Embedding: {EMBEDDING_NAME}")
print(f"Raw PrimeKG file: {raw_file}")
print(f"Master table: {master_file}")

usecols = [
    "relation",
    "display_relation",
    "x_id",
    "x_type",
    "x_name",
    "x_source",
    "y_id",
    "y_type",
    "y_name",
    "y_source",
]

gene_to_row = {}
gene_symbol_by_entrez = {}
feature_to_col = {}
feature_records = []

row_idx = []
col_idx = []
data = []

included_relation_counts = defaultdict(int)
included_neighbor_type_counts = defaultdict(int)
excluded_counts = defaultdict(int)

total_rows = 0
included_rows = 0
chunk_size = 500_000

def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def get_gene_row(entrez_id, symbol):
    entrez_id = clean_str(entrez_id)
    symbol = clean_str(symbol)

    if not entrez_id:
        return None

    if entrez_id not in gene_to_row:
        gene_to_row[entrez_id] = len(gene_to_row)
        gene_symbol_by_entrez[entrez_id] = symbol

    return gene_to_row[entrez_id]

def get_feature_col(relation, display_relation, neighbor_type, neighbor_id, neighbor_name, neighbor_source):
    relation = clean_str(relation)
    display_relation = clean_str(display_relation)
    neighbor_type = clean_str(neighbor_type)
    neighbor_id = clean_str(neighbor_id)
    neighbor_name = clean_str(neighbor_name)
    neighbor_source = clean_str(neighbor_source)

    key = "||".join([relation, display_relation, neighbor_type, neighbor_id])

    if key not in feature_to_col:
        feature_to_col[key] = len(feature_to_col)
        feature_records.append(
            {
                "feature_col": feature_to_col[key],
                "feature_key": key,
                "relation": relation,
                "display_relation": display_relation,
                "neighbor_type": neighbor_type,
                "neighbor_id": neighbor_id,
                "neighbor_name": neighbor_name,
                "neighbor_source": neighbor_source,
            }
        )

    return feature_to_col[key]

def process_side(df, gene_side):
    global included_rows

    if gene_side == "x":
        gene_id_col = "x_id"
        gene_name_col = "x_name"
        neighbor_type_col = "y_type"
        neighbor_id_col = "y_id"
        neighbor_name_col = "y_name"
        neighbor_source_col = "y_source"
    else:
        gene_id_col = "y_id"
        gene_name_col = "y_name"
        neighbor_type_col = "x_type"
        neighbor_id_col = "x_id"
        neighbor_name_col = "x_name"
        neighbor_source_col = "x_source"

    for row in df.itertuples(index=False):
        d = row._asdict()

        gene_row = get_gene_row(d[gene_id_col], d[gene_name_col])
        if gene_row is None:
            continue

        feature_col = get_feature_col(
            d["relation"],
            d["display_relation"],
            d[neighbor_type_col],
            d[neighbor_id_col],
            d[neighbor_name_col],
            d[neighbor_source_col],
        )

        row_idx.append(gene_row)
        col_idx.append(feature_col)
        data.append(1.0)

        included_relation_counts[clean_str(d["relation"])] += 1
        included_neighbor_type_counts[clean_str(d[neighbor_type_col])] += 1
        included_rows += 1

print("\nReading PrimeKG in chunks...")

for i, chunk in enumerate(pd.read_csv(raw_file, usecols=usecols, dtype=str, chunksize=chunk_size, low_memory=False), start=1):
    total_rows += chunk.shape[0]

    x_gene = chunk["x_type"].eq("gene/protein")
    y_gene = chunk["y_type"].eq("gene/protein")

    # Important design decision:
    # Keep only edges with exactly one gene/protein endpoint.
    # This removes direct gene-gene / PPI-like edges so this embedding captures heterogeneous
    # biomedical context rather than duplicating the existing PPI node2vec embedding.
    exactly_one_gene = x_gene ^ y_gene

    excluded_counts["both_gene_or_ppi_like_edges"] += int((x_gene & y_gene).sum())
    excluded_counts["no_gene_endpoint_edges"] += int((~x_gene & ~y_gene).sum())

    keep = chunk.loc[exactly_one_gene].copy()

    if keep.empty:
        print(f"Chunk {i}: total rows so far {total_rows:,}; included edges so far {included_rows:,}")
        continue

    keep_x_gene = keep.loc[keep["x_type"].eq("gene/protein")]
    keep_y_gene = keep.loc[keep["y_type"].eq("gene/protein")]

    process_side(keep_x_gene, "x")
    process_side(keep_y_gene, "y")

    print(
        f"Chunk {i}: total rows so far {total_rows:,}; "
        f"included gene-context edges so far {included_rows:,}; "
        f"genes {len(gene_to_row):,}; features {len(feature_to_col):,}"
    )

print("\nFinished reading.")
print(f"Total PrimeKG rows: {total_rows:,}")
print(f"Included non-PPI gene-context edges: {included_rows:,}")
print(f"Source genes with at least one included context edge: {len(gene_to_row):,}")
print(f"Typed biomedical context features: {len(feature_to_col):,}")
print("Excluded counts:", dict(excluded_counts))

if included_rows == 0:
    raise RuntimeError("No PrimeKG gene-context edges were included. Check input format.")

# Build sparse gene x feature matrix.
n_genes = len(gene_to_row)
n_features = len(feature_to_col)

M = coo_matrix(
    (np.asarray(data, dtype=np.float32), (np.asarray(row_idx, dtype=np.int32), np.asarray(col_idx, dtype=np.int32))),
    shape=(n_genes, n_features),
    dtype=np.float32,
).tocsr()
M.sum_duplicates()

print("\nSparse matrix:")
print("shape:", M.shape)
print("nnz:", M.nnz)
print("density:", M.nnz / (M.shape[0] * M.shape[1]))

# TF-IDF weighting downweights extremely common biomedical features.
print("\nApplying TF-IDF weighting...")
tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
M_tfidf = tfidf.fit_transform(M)

# SVD embedding.
n_components = min(256, M_tfidf.shape[0] - 1, M_tfidf.shape[1] - 1)
if n_components < 256:
    print(f"Warning: using {n_components} SVD components due to matrix shape.")

print("\nRunning TruncatedSVD...")
svd = TruncatedSVD(n_components=n_components, random_state=42, algorithm="randomized")
X_source = svd.fit_transform(M_tfidf).astype(np.float32)

# Z-score SVD dimensions across source genes.
mean = X_source.mean(axis=0, keepdims=True)
std = X_source.std(axis=0, ddof=0, keepdims=True)
std[std == 0] = 1.0
X_source_z = ((X_source - mean) / std).astype(np.float32)

print("Source embedding shape:", X_source_z.shape)
print("SVD total explained variance ratio:", float(svd.explained_variance_ratio_.sum()))

# Source gene table.
source_gene_rows = []
for entrez_id, r in gene_to_row.items():
    source_gene_rows.append(
        {
            "source_row": r,
            "source_entrez_id": entrez_id,
            "source_gene_symbol": gene_symbol_by_entrez.get(entrez_id, ""),
        }
    )

source_genes = pd.DataFrame(source_gene_rows).sort_values("source_row")
source_genes.to_csv(out_dir / "source_genes.tsv", sep="\t", index=False)

# Save source-level feature tables.
features = pd.DataFrame(feature_records).sort_values("feature_col")
features.to_csv(out_dir / "feature_columns.tsv", sep="\t", index=False)

pd.DataFrame(
    sorted(included_relation_counts.items()),
    columns=["relation", "included_edge_count"],
).to_csv(out_dir / "included_relation_counts.tsv", sep="\t", index=False)

pd.DataFrame(
    sorted(included_neighbor_type_counts.items()),
    columns=["neighbor_type", "included_edge_count"],
).to_csv(out_dir / "included_neighbor_type_counts.tsv", sep="\t", index=False)

pd.DataFrame(
    {
        "component": np.arange(1, n_components + 1),
        "explained_variance_ratio": svd.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(svd.explained_variance_ratio_),
    }
).to_csv(out_dir / "svd_explained_variance.tsv", sep="\t", index=False)

# Map to master.
master = pd.read_csv(master_file, dtype=str).fillna("")
needed_master_cols = ["ensembl_gene_id", "gene_symbol", "gene_type", "entrez_id", "uniprot_id"]
missing_master = [c for c in needed_master_cols if c not in master.columns]
if missing_master:
    raise ValueError(f"Missing master columns: {missing_master}")

master = master[needed_master_cols].copy()

source_embedding_df = pd.DataFrame(
    X_source_z,
    columns=[f"svd_{i+1}" for i in range(X_source_z.shape[1])],
)
source_embedding_df.insert(0, "source_entrez_id", source_genes["source_entrez_id"].astype(str).values)
source_embedding_df.insert(1, "source_gene_symbol", source_genes["source_gene_symbol"].astype(str).values)

merged = master.merge(
    source_embedding_df,
    left_on="entrez_id",
    right_on="source_entrez_id",
    how="left",
    indicator=True,
)

mapped = merged[merged["_merge"] == "both"].copy()
unmapped_master = merged[merged["_merge"] == "left_only"].copy()

embedding_cols = [c for c in source_embedding_df.columns if c.startswith("svd_")]
X = mapped[embedding_cols].astype(float).to_numpy(dtype=np.float32)

np.save(out_dir / "embeddings.npy", X)

genes = mapped[needed_master_cols + ["source_entrez_id", "source_gene_symbol"]].copy()
genes.insert(0, "embedding_row", np.arange(len(genes), dtype=int))
genes.to_csv(out_dir / "genes.tsv", sep="\t", index=False)

unmapped_master[needed_master_cols].to_csv(
    report_dir / "unmapped_master_genes.tsv",
    sep="\t",
    index=False,
)

mapped_source_entrez = set(mapped["source_entrez_id"].astype(str))
unmapped_source = source_genes[~source_genes["source_entrez_id"].astype(str).isin(mapped_source_entrez)].copy()
unmapped_source.to_csv(report_dir / "unmapped_source_genes.tsv", sep="\t", index=False)

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "status": "processed_qc_passed",
    "modality": "heterogeneous biomedical knowledge graph / non-PPI gene context",
    "species": "human",
    "source": "PrimeKG kg.csv",
    "source_file": str(raw_file),
    "master_table": str(master_file),
    "identifier_mapping": "PrimeKG gene/protein x_id/y_id interpreted as NCBI Entrez IDs and mapped to master entrez_id.",
    "important_design_decision": "Edges where both endpoints are gene/protein were excluded to avoid duplicating the existing PPI node2vec embedding. This embedding represents non-PPI biomedical context only.",
    "feature_definition": "Each sparse feature is relation + display_relation + neighbor_type + neighbor_id for a non-gene biomedical neighbor connected to a gene.",
    "preprocessing": "Built sparse gene x typed-neighbor-feature count matrix; collapsed duplicate gene-feature edges by summation; applied sublinear TF-IDF weighting; reduced with randomized TruncatedSVD; z-scored SVD dimensions across source genes; mapped to master by Entrez ID.",
    "embedding_shape": list(X.shape),
    "embedding_dimension": int(X.shape[1]),
    "source_sparse_matrix_shape": [int(M.shape[0]), int(M.shape[1])],
    "source_sparse_matrix_nnz": int(M.nnz),
    "n_total_primekg_rows": int(total_rows),
    "n_included_nonppi_gene_context_edges": int(included_rows),
    "n_excluded_both_gene_or_ppi_like_edges": int(excluded_counts["both_gene_or_ppi_like_edges"]),
    "n_excluded_no_gene_endpoint_edges": int(excluded_counts["no_gene_endpoint_edges"]),
    "n_source_genes_with_context": int(n_genes),
    "n_typed_biomedical_context_features": int(n_features),
    "svd_n_components": int(n_components),
    "svd_total_explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
    "n_master_genes": int(master.shape[0]),
    "n_mapped_master_genes": int(mapped.shape[0]),
    "n_unmapped_master_genes": int(unmapped_master.shape[0]),
    "n_unmapped_source_genes": int(unmapped_source.shape[0]),
    "output_files": {
        "embeddings": str(out_dir / "embeddings.npy"),
        "genes": str(out_dir / "genes.tsv"),
        "source_genes": str(out_dir / "source_genes.tsv"),
        "feature_columns": str(out_dir / "feature_columns.tsv"),
        "included_relation_counts": str(out_dir / "included_relation_counts.tsv"),
        "included_neighbor_type_counts": str(out_dir / "included_neighbor_type_counts.tsv"),
        "svd_explained_variance": str(out_dir / "svd_explained_variance.tsv"),
        "metadata": str(out_dir / "metadata.json"),
        "mapping_report": str(out_dir / "mapping_report.txt"),
        "unmapped_master_genes": str(report_dir / "unmapped_master_genes.tsv"),
        "unmapped_source_genes": str(report_dir / "unmapped_source_genes.tsv"),
    },
}

with open(out_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

report = f"""Mapping/QC report for {EMBEDDING_NAME}
====================================================================

Status: processed_qc_passed

Input
-----
PrimeKG file: {raw_file}
Master table: {master_file}

Design decision
---------------
This is a non-PPI heterogeneous biomedical context embedding.
All edges where both endpoints are gene/protein were excluded.
This avoids duplicating Node2Vec-consensus-PPI-v1.1 and focuses the embedding on
drug, disease, anatomy, phenotype, pathway, biological process, molecular function,
cellular component, and exposure context.

Source data
-----------
Total PrimeKG rows read: {total_rows}
Included non-PPI gene-context edges: {included_rows}
Excluded both-gene/PPI-like edges: {excluded_counts["both_gene_or_ppi_like_edges"]}
Excluded no-gene-endpoint edges: {excluded_counts["no_gene_endpoint_edges"]}

Sparse feature matrix
---------------------
Source genes with at least one included context edge: {n_genes}
Typed biomedical context features: {n_features}
Sparse matrix nnz after duplicate collapse: {M.nnz}
Sparse matrix density: {M.nnz / (M.shape[0] * M.shape[1]):.8f}

Embedding
---------
Weighting: sublinear TF-IDF
Reduction: randomized TruncatedSVD
SVD components: {n_components}
SVD total explained variance ratio: {svd.explained_variance_ratio_.sum():.6f}
Final mapped shape: {X.shape[0]} genes x {X.shape[1]} dimensions
dtype: float32

Mapping
-------
Master genes: {master.shape[0]}
Mapped master genes: {mapped.shape[0]}
Unmapped master genes: {unmapped_master.shape[0]}
Unmapped source genes: {unmapped_source.shape[0]}

Outputs
-------
{out_dir / "embeddings.npy"}
{out_dir / "genes.tsv"}
{out_dir / "source_genes.tsv"}
{out_dir / "feature_columns.tsv"}
{out_dir / "included_relation_counts.tsv"}
{out_dir / "included_neighbor_type_counts.tsv"}
{out_dir / "svd_explained_variance.tsv"}
{out_dir / "metadata.json"}
{out_dir / "mapping_report.txt"}
{report_dir / "unmapped_master_genes.tsv"}
{report_dir / "unmapped_source_genes.tsv"}
"""

(out_dir / "mapping_report.txt").write_text(report)
(report_dir / "mapping_report.txt").write_text(report)

print()
print(report)
