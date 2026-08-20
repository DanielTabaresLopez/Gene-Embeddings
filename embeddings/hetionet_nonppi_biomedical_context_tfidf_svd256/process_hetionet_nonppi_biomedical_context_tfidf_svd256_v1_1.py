#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

EMBEDDING_NAME = "Hetionet-nonPPI-biomedical-context-TFIDF-SVD256-v1.1"

home = Path.home()

raw_dir = home / "data/raw_embeddings/hetionet_v1_1"
edges_file = raw_dir / "hetionet-v1.0-edges.sif.gz"
nodes_file = raw_dir / "hetionet-v1.0-nodes.tsv"
master_file = home / "metadata/master_gene_table_v1_1_enriched.csv"

out_dir = home / "data/processed_embeddings/hetionet_nonppi_biomedical_context_tfidf_svd256_v1_1"
report_dir = home / "reports/mapping_reports/hetionet_nonppi_biomedical_context_tfidf_svd256_v1_1"

out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

for p in [edges_file, nodes_file, master_file]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

print(f"Embedding: {EMBEDDING_NAME}")
print(f"Edges: {edges_file}")
print(f"Nodes: {nodes_file}")
print(f"Master: {master_file}")

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def node_kind_from_id(node_id):
    node_id = clean(node_id)
    if "::" in node_id:
        return node_id.split("::", 1)[0]
    return ""

def node_local_id(node_id):
    node_id = clean(node_id)
    if "::" in node_id:
        return node_id.split("::", 1)[1]
    return node_id

def is_gene_node(node_id, kind):
    return clean(kind) == "Gene" or clean(node_id).startswith("Gene::")

# Load node metadata.
nodes = pd.read_csv(nodes_file, sep="\t", dtype=str).fillna("")

# Expected Hetionet columns are usually id, name, kind.
# This fallback keeps the script robust if headers are missing.
if not {"id", "name", "kind"}.issubset(set(nodes.columns)):
    nodes = pd.read_csv(nodes_file, sep="\t", dtype=str, header=None).fillna("")
    if nodes.shape[1] >= 3:
        nodes = nodes.iloc[:, :3]
        nodes.columns = ["id", "name", "kind"]
    else:
        raise ValueError(f"Could not interpret node table columns: {list(nodes.columns)}")

nodes["id"] = nodes["id"].astype(str)
nodes["name"] = nodes["name"].astype(str)
nodes["kind"] = nodes["kind"].astype(str)

node_meta = nodes.set_index("id")[["name", "kind"]].to_dict(orient="index")

print("Node table shape:", nodes.shape)
print("Node kind counts:")
print(nodes["kind"].value_counts().to_string())

gene_to_row = {}
gene_symbol_by_entrez = {}
feature_to_col = {}
feature_records = []

row_idx = []
col_idx = []
data = []

included_metaedge_counts = defaultdict(int)
included_neighbor_kind_counts = defaultdict(int)
excluded_counts = defaultdict(int)

total_edges = 0
included_edges = 0

def get_gene_row(gene_node_id):
    gene_node_id = clean(gene_node_id)
    entrez_id = node_local_id(gene_node_id)

    if not entrez_id:
        return None

    if entrez_id not in gene_to_row:
        gene_to_row[entrez_id] = len(gene_to_row)
        gene_symbol_by_entrez[entrez_id] = node_meta.get(gene_node_id, {}).get("name", "")

    return gene_to_row[entrez_id]

def get_feature_col(metaedge, gene_side, neighbor_id):
    neighbor_id = clean(neighbor_id)
    neighbor_info = node_meta.get(neighbor_id, {})
    neighbor_name = neighbor_info.get("name", "")
    neighbor_kind = neighbor_info.get("kind", node_kind_from_id(neighbor_id))

    key = "||".join([clean(metaedge), clean(gene_side), clean(neighbor_kind), clean(neighbor_id)])

    if key not in feature_to_col:
        feature_to_col[key] = len(feature_to_col)
        feature_records.append(
            {
                "feature_col": feature_to_col[key],
                "feature_key": key,
                "metaedge": clean(metaedge),
                "gene_side": clean(gene_side),
                "neighbor_kind": clean(neighbor_kind),
                "neighbor_id": clean(neighbor_id),
                "neighbor_name": clean(neighbor_name),
            }
        )

    return feature_to_col[key], neighbor_kind

chunk_size = 500_000

print("\nReading Hetionet edges in chunks...")

for i, chunk in enumerate(
    pd.read_csv(
        edges_file,
        sep="\t",
        compression="gzip",
        header=None,
        names=["source_id", "metaedge", "target_id"],
        dtype=str,
        chunksize=chunk_size,
    ),
    start=1,
):
    chunk = chunk.fillna("")
    total_edges += chunk.shape[0]

    source_kind = chunk["source_id"].map(lambda x: node_meta.get(x, {}).get("kind", node_kind_from_id(x)))
    target_kind = chunk["target_id"].map(lambda x: node_meta.get(x, {}).get("kind", node_kind_from_id(x)))

    source_gene = [is_gene_node(n, k) for n, k in zip(chunk["source_id"], source_kind)]
    target_gene = [is_gene_node(n, k) for n, k in zip(chunk["target_id"], target_kind)]

    source_gene = pd.Series(source_gene, index=chunk.index)
    target_gene = pd.Series(target_gene, index=chunk.index)

    exactly_one_gene = source_gene ^ target_gene

    excluded_counts["both_gene_or_GiG_like_edges"] += int((source_gene & target_gene).sum())
    excluded_counts["no_gene_endpoint_edges"] += int((~source_gene & ~target_gene).sum())

    keep = chunk.loc[exactly_one_gene].copy()
    keep_source_gene = source_gene.loc[exactly_one_gene]

    for row, gene_is_source in zip(keep.itertuples(index=False), keep_source_gene.tolist()):
        if gene_is_source:
            gene_node_id = row.source_id
            neighbor_node_id = row.target_id
            gene_side = "source"
        else:
            gene_node_id = row.target_id
            neighbor_node_id = row.source_id
            gene_side = "target"

        gene_row = get_gene_row(gene_node_id)
        if gene_row is None:
            continue

        feature_col, neighbor_kind = get_feature_col(row.metaedge, gene_side, neighbor_node_id)

        row_idx.append(gene_row)
        col_idx.append(feature_col)
        data.append(1.0)

        included_metaedge_counts[clean(row.metaedge)] += 1
        included_neighbor_kind_counts[clean(neighbor_kind)] += 1
        included_edges += 1

    print(
        f"Chunk {i}: total edges so far {total_edges:,}; "
        f"included gene-context edges so far {included_edges:,}; "
        f"genes {len(gene_to_row):,}; features {len(feature_to_col):,}"
    )

print("\nFinished reading.")
print(f"Total Hetionet edges: {total_edges:,}")
print(f"Included non-PPI gene-context edges: {included_edges:,}")
print(f"Source genes with context: {len(gene_to_row):,}")
print(f"Typed biomedical context features: {len(feature_to_col):,}")
print("Excluded counts:", dict(excluded_counts))

if included_edges == 0:
    raise RuntimeError("No Hetionet gene-context edges were included. Check input format.")

# Sparse gene x feature matrix.
n_genes = len(gene_to_row)
n_features = len(feature_to_col)

M = coo_matrix(
    (
        np.asarray(data, dtype=np.float32),
        (np.asarray(row_idx, dtype=np.int32), np.asarray(col_idx, dtype=np.int32)),
    ),
    shape=(n_genes, n_features),
    dtype=np.float32,
).tocsr()
M.sum_duplicates()

print("\nSparse matrix:")
print("shape:", M.shape)
print("nnz:", M.nnz)
print("density:", M.nnz / (M.shape[0] * M.shape[1]))

# TF-IDF + SVD.
print("\nApplying TF-IDF...")
tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
M_tfidf = tfidf.fit_transform(M)

n_components = min(256, M_tfidf.shape[0] - 1, M_tfidf.shape[1] - 1)
if n_components < 256:
    print(f"Warning: using only {n_components} SVD components due to matrix shape.")

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

# Save source-level tables.
source_genes = pd.DataFrame(
    [
        {
            "source_row": row,
            "source_entrez_id": entrez_id,
            "source_gene_symbol": gene_symbol_by_entrez.get(entrez_id, ""),
            "source_node_id": f"Gene::{entrez_id}",
        }
        for entrez_id, row in gene_to_row.items()
    ]
).sort_values("source_row")

source_genes.to_csv(out_dir / "source_genes.tsv", sep="\t", index=False)

features = pd.DataFrame(feature_records).sort_values("feature_col")
features.to_csv(out_dir / "feature_columns.tsv", sep="\t", index=False)

pd.DataFrame(
    sorted(included_metaedge_counts.items()),
    columns=["metaedge", "included_edge_count"],
).to_csv(out_dir / "included_metaedge_counts.tsv", sep="\t", index=False)

pd.DataFrame(
    sorted(included_neighbor_kind_counts.items()),
    columns=["neighbor_kind", "included_edge_count"],
).to_csv(out_dir / "included_neighbor_kind_counts.tsv", sep="\t", index=False)

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
    "source": "Hetionet v1.0",
    "source_edges_file": str(edges_file),
    "source_nodes_file": str(nodes_file),
    "master_table": str(master_file),
    "identifier_mapping": "Hetionet Gene::Entrez node IDs mapped to master entrez_id.",
    "important_design_decision": "Edges where both endpoints are Gene nodes were excluded to avoid duplicating PPI/node2vec-style information. This embedding represents non-PPI biomedical context only.",
    "feature_definition": "Each sparse feature is metaedge + gene_side + neighbor_kind + neighbor_id for a non-gene biomedical neighbor connected to a gene.",
    "preprocessing": "Built sparse gene x typed-neighbor-feature count matrix; collapsed duplicate gene-feature edges by summation; applied sublinear TF-IDF weighting; reduced with randomized TruncatedSVD; z-scored SVD dimensions across source genes; mapped to master by Entrez ID.",
    "embedding_shape": list(X.shape),
    "embedding_dimension": int(X.shape[1]),
    "source_sparse_matrix_shape": [int(M.shape[0]), int(M.shape[1])],
    "source_sparse_matrix_nnz": int(M.nnz),
    "n_total_hetionet_edges": int(total_edges),
    "n_included_nonppi_gene_context_edges": int(included_edges),
    "n_excluded_both_gene_or_GiG_like_edges": int(excluded_counts["both_gene_or_GiG_like_edges"]),
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
        "included_metaedge_counts": str(out_dir / "included_metaedge_counts.tsv"),
        "included_neighbor_kind_counts": str(out_dir / "included_neighbor_kind_counts.tsv"),
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
======================================================================

Status: processed_qc_passed

Input
-----
Hetionet edges file: {edges_file}
Hetionet nodes file: {nodes_file}
Master table:        {master_file}

Design decision
---------------
This is a non-PPI heterogeneous biomedical context embedding.
All edges where both endpoints are Gene nodes were excluded.
This avoids duplicating Node2Vec-consensus-PPI-v1.1 and focuses the embedding on
compound, disease, anatomy, biological process, molecular function, cellular
component, pathway, side effect, symptom, and pharmacologic-class context.

Source data
-----------
Total Hetionet edges read: {total_edges}
Included non-PPI gene-context edges: {included_edges}
Excluded both-gene/GiG-like edges: {excluded_counts["both_gene_or_GiG_like_edges"]}
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
{out_dir / "included_metaedge_counts.tsv"}
{out_dir / "included_neighbor_kind_counts.tsv"}
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
