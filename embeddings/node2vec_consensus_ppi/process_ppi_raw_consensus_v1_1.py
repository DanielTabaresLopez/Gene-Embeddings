from pathlib import Path
from collections import defaultdict
import json
import re

import numpy as np
import pandas as pd
from scipy import sparse


home = Path.home()

MASTER_PATH = home / "metadata/master_gene_table_v1_1_enriched.csv"
EDGE_PATH = home / "data/raw_embeddings/node2vec_ppi_original_try/zhong_repo/src/other/preprocess_embedding/consensus.dat"

OUT_STEM = "ppi_raw_consensus_v1_1"
DISPLAY_NAME = "PPI-RAW-consensus-v1.1"

OUT_DIR = home / "data/processed_embeddings" / OUT_STEM
REPORT_DIR = home / "reports/mapping_reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SPARSE = OUT_DIR / f"{OUT_STEM}_embeddings_sparse.npz"
OUT_GENES = OUT_DIR / f"{OUT_STEM}_genes.tsv"
OUT_FEATURES = OUT_DIR / f"{OUT_STEM}_feature_nodes.tsv"
OUT_META = OUT_DIR / f"{OUT_STEM}_metadata.json"

REPORT_TXT = REPORT_DIR / f"{OUT_STEM}_mapping_report.txt"
UNMAPPED_TSV = REPORT_DIR / f"{OUT_STEM}_unmapped_source_identifiers.tsv"
AMBIGUOUS_TSV = REPORT_DIR / f"{OUT_STEM}_ambiguous_source_identifiers.tsv"
DUPLICATES_TSV = REPORT_DIR / f"{OUT_STEM}_duplicate_final_mappings.tsv"


def split_values(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return []
    return [t.strip() for t in re.split(r"[|,;\s]+", s) if t.strip()]


def clean_entrez(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s if re.fullmatch(r"\d+", s) else ""


def first_existing(row, cols):
    for c in cols:
        if c in row.index and str(row[c]).strip() and str(row[c]).lower() != "nan":
            return str(row[c]).strip()
    return ""


def build_entrez_mapping(master):
    mapping = defaultdict(set)
    candidate_cols = ["entrez_id", "hgnc_entrez_id", "map_entrez_ids_all"]

    for i, row in master.iterrows():
        for c in candidate_cols:
            if c not in master.columns:
                continue
            for token in split_values(row[c]):
                eid = clean_entrez(token)
                if eid:
                    mapping[eid].add(i)

    unique = {eid: next(iter(rows)) for eid, rows in mapping.items() if len(rows) == 1}
    ambiguous = {eid: sorted(rows) for eid, rows in mapping.items() if len(rows) > 1}

    return unique, ambiguous


def read_edges(edge_path):
    edges = pd.read_csv(edge_path, sep="\t", header=None, names=["gene1", "gene2"], dtype=str)

    cleaned_edges = []
    node_set = set()
    skipped = 0

    for a, b in edges[["gene1", "gene2"]].itertuples(index=False):
        a = clean_entrez(a)
        b = clean_entrez(b)

        if not a or not b or a == b:
            skipped += 1
            continue

        cleaned_edges.append((a, b))
        node_set.add(a)
        node_set.add(b)

    nodes = sorted(node_set, key=lambda x: int(x))
    node_to_col = {node: i for i, node in enumerate(nodes)}

    rows = []
    cols = []

    seen = set()
    duplicated_edge_rows = 0

    for a, b in cleaned_edges:
        ia = node_to_col[a]
        ib = node_to_col[b]

        key = (min(ia, ib), max(ia, ib))
        if key in seen:
            duplicated_edge_rows += 1
            continue
        seen.add(key)

        rows.extend([ia, ib])
        cols.extend([ib, ia])

    data = np.ones(len(rows), dtype=np.float32)

    A = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(nodes), len(nodes)),
        dtype=np.float32,
    )

    stats = {
        "raw_edge_rows": int(len(edges)),
        "skipped_self_or_bad_edges": int(skipped),
        "unique_undirected_edges": int(len(seen)),
        "duplicated_clean_edge_rows_removed": int(duplicated_edge_rows),
        "source_nodes": int(len(nodes)),
        "source_dimensions": int(len(nodes)),
        "sparse_nonzero_entries": int(A.nnz),
    }

    return nodes, A, stats


def main():
    print("Loading master table...")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

    unique_map, ambiguous_map = build_entrez_mapping(master)

    print("Unique Entrez mappings:", len(unique_map))
    print("Ambiguous Entrez mappings:", len(ambiguous_map))

    print("Reading consensus PPI graph and building sparse adjacency...")
    source_nodes, A, graph_stats = read_edges(EDGE_PATH)

    print("Source adjacency:", A.shape)
    print("Nonzero entries:", A.nnz)

    mapped_records = []
    unmapped_records = []
    ambiguous_records = []

    for src_i, eid in enumerate(source_nodes):
        if eid in unique_map:
            mapped_records.append({
                "source_row_index": src_i,
                "source_identifier": eid,
                "normalized_entrez_id": eid,
                "master_row_index": unique_map[eid],
                "mapping_method": "entrez_id_exact_unique",
            })
        elif eid in ambiguous_map:
            rows = ambiguous_map[eid]
            ambiguous_records.append({
                "source_row_index": src_i,
                "source_identifier": eid,
                "normalized_entrez_id": eid,
                "reason": "entrez_id_maps_to_multiple_master_rows",
                "master_row_indices": ";".join(map(str, rows)),
                "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
            })
        else:
            unmapped_records.append({
                "source_row_index": src_i,
                "source_identifier": eid,
                "normalized_entrez_id": eid,
                "reason": "entrez_id_not_in_master_gene_table",
            })

    master_to_source = defaultdict(list)
    for rec in mapped_records:
        master_to_source[rec["master_row_index"]].append(rec)

    final_master_indices = sorted(master_to_source.keys())

    final_rows = []
    gene_rows = []
    duplicate_rows = []

    print("Collapsing rows to final master genes...")
    for master_i in final_master_indices:
        source_items = master_to_source[master_i]
        source_row_indices = [r["source_row_index"] for r in source_items]
        source_keys = [r["source_identifier"] for r in source_items]
        methods = sorted(set(r["mapping_method"] for r in source_items))

        row_vec = A[source_row_indices].mean(axis=0)
        row_vec = sparse.csr_matrix(row_vec, dtype=np.float32)
        final_rows.append(row_vec)

        row = master.loc[master_i]

        if len(source_row_indices) > 1:
            duplicate_rows.append({
                "ensembl_gene_id": row["ensembl_gene_id"],
                "gene_symbol": row.get(symbol_col, ""),
                "source_key_count": len(source_row_indices),
                "source_keys": ";".join(source_keys),
                "source_row_indices": ";".join(map(str, source_row_indices)),
                "mapping_methods": ";".join(methods),
                "action": "averaged_duplicate_source_rows_mapping_to_same_master_gene",
            })

        gene_rows.append({
            "ensembl_gene_id": row["ensembl_gene_id"],
            "gene_symbol": row.get(symbol_col, ""),
            "gene_type": row.get("gene_type", ""),
            "entrez_id": first_existing(row, ["entrez_id", "hgnc_entrez_id"]),
            "uniprot_id": first_existing(row, ["uniprot_id", "hgnc_uniprot_ids"]),
            "canonical_transcript_id": row.get("canonical_transcript_id", ""),
            "canonical_protein_id": row.get("canonical_protein_id", ""),
            "source_embedding": DISPLAY_NAME,
            "source_identifier_type": "Entrez Gene ID from consensus PPI graph node",
            "source_key_count": len(source_row_indices),
            "source_keys_examples": ";".join(source_keys[:10]),
            "source_row_indices_examples": ";".join(map(str, source_row_indices[:10])),
            "mapping_methods": ";".join(methods),
        })

    X_final = sparse.vstack(final_rows, format="csr", dtype=np.float32)
    genes_df = pd.DataFrame(gene_rows)

    print("Final sparse X:", X_final.shape)
    print("Final nonzero entries:", X_final.nnz)

    sparse.save_npz(OUT_SPARSE, X_final, compressed=True)

    genes_df.to_csv(OUT_GENES, sep="\t", index=False)

    features_df = pd.DataFrame({
        "feature_index": np.arange(len(source_nodes), dtype=int),
        "feature_source_entrez_id": source_nodes,
        "feature_description": ["PPI adjacency column for source graph node Entrez ID"] * len(source_nodes),
    })
    features_df.to_csv(OUT_FEATURES, sep="\t", index=False)

    pd.DataFrame(unmapped_records).to_csv(UNMAPPED_TSV, sep="\t", index=False)
    pd.DataFrame(ambiguous_records).to_csv(AMBIGUOUS_TSV, sep="\t", index=False)
    pd.DataFrame(duplicate_rows).to_csv(DUPLICATES_TSV, sep="\t", index=False)

    metadata = {
        "embedding_name": DISPLAY_NAME,
        "source_embedding": DISPLAY_NAME,
        "core_status": "optional_non_core_baseline",
        "modality": "PPI network raw adjacency baseline",
        "source_identifier_type": "Entrez Gene ID",
        "feature_identifier_type": "Entrez Gene ID source graph nodes",
        "download_source": "ylaboratory/gene-embedding-benchmarks GitHub repository: src/other/preprocess_embedding/consensus.dat",
        "original_model_or_method": "PPI-RAW raw adjacency matrix baseline",
        "embedding_generated_by": "Daniel locally from consensus.dat",
        "input_graph": "Consensus human PPI edge list used by Zhong preprocessing; two-column Entrez-ID interaction file",
        "storage_format": "scipy.sparse CSR saved with scipy.sparse.save_npz; not dense standard X because PPI-RAW is high-dimensional and sparse",
        "not_core_reason": "Raw adjacency baseline, non-ML, very high-dimensional. Useful for controls but not counted in the core final embedding set unless explicitly needed.",
        "source_files": {
            "edge_path": str(EDGE_PATH),
        },
        "master_table": str(MASTER_PATH),
        "mapping_strategy": "Exact Entrez ID mapping to master_gene_table_v1_1_enriched.csv using entrez_id, hgnc_entrez_id, and map_entrez_ids_all. Only unique mappings are accepted. Duplicate source rows mapping to the same final master gene are averaged.",
        "counts": {
            **graph_stats,
            "mapped_source_nodes_before_duplicate_collapse": int(len(mapped_records)),
            "unmapped_source_nodes": int(len(unmapped_records)),
            "ambiguous_source_nodes": int(len(ambiguous_records)),
            "final_mapped_genes": int(X_final.shape[0]),
            "final_dimensions": int(X_final.shape[1]),
            "final_sparse_nonzero_entries": int(X_final.nnz),
            "master_rows": int(len(master)),
            "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
        },
        "coverage": {
            "source_node_mapping_fraction": float(len(mapped_records) / len(source_nodes)) if source_nodes else 0.0,
            "master_gene_coverage_fraction": float(X_final.shape[0] / len(master)),
        },
        "outputs": {
            "sparse_npz": str(OUT_SPARSE),
            "genes_tsv": str(OUT_GENES),
            "feature_nodes_tsv": str(OUT_FEATURES),
            "metadata_json": str(OUT_META),
            "mapping_report": str(REPORT_TXT),
            "unmapped_tsv": str(UNMAPPED_TSV),
            "ambiguous_tsv": str(AMBIGUOUS_TSV),
            "duplicates_tsv": str(DUPLICATES_TSV),
        },
    }

    with open(OUT_META, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(REPORT_TXT, "w") as f:
        f.write(f"{DISPLAY_NAME} mapping report\n")
        f.write("=" * (len(DISPLAY_NAME) + 16) + "\n\n")
        f.write(f"Edge path: {EDGE_PATH}\n")
        f.write(f"Master table: {MASTER_PATH}\n\n")
        f.write("Provenance note:\n")
        f.write("This optional baseline was generated locally from the consensus.dat PPI graph input found in the Zhong benchmark repository.\n")
        f.write("It is the raw adjacency representation of the consensus PPI graph, saved as sparse CSR because the matrix is high-dimensional and mostly zeros.\n")
        f.write("It is not counted in the core embedding set unless explicitly needed for raw-network controls.\n\n")

        f.write("Counts:\n")
        for k, v in metadata["counts"].items():
            f.write(f"{k}: {v}\n")

        f.write("\nCoverage:\n")
        for k, v in metadata["coverage"].items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\nOutputs:\n")
        for k, v in metadata["outputs"].items():
            f.write(f"{k}: {v}\n")

    print("\nDone")
    print("Sparse X:", X_final.shape)
    print("Nonzero entries:", X_final.nnz)
    print("Report:", REPORT_TXT)


if __name__ == "__main__":
    main()
