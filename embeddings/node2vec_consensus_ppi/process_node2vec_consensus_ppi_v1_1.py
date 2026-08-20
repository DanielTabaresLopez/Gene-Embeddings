from pathlib import Path
import json
import re
import os
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


home = Path.home()

MASTER_PATH = home / "metadata/master_gene_table_v1_1_enriched.csv"
EDGE_PATH = home / "data/raw_embeddings/node2vec_ppi_original_try/zhong_repo/src/other/preprocess_embedding/consensus.dat"

OUT_STEM = "node2vec_consensus_ppi_v1_1"
DISPLAY_NAME = "Node2Vec-consensus-PPI-v1.1"

OUT_DIR = home / "data/processed_embeddings" / OUT_STEM
REPORT_DIR = home / "reports/mapping_reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / f"{OUT_STEM}_embeddings.npz"
OUT_GENES = OUT_DIR / f"{OUT_STEM}_genes.tsv"
OUT_META = OUT_DIR / f"{OUT_STEM}_metadata.json"

REPORT_TXT = REPORT_DIR / f"{OUT_STEM}_mapping_report.txt"
UNMAPPED_TSV = REPORT_DIR / f"{OUT_STEM}_unmapped_source_identifiers.tsv"
AMBIGUOUS_TSV = REPORT_DIR / f"{OUT_STEM}_ambiguous_source_identifiers.tsv"
DUPLICATES_TSV = REPORT_DIR / f"{OUT_STEM}_duplicate_final_mappings.tsv"

# Reproducible local node2vec/DeepWalk-style settings.
# p=q=1 corresponds to unbiased node2vec random walks.
SEED = 42
DIMENSIONS = 128
WALK_LENGTH = 80
NUM_WALKS = 10
WINDOW = 10
EPOCHS = 5
MIN_COUNT = 1
SG = 1
WORKERS = min(8, max(1, os.cpu_count() or 1))


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

    adjacency = defaultdict(set)
    skipped = 0

    for a, b in edges[["gene1", "gene2"]].itertuples(index=False):
        a = clean_entrez(a)
        b = clean_entrez(b)

        if not a or not b or a == b:
            skipped += 1
            continue

        adjacency[a].add(b)
        adjacency[b].add(a)

    nodes = sorted(adjacency.keys(), key=lambda x: int(x))
    neighbors = {
        node: np.array(sorted(adjacency[node], key=lambda x: int(x)), dtype=object)
        for node in nodes
    }

    edge_count_undirected = sum(len(v) for v in adjacency.values()) // 2

    return nodes, neighbors, int(len(edges)), edge_count_undirected, skipped


def generate_random_walks(nodes, neighbors):
    rng = np.random.default_rng(SEED)
    walks = []

    nodes_array = np.array(nodes, dtype=object)

    for walk_round in range(NUM_WALKS):
        order = nodes_array.copy()
        rng.shuffle(order)

        for start in order:
            walk = [start]
            current = start

            for _ in range(WALK_LENGTH - 1):
                neigh = neighbors.get(current)
                if neigh is None or len(neigh) == 0:
                    break
                current = neigh[rng.integers(0, len(neigh))]
                walk.append(current)

            walks.append(walk)

        print(f"Generated walk round {walk_round + 1}/{NUM_WALKS}; total walks so far: {len(walks)}")

    return walks


def train_node2vec_embeddings(walks, nodes):
    print("Training Word2Vec skip-gram model...")
    model = Word2Vec(
        sentences=walks,
        vector_size=DIMENSIONS,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
        epochs=EPOCHS,
        seed=SEED,
    )

    X_source = np.vstack([model.wv[node] for node in nodes]).astype(np.float32)

    return X_source


def map_and_save(master, unique_map, ambiguous_map, nodes, X_source, source_edge_stats):
    symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

    mapped_records = []
    unmapped_records = []
    ambiguous_records = []

    for src_i, eid in enumerate(nodes):
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

    X_final = []
    gene_rows = []
    duplicate_rows = []

    for master_i in final_master_indices:
        source_items = master_to_source[master_i]
        source_row_indices = [r["source_row_index"] for r in source_items]
        source_keys = [r["source_identifier"] for r in source_items]
        methods = sorted(set(r["mapping_method"] for r in source_items))

        row = master.loc[master_i]

        X_final.append(X_source[source_row_indices].mean(axis=0).astype(np.float32))

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

    X_final = np.vstack(X_final).astype(np.float32)
    genes_df = pd.DataFrame(gene_rows)

    np.savez_compressed(
        OUT_NPZ,
        X=X_final,
        ensembl_gene_id=genes_df["ensembl_gene_id"].to_numpy(dtype=str),
        gene_symbol=genes_df["gene_symbol"].to_numpy(dtype=str),
    )

    genes_df.to_csv(OUT_GENES, sep="\t", index=False)
    pd.DataFrame(unmapped_records).to_csv(UNMAPPED_TSV, sep="\t", index=False)
    pd.DataFrame(ambiguous_records).to_csv(AMBIGUOUS_TSV, sep="\t", index=False)
    pd.DataFrame(duplicate_rows).to_csv(DUPLICATES_TSV, sep="\t", index=False)

    metadata = {
        "embedding_name": DISPLAY_NAME,
        "source_embedding": DISPLAY_NAME,
        "modality": "PPI network embedding",
        "source_identifier_type": "Entrez Gene ID",
        "download_source": "ylaboratory/gene-embedding-benchmarks GitHub repository: src/other/preprocess_embedding/consensus.dat",
        "original_model_or_method": "node2vec: scalable feature learning for networks",
        "embedding_generated_by": "Daniel locally from consensus.dat using reproducible random walks and Word2Vec skip-gram",
        "input_graph": "Consensus human PPI edge list used by Zhong preprocessing; two-column Entrez-ID interaction file",
        "exact_zhong_reproduction": False,
        "exact_zhong_reproduction_note": "The graph input was recovered, but exact NODE2VEC hyperparameters were not found in the inspected Zhong preprocessing notebook. This is therefore a local reproducible node2vec-on-consensus-PPI embedding, not guaranteed byte-identical to Zhong's archived NODE2VEC.",
        "node2vec_parameters": {
            "dimensions": DIMENSIONS,
            "walk_length": WALK_LENGTH,
            "num_walks": NUM_WALKS,
            "p": 1,
            "q": 1,
            "window": WINDOW,
            "epochs": EPOCHS,
            "min_count": MIN_COUNT,
            "sg": SG,
            "seed": SEED,
            "workers": WORKERS,
        },
        "source_files": {
            "edge_path": str(EDGE_PATH),
        },
        "master_table": str(MASTER_PATH),
        "mapping_strategy": "Exact Entrez ID mapping to master_gene_table_v1_1_enriched.csv using entrez_id, hgnc_entrez_id, and map_entrez_ids_all. Only unique mappings are accepted. Duplicate source rows mapping to the same final master gene are averaged.",
        "counts": {
            **source_edge_stats,
            "source_nodes": int(len(nodes)),
            "source_dimensions": int(X_source.shape[1]),
            "mapped_source_nodes_before_duplicate_collapse": int(len(mapped_records)),
            "unmapped_source_nodes": int(len(unmapped_records)),
            "ambiguous_source_nodes": int(len(ambiguous_records)),
            "final_mapped_genes": int(X_final.shape[0]),
            "final_dimensions": int(X_final.shape[1]),
            "master_rows": int(len(master)),
            "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
        },
        "coverage": {
            "source_node_mapping_fraction": float(len(mapped_records) / len(nodes)) if nodes else 0.0,
            "master_gene_coverage_fraction": float(X_final.shape[0] / len(master)),
        },
        "outputs": {
            "npz": str(OUT_NPZ),
            "genes_tsv": str(OUT_GENES),
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
        f.write("This embedding was generated locally from the consensus.dat PPI graph input found in the Zhong benchmark repository.\n")
        f.write("The exact graph input is recovered, but exact Zhong NODE2VEC hyperparameters were not found in the inspected notebook.\n")
        f.write("Therefore this is a local reproducible node2vec-on-consensus-PPI embedding, not guaranteed byte-identical to Zhong's NODE2VEC archive.\n\n")

        f.write("Node2Vec parameters:\n")
        for k, v in metadata["node2vec_parameters"].items():
            f.write(f"{k}: {v}\n")

        f.write("\nCounts:\n")
        for k, v in metadata["counts"].items():
            f.write(f"{k}: {v}\n")

        f.write("\nCoverage:\n")
        for k, v in metadata["coverage"].items():
            f.write(f"{k}: {v:.6f}\n")

        f.write("\nOutputs:\n")
        for k, v in metadata["outputs"].items():
            f.write(f"{k}: {v}\n")

    print("\nDone")
    print("X_final:", X_final.shape)
    print("genes:", len(genes_df))
    print("Report:", REPORT_TXT)


def main():
    print("Loading master table...")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    unique_map, ambiguous_map = build_entrez_mapping(master)

    print("Unique Entrez mappings:", len(unique_map))
    print("Ambiguous Entrez mappings:", len(ambiguous_map))

    print("Reading PPI edge list...")
    nodes, neighbors, raw_edge_rows, undirected_edges, skipped_edges = read_edges(EDGE_PATH)

    print("Source nodes:", len(nodes))
    print("Raw edge rows:", raw_edge_rows)
    print("Undirected non-self edges:", undirected_edges)
    print("Skipped/self/bad edges:", skipped_edges)

    print("Generating random walks...")
    walks = generate_random_walks(nodes, neighbors)

    print("Number of walks:", len(walks))
    print("Approx token count:", sum(len(w) for w in walks))

    X_source = train_node2vec_embeddings(walks, nodes)

    del walks
    gc.collect()

    source_edge_stats = {
        "raw_edge_rows": int(raw_edge_rows),
        "undirected_nonself_edges": int(undirected_edges),
        "skipped_self_or_bad_edges": int(skipped_edges),
    }

    map_and_save(master, unique_map, ambiguous_map, nodes, X_source, source_edge_stats)


if __name__ == "__main__":
    main()
