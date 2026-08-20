from pathlib import Path
from collections import defaultdict
import gzip
import json
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler


home = Path.home()

MASTER_PATH = home / "metadata/master_gene_table_v1_1_enriched.csv"

RAW_DIR = home / "data/raw_embeddings/deepnf_original_try/string_v12"
EDGELIST_DIR = RAW_DIR / "deepnf_edgelists"
ALIASES_PATH = RAW_DIR / "9606.protein.aliases.v12.0.txt.gz"
INFO_PATH = RAW_DIR / "9606.protein.info.v12.0.txt.gz"

OUT_STEM = "deepnf_string_v12_ppmi_svd_fusion_v1_1"
DISPLAY_NAME = "deepNF-STRING-v12-PPMI-SVD-fusion-v1.1"

OUT_DIR = home / "data/processed_embeddings" / OUT_STEM
REPORT_DIR = home / "reports/mapping_reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / f"{OUT_STEM}_embeddings.npz"
OUT_GENES = OUT_DIR / f"{OUT_STEM}_genes.tsv"
OUT_SOURCE_NODES = OUT_DIR / f"{OUT_STEM}_source_string_nodes.tsv"
OUT_META = OUT_DIR / f"{OUT_STEM}_metadata.json"

REPORT_TXT = REPORT_DIR / f"{OUT_STEM}_mapping_report.txt"
UNMAPPED_TSV = REPORT_DIR / f"{OUT_STEM}_unmapped_source_identifiers.tsv"
AMBIGUOUS_TSV = REPORT_DIR / f"{OUT_STEM}_ambiguous_source_identifiers.tsv"
DUPLICATES_TSV = REPORT_DIR / f"{OUT_STEM}_duplicate_final_mappings.tsv"

SEED = 42
PER_CHANNEL_DIM = 128
FINAL_DIM = 256


def split_values(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return []
    return [t.strip() for t in re.split(r"[|,;\s]+", s) if t.strip()]


def clean_entrez(x):
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s if re.fullmatch(r"\d+", s) else ""


def clean_ensp(x):
    s = str(x).strip()
    if s.startswith("9606."):
        s = s.split(".", 1)[1]
    if re.fullmatch(r"ENSP\d+(\.\d+)?", s):
        return s.split(".")[0]
    return ""


def first_existing(row, cols):
    for c in cols:
        if c in row.index and str(row[c]).strip():
            return str(row[c]).strip()
    return ""


def read_string_info():
    preferred = {}
    with gzip.open(INFO_PATH, "rt") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t", 3)
            if len(parts) >= 2:
                preferred[parts[0]] = parts[1]
    return preferred


def read_string_alias_entrez():
    out = defaultdict(set)
    good_source_terms = ["geneid", "entrez", "hgnc_entrez", "kegg_geneid"]

    with gzip.open(ALIASES_PATH, "rt") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid, alias, source = parts[0], parts[1], parts[2]
            if not any(t in source.lower() for t in good_source_terms):
                continue
            eid = clean_entrez(alias)
            if eid:
                out[sid].add(eid)

    return out


def build_master_maps(master):
    entrez_to_rows = defaultdict(set)
    ensp_to_rows = defaultdict(set)
    string_to_rows = defaultdict(set)
    symbol_to_rows = defaultdict(set)

    entrez_cols = ["entrez_id", "hgnc_entrez_id", "map_entrez_ids_all"]
    ensp_cols = [
        "canonical_protein_id",
        "all_ensembl_protein_ids_grch38_current",
        "all_ensembl_protein_ids_grch37",
        "map_ensembl_protein_ids_all",
    ]
    string_cols = ["map_string_protein_ids_all"]
    symbol_cols = ["gene_symbol", "hgnc_approved_symbol"]

    for i, row in master.iterrows():
        for c in entrez_cols:
            if c in master.columns:
                for tok in split_values(row[c]):
                    eid = clean_entrez(tok)
                    if eid:
                        entrez_to_rows[eid].add(i)

        for c in ensp_cols:
            if c in master.columns:
                for tok in split_values(row[c]):
                    ensp = clean_ensp(tok)
                    if ensp:
                        ensp_to_rows[ensp].add(i)

        for c in string_cols:
            if c in master.columns:
                for tok in split_values(row[c]):
                    tok = str(tok).strip()
                    if tok:
                        string_to_rows[tok].add(i)
                    ensp = clean_ensp(tok)
                    if ensp:
                        ensp_to_rows[ensp].add(i)

        for c in symbol_cols:
            if c in master.columns:
                sym = str(row.get(c, "")).strip()
                if sym:
                    symbol_to_rows[sym.upper()].add(i)

    return {
        "entrez": entrez_to_rows,
        "ensp": ensp_to_rows,
        "string": string_to_rows,
        "symbol": symbol_to_rows,
    }


def map_string_node(sid, master_maps, alias_entrez, preferred_symbol):
    candidates = defaultdict(set)

    if sid in master_maps["string"]:
        candidates["string_id_exact"].update(master_maps["string"][sid])

    ensp = clean_ensp(sid)
    if ensp and ensp in master_maps["ensp"]:
        candidates["ensembl_protein_id_exact"].update(master_maps["ensp"][ensp])

    for eid in alias_entrez.get(sid, []):
        if eid in master_maps["entrez"]:
            candidates["string_alias_entrez"].update(master_maps["entrez"][eid])

    sym = preferred_symbol.get(sid, "")
    if sym and sym.upper() in master_maps["symbol"]:
        candidates["string_preferred_symbol"].update(master_maps["symbol"][sym.upper()])

    all_rows = set()
    methods = []
    for method, rows in candidates.items():
        if rows:
            all_rows.update(rows)
            methods.append(method)

    if len(all_rows) == 1:
        return next(iter(all_rows)), ";".join(sorted(methods)), ""
    if len(all_rows) > 1:
        return None, ";".join(sorted(methods)), "maps_to_multiple_master_rows"
    return None, "", "unmapped"


def build_node_universe(edgelist_paths):
    nodes = set()
    channel_counts = {}

    for p in edgelist_paths:
        edges = 0
        with open(p) as f:
            for line in f:
                a, b, w = line.strip().split()
                nodes.add(a)
                nodes.add(b)
                edges += 1
        channel_counts[p.stem] = edges

    return sorted(nodes), channel_counts


def build_adjacency(path, node_to_idx, n):
    rows = []
    cols = []
    data = []

    with open(path) as f:
        for line in f:
            a, b, w = line.strip().split()
            i = node_to_idx[a]
            j = node_to_idx[b]
            val = float(w)
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([val, val])

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


def sparse_ppmi(A):
    """
    Sparse PPMI on nonzero entries only:
      PPMI_ij = max(log(total * A_ij / (row_i * col_j)), 0)

    This is stable and avoids creating a dense 19k x 19k matrix.
    """
    A = A.tocsr().astype(np.float32)
    A.eliminate_zeros()

    total = float(A.sum())
    row_sum = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    col_sum = np.asarray(A.sum(axis=0)).ravel().astype(np.float64)

    coo = A.tocoo()
    denom = row_sum[coo.row] * col_sum[coo.col]

    valid = (coo.data > 0) & (denom > 0)
    rows = coo.row[valid]
    cols = coo.col[valid]
    vals = coo.data[valid].astype(np.float64)
    denom = denom[valid]

    ppmi_vals = np.log((total * vals) / denom)
    keep = ppmi_vals > 0

    P = sparse.csr_matrix(
        (ppmi_vals[keep].astype(np.float32), (rows[keep], cols[keep])),
        shape=A.shape,
        dtype=np.float32,
    )
    P.sum_duplicates()
    P.eliminate_zeros()
    return P


def standardize_dense(X):
    X = X.astype(np.float32)
    scaler = StandardScaler()
    return scaler.fit_transform(X).astype(np.float32)


def main():
    print("Loading master table...")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

    print("Loading STRING alias/info mappings...")
    alias_entrez = read_string_alias_entrez()
    preferred_symbol = read_string_info()
    master_maps = build_master_maps(master)

    edgelist_paths = sorted(EDGELIST_DIR.glob("string_v12_*_min150.edgelist"))
    if not edgelist_paths:
        raise FileNotFoundError(f"No edgelists found in {EDGELIST_DIR}")

    print("Edgelists:")
    for p in edgelist_paths:
        print(" ", p.name)

    print("Building source STRING node universe...")
    source_nodes, channel_edge_counts = build_node_universe(edgelist_paths)
    node_to_idx = {node: i for i, node in enumerate(source_nodes)}
    n = len(source_nodes)

    print("Total STRING source nodes:", n)

    channel_embeddings = []
    channel_summaries = []

    for p in edgelist_paths:
        channel = p.stem.replace("string_v12_", "").replace("_min150", "")
        print(f"\nProcessing channel: {channel}")

        A = build_adjacency(p, node_to_idx, n)
        print("  raw adjacency:", A.shape, "nnz:", A.nnz)

        P = sparse_ppmi(A)
        print("  sparse PPMI:", P.shape, "nnz:", P.nnz)

        P = normalize(P, norm="l2", axis=1, copy=False)

        svd = TruncatedSVD(
            n_components=PER_CHANNEL_DIM,
            n_iter=7,
            random_state=SEED,
        )
        E = svd.fit_transform(P).astype(np.float32)
        E = standardize_dense(E)

        channel_embeddings.append(E)

        explained = float(svd.explained_variance_ratio_.sum())
        channel_summaries.append({
            "channel": channel,
            "edgelist": str(p),
            "edges_input": int(channel_edge_counts[p.stem]),
            "raw_adjacency_nnz_symmetric": int(A.nnz),
            "ppmi_nnz": int(P.nnz),
            "nodes_total": int(n),
            "per_channel_dim": int(PER_CHANNEL_DIM),
            "svd_explained_variance_ratio_sum": explained,
        })

        print("  channel embedding:", E.shape)
        print("  SVD explained variance sum:", explained)

    Z = np.concatenate(channel_embeddings, axis=1).astype(np.float32)
    Z = standardize_dense(Z)
    print("\nConcatenated channel embedding:", Z.shape)

    print("Final SVD fusion...")
    final_svd = TruncatedSVD(
        n_components=FINAL_DIM,
        n_iter=10,
        random_state=SEED,
    )
    X_source = final_svd.fit_transform(Z).astype(np.float32)
    X_source = standardize_dense(X_source)

    print("Source fused embedding:", X_source.shape)
    print("Final SVD explained variance sum:", float(final_svd.explained_variance_ratio_.sum()))

    print("Mapping STRING source nodes to master genes...")
    mapped_records = []
    unmapped_records = []
    ambiguous_records = []

    for src_i, sid in enumerate(source_nodes):
        master_i, method, reason = map_string_node(
            sid=sid,
            master_maps=master_maps,
            alias_entrez=alias_entrez,
            preferred_symbol=preferred_symbol,
        )

        if master_i is not None:
            mapped_records.append({
                "source_row_index": src_i,
                "source_identifier": sid,
                "source_ensembl_protein_id": clean_ensp(sid),
                "preferred_symbol": preferred_symbol.get(sid, ""),
                "master_row_index": master_i,
                "mapping_method": method,
            })
        elif reason == "maps_to_multiple_master_rows":
            ambiguous_records.append({
                "source_row_index": src_i,
                "source_identifier": sid,
                "source_ensembl_protein_id": clean_ensp(sid),
                "preferred_symbol": preferred_symbol.get(sid, ""),
                "reason": reason,
                "mapping_methods": method,
            })
        else:
            unmapped_records.append({
                "source_row_index": src_i,
                "source_identifier": sid,
                "source_ensembl_protein_id": clean_ensp(sid),
                "preferred_symbol": preferred_symbol.get(sid, ""),
                "reason": "no_unique_mapping_to_master",
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
                "action": "averaged_duplicate_STRING_protein_nodes_mapping_to_same_master_gene",
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
            "source_identifier_type": "STRING protein ID / Ensembl protein ID",
            "source_key_count": len(source_row_indices),
            "source_keys_examples": ";".join(source_keys[:10]),
            "source_row_indices_examples": ";".join(map(str, source_row_indices[:10])),
            "mapping_methods": ";".join(methods),
        })

    X_final = np.vstack(X_final).astype(np.float32)
    X_final = standardize_dense(X_final)

    genes_df = pd.DataFrame(gene_rows)

    print("Final X:", X_final.shape)

    np.savez_compressed(
        OUT_NPZ,
        X=X_final,
        ensembl_gene_id=genes_df["ensembl_gene_id"].to_numpy(dtype=str),
        gene_symbol=genes_df["gene_symbol"].to_numpy(dtype=str),
    )

    genes_df.to_csv(OUT_GENES, sep="\t", index=False)

    source_nodes_df = pd.DataFrame({
        "source_row_index": np.arange(len(source_nodes), dtype=int),
        "string_protein_id": source_nodes,
        "ensembl_protein_id": [clean_ensp(x) for x in source_nodes],
        "preferred_symbol": [preferred_symbol.get(x, "") for x in source_nodes],
        "alias_entrez_ids": [";".join(sorted(alias_entrez.get(x, []))) for x in source_nodes],
    })
    source_nodes_df.to_csv(OUT_SOURCE_NODES, sep="\t", index=False)

    pd.DataFrame(unmapped_records).to_csv(UNMAPPED_TSV, sep="\t", index=False)
    pd.DataFrame(ambiguous_records).to_csv(AMBIGUOUS_TSV, sep="\t", index=False)
    pd.DataFrame(duplicate_rows).to_csv(DUPLICATES_TSV, sep="\t", index=False)

    metadata = {
        "embedding_name": DISPLAY_NAME,
        "source_embedding": DISPLAY_NAME,
        "core_status": "candidate_core_pending_supervisor_approval",
        "modality": "STRING network multimodal sparse PPMI/SVD fusion",
        "source_identifier_type": "STRING protein ID / Ensembl protein ID",
        "download_source": "STRING v12 human protein.links.detailed, protein.info, and protein.aliases files",
        "original_model_or_method": "deepNF: deep network fusion for protein function prediction",
        "embedding_generated_by": "Daniel locally from STRING v12 channel networks",
        "exact_original_reproduction": False,
        "exact_original_reproduction_note": "The original deepNF data link returned 404, no archived Wayback files were found, GitHub issues report the data link unavailable, and inspected forks did not contain the original matrices. This is a stable regenerated deepNF-derived representation using sparse PPMI and SVD fusion.",
        "input_data_or_sequence_source": "STRING v12 Homo sapiens evidence-channel network scores",
        "string_channels": [s["channel"] for s in channel_summaries],
        "min_channel_score": 150,
        "per_channel_svd_dim": PER_CHANNEL_DIM,
        "final_svd_fusion_dim": FINAL_DIM,
        "pipeline": [
            "Split STRING v12 detailed network into evidence channels",
            "Build one sparse weighted adjacency per evidence channel",
            "Compute sparse PPMI on nonzero entries",
            "L2-normalize sparse PPMI rows",
            "TruncatedSVD per channel to 128 dimensions",
            "Concatenate seven channel embeddings to 896 dimensions",
            "Final TruncatedSVD fusion to 256 dimensions",
            "Standardize final dimensions",
            "Map STRING protein nodes to master genes and average duplicate STRING proteins mapping to the same gene",
        ],
        "final_svd": {
            "n_components": FINAL_DIM,
            "n_iter": 10,
            "random_state": SEED,
            "explained_variance_ratio_sum": float(final_svd.explained_variance_ratio_.sum()),
        },
        "source_files": {
            "string_links_detailed": str(RAW_DIR / "9606.protein.links.detailed.v12.0.txt.gz"),
            "string_info": str(INFO_PATH),
            "string_aliases": str(ALIASES_PATH),
            "edgelist_dir": str(EDGELIST_DIR),
        },
        "channel_summaries": channel_summaries,
        "master_table": str(MASTER_PATH),
        "mapping_strategy": "Map STRING protein nodes to master genes using exact STRING protein IDs, Ensembl protein IDs, STRING alias Entrez IDs, and preferred symbols. Only unique final master mappings are accepted. Multiple STRING protein nodes mapping to the same master gene are averaged.",
        "counts": {
            "source_string_nodes": int(len(source_nodes)),
            "source_concat_dimensions": int(Z.shape[1]),
            "source_final_dimensions": int(X_source.shape[1]),
            "mapped_source_nodes_before_duplicate_collapse": int(len(mapped_records)),
            "unmapped_source_nodes": int(len(unmapped_records)),
            "ambiguous_source_nodes": int(len(ambiguous_records)),
            "final_mapped_genes": int(X_final.shape[0]),
            "final_dimensions": int(X_final.shape[1]),
            "master_rows": int(len(master)),
            "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
        },
        "coverage": {
            "source_node_mapping_fraction": float(len(mapped_records) / len(source_nodes)) if source_nodes else 0.0,
            "master_gene_coverage_fraction": float(X_final.shape[0] / len(master)),
        },
        "outputs": {
            "npz": str(OUT_NPZ),
            "genes_tsv": str(OUT_GENES),
            "source_nodes_tsv": str(OUT_SOURCE_NODES),
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
        f.write("Provenance note:\n")
        f.write("This is a locally regenerated deepNF-derived STRING v12 multimodal network embedding.\n")
        f.write("It is not an exact original deepNF reproduction because the original released data could not be recovered.\n")
        f.write("This version uses sparse PPMI and SVD fusion instead of the unstable local autoencoder attempt.\n\n")

        f.write("Channel summaries:\n")
        for s in channel_summaries:
            f.write(json.dumps(s) + "\n")

        f.write("\nFinal SVD:\n")
        for k, v in metadata["final_svd"].items():
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


if __name__ == "__main__":
    main()
