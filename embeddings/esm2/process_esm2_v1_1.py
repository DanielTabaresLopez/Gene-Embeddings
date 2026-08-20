from pathlib import Path
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd


EMBEDDING_NAME = "ESM2-v1.1"
SOURCE_EMBEDDING = "Zhong_all_genes_ESM2"
SOURCE_IDENTIFIER_TYPE = "Entrez"

home = Path.home()

master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"

raw_dir = home / "data/raw_embeddings/zhong_gene_embedding_benchmark/full_embeddings/all_genes/ESM2"
emb_path = raw_dir / "ESM2_emb.csv"
genelist_path = raw_dir / "ESM2_genelist.txt"

out_dir = home / "data/processed_embeddings/esm2_v1_1"
report_dir = home / "reports/mapping_reports"
out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

out_npz = out_dir / "esm2_v1_1_embeddings.npz"
out_genes = out_dir / "esm2_v1_1_genes.tsv"
out_meta = out_dir / "esm2_v1_1_metadata.json"

report_txt = report_dir / "esm2_v1_1_mapping_report.txt"
unmapped_tsv = report_dir / "esm2_v1_1_unmapped_source_identifiers.tsv"
ambiguous_tsv = report_dir / "esm2_v1_1_ambiguous_source_identifiers.tsv"
duplicates_tsv = report_dir / "esm2_v1_1_duplicate_final_mappings.tsv"


def split_ids(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return []
    return [t.strip() for t in re.split(r"[|,;\s]+", s) if t.strip()]


def normalize_entrez(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    if re.fullmatch(r"\d+(\.0)?", s):
        return str(int(float(s)))
    return s


def first_existing(row, cols):
    for c in cols:
        if c in row.index and str(row[c]).strip() and str(row[c]).lower() != "nan":
            return str(row[c]).strip()
    return ""


def read_genelist(path):
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(normalize_entrez(s))
    return ids


def read_embedding_matrix(path, n_ids, expected_dim=1280):
    df = pd.read_csv(path, header=None)

    # If a header row was read as data, drop it.
    if df.shape[0] == n_ids + 1:
        df = df.iloc[1:].reset_index(drop=True)

    # Convert to numeric. Non-numeric identifier columns become NaN.
    num = df.apply(pd.to_numeric, errors="coerce")

    # Drop completely empty rows/columns.
    num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # If there is an extra index/ID column, drop the first column.
    if num.shape[1] == expected_dim + 1:
        num = num.iloc[:, 1:]

    if num.shape[0] != n_ids:
        raise ValueError(f"Embedding rows ({num.shape[0]}) != genelist length ({n_ids}).")

    if num.shape[1] != expected_dim:
        print(f"WARNING: expected {expected_dim} dims, observed {num.shape[1]} dims. Continuing.")

    if num.isna().any().any():
        n_nan = int(num.isna().sum().sum())
        raise ValueError(f"Embedding matrix contains {n_nan} NaN values after numeric conversion.")

    return num.to_numpy(dtype=np.float32)


print("Loading source files...")
source_ids = read_genelist(genelist_path)
X_source = read_embedding_matrix(emb_path, len(source_ids), expected_dim=1280)

print(f"Source identifiers: {len(source_ids):,}")
print(f"Source matrix: {X_source.shape[0]:,} genes x {X_source.shape[1]:,} dims")

print("Loading master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")

required = ["ensembl_gene_id"]
for c in required:
    if c not in master.columns:
        raise ValueError(f"Missing required master column: {c}")

symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

entrez_cols = [
    c for c in [
        "entrez_id",
        "hgnc_entrez_id",
        "map_entrez_ids_all",
    ]
    if c in master.columns
]

if not entrez_cols:
    raise ValueError("No Entrez mapping columns found in master table.")

print("Building Entrez -> master index...")
entrez_to_master_rows = defaultdict(set)

for i, row in master.iterrows():
    for c in entrez_cols:
        for token in split_ids(row[c]):
            e = normalize_entrez(token)
            if e:
                entrez_to_master_rows[e].add(i)

unique_entrez_to_master = {
    e: next(iter(rows))
    for e, rows in entrez_to_master_rows.items()
    if len(rows) == 1
}

ambiguous_entrez_to_master = {
    e: sorted(rows)
    for e, rows in entrez_to_master_rows.items()
    if len(rows) > 1
}

mapped_records = []
unmapped_records = []
ambiguous_records = []

for src_i, src_id in enumerate(source_ids):
    entrez = normalize_entrez(src_id)

    if entrez in unique_entrez_to_master:
        master_i = unique_entrez_to_master[entrez]
        mapped_records.append((src_i, entrez, master_i))
    elif entrez in ambiguous_entrez_to_master:
        ambiguous_records.append({
            "source_identifier": entrez,
            "reason": "entrez_maps_to_multiple_master_rows",
            "master_row_indices": ";".join(map(str, ambiguous_entrez_to_master[entrez])),
            "master_ensembl_gene_ids": ";".join(master.loc[ambiguous_entrez_to_master[entrez], "ensembl_gene_id"].astype(str)),
        })
    else:
        unmapped_records.append({
            "source_identifier": entrez,
            "reason": "no_unique_entrez_match_in_master",
        })

print(f"Mapped source rows before duplicate collapse: {len(mapped_records):,}")
print(f"Unmapped source rows: {len(unmapped_records):,}")
print(f"Ambiguous source rows: {len(ambiguous_records):,}")

# Collapse duplicate final master mappings by averaging source rows.
master_to_source = defaultdict(list)
master_to_entrez = defaultdict(list)

for src_i, entrez, master_i in mapped_records:
    master_to_source[master_i].append(src_i)
    master_to_entrez[master_i].append(entrez)

final_master_indices = sorted(master_to_source.keys())
X_final = []
gene_rows = []
duplicate_rows = []

for master_i in final_master_indices:
    src_indices = master_to_source[master_i]
    entrez_ids = master_to_entrez[master_i]
    vec = X_source[src_indices].mean(axis=0).astype(np.float32)
    X_final.append(vec)

    row = master.loc[master_i]
    source_examples = ";".join(entrez_ids[:10])

    if len(src_indices) > 1:
        duplicate_rows.append({
            "ensembl_gene_id": row["ensembl_gene_id"],
            "gene_symbol": row.get(symbol_col, ""),
            "source_key_count": len(src_indices),
            "source_keys": ";".join(entrez_ids),
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
        "source_embedding": SOURCE_EMBEDDING,
        "source_identifier_type": SOURCE_IDENTIFIER_TYPE,
        "source_key_count": len(src_indices),
        "source_keys_examples": source_examples,
        "mapping_methods": "entrez_exact_unique",
    })

X_final = np.vstack(X_final).astype(np.float32)
genes_df = pd.DataFrame(gene_rows)

print(f"Final mapped genes after duplicate collapse: {X_final.shape[0]:,}")
print(f"Final dimensions: {X_final.shape[1]:,}")

np.savez_compressed(
    out_npz,
    X=X_final,
    ensembl_gene_id=genes_df["ensembl_gene_id"].to_numpy(dtype=str),
    gene_symbol=genes_df["gene_symbol"].to_numpy(dtype=str),
)

genes_df.to_csv(out_genes, sep="\t", index=False)

pd.DataFrame(unmapped_records).to_csv(unmapped_tsv, sep="\t", index=False)
pd.DataFrame(ambiguous_records).to_csv(ambiguous_tsv, sep="\t", index=False)
pd.DataFrame(duplicate_rows).to_csv(duplicates_tsv, sep="\t", index=False)

coverage_master = X_final.shape[0] / len(master)
coverage_source = len(mapped_records) / len(source_ids)

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "source_embedding": SOURCE_EMBEDDING,
    "modality": "protein_sequence",
    "algorithm": "ESM2 Transformer",
    "source_identifier_type": SOURCE_IDENTIFIER_TYPE,
    "source_files": {
        "embedding_matrix": str(emb_path),
        "genelist": str(genelist_path),
    },
    "master_table": str(master_path),
    "mapping_strategy": "Exact Entrez ID mapping to master_gene_table_v1_1_enriched.csv. Only Entrez IDs mapping to exactly one master gene are accepted. Duplicate source rows mapping to the same final master gene are averaged.",
    "counts": {
        "source_rows": len(source_ids),
        "source_dimensions": int(X_source.shape[1]),
        "mapped_source_rows_before_duplicate_collapse": len(mapped_records),
        "unmapped_source_rows": len(unmapped_records),
        "ambiguous_source_rows": len(ambiguous_records),
        "final_mapped_genes": int(X_final.shape[0]),
        "final_dimensions": int(X_final.shape[1]),
        "master_rows": int(len(master)),
        "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
    },
    "coverage": {
        "source_row_mapping_fraction": coverage_source,
        "master_gene_coverage_fraction": coverage_master,
    },
    "outputs": {
        "npz": str(out_npz),
        "genes_tsv": str(out_genes),
        "metadata_json": str(out_meta),
        "mapping_report": str(report_txt),
        "unmapped_tsv": str(unmapped_tsv),
        "ambiguous_tsv": str(ambiguous_tsv),
        "duplicates_tsv": str(duplicates_tsv),
    },
}

with open(out_meta, "w") as f:
    json.dump(metadata, f, indent=2)

with open(report_txt, "w") as f:
    f.write("ESM2-v1.1 mapping report\n")
    f.write("========================\n\n")
    f.write(f"Source embedding matrix: {emb_path}\n")
    f.write(f"Source genelist:         {genelist_path}\n")
    f.write(f"Master table:            {master_path}\n\n")
    for k, v in metadata["counts"].items():
        f.write(f"{k}: {v}\n")
    f.write("\n")
    for k, v in metadata["coverage"].items():
        f.write(f"{k}: {v:.6f}\n")
    f.write("\nOutputs:\n")
    for k, v in metadata["outputs"].items():
        f.write(f"{k}: {v}\n")

print("\nDONE")
print(report_txt)
print(out_npz)
print(out_genes)
print(out_meta)
