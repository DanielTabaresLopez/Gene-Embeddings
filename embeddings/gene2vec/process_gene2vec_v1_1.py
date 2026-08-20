from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s in {"", "nan", "NaN", "None", "NONE"}:
        return ""
    return s


def norm_symbol(x):
    return clean(x).upper()


def split_pipe(x):
    s = clean(x)
    return [v for v in s.split("|") if v] if s else []


def build_unique_map(master, cols, normalizer):
    d = defaultdict(set)

    for idx, row in master.iterrows():
        for col in cols:
            if col not in master.columns:
                continue

            for value in split_pipe(row[col]):
                key = normalizer(value)
                if key:
                    d[key].add(idx)

    unique = {k: next(iter(v)) for k, v in d.items() if len(v) == 1}
    ambiguous = {k for k, v in d.items() if len(v) > 1}

    return unique, ambiguous


def load_gene2vec_txt(path):
    rows = []
    bad_rows = []
    dim = None

    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            # Optional word2vec-style header
            if line_no == 1 and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                dim = int(parts[1])
                continue

            source_id = parts[0]
            values = parts[1:]

            try:
                vec = np.asarray(values, dtype=np.float32)
            except Exception:
                bad_rows.append({
                    "line_no": line_no,
                    "source_id": source_id,
                    "reason": "cannot_parse_floats",
                })
                continue

            if dim is None:
                dim = vec.shape[0]

            if vec.shape[0] != dim:
                bad_rows.append({
                    "line_no": line_no,
                    "source_id": source_id,
                    "reason": f"wrong_dim_{vec.shape[0]}_expected_{dim}",
                })
                continue

            rows.append((source_id, vec))

    return rows, bad_rows, dim


home = Path.home()

raw_path = home / "data/raw_embeddings/gene2vec/gene2vec_dim_200_iter_9.txt"
master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"

out_dir = home / "data/processed_embeddings/gene2vec_v1_1"
report_dir = home / "reports/mapping_reports"

out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

assert raw_path.exists(), f"Missing raw Gene2Vec file: {raw_path}"
assert master_path.exists(), f"Missing enriched master table: {master_path}"

print("Loading enriched master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")

required_cols = [
    "ensembl_gene_id",
    "gene_symbol",
    "gene_type",
    "entrez_id",
    "uniprot_id",
    "canonical_transcript_id",
    "canonical_protein_id",
    "hgnc_approved_symbol",
    "hgnc_previous_symbols",
    "hgnc_alias_symbols",
]

missing_cols = [c for c in required_cols if c not in master.columns]
if missing_cols:
    raise ValueError(f"Missing columns in enriched master table: {missing_cols}")

print("Building priority symbol mappers...")

current_symbol_map, ambiguous_current_symbol = build_unique_map(
    master,
    ["gene_symbol", "hgnc_approved_symbol"],
    norm_symbol,
)

previous_symbol_map, ambiguous_previous_symbol = build_unique_map(
    master,
    ["hgnc_previous_symbols"],
    norm_symbol,
)

alias_symbol_map, ambiguous_alias_symbol = build_unique_map(
    master,
    ["hgnc_alias_symbols"],
    norm_symbol,
)

print("Loading Gene2Vec raw file...")
source_rows, bad_rows, dim = load_gene2vec_txt(raw_path)

print(f"Raw Gene2Vec entries: {len(source_rows):,}")
print(f"Embedding dimension: {dim}")
print(f"Bad rows skipped: {len(bad_rows):,}")

target_vectors = defaultdict(list)
target_source_ids = defaultdict(list)
target_mapping_methods = defaultdict(set)

mapped_source_rows = []
unmapped_rows = []
ambiguous_rows = []
nonfinite_rows = []

method_counts = defaultdict(int)

for source_id, vec in source_rows:
    key = norm_symbol(source_id)

    if not np.isfinite(vec).all():
        nonfinite_rows.append({
            "source_id": source_id,
            "normalized_key": key,
            "reason": "nonfinite_vector",
        })
        continue

    target_idx = None
    method = None

    if key in current_symbol_map:
        target_idx = current_symbol_map[key]
        method = "direct_current_gene_symbol"
    elif key in previous_symbol_map:
        target_idx = previous_symbol_map[key]
        method = "hgnc_previous_symbol"
    elif key in alias_symbol_map:
        target_idx = alias_symbol_map[key]
        method = "hgnc_alias_symbol"
    elif key in ambiguous_current_symbol or key in ambiguous_previous_symbol or key in ambiguous_alias_symbol:
        ambiguous_rows.append({
            "source_id": source_id,
            "normalized_key": key,
            "reason": "ambiguous_symbol_to_master_gene",
        })
        continue
    else:
        unmapped_rows.append({
            "source_id": source_id,
            "normalized_key": key,
            "reason": "no_symbol_to_master_gene_match",
        })
        continue

    target_vectors[target_idx].append(vec)
    target_source_ids[target_idx].append(source_id)
    target_mapping_methods[target_idx].add(method)
    method_counts[method] += 1

    mapped_source_rows.append({
        "source_id": source_id,
        "normalized_key": key,
        "mapping_method": method,
        "target_ensembl_gene_id": master.loc[target_idx, "ensembl_gene_id"],
        "target_gene_symbol": master.loc[target_idx, "gene_symbol"],
    })

mapped_indices = sorted(target_vectors.keys())

X = np.vstack([
    np.mean(np.vstack(target_vectors[idx]), axis=0).astype(np.float32)
    for idx in mapped_indices
])

mapped_master = master.loc[mapped_indices].copy()

genes_out = mapped_master[
    [
        "ensembl_gene_id",
        "gene_symbol",
        "gene_type",
        "entrez_id",
        "uniprot_id",
        "canonical_transcript_id",
        "canonical_protein_id",
    ]
].copy()

genes_out["source_embedding"] = "Gene2Vec-v1.1"
genes_out["source_identifier_type"] = "gene_symbol"
genes_out["source_key_count"] = [
    len(target_source_ids[idx])
    for idx in mapped_indices
]
genes_out["source_keys_examples"] = [
    ";".join(target_source_ids[idx][:5])
    for idx in mapped_indices
]
genes_out["mapping_methods"] = [
    ";".join(sorted(target_mapping_methods[idx]))
    for idx in mapped_indices
]

zero_vector_mask = np.linalg.norm(X, axis=1) == 0
n_zero_vectors = int(zero_vector_mask.sum())

npz_path = out_dir / "gene2vec_v1_1_embeddings.npz"
genes_tsv_path = out_dir / "gene2vec_v1_1_genes.tsv"
metadata_path = out_dir / "gene2vec_v1_1_metadata.json"

np.savez_compressed(
    npz_path,
    X=X,
    ensembl_gene_id=genes_out["ensembl_gene_id"].astype(str).values,
    gene_symbol=genes_out["gene_symbol"].astype(str).values,
)

genes_out.to_csv(genes_tsv_path, sep="\t", index=False)

# Compare against old strict Gene2Vec if available.
old_genes_path = home / "data/processed_embeddings/gene2vec/gene2vec_genes.tsv"
n_genes_added_vs_old = None
if old_genes_path.exists():
    old_genes = pd.read_csv(old_genes_path, sep="\t", dtype=str)
    old_set = set(old_genes["ensembl_gene_id"].astype(str))
    new_set = set(genes_out["ensembl_gene_id"].astype(str))
    n_genes_added_vs_old = len(new_set - old_set)

metadata = {
    "embedding_name": "Gene2Vec-v1.1",
    "modality": "coexpression_expression",
    "raw_file": str(raw_path),
    "master_table": str(master_path),
    "original_identifier_type": "gene_symbol",
    "mapping_strategy": "priority symbol mapping using enriched master table: current gene symbols, HGNC previous symbols, HGNC alias symbols",
    "duplicate_source_rows": "averaged when multiple source rows mapped to the same master gene",
    "ambiguous_source_rows": "excluded and reported",
    "dtype_saved": "float32",
    "n_source_rows_raw": int(len(source_rows)),
    "n_bad_rows_skipped": int(len(bad_rows)),
    "n_mapped_source_rows": int(len(mapped_source_rows)),
    "n_unmapped_source_rows": int(len(unmapped_rows)),
    "n_ambiguous_source_rows_excluded": int(len(ambiguous_rows)),
    "n_nonfinite_source_rows_skipped": int(len(nonfinite_rows)),
    "mapping_method_counts": dict(sorted(method_counts.items())),
    "n_master_rows": int(len(master)),
    "n_mapped_genes": int(X.shape[0]),
    "n_genes_added_vs_gene2vec_strict_v1": None if n_genes_added_vs_old is None else int(n_genes_added_vs_old),
    "embedding_dim": int(X.shape[1]),
    "coverage_of_master_rows": float(X.shape[0] / len(master)),
    "n_zero_vectors": n_zero_vectors,
    "output_npz": str(npz_path),
    "output_genes_tsv": str(genes_tsv_path),
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

mapped_rows_path = report_dir / "gene2vec_v1_1_mapped_source_rows.tsv"
pd.DataFrame(mapped_source_rows).to_csv(mapped_rows_path, sep="\t", index=False)

unmapped_path = report_dir / "gene2vec_v1_1_unmapped_source_symbols.tsv"
pd.DataFrame(unmapped_rows).to_csv(unmapped_path, sep="\t", index=False)

ambiguous_path = report_dir / "gene2vec_v1_1_ambiguous_source_symbols_excluded.tsv"
pd.DataFrame(ambiguous_rows).to_csv(ambiguous_path, sep="\t", index=False)

bad_rows_path = report_dir / "gene2vec_v1_1_bad_rows_skipped.tsv"
pd.DataFrame(bad_rows).to_csv(bad_rows_path, sep="\t", index=False)

nonfinite_path = report_dir / "gene2vec_v1_1_nonfinite_source_rows_skipped.tsv"
pd.DataFrame(nonfinite_rows).to_csv(nonfinite_path, sep="\t", index=False)

missing_master_rows = master.loc[~master.index.isin(mapped_indices)].copy()
missing_master_path = report_dir / "gene2vec_v1_1_missing_master_genes.tsv"
missing_master_rows.to_csv(missing_master_path, sep="\t", index=False)

report_path = report_dir / "gene2vec_v1_1_mapping_report.txt"
with open(report_path, "w") as f:
    f.write("Gene2Vec v1.1 mapping report\n")
    f.write("============================\n\n")
    for k, v in metadata.items():
        f.write(f"{k}: {v}\n")
    f.write("\nAdditional report files\n")
    f.write("-----------------------\n")
    f.write(f"mapped_source_rows: {mapped_rows_path}\n")
    f.write(f"unmapped_source_symbols: {unmapped_path}\n")
    f.write(f"ambiguous_source_symbols_excluded: {ambiguous_path}\n")
    f.write(f"bad_rows_skipped: {bad_rows_path}\n")
    f.write(f"nonfinite_source_rows_skipped: {nonfinite_path}\n")
    f.write(f"missing_master_genes: {missing_master_path}\n")

print("\nDONE")
print(f"Saved matrix: {npz_path}")
print(f"Saved genes table: {genes_tsv_path}")
print(f"Saved metadata: {metadata_path}")
print(f"Saved report: {report_path}")

print("\nSummary:")
for k, v in metadata.items():
    print(f"{k}: {v}")
