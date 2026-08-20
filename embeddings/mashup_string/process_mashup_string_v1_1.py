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


def strip_version(x):
    return clean(x).split(".")[0] if clean(x) else ""


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


home = Path.home()

genes_path = home / "data/raw_embeddings/mashup/string_human_genes.txt"
vectors_path = home / "data/raw_embeddings/mashup/string_human_mashup_vectors_d800.txt"
master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"

out_dir = home / "data/processed_embeddings/mashup_string_v1_1"
report_dir = home / "reports/mapping_reports"

out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

assert genes_path.exists(), f"Missing genes file: {genes_path}"
assert vectors_path.exists(), f"Missing vectors file: {vectors_path}"
assert master_path.exists(), f"Missing enriched master table: {master_path}"

print("Loading enriched master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")

print("Building priority mappers...")

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

canonical_protein_map, ambiguous_canonical_protein = build_unique_map(
    master,
    ["canonical_protein_id"],
    strip_version,
)

grch38_protein_map, ambiguous_grch38_protein = build_unique_map(
    master,
    ["all_ensembl_protein_ids_grch38_current"],
    strip_version,
)

grch37_protein_map, ambiguous_grch37_protein = build_unique_map(
    master,
    ["all_ensembl_protein_ids_grch37"],
    strip_version,
)

print("Loading Mashup source IDs...")
source_ids = [line.strip() for line in open(genes_path) if line.strip()]

print("Loading Mashup vectors...")
X_raw = np.loadtxt(vectors_path, dtype=np.float32)

if len(source_ids) != X_raw.shape[0]:
    raise ValueError(f"Gene count and vector rows differ: {len(source_ids)} vs {X_raw.shape[0]}")

target_vectors = defaultdict(list)
target_source_ids = defaultdict(list)
target_source_types = defaultdict(set)
target_mapping_methods = defaultdict(set)

mapped_source_rows = []
unmapped_rows = []
ambiguous_rows = []
nonfinite_rows = []

method_counts = defaultdict(int)
source_type_counts = defaultdict(int)

for source_id, vec in zip(source_ids, X_raw):
    if not np.isfinite(vec).all():
        nonfinite_rows.append({"source_id": source_id, "reason": "nonfinite_vector"})
        continue

    key = norm_symbol(source_id)

    target_idx = None
    method = None
    source_type = None

    if key.startswith("ENSP"):
        source_type = "ensembl_protein_id"
        source_type_counts[source_type] += 1
        p = strip_version(key)

        if p in canonical_protein_map:
            target_idx = canonical_protein_map[p]
            method = "direct_canonical_protein_id"
        elif p in grch38_protein_map:
            target_idx = grch38_protein_map[p]
            method = "ensembl_grch38_current_gtf_protein_id"
        elif p in grch37_protein_map:
            target_idx = grch37_protein_map[p]
            method = "ensembl_grch37_gtf_protein_id"
        elif p in ambiguous_canonical_protein or p in ambiguous_grch38_protein or p in ambiguous_grch37_protein:
            ambiguous_rows.append({
                "source_id": source_id,
                "source_type": source_type,
                "normalized_key": p,
                "reason": "ambiguous_protein_to_master_gene",
            })
            continue
        else:
            unmapped_rows.append({
                "source_id": source_id,
                "source_type": source_type,
                "normalized_key": p,
                "reason": "no_protein_to_master_gene_match",
            })
            continue

    else:
        source_type = "gene_symbol"
        source_type_counts[source_type] += 1

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
                "source_type": source_type,
                "normalized_key": key,
                "reason": "ambiguous_symbol_to_master_gene",
            })
            continue
        else:
            unmapped_rows.append({
                "source_id": source_id,
                "source_type": source_type,
                "normalized_key": key,
                "reason": "no_symbol_to_master_gene_match",
            })
            continue

    target_vectors[target_idx].append(vec)
    target_source_ids[target_idx].append(source_id)
    target_source_types[target_idx].add(source_type)
    target_mapping_methods[target_idx].add(method)
    method_counts[method] += 1

    mapped_source_rows.append({
        "source_id": source_id,
        "source_type": source_type,
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

genes_out["source_embedding"] = "Mashup-STRING-v1.1"
genes_out["source_identifier_type"] = [
    ";".join(sorted(target_source_types[idx]))
    for idx in mapped_indices
]
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

npz_path = out_dir / "mashup_string_v1_1_embeddings.npz"
genes_tsv_path = out_dir / "mashup_string_v1_1_genes.tsv"
metadata_path = out_dir / "mashup_string_v1_1_metadata.json"

np.savez_compressed(
    npz_path,
    X=X,
    ensembl_gene_id=genes_out["ensembl_gene_id"].astype(str).values,
    gene_symbol=genes_out["gene_symbol"].astype(str).values,
)

genes_out.to_csv(genes_tsv_path, sep="\t", index=False)

metadata = {
    "embedding_name": "Mashup-STRING-v1.1",
    "modality": "network_ppi",
    "raw_genes_file": str(genes_path),
    "raw_vectors_file": str(vectors_path),
    "master_table": str(master_path),
    "mapping_strategy": "priority mapping using enriched master table: current symbols, HGNC previous symbols, HGNC alias symbols, canonical protein IDs, GRCh38 protein IDs, GRCh37 protein IDs",
    "duplicate_source_rows": "averaged when multiple source rows mapped to the same master gene",
    "ambiguous_source_rows": "excluded and reported",
    "dtype_saved": "float32",
    "n_source_rows_raw": int(len(source_ids)),
    "n_source_symbol_like_rows": int(source_type_counts["gene_symbol"]),
    "n_source_ensp_like_rows": int(source_type_counts["ensembl_protein_id"]),
    "n_mapped_source_rows": int(len(mapped_source_rows)),
    "n_unmapped_source_rows": int(len(unmapped_rows)),
    "n_ambiguous_source_rows_excluded": int(len(ambiguous_rows)),
    "n_nonfinite_source_rows_skipped": int(len(nonfinite_rows)),
    "mapping_method_counts": dict(sorted(method_counts.items())),
    "n_master_rows": int(len(master)),
    "n_mapped_genes": int(X.shape[0]),
    "embedding_dim": int(X.shape[1]),
    "coverage_of_master_rows": float(X.shape[0] / len(master)),
    "n_zero_vectors": n_zero_vectors,
    "output_npz": str(npz_path),
    "output_genes_tsv": str(genes_tsv_path),
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

mapped_rows_path = report_dir / "mashup_string_v1_1_mapped_source_rows.tsv"
pd.DataFrame(mapped_source_rows).to_csv(mapped_rows_path, sep="\t", index=False)

unmapped_path = report_dir / "mashup_string_v1_1_unmapped_source_ids.tsv"
pd.DataFrame(unmapped_rows).to_csv(unmapped_path, sep="\t", index=False)

ambiguous_path = report_dir / "mashup_string_v1_1_ambiguous_source_ids_excluded.tsv"
pd.DataFrame(ambiguous_rows).to_csv(ambiguous_path, sep="\t", index=False)

nonfinite_path = report_dir / "mashup_string_v1_1_nonfinite_source_rows_skipped.tsv"
pd.DataFrame(nonfinite_rows).to_csv(nonfinite_path, sep="\t", index=False)

missing_master_rows = master.loc[~master.index.isin(mapped_indices)].copy()
missing_master_path = report_dir / "mashup_string_v1_1_missing_master_genes.tsv"
missing_master_rows.to_csv(missing_master_path, sep="\t", index=False)

report_path = report_dir / "mashup_string_v1_1_mapping_report.txt"
with open(report_path, "w") as f:
    f.write("Mashup-STRING v1.1 mapping report\n")
    f.write("=================================\n\n")
    for k, v in metadata.items():
        f.write(f"{k}: {v}\n")
    f.write("\nAdditional report files\n")
    f.write("-----------------------\n")
    f.write(f"mapped_source_rows: {mapped_rows_path}\n")
    f.write(f"unmapped_source_ids: {unmapped_path}\n")
    f.write(f"ambiguous_source_ids_excluded: {ambiguous_path}\n")
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
