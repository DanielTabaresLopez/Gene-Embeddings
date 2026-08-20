from pathlib import Path
from collections import defaultdict
import json
import pickle

import numpy as np
import pandas as pd


EMBEDDING_NAME = "GenePT-ADA-v1.1-conservative"
PREFIX = "genept_ada_v1_1_conservative"
MODALITY = "biomedical_literature_text"


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


home = Path.home()

raw_path = home / "data/raw_embeddings/genept/unzipped/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle"
old_genes_path = home / "data/processed_embeddings/genept_ada/genept_ada_genes.tsv"
master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"

out_dir = home / "data/processed_embeddings" / PREFIX
report_dir = home / "reports/mapping_reports"

out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

assert raw_path.exists(), f"Missing raw file: {raw_path}"
assert master_path.exists(), f"Missing enriched master table: {master_path}"

print("Loading enriched master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")

print("Building priority symbol mappers...")
current_symbol_map, ambiguous_current_symbol = build_unique_map(
    master, ["gene_symbol", "hgnc_approved_symbol"], norm_symbol
)
previous_symbol_map, ambiguous_previous_symbol = build_unique_map(
    master, ["hgnc_previous_symbols"], norm_symbol
)
alias_symbol_map, ambiguous_alias_symbol = build_unique_map(
    master, ["hgnc_alias_symbols"], norm_symbol
)

print("Loading raw GenePT pickle...")
with open(raw_path, "rb") as f:
    raw = pickle.load(f)

if not isinstance(raw, dict):
    raise TypeError(f"Expected dict, got {type(raw)}")

print(f"Raw entries: {len(raw):,}")

candidate_vectors = defaultdict(lambda: defaultdict(list))
candidate_source_ids = defaultdict(lambda: defaultdict(list))

all_candidate_rows = []
unmapped_rows = []
ambiguous_rows = []
invalid_dim_rows = []
nonfinite_rows = []

dim = None

for source_id, vec in raw.items():
    key = norm_symbol(source_id)
    arr = np.asarray(vec, dtype=np.float32)

    if arr.ndim != 1:
        invalid_dim_rows.append({"source_id": source_id, "normalized_key": key, "reason": f"not_1d_shape_{arr.shape}"})
        continue

    if dim is None:
        dim = arr.shape[0]

    if arr.shape[0] != dim:
        invalid_dim_rows.append({"source_id": source_id, "normalized_key": key, "reason": f"wrong_dim_{arr.shape[0]}_expected_{dim}"})
        continue

    if not np.isfinite(arr).all():
        nonfinite_rows.append({"source_id": source_id, "normalized_key": key, "reason": "nonfinite_vector"})
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
        ambiguous_rows.append({"source_id": source_id, "normalized_key": key, "reason": "ambiguous_symbol_to_master_gene"})
        continue
    else:
        unmapped_rows.append({"source_id": source_id, "normalized_key": key, "reason": "no_symbol_to_master_gene_match"})
        continue

    candidate_vectors[target_idx][method].append(arr)
    candidate_source_ids[target_idx][method].append(source_id)

    all_candidate_rows.append({
        "source_id": source_id,
        "normalized_key": key,
        "candidate_mapping_method": method,
        "target_ensembl_gene_id": master.loc[target_idx, "ensembl_gene_id"],
        "target_gene_symbol": master.loc[target_idx, "gene_symbol"],
    })

priority = ["direct_current_gene_symbol", "hgnc_previous_symbol", "hgnc_alias_symbol"]

mapped_indices = []
used_methods = {}
used_source_ids = {}
unused_rescue_rows = []

for idx in sorted(candidate_vectors.keys()):
    chosen_method = None
    for method in priority:
        if len(candidate_vectors[idx].get(method, [])) > 0:
            chosen_method = method
            break

    if chosen_method is None:
        continue

    mapped_indices.append(idx)
    used_methods[idx] = chosen_method
    used_source_ids[idx] = candidate_source_ids[idx][chosen_method]

    for method in priority:
        if method == chosen_method:
            continue
        for sid in candidate_source_ids[idx].get(method, []):
            unused_rescue_rows.append({
                "source_id": sid,
                "candidate_mapping_method": method,
                "target_ensembl_gene_id": master.loc[idx, "ensembl_gene_id"],
                "target_gene_symbol": master.loc[idx, "gene_symbol"],
                "reason": f"not_used_because_{chosen_method}_available",
            })

X = np.vstack([
    np.mean(np.vstack(candidate_vectors[idx][used_methods[idx]]), axis=0).astype(np.float32)
    for idx in mapped_indices
])

mapped_master = master.loc[mapped_indices].copy()

genes_out = mapped_master[
    ["ensembl_gene_id", "gene_symbol", "gene_type", "entrez_id", "uniprot_id", "canonical_transcript_id", "canonical_protein_id"]
].copy()

genes_out["source_embedding"] = EMBEDDING_NAME
genes_out["source_identifier_type"] = "gene_symbol"
genes_out["source_key_count"] = [len(used_source_ids[idx]) for idx in mapped_indices]
genes_out["source_keys_examples"] = [";".join(used_source_ids[idx][:5]) for idx in mapped_indices]
genes_out["mapping_methods"] = [used_methods[idx] for idx in mapped_indices]

n_zero_vectors = int((np.linalg.norm(X, axis=1) == 0).sum())

n_genes_added_vs_old = None
n_genes_lost_vs_old = None
if old_genes_path.exists():
    old_genes = pd.read_csv(old_genes_path, sep="\t", dtype=str)
    old_set = set(old_genes["ensembl_gene_id"].astype(str))
    new_set = set(genes_out["ensembl_gene_id"].astype(str))
    n_genes_added_vs_old = len(new_set - old_set)
    n_genes_lost_vs_old = len(old_set - new_set)

npz_path = out_dir / f"{PREFIX}_embeddings.npz"
genes_tsv_path = out_dir / f"{PREFIX}_genes.tsv"
metadata_path = out_dir / f"{PREFIX}_metadata.json"

np.savez_compressed(
    npz_path,
    X=X,
    ensembl_gene_id=genes_out["ensembl_gene_id"].astype(str).values,
    gene_symbol=genes_out["gene_symbol"].astype(str).values,
)

genes_out.to_csv(genes_tsv_path, sep="\t", index=False)

method_counts = genes_out["mapping_methods"].value_counts().to_dict()

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "modality": MODALITY,
    "raw_file": str(raw_path),
    "master_table": str(master_path),
    "original_identifier_type": "gene_symbol",
    "mapping_strategy": "conservative direct-first mapping using enriched master table: use direct current-symbol vectors when available; use HGNC previous/alias symbols only for genes without direct current-symbol vectors",
    "duplicate_source_rows": "averaged only within the selected priority class for each gene",
    "unused_lower_priority_source_rows": "reported but not averaged into genes with higher-priority direct mappings",
    "ambiguous_source_rows": "excluded and reported",
    "dtype_saved": "float32",
    "n_source_rows_raw": int(len(raw)),
    "n_candidate_source_rows": int(len(all_candidate_rows)),
    "n_used_source_rows": int(sum(genes_out["source_key_count"].astype(int))),
    "n_unused_lower_priority_source_rows": int(len(unused_rescue_rows)),
    "n_unmapped_source_rows": int(len(unmapped_rows)),
    "n_ambiguous_source_rows_excluded": int(len(ambiguous_rows)),
    "n_invalid_dim_source_rows_skipped": int(len(invalid_dim_rows)),
    "n_nonfinite_source_rows_skipped": int(len(nonfinite_rows)),
    "mapping_method_counts_by_output_gene": method_counts,
    "n_master_rows": int(len(master)),
    "n_mapped_genes": int(X.shape[0]),
    "n_genes_added_vs_strict_v1": None if n_genes_added_vs_old is None else int(n_genes_added_vs_old),
    "n_genes_lost_vs_strict_v1": None if n_genes_lost_vs_old is None else int(n_genes_lost_vs_old),
    "embedding_dim": int(X.shape[1]),
    "coverage_of_master_rows": float(X.shape[0] / len(master)),
    "n_zero_vectors": n_zero_vectors,
    "output_npz": str(npz_path),
    "output_genes_tsv": str(genes_tsv_path),
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

pd.DataFrame(all_candidate_rows).to_csv(report_dir / f"{PREFIX}_all_candidate_source_rows.tsv", sep="\t", index=False)
pd.DataFrame(unused_rescue_rows).to_csv(report_dir / f"{PREFIX}_unused_lower_priority_source_rows.tsv", sep="\t", index=False)
pd.DataFrame(unmapped_rows).to_csv(report_dir / f"{PREFIX}_unmapped_source_symbols.tsv", sep="\t", index=False)
pd.DataFrame(ambiguous_rows).to_csv(report_dir / f"{PREFIX}_ambiguous_source_symbols_excluded.tsv", sep="\t", index=False)
pd.DataFrame(invalid_dim_rows).to_csv(report_dir / f"{PREFIX}_invalid_dim_source_rows_skipped.tsv", sep="\t", index=False)
pd.DataFrame(nonfinite_rows).to_csv(report_dir / f"{PREFIX}_nonfinite_source_rows_skipped.tsv", sep="\t", index=False)
master.loc[~master.index.isin(mapped_indices)].to_csv(report_dir / f"{PREFIX}_missing_master_genes.tsv", sep="\t", index=False)

report_path = report_dir / f"{PREFIX}_mapping_report.txt"
with open(report_path, "w") as f:
    f.write(f"{EMBEDDING_NAME} mapping report\n")
    f.write("=" * (len(EMBEDDING_NAME) + 15) + "\n\n")
    for k, v in metadata.items():
        f.write(f"{k}: {v}\n")

print("\nDONE")
print(f"Saved matrix: {npz_path}")
print(f"Saved genes table: {genes_tsv_path}")
print(f"Saved metadata: {metadata_path}")
print(f"Saved report: {report_path}")
print("\nSummary:")
for k, v in metadata.items():
    print(f"{k}: {v}")
