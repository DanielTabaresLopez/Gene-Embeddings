from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd


EMBEDDING_NAME = "BioConceptVec-SkipGram-v1.1"
PREFIX = "bioconceptvec_skipgram_v1_1"
RAW_FILENAME = "concept_skip.json"
MODALITY = "biomedical_literature_text"


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s in {"", "nan", "NaN", "None", "NONE"}:
        return ""
    return s


def norm_entrez(x):
    s = clean(x)
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def split_pipe(x):
    s = clean(x)
    return [v for v in s.split("|") if v] if s else []


def build_unique_entrez_map(master):
    d = defaultdict(set)

    for idx, row in master.iterrows():
        for value in split_pipe(row["map_entrez_ids_all"]):
            key = norm_entrez(value)
            if key:
                d[key].add(idx)

    unique = {k: next(iter(v)) for k, v in d.items() if len(v) == 1}
    ambiguous = {k for k, v in d.items() if len(v) > 1}

    return unique, ambiguous


def parse_gene_concept_key(key):
    """
    BioConceptVec gene concepts look like:
    Gene_7157
    Gene_64066_4313_4316

    Strict v1.1:
    - single Entrez ID accepted
    - multi-Entrez concepts excluded
    """
    if not key.startswith("Gene_"):
        return None, "non_gene"

    rest = key[len("Gene_"):]
    parts = [p for p in rest.split("_") if p]

    if len(parts) == 1 and parts[0].isdigit():
        return parts[0], "single_entrez"

    if len(parts) > 1 and all(p.isdigit() for p in parts):
        return parts, "multi_entrez"

    return rest, "bad_gene_concept"


home = Path.home()

raw_path = home / "data/raw_embeddings/bioconceptvec" / RAW_FILENAME
master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"

out_dir = home / "data/processed_embeddings" / PREFIX
report_dir = home / "reports/mapping_reports"

out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

assert raw_path.exists(), f"Missing raw BioConceptVec file: {raw_path}"
assert master_path.exists(), f"Missing enriched master table: {master_path}"

print("Loading enriched master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")

print("Building Entrez mapper from map_entrez_ids_all...")
entrez_map, ambiguous_entrez = build_unique_entrez_map(master)

print("Loading BioConceptVec JSON...")
with open(raw_path) as f:
    raw = json.load(f)

if not isinstance(raw, dict):
    raise TypeError(f"Expected dict, got {type(raw)}")

print(f"Total concepts: {len(raw):,}")

target_vectors = defaultdict(list)
target_source_ids = defaultdict(list)

mapped_source_rows = []
unmapped_rows = []
ambiguous_rows = []
multi_entrez_rows = []
bad_gene_rows = []
nonfinite_rows = []
invalid_dim_rows = []

n_non_gene_skipped = 0
n_single_entrez_gene_concepts = 0

dim = None

for source_id, vec in raw.items():
    parsed, status = parse_gene_concept_key(source_id)

    if status == "non_gene":
        n_non_gene_skipped += 1
        continue

    if status == "multi_entrez":
        multi_entrez_rows.append({
            "source_id": source_id,
            "entrez_ids": "|".join(parsed),
            "reason": "multi_entrez_gene_concept_excluded",
        })
        continue

    if status == "bad_gene_concept":
        bad_gene_rows.append({
            "source_id": source_id,
            "parsed_value": parsed,
            "reason": "bad_gene_concept_key",
        })
        continue

    entrez = norm_entrez(parsed)
    n_single_entrez_gene_concepts += 1

    arr = np.asarray(vec, dtype=np.float32)

    if arr.ndim != 1:
        invalid_dim_rows.append({
            "source_id": source_id,
            "entrez_id": entrez,
            "reason": f"not_1d_shape_{arr.shape}",
        })
        continue

    if dim is None:
        dim = arr.shape[0]

    if arr.shape[0] != dim:
        invalid_dim_rows.append({
            "source_id": source_id,
            "entrez_id": entrez,
            "reason": f"wrong_dim_{arr.shape[0]}_expected_{dim}",
        })
        continue

    if not np.isfinite(arr).all():
        nonfinite_rows.append({
            "source_id": source_id,
            "entrez_id": entrez,
            "reason": "nonfinite_vector",
        })
        continue

    if entrez in entrez_map:
        target_idx = entrez_map[entrez]
    elif entrez in ambiguous_entrez:
        ambiguous_rows.append({
            "source_id": source_id,
            "entrez_id": entrez,
            "reason": "ambiguous_entrez_to_master_gene",
        })
        continue
    else:
        unmapped_rows.append({
            "source_id": source_id,
            "entrez_id": entrez,
            "reason": "no_entrez_to_master_gene_match",
        })
        continue

    target_vectors[target_idx].append(arr)
    target_source_ids[target_idx].append(source_id)

    mapped_source_rows.append({
        "source_id": source_id,
        "entrez_id": entrez,
        "mapping_method": "entrez_id_from_map_entrez_ids_all",
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

genes_out["source_embedding"] = EMBEDDING_NAME
genes_out["source_identifier_type"] = "entrez_id"
genes_out["source_key_count"] = [len(target_source_ids[idx]) for idx in mapped_indices]
genes_out["source_keys_examples"] = [";".join(target_source_ids[idx][:5]) for idx in mapped_indices]
genes_out["mapping_methods"] = "entrez_id_from_map_entrez_ids_all"

n_zero_vectors = int((np.linalg.norm(X, axis=1) == 0).sum())

old_prefix = PREFIX.replace("_v1_1", "")
old_genes_path = home / "data/processed_embeddings" / old_prefix / f"{old_prefix}_genes.tsv"

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

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "modality": MODALITY,
    "raw_file": str(raw_path),
    "master_table": str(master_path),
    "original_identifier_type": "BioConceptVec concept key with Entrez/NCBI Gene ID, e.g. Gene_7157",
    "mapping_strategy": "strict single-Entrez BioConceptVec gene concept mapping using enriched master table column map_entrez_ids_all",
    "multi_entrez_gene_concepts": "excluded and reported",
    "duplicate_source_rows": "averaged when multiple source rows mapped to the same master gene",
    "ambiguous_source_rows": "excluded and reported",
    "dtype_saved": "float32",
    "n_total_concepts_raw": int(len(raw)),
    "n_non_gene_concepts_skipped": int(n_non_gene_skipped),
    "n_single_entrez_gene_concepts": int(n_single_entrez_gene_concepts),
    "n_multi_entrez_gene_concepts_excluded": int(len(multi_entrez_rows)),
    "n_bad_gene_concepts_excluded": int(len(bad_gene_rows)),
    "n_mapped_source_rows": int(len(mapped_source_rows)),
    "n_unmapped_source_rows": int(len(unmapped_rows)),
    "n_ambiguous_source_rows_excluded": int(len(ambiguous_rows)),
    "n_invalid_dim_source_rows_skipped": int(len(invalid_dim_rows)),
    "n_nonfinite_source_rows_skipped": int(len(nonfinite_rows)),
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

pd.DataFrame(mapped_source_rows).to_csv(report_dir / f"{PREFIX}_mapped_source_rows.tsv", sep="\t", index=False)
pd.DataFrame(unmapped_rows).to_csv(report_dir / f"{PREFIX}_unmapped_source_entrez.tsv", sep="\t", index=False)
pd.DataFrame(ambiguous_rows).to_csv(report_dir / f"{PREFIX}_ambiguous_source_entrez_excluded.tsv", sep="\t", index=False)
pd.DataFrame(multi_entrez_rows).to_csv(report_dir / f"{PREFIX}_multi_entrez_gene_concepts_excluded.tsv", sep="\t", index=False)
pd.DataFrame(bad_gene_rows).to_csv(report_dir / f"{PREFIX}_bad_gene_concepts_excluded.tsv", sep="\t", index=False)
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
