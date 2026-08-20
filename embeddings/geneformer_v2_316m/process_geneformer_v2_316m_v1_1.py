from pathlib import Path
import json
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd


EMBEDDING_NAME = "Geneformer-V2-316M-v1.1"
SOURCE_EMBEDDING = "ctheodoris_Geneformer_default_model"
SOURCE_IDENTIFIER_TYPE = "Geneformer token dictionary Ensembl gene ID"
MODALITY = "single_cell_expression_foundation_model"
ALGORITHM = "Geneformer V2 316M input gene-token embedding layer"
OUT_STEM = "geneformer_v2_316m_v1_1"

home = Path.home()

master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"
raw_dir = home / "data/raw_embeddings/geneformer_official"

out_dir = home / "data/processed_embeddings" / OUT_STEM
report_dir = home / "reports/mapping_reports"
out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

out_npz = out_dir / f"{OUT_STEM}_embeddings.npz"
out_genes = out_dir / f"{OUT_STEM}_genes.tsv"
out_meta = out_dir / f"{OUT_STEM}_metadata.json"

report_txt = report_dir / f"{OUT_STEM}_mapping_report.txt"
unmapped_tsv = report_dir / f"{OUT_STEM}_unmapped_source_identifiers.tsv"
ambiguous_tsv = report_dir / f"{OUT_STEM}_ambiguous_source_identifiers.tsv"
duplicates_tsv = report_dir / f"{OUT_STEM}_duplicate_final_mappings.tsv"


def clean_ensembl_gene_id(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    s = re.sub(r"\.\d+$", "", s)
    return s


def split_ids(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return []
    return [clean_ensembl_gene_id(t) for t in re.split(r"[|,;\s]+", s) if t.strip()]


def first_existing(row, cols):
    for c in cols:
        if c in row.index and str(row[c]).strip() and str(row[c]).lower() != "nan":
            return str(row[c]).strip()
    return ""


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_token_dictionary(raw_dir):
    candidates = []
    for p in sorted(raw_dir.rglob("*.pkl")):
        try:
            obj = load_pickle(p)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        ens_to_int = 0
        int_to_ens = 0

        for k, v in list(obj.items())[:100000]:
            if isinstance(k, str) and clean_ensembl_gene_id(k).startswith("ENSG") and isinstance(v, int):
                ens_to_int += 1
            if isinstance(v, str) and clean_ensembl_gene_id(v).startswith("ENSG") and isinstance(k, int):
                int_to_ens += 1

        score = max(ens_to_int, int_to_ens)
        if score > 1000:
            direction = "ens_to_int" if ens_to_int >= int_to_ens else "int_to_ens"
            candidates.append((score, direction, p, obj))

    if not candidates:
        raise FileNotFoundError("Could not find a Geneformer token dictionary pickle with Ensembl gene IDs.")

    candidates.sort(reverse=True, key=lambda x: x[0])
    score, direction, path, obj = candidates[0]
    return path, direction, obj


def find_embedding_weight(raw_dir):
    # Prefer root model.safetensors, then any safetensors.
    st_files = []
    root_st = raw_dir / "model.safetensors"
    if root_st.exists():
        st_files.append(root_st)
    st_files.extend([p for p in sorted(raw_dir.rglob("*.safetensors")) if p != root_st])

    if st_files:
        from safetensors import safe_open

        for st in st_files:
            with safe_open(st, framework="np", device="cpu") as f:
                keys = list(f.keys())

                preferred = [
                    k for k in keys
                    if k.endswith("word_embeddings.weight") or "word_embeddings.weight" in k
                ]
                fallback = [
                    k for k in keys
                    if "embeddings" in k.lower() and "weight" in k.lower() and len(f.get_tensor(k).shape) == 2
                ]

                candidates = preferred + [k for k in fallback if k not in preferred]

                if candidates:
                    key = candidates[0]
                    W = f.get_tensor(key).astype(np.float32)
                    if W.ndim != 2:
                        continue
                    return st, key, W

    # Fallback for pytorch_model.bin if present.
    bin_files = []
    root_bin = raw_dir / "pytorch_model.bin"
    if root_bin.exists():
        bin_files.append(root_bin)
    bin_files.extend([p for p in sorted(raw_dir.rglob("*.bin")) if p != root_bin])

    if bin_files:
        import torch

        for bp in bin_files:
            sd = torch.load(bp, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]

            keys = list(sd.keys())
            preferred = [
                k for k in keys
                if k.endswith("word_embeddings.weight") or "word_embeddings.weight" in k
            ]
            fallback = [
                k for k in keys
                if "embeddings" in k.lower() and "weight" in k.lower() and len(sd[k].shape) == 2
            ]

            candidates = preferred + [k for k in fallback if k not in preferred]

            if candidates:
                key = candidates[0]
                W = sd[key].detach().cpu().numpy().astype(np.float32)
                if W.ndim != 2:
                    continue
                return bp, key, W

    raise FileNotFoundError("Could not find model embedding weight in safetensors or pytorch_model.bin.")


print("Finding Geneformer token dictionary...")
token_dict_path, token_dict_direction, token_dict = find_token_dictionary(raw_dir)
print("Token dictionary:", token_dict_path)
print("Token dictionary direction:", token_dict_direction)
print("Token dictionary entries:", len(token_dict))

print("Finding model embedding weight...")
weight_path, weight_key, W = find_embedding_weight(raw_dir)
print("Weight file:", weight_path)
print("Weight key:", weight_key)
print("Embedding matrix:", W.shape)

print("Building source gene-token table...")
source_records = []

if token_dict_direction == "ens_to_int":
    items = token_dict.items()
    for ens, tok in items:
        ens_clean = clean_ensembl_gene_id(ens)
        if not ens_clean.startswith("ENSG"):
            continue
        if not isinstance(tok, int):
            continue
        if tok < 0 or tok >= W.shape[0]:
            continue
        source_records.append({
            "source_identifier": ens_clean,
            "token_id": tok,
        })
else:
    items = token_dict.items()
    for tok, ens in items:
        ens_clean = clean_ensembl_gene_id(ens)
        if not ens_clean.startswith("ENSG"):
            continue
        if not isinstance(tok, int):
            continue
        if tok < 0 or tok >= W.shape[0]:
            continue
        source_records.append({
            "source_identifier": ens_clean,
            "token_id": tok,
        })

source_df = pd.DataFrame(source_records).drop_duplicates()
print(f"Usable source Ensembl-token rows: {len(source_df):,}")

print("Loading master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")

if "ensembl_gene_id" not in master.columns:
    raise ValueError("Missing ensembl_gene_id in master table.")

symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

ensembl_cols = [c for c in [
    "ensembl_gene_id",
    "hgnc_ensembl_gene_id",
    "map_ensembl_gene_ids_all",
    "all_ensembl_gene_ids_all",
] if c in master.columns]

if not ensembl_cols:
    raise ValueError("No Ensembl gene ID columns found in master table.")

print("Building Ensembl gene ID -> master index...")
ens_to_master_rows = defaultdict(set)

for i, row in master.iterrows():
    for c in ensembl_cols:
        for token in split_ids(row[c]):
            e = clean_ensembl_gene_id(token)
            if e.startswith("ENSG"):
                ens_to_master_rows[e].add(i)

unique_ens_to_master = {
    e: next(iter(rows))
    for e, rows in ens_to_master_rows.items()
    if len(rows) == 1
}

ambiguous_ens_to_master = {
    e: sorted(rows)
    for e, rows in ens_to_master_rows.items()
    if len(rows) > 1
}

mapped_records = []
unmapped_records = []
ambiguous_records = []

for _, src in source_df.iterrows():
    ens = clean_ensembl_gene_id(src["source_identifier"])
    tok = int(src["token_id"])

    if ens in unique_ens_to_master:
        mapped_records.append((ens, tok, unique_ens_to_master[ens]))
    elif ens in ambiguous_ens_to_master:
        rows = ambiguous_ens_to_master[ens]
        ambiguous_records.append({
            "source_identifier": ens,
            "token_id": tok,
            "reason": "ensembl_gene_id_maps_to_multiple_master_rows",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    else:
        unmapped_records.append({
            "source_identifier": ens,
            "token_id": tok,
            "reason": "no_unique_ensembl_gene_id_match_in_master",
        })

print(f"Mapped source rows before duplicate collapse: {len(mapped_records):,}")
print(f"Unmapped source rows: {len(unmapped_records):,}")
print(f"Ambiguous source rows: {len(ambiguous_records):,}")

master_to_source = defaultdict(list)

for ens, tok, master_i in mapped_records:
    master_to_source[master_i].append((ens, tok))

final_master_indices = sorted(master_to_source.keys())
X_final = []
gene_rows = []
duplicate_rows = []

for master_i in final_master_indices:
    source_items = master_to_source[master_i]
    token_ids = [tok for ens, tok in source_items]
    ens_ids = [ens for ens, tok in source_items]
    row = master.loc[master_i]

    X_final.append(W[token_ids].mean(axis=0).astype(np.float32))

    if len(token_ids) > 1:
        duplicate_rows.append({
            "ensembl_gene_id": row["ensembl_gene_id"],
            "gene_symbol": row.get(symbol_col, ""),
            "source_key_count": len(token_ids),
            "source_keys": ";".join(ens_ids),
            "token_ids": ";".join(map(str, token_ids)),
            "action": "averaged_duplicate_source_tokens_mapping_to_same_master_gene",
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
        "source_key_count": len(token_ids),
        "source_keys_examples": ";".join(ens_ids[:10]),
        "source_token_ids_examples": ";".join(map(str, token_ids[:10])),
        "mapping_methods": "ensembl_gene_id_exact_unique",
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

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "source_embedding": SOURCE_EMBEDDING,
    "modality": MODALITY,
    "algorithm": ALGORITHM,
    "source_identifier_type": SOURCE_IDENTIFIER_TYPE,
    "download_source": "Official ctheodoris/Geneformer Hugging Face repository",
    "embedding_generated_by": "Extracted locally from official Geneformer model input gene-token embedding layer",
    "source_files": {
        "raw_dir": str(raw_dir),
        "token_dictionary": str(token_dict_path),
        "model_weight_file": str(weight_path),
        "model_weight_key": weight_key,
    },
    "master_table": str(master_path),
    "mapping_strategy": "Exact Ensembl gene ID mapping to master_gene_table_v1_1_enriched.csv. Only Ensembl IDs mapping to exactly one master gene are accepted. Duplicate source tokens mapping to the same final master gene are averaged.",
    "counts": {
        "token_dictionary_entries": int(len(token_dict)),
        "source_usable_gene_tokens": int(len(source_df)),
        "model_vocab_rows": int(W.shape[0]),
        "source_dimensions": int(W.shape[1]),
        "mapped_source_rows_before_duplicate_collapse": int(len(mapped_records)),
        "unmapped_source_rows": int(len(unmapped_records)),
        "ambiguous_source_rows": int(len(ambiguous_records)),
        "final_mapped_genes": int(X_final.shape[0]),
        "final_dimensions": int(X_final.shape[1]),
        "master_rows": int(len(master)),
        "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
    },
    "coverage": {
        "source_row_mapping_fraction": float(len(mapped_records) / len(source_df)) if len(source_df) else 0.0,
        "master_gene_coverage_fraction": float(X_final.shape[0] / len(master)),
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
    f.write(f"{EMBEDDING_NAME} mapping report\n")
    f.write("=" * (len(EMBEDDING_NAME) + 15) + "\n\n")
    f.write(f"Raw directory:       {raw_dir}\n")
    f.write(f"Token dictionary:    {token_dict_path}\n")
    f.write(f"Token dict direction:{token_dict_direction}\n")
    f.write(f"Model weight file:   {weight_path}\n")
    f.write(f"Model weight key:    {weight_key}\n")
    f.write(f"Master table:        {master_path}\n\n")
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
