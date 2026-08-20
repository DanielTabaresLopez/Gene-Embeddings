from pathlib import Path
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


EMBEDDING_NAME = "scGPT-whole-human-v1.1"
SOURCE_EMBEDDING = "official_scGPT_whole_human"
SOURCE_IDENTIFIER_TYPE = "scGPT vocab gene symbol"
MODALITY = "single_cell_expression_foundation_model"
ALGORITHM = "scGPT whole-human input gene-token embedding layer"
OUT_STEM = "scgpt_whole_human_v1_1"

home = Path.home()

master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"
raw_dir = home / "data/raw_embeddings/scgpt_official"

vocab_path = raw_dir / "vocab.json"
model_path = raw_dir / "best_model.pt"
args_path = raw_dir / "args.json"

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


def clean_symbol(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    return s


def is_special_token(s):
    s = str(s).strip()
    if not s:
        return True
    if s.startswith("<") and s.endswith(">"):
        return True
    if s in {"[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"}:
        return True
    return False


def split_symbols(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return []
    return [clean_symbol(t) for t in re.split(r"[|,;]+", s) if clean_symbol(t)]


def first_existing(row, cols):
    for c in cols:
        if c in row.index and str(row[c]).strip() and str(row[c]).lower() != "nan":
            return str(row[c]).strip()
    return ""


def add_symbol(mapping, symbol, master_i):
    symbol = clean_symbol(symbol)
    if symbol:
        mapping[symbol].add(master_i)


def build_unique_and_ambiguous(mapping):
    unique = {
        s: next(iter(rows))
        for s, rows in mapping.items()
        if len(rows) == 1
    }
    ambiguous = {
        s: sorted(rows)
        for s, rows in mapping.items()
        if len(rows) > 1
    }
    return unique, ambiguous


print("Loading scGPT vocab...")
with open(vocab_path) as f:
    vocab = json.load(f)

if not isinstance(vocab, dict):
    raise ValueError("Expected vocab.json to be a dictionary of token -> token_id.")

print(f"Vocab entries: {len(vocab):,}")

print("Loading scGPT model...")
state = torch.load(model_path, map_location="cpu")

if "encoder.embedding.weight" not in state:
    keys = list(state.keys())
    candidates = [k for k in keys if "embedding.weight" in k and hasattr(state[k], "shape") and len(state[k].shape) == 2]
    raise KeyError(f"encoder.embedding.weight not found. Candidate embedding keys: {candidates[:20]}")

W = state["encoder.embedding.weight"].detach().cpu().numpy().astype(np.float32)
print(f"Embedding matrix: {W.shape[0]:,} tokens x {W.shape[1]:,} dims")

print("Building source symbol-token table...")
source_records = []

for symbol, token_id in vocab.items():
    symbol_clean = clean_symbol(symbol)

    if is_special_token(symbol_clean):
        continue
    if not isinstance(token_id, int):
        continue
    if token_id < 0 or token_id >= W.shape[0]:
        continue

    source_records.append({
        "source_identifier": symbol_clean,
        "token_id": token_id,
    })

source_df = pd.DataFrame(source_records).drop_duplicates()
print(f"Usable source symbol tokens: {len(source_df):,}")

print("Loading master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")

if "ensembl_gene_id" not in master.columns:
    raise ValueError("Missing ensembl_gene_id in master table.")

symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

print("Building prioritized symbol mappings...")

current_map = defaultdict(set)
previous_map = defaultdict(set)
alias_map = defaultdict(set)

for i, row in master.iterrows():
    # Priority 1: current / approved symbols.
    for c in ["gene_symbol", "hgnc_approved_symbol"]:
        if c in master.columns:
            add_symbol(current_map, row[c], i)

    # Priority 2: previous symbols.
    if "hgnc_previous_symbols" in master.columns:
        for s in split_symbols(row["hgnc_previous_symbols"]):
            add_symbol(previous_map, s, i)

    # Priority 3: alias symbols.
    if "hgnc_alias_symbols" in master.columns:
        for s in split_symbols(row["hgnc_alias_symbols"]):
            add_symbol(alias_map, s, i)

current_unique, current_ambiguous = build_unique_and_ambiguous(current_map)
previous_unique, previous_ambiguous = build_unique_and_ambiguous(previous_map)
alias_unique, alias_ambiguous = build_unique_and_ambiguous(alias_map)

mapped_records = []
unmapped_records = []
ambiguous_records = []

for _, src in source_df.iterrows():
    symbol = clean_symbol(src["source_identifier"])
    token_id = int(src["token_id"])

    if symbol in current_unique:
        mapped_records.append((symbol, token_id, current_unique[symbol], "symbol_current_or_hgnc_approved_exact_unique"))
    elif symbol in current_ambiguous:
        rows = current_ambiguous[symbol]
        ambiguous_records.append({
            "source_identifier": symbol,
            "token_id": token_id,
            "reason": "current_symbol_maps_to_multiple_master_rows",
            "mapping_tier": "current_or_hgnc_approved",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    elif symbol in previous_unique:
        mapped_records.append((symbol, token_id, previous_unique[symbol], "symbol_hgnc_previous_exact_unique"))
    elif symbol in previous_ambiguous:
        rows = previous_ambiguous[symbol]
        ambiguous_records.append({
            "source_identifier": symbol,
            "token_id": token_id,
            "reason": "previous_symbol_maps_to_multiple_master_rows",
            "mapping_tier": "hgnc_previous_symbols",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    elif symbol in alias_unique:
        mapped_records.append((symbol, token_id, alias_unique[symbol], "symbol_hgnc_alias_exact_unique"))
    elif symbol in alias_ambiguous:
        rows = alias_ambiguous[symbol]
        ambiguous_records.append({
            "source_identifier": symbol,
            "token_id": token_id,
            "reason": "alias_symbol_maps_to_multiple_master_rows",
            "mapping_tier": "hgnc_alias_symbols",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    else:
        unmapped_records.append({
            "source_identifier": symbol,
            "token_id": token_id,
            "reason": "no_unique_symbol_match_in_master",
        })

print(f"Mapped source rows before duplicate collapse: {len(mapped_records):,}")
print(f"Unmapped source rows: {len(unmapped_records):,}")
print(f"Ambiguous source rows: {len(ambiguous_records):,}")

master_to_source = defaultdict(list)

for symbol, token_id, master_i, method in mapped_records:
    master_to_source[master_i].append((symbol, token_id, method))

final_master_indices = sorted(master_to_source.keys())
X_final = []
gene_rows = []
duplicate_rows = []

for master_i in final_master_indices:
    source_items = master_to_source[master_i]
    symbols = [s for s, token_id, method in source_items]
    token_ids = [token_id for s, token_id, method in source_items]
    methods = sorted(set(method for s, token_id, method in source_items))

    row = master.loc[master_i]

    X_final.append(W[token_ids].mean(axis=0).astype(np.float32))

    if len(token_ids) > 1:
        duplicate_rows.append({
            "ensembl_gene_id": row["ensembl_gene_id"],
            "gene_symbol": row.get(symbol_col, ""),
            "source_key_count": len(token_ids),
            "source_keys": ";".join(symbols),
            "token_ids": ";".join(map(str, token_ids)),
            "mapping_methods": ";".join(methods),
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
        "source_keys_examples": ";".join(symbols[:10]),
        "source_token_ids_examples": ";".join(map(str, token_ids[:10])),
        "mapping_methods": ";".join(methods),
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

mapping_method_counts = genes_df["mapping_methods"].value_counts().to_dict()

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "source_embedding": SOURCE_EMBEDDING,
    "modality": MODALITY,
    "algorithm": ALGORITHM,
    "source_identifier_type": SOURCE_IDENTIFIER_TYPE,
    "download_source": "Official scGPT whole-human Google Drive checkpoint linked from the scGPT project",
    "embedding_generated_by": "Extracted locally from official scGPT whole-human model input gene-token embedding layer",
    "source_files": {
        "raw_dir": str(raw_dir),
        "vocab_json": str(vocab_path),
        "model_file": str(model_path),
        "args_json": str(args_path),
        "model_weight_key": "encoder.embedding.weight",
    },
    "master_table": str(master_path),
    "mapping_strategy": "Priority symbol mapping to master_gene_table_v1_1_enriched.csv: current gene_symbol / hgnc_approved_symbol first, then HGNC previous symbols, then HGNC alias symbols. Only unique mappings are accepted. Duplicate source tokens mapping to the same final master gene are averaged.",
    "counts": {
        "vocab_entries": int(len(vocab)),
        "source_usable_symbol_tokens": int(len(source_df)),
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
    "mapping_method_counts": mapping_method_counts,
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
    f.write(f"Raw directory:     {raw_dir}\n")
    f.write(f"Vocab file:        {vocab_path}\n")
    f.write(f"Model file:        {model_path}\n")
    f.write(f"Model weight key:  encoder.embedding.weight\n")
    f.write(f"Master table:      {master_path}\n\n")

    for k, v in metadata["counts"].items():
        f.write(f"{k}: {v}\n")

    f.write("\n")
    for k, v in metadata["coverage"].items():
        f.write(f"{k}: {v:.6f}\n")

    f.write("\nMapping method counts:\n")
    for k, v in mapping_method_counts.items():
        f.write(f"{k}: {v}\n")

    f.write("\nOutputs:\n")
    for k, v in metadata["outputs"].items():
        f.write(f"{k}: {v}\n")

print("\nDONE")
print(report_txt)
print(out_npz)
print(out_genes)
print(out_meta)
