from pathlib import Path
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


EMBEDDING_NAME = "scFoundation-gene-pos-v1.1"
SOURCE_EMBEDDING = "official_scFoundation_gene_checkpoint_pos_emb"
SOURCE_IDENTIFIER_TYPE = "scFoundation OS_scRNA_gene_index gene symbol"
MODALITY = "single_cell_expression_foundation_model / learned gene identity-position embedding"
ALGORITHM = "scFoundation gene sub-checkpoint model.pos_emb.weight fixed gene rows"
OUT_STEM = "scfoundation_gene_pos_v1_1"

home = Path.home()

master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"
raw_dir = home / "data/raw_embeddings/scfoundation_official"
repo_dir = raw_dir / "repo"
model_dir = repo_dir / "model"

gene_index_path = repo_dir / "OS_scRNA_gene_index.19264.tsv"
if not gene_index_path.exists():
    gene_index_path = model_dir / "OS_scRNA_gene_index.19264.tsv"

checkpoint_path = model_dir / "models/models.ckpt"

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
excluded_special_tsv = report_dir / f"{OUT_STEM}_excluded_special_checkpoint_rows.tsv"


def clean_symbol(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    return s


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


print("Loading scFoundation gene index...")
gene_index = pd.read_csv(gene_index_path, sep="\t", dtype=str).fillna("")
print(f"Gene index rows: {len(gene_index):,}")
print(gene_index.head())

if not {"gene_name", "index"}.issubset(gene_index.columns):
    raise ValueError(f"Expected columns gene_name and index in {gene_index_path}")

gene_index["index_int"] = gene_index["index"].astype(int)

expected = list(range(len(gene_index)))
observed = gene_index["index_int"].tolist()
if observed != expected:
    raise ValueError("scFoundation gene index is not exactly 0..N-1 in file order. Inspect before processing.")

print("Loading scFoundation checkpoint...")
obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

if "gene" not in obj:
    raise KeyError("Checkpoint does not contain top-level key 'gene'.")

gene_ckpt = obj["gene"]
config = gene_ckpt.get("config", {})
state_dict = gene_ckpt.get("state_dict", {})

weight_key = "model.pos_emb.weight"
if weight_key not in state_dict:
    raise KeyError(f"Missing {weight_key} in gene state_dict.")

pos_emb = state_dict[weight_key].detach().cpu().float().numpy().astype(np.float32)
print(f"{weight_key} shape: {pos_emb.shape}")

source_gene_count = len(gene_index)
if pos_emb.shape[0] < source_gene_count:
    raise ValueError(f"pos_emb has only {pos_emb.shape[0]} rows but gene index has {source_gene_count} rows.")

X_source = pos_emb[gene_index["index_int"].to_numpy()].astype(np.float32)
print(f"Source gene matrix: {X_source.shape[0]:,} genes x {X_source.shape[1]:,} dims")

excluded_rows = []
for i in range(source_gene_count, pos_emb.shape[0]):
    excluded_rows.append({
        "checkpoint_row_index": i,
        "reason": "checkpoint_pos_emb_row_not_in_OS_scRNA_gene_index_19264",
    })
pd.DataFrame(excluded_rows).to_csv(excluded_special_tsv, sep="\t", index=False)

del obj, gene_ckpt, state_dict, pos_emb

print("Loading master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")

if "ensembl_gene_id" not in master.columns:
    raise ValueError("Missing ensembl_gene_id in master table.")

symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

print("Building prioritized master symbol mappings...")

current_map = defaultdict(set)
previous_map = defaultdict(set)
alias_map = defaultdict(set)

for i, row in master.iterrows():
    for c in ["gene_symbol", "hgnc_approved_symbol"]:
        if c in master.columns:
            add_symbol(current_map, row[c], i)

    if "hgnc_previous_symbols" in master.columns:
        for s in split_symbols(row["hgnc_previous_symbols"]):
            add_symbol(previous_map, s, i)

    if "hgnc_alias_symbols" in master.columns:
        for s in split_symbols(row["hgnc_alias_symbols"]):
            add_symbol(alias_map, s, i)

current_unique, current_ambiguous = build_unique_and_ambiguous(current_map)
previous_unique, previous_ambiguous = build_unique_and_ambiguous(previous_map)
alias_unique, alias_ambiguous = build_unique_and_ambiguous(alias_map)

mapped_records = []
unmapped_records = []
ambiguous_records = []

for src_i, src in gene_index.iterrows():
    symbol = clean_symbol(src["gene_name"])
    idx = int(src["index_int"])

    if symbol in current_unique:
        mapped_records.append((src_i, idx, symbol, current_unique[symbol], "symbol_current_or_hgnc_approved_exact_unique"))
    elif symbol in current_ambiguous:
        rows = current_ambiguous[symbol]
        ambiguous_records.append({
            "source_identifier": symbol,
            "source_row_index": src_i,
            "source_checkpoint_index": idx,
            "reason": "current_symbol_maps_to_multiple_master_rows",
            "mapping_tier": "current_or_hgnc_approved",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    elif symbol in previous_unique:
        mapped_records.append((src_i, idx, symbol, previous_unique[symbol], "symbol_hgnc_previous_exact_unique"))
    elif symbol in previous_ambiguous:
        rows = previous_ambiguous[symbol]
        ambiguous_records.append({
            "source_identifier": symbol,
            "source_row_index": src_i,
            "source_checkpoint_index": idx,
            "reason": "previous_symbol_maps_to_multiple_master_rows",
            "mapping_tier": "hgnc_previous_symbols",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    elif symbol in alias_unique:
        mapped_records.append((src_i, idx, symbol, alias_unique[symbol], "symbol_hgnc_alias_exact_unique"))
    elif symbol in alias_ambiguous:
        rows = alias_ambiguous[symbol]
        ambiguous_records.append({
            "source_identifier": symbol,
            "source_row_index": src_i,
            "source_checkpoint_index": idx,
            "reason": "alias_symbol_maps_to_multiple_master_rows",
            "mapping_tier": "hgnc_alias_symbols",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    else:
        unmapped_records.append({
            "source_identifier": symbol,
            "source_row_index": src_i,
            "source_checkpoint_index": idx,
            "reason": "no_unique_symbol_match_in_master",
        })

print(f"Mapped source rows before duplicate collapse: {len(mapped_records):,}")
print(f"Unmapped source rows: {len(unmapped_records):,}")
print(f"Ambiguous source rows: {len(ambiguous_records):,}")

master_to_source = defaultdict(list)
for src_i, idx, symbol, master_i, method in mapped_records:
    master_to_source[master_i].append((src_i, idx, symbol, method))

final_master_indices = sorted(master_to_source.keys())
X_final = []
gene_rows = []
duplicate_rows = []

for master_i in final_master_indices:
    source_items = master_to_source[master_i]
    src_indices = [src_i for src_i, idx, symbol, method in source_items]
    checkpoint_indices = [idx for src_i, idx, symbol, method in source_items]
    symbols = [symbol for src_i, idx, symbol, method in source_items]
    methods = sorted(set(method for src_i, idx, symbol, method in source_items))
    row = master.loc[master_i]

    X_final.append(X_source[src_indices].mean(axis=0).astype(np.float32))

    if len(src_indices) > 1:
        duplicate_rows.append({
            "ensembl_gene_id": row["ensembl_gene_id"],
            "gene_symbol": row.get(symbol_col, ""),
            "source_key_count": len(src_indices),
            "source_keys": ";".join(symbols),
            "source_row_indices": ";".join(map(str, src_indices)),
            "source_checkpoint_indices": ";".join(map(str, checkpoint_indices)),
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
        "source_embedding": SOURCE_EMBEDDING,
        "source_identifier_type": SOURCE_IDENTIFIER_TYPE,
        "source_key_count": len(src_indices),
        "source_keys_examples": ";".join(symbols[:10]),
        "source_row_indices_examples": ";".join(map(str, src_indices[:10])),
        "source_checkpoint_indices_examples": ";".join(map(str, checkpoint_indices[:10])),
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

model_config = config.get("model_config", {}).get("mae_autobin", {})
dataset_config = config.get("dataset_config", {}).get("rnaseq", {})

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "source_embedding": SOURCE_EMBEDDING,
    "modality": MODALITY,
    "algorithm": ALGORITHM,
    "source_identifier_type": SOURCE_IDENTIFIER_TYPE,
    "download_source": "Official scFoundation GitHub repository and official SharePoint checkpoint linked from the repository",
    "original_model_or_method": "scFoundation / xTrimoGene",
    "embedding_generated_by": "Extracted locally from official scFoundation gene sub-checkpoint model.pos_emb.weight",
    "fixed_embedding_definition": "Rows 0..19263 of model.pos_emb.weight, matched to OS_scRNA_gene_index.19264.tsv. Rows beyond the 19,264-gene index are treated as non-gene/special checkpoint rows and excluded.",
    "contextual_output_note": "The official scFoundation gene-mode inference can produce context-dependent gene outputs of shape N x 19264 x h for input expression profiles. This processed embedding is not that contextual output; it is the fixed learned gene identity-position embedding from the gene checkpoint.",
    "source_files": {
        "raw_dir": str(raw_dir),
        "repo_dir": str(repo_dir),
        "gene_index": str(gene_index_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_top_key": "gene",
        "checkpoint_weight_key": weight_key,
    },
    "model_config_used": {
        "model": model_config.get("model", ""),
        "gene_num": model_config.get("gene_num", ""),
        "seq_len": model_config.get("seq_len", ""),
        "encoder_hidden_dim": model_config.get("encoder", {}).get("hidden_dim", ""),
        "encoder_depth": model_config.get("encoder", {}).get("depth", ""),
        "encoder_heads": model_config.get("encoder", {}).get("heads", ""),
        "decoder_hidden_dim": model_config.get("decoder", {}).get("hidden_dim", ""),
        "decoder_depth": model_config.get("decoder", {}).get("depth", ""),
        "n_class": model_config.get("n_class", ""),
        "bin_num": model_config.get("bin_num", ""),
        "pad_token_id": model_config.get("pad_token_id", ""),
        "mask_token_id": model_config.get("mask_token_id", ""),
        "training_gene_universe_size": dataset_config.get("seq_len", ""),
    },
    "sequence_and_reference_provenance": {
        "input_sequence_type": "not sequence-derived during local processing",
        "input_sequence_source": "not directly applicable; learned from scRNA-seq expression data",
        "input_sequence_release": "not applicable",
        "species": "human",
        "gene_index_file": "OS_scRNA_gene_index.19264.tsv",
        "model_checkpoint": "models.ckpt",
        "model_checkpoint_subkey": "gene",
        "pooling_strategy": "fixed checkpoint embedding table; no cell/context pooling",
        "sequence_reference_status": "not applicable for sequence versioning; gene-index provenance recorded",
    },
    "master_table": str(master_path),
    "mapping_strategy": "Priority symbol mapping to master_gene_table_v1_1_enriched.csv: current gene_symbol / hgnc_approved_symbol first, then HGNC previous symbols, then HGNC alias symbols. Only unique mappings are accepted. Duplicate source rows mapping to the same final master gene are averaged.",
    "counts": {
        "source_gene_index_rows": int(len(gene_index)),
        "checkpoint_pos_emb_rows": int(source_gene_count + len(excluded_rows)),
        "excluded_special_checkpoint_rows": int(len(excluded_rows)),
        "source_dimensions": int(X_source.shape[1]),
        "mapped_source_rows_before_duplicate_collapse": int(len(mapped_records)),
        "unmapped_source_rows": int(len(unmapped_records)),
        "ambiguous_source_rows": int(len(ambiguous_records)),
        "final_mapped_genes": int(X_final.shape[0]),
        "final_dimensions": int(X_final.shape[1]),
        "master_rows": int(len(master)),
        "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
    },
    "coverage": {
        "source_row_mapping_fraction": float(len(mapped_records) / len(gene_index)) if len(gene_index) else 0.0,
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
        "excluded_special_checkpoint_rows_tsv": str(excluded_special_tsv),
    },
}

with open(out_meta, "w") as f:
    json.dump(metadata, f, indent=2)

with open(report_txt, "w") as f:
    f.write(f"{EMBEDDING_NAME} mapping report\n")
    f.write("=" * (len(EMBEDDING_NAME) + 15) + "\n\n")
    f.write(f"Raw directory:      {raw_dir}\n")
    f.write(f"Repository:         {repo_dir}\n")
    f.write(f"Gene index:         {gene_index_path}\n")
    f.write(f"Checkpoint:         {checkpoint_path}\n")
    f.write(f"Checkpoint subkey:  gene\n")
    f.write(f"Weight key:         {weight_key}\n")
    f.write(f"Master table:       {master_path}\n\n")

    f.write("scFoundation provenance note:\n")
    f.write("The processed vector is the fixed learned gene identity-position embedding from the gene sub-checkpoint.\n")
    f.write("Specifically, it uses rows 0..19263 of model.pos_emb.weight matched to OS_scRNA_gene_index.19264.tsv.\n")
    f.write("Official scFoundation gene-mode inference can produce context-dependent N x 19264 x h gene outputs, but those depend on input expression profiles and are not used here.\n\n")

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
