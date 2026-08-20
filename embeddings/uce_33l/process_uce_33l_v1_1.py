from pathlib import Path
import json
import pickle
import re
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


EMBEDDING_NAME = "UCE-33L-v1.1"
SOURCE_EMBEDDING = "official_UCE_33L_8ep_1024t_1280"
SOURCE_IDENTIFIER_TYPE = "UCE species_chrom human gene symbol"
MODALITY = "single_cell_expression_foundation_model / protein-informed gene-token embedding"
ALGORITHM = "UCE 33-layer model gene encoder projection of ESM2-derived all_tokens"
OUT_STEM = "uce_33l_v1_1"

home = Path.home()

master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"
raw_dir = home / "data/raw_embeddings/uce_official"

species_chrom_path = raw_dir / "species_chrom.csv"
species_offsets_path = raw_dir / "species_offsets.pkl"
all_tokens_path = raw_dir / "all_tokens.torch"
checkpoint_path = raw_dir / "33l_8ep_1024t_1280.torch"
repo_dir = raw_dir / "repo"

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


def get_next_species_offset(offsets, species, max_rows):
    start = int(offsets[species])
    later = sorted(int(v) for v in offsets.values() if int(v) > start)
    return later[0] if later else max_rows


print("Loading UCE species table...")
species_chrom = pd.read_csv(species_chrom_path, dtype={"gene_symbol": str, "chromosome": str, "species": str})
human_df_all = species_chrom[species_chrom["species"] == "human"].copy().reset_index(drop=True)
print(f"species_chrom rows: {len(species_chrom):,}")
print(f"human species rows before token alignment: {len(human_df_all):,}")

print("Loading UCE species offsets...")
with open(species_offsets_path, "rb") as f:
    species_offsets = pickle.load(f)

if "human" not in species_offsets:
    raise ValueError("No human offset found in species_offsets.pkl")

print("Loading all_tokens.torch...")
all_tokens = torch.load(all_tokens_path, map_location="cpu")
if not isinstance(all_tokens, torch.Tensor):
    raise ValueError("Expected all_tokens.torch to contain a torch.Tensor.")

print(f"all_tokens shape: {tuple(all_tokens.shape)}")

human_start = int(species_offsets["human"])
human_end = get_next_species_offset(species_offsets, "human", all_tokens.shape[0])
human_token_count = human_end - human_start

print(f"human token range in all_tokens: [{human_start}, {human_end})")
print(f"human token count from offsets: {human_token_count:,}")

excluded_unembedded_tsv = report_dir / f"{OUT_STEM}_excluded_unembedded_human_source_rows.tsv"

if human_token_count > len(human_df_all):
    raise ValueError(
        f"Human token count from offsets ({human_token_count}) is larger than "
        f"human rows in species_chrom.csv ({len(human_df_all)}). Do not process until this is resolved."
    )

if human_token_count != len(human_df_all):
    excluded_unembedded = human_df_all.iloc[human_token_count:].copy()
    excluded_unembedded["reason"] = "present_in_species_chrom_but_no_matching_all_tokens_row_from_species_offsets"
    excluded_unembedded.to_csv(excluded_unembedded_tsv, sep="\t", index=False)

    print(
        f"WARNING: species_chrom has {len(human_df_all):,} human rows but "
        f"all_tokens offset segment has {human_token_count:,}. "
        f"Keeping the first {human_token_count:,} token-aligned rows and excluding "
        f"{len(excluded_unembedded):,} trailing rows."
    )
    print(f"Excluded human source rows report: {excluded_unembedded_tsv}")

human_df = human_df_all.iloc[:human_token_count].copy().reset_index(drop=True)
print(f"human token-aligned rows used: {len(human_df):,}")

print("Loading UCE 33L checkpoint...")
ckpt = torch.load(checkpoint_path, map_location="cpu")
if not isinstance(ckpt, (dict, OrderedDict)):
    raise ValueError("Expected checkpoint to be a state_dict-like dictionary.")

required_keys = [
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
]
missing = [k for k in required_keys if k not in ckpt]
if missing:
    raise KeyError(f"Missing required UCE encoder keys: {missing}")

linear_w = ckpt["encoder.0.weight"].float()
linear_b = ckpt["encoder.0.bias"].float()
ln_w = ckpt["encoder.2.weight"].float()
ln_b = ckpt["encoder.2.bias"].float()

input_dim = int(linear_w.shape[1])
output_dim = int(linear_w.shape[0])

if all_tokens.shape[1] != input_dim:
    raise ValueError(f"all_tokens dim {all_tokens.shape[1]} != encoder input dim {input_dim}")

n_transformer_layers = len({
    int(k.split(".")[2])
    for k in ckpt.keys()
    if k.startswith("transformer_encoder.layers.") and k.split(".")[2].isdigit()
})

print(f"UCE encoder input dim: {input_dim:,}")
print(f"UCE encoder output dim: {output_dim:,}")
print(f"Transformer layers found in checkpoint: {n_transformer_layers}")

# Free memory from unused checkpoint tensors.
del ckpt

print("Computing UCE encoded human gene-token embeddings...")
human_tokens = all_tokens[human_start:human_end].float()
del all_tokens

encoded_chunks = []
batch_size = 512

with torch.no_grad():
    for i in range(0, human_tokens.shape[0], batch_size):
        x = human_tokens[i:i + batch_size]
        x = F.linear(x, linear_w, linear_b)
        x = F.relu(x)
        x = F.layer_norm(x, normalized_shape=(output_dim,), weight=ln_w, bias=ln_b)
        encoded_chunks.append(x.cpu().numpy().astype(np.float32))

X_source = np.vstack(encoded_chunks).astype(np.float32)
print(f"Encoded human source matrix: {X_source.shape[0]:,} genes x {X_source.shape[1]:,} dims")

del human_tokens, linear_w, linear_b, ln_w, ln_b, encoded_chunks

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

for src_i, src in human_df.iterrows():
    symbol = clean_symbol(src["gene_symbol"])

    if symbol in current_unique:
        mapped_records.append((src_i, symbol, current_unique[symbol], "symbol_current_or_hgnc_approved_exact_unique"))
    elif symbol in current_ambiguous:
        rows = current_ambiguous[symbol]
        ambiguous_records.append({
            "source_identifier": symbol,
            "source_row_index": src_i,
            "chromosome": src.get("chromosome", ""),
            "start": src.get("start", ""),
            "reason": "current_symbol_maps_to_multiple_master_rows",
            "mapping_tier": "current_or_hgnc_approved",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    elif symbol in previous_unique:
        mapped_records.append((src_i, symbol, previous_unique[symbol], "symbol_hgnc_previous_exact_unique"))
    elif symbol in previous_ambiguous:
        rows = previous_ambiguous[symbol]
        ambiguous_records.append({
            "source_identifier": symbol,
            "source_row_index": src_i,
            "chromosome": src.get("chromosome", ""),
            "start": src.get("start", ""),
            "reason": "previous_symbol_maps_to_multiple_master_rows",
            "mapping_tier": "hgnc_previous_symbols",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    elif symbol in alias_unique:
        mapped_records.append((src_i, symbol, alias_unique[symbol], "symbol_hgnc_alias_exact_unique"))
    elif symbol in alias_ambiguous:
        rows = alias_ambiguous[symbol]
        ambiguous_records.append({
            "source_identifier": symbol,
            "source_row_index": src_i,
            "chromosome": src.get("chromosome", ""),
            "start": src.get("start", ""),
            "reason": "alias_symbol_maps_to_multiple_master_rows",
            "mapping_tier": "hgnc_alias_symbols",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    else:
        unmapped_records.append({
            "source_identifier": symbol,
            "source_row_index": src_i,
            "chromosome": src.get("chromosome", ""),
            "start": src.get("start", ""),
            "reason": "no_unique_symbol_match_in_master",
        })

print(f"Mapped source rows before duplicate collapse: {len(mapped_records):,}")
print(f"Unmapped source rows: {len(unmapped_records):,}")
print(f"Ambiguous source rows: {len(ambiguous_records):,}")

master_to_source = defaultdict(list)

for src_i, symbol, master_i, method in mapped_records:
    master_to_source[master_i].append((src_i, symbol, method))

final_master_indices = sorted(master_to_source.keys())
X_final = []
gene_rows = []
duplicate_rows = []

for master_i in final_master_indices:
    source_items = master_to_source[master_i]
    src_indices = [src_i for src_i, symbol, method in source_items]
    symbols = [symbol for src_i, symbol, method in source_items]
    methods = sorted(set(method for src_i, symbol, method in source_items))
    row = master.loc[master_i]

    X_final.append(X_source[src_indices].mean(axis=0).astype(np.float32))

    source_chromosomes = human_df.loc[src_indices, "chromosome"].astype(str).tolist()
    source_starts = human_df.loc[src_indices, "start"].astype(str).tolist()

    if len(src_indices) > 1:
        duplicate_rows.append({
            "ensembl_gene_id": row["ensembl_gene_id"],
            "gene_symbol": row.get(symbol_col, ""),
            "source_key_count": len(src_indices),
            "source_keys": ";".join(symbols),
            "source_row_indices": ";".join(map(str, src_indices)),
            "source_chromosomes": ";".join(source_chromosomes),
            "source_starts": ";".join(source_starts),
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
        "source_key_count": len(src_indices),
        "source_keys_examples": ";".join(symbols[:10]),
        "source_row_indices_examples": ";".join(map(str, src_indices[:10])),
        "source_chromosomes_examples": ";".join(source_chromosomes[:10]),
        "source_starts_examples": ";".join(source_starts[:10]),
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
    "download_source": "Official UCE GitHub repository plus official Figshare model files",
    "original_model_or_method": "Universal Cell Embeddings (UCE)",
    "embedding_generated_by": "Extracted locally from official UCE all_tokens and 33-layer checkpoint gene encoder",
    "exception_status": "Included after supervisor approval despite citation/age rule uncertainty",
    "sequence_and_token_provenance": {
        "input_sequence_type": "protein-informed gene token",
        "input_sequence_source": "UCE all_tokens.torch, described by UCE as derived from ESM2 protein embeddings",
        "input_sequence_release": "not explicitly verified from source files",
        "species": "human subset from multi-species UCE release",
        "transcript_choice": "not directly applicable / not specified by UCE source files",
        "protein_choice": "not directly applicable to processed UCE token table / ESM2-derived provenance should be recorded as partially verified",
        "genome_build": "not explicitly stated in downloaded source files",
        "model_checkpoint": "33l_8ep_1024t_1280.torch",
        "pooling_strategy": "UCE gene encoder projection: Linear(5120->1280), ReLU, LayerNorm",
        "sequence_reference_status": "partially verified; exact protein/reference release not specified by downloaded files",
    },
    "source_files": {
        "raw_dir": str(raw_dir),
        "repo_dir": str(repo_dir),
        "species_chrom_csv": str(species_chrom_path),
        "species_offsets_pkl": str(species_offsets_path),
        "all_tokens_torch": str(all_tokens_path),
        "checkpoint": str(checkpoint_path),
        "figshare_article": "24320806",
        "figshare_doi": "10.6084/m9.figshare.24320806.v5",
    },
    "model_structure_used": {
        "raw_all_tokens_shape": [145469, input_dim],
        "human_token_start": human_start,
        "human_token_end": human_end,
        "human_token_count": human_token_count,
        "encoder_input_dim": input_dim,
        "encoder_output_dim": output_dim,
        "encoder_keys_used": required_keys,
        "transformer_layers_in_checkpoint": n_transformer_layers,
        "transformer_layers_used_for_fixed_gene_embedding": 0,
    },
    "master_table": str(master_path),
    "mapping_strategy": "Priority symbol mapping to master_gene_table_v1_1_enriched.csv: current gene_symbol / hgnc_approved_symbol first, then HGNC previous symbols, then HGNC alias symbols. Only unique mappings are accepted. Duplicate source tokens mapping to the same final master gene are averaged.",
    "counts": {
        "species_chrom_rows": int(len(species_chrom)),
        "human_species_chrom_rows_before_token_alignment": int(len(human_df_all)),
        "human_source_rows": int(len(human_df)),
        "excluded_unembedded_human_rows": int(len(human_df_all) - len(human_df)),
        "raw_all_tokens_rows": 145469,
        "raw_input_dimensions": int(input_dim),
        "encoded_source_dimensions": int(output_dim),
        "mapped_source_rows_before_duplicate_collapse": int(len(mapped_records)),
        "unmapped_source_rows": int(len(unmapped_records)),
        "ambiguous_source_rows": int(len(ambiguous_records)),
        "final_mapped_genes": int(X_final.shape[0]),
        "final_dimensions": int(X_final.shape[1]),
        "master_rows": int(len(master)),
        "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
    },
    "coverage": {
        "source_row_mapping_fraction": float(len(mapped_records) / len(human_df)) if len(human_df) else 0.0,
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
        "excluded_unembedded_human_rows_tsv": str(excluded_unembedded_tsv),
    },
}

with open(out_meta, "w") as f:
    json.dump(metadata, f, indent=2)

with open(report_txt, "w") as f:
    f.write(f"{EMBEDDING_NAME} mapping report\n")
    f.write("=" * (len(EMBEDDING_NAME) + 15) + "\n\n")
    f.write(f"Raw directory:       {raw_dir}\n")
    f.write(f"species_chrom.csv:   {species_chrom_path}\n")
    f.write(f"species_offsets.pkl: {species_offsets_path}\n")
    f.write(f"all_tokens.torch:    {all_tokens_path}\n")
    f.write(f"checkpoint:          {checkpoint_path}\n")
    f.write(f"Master table:        {master_path}\n\n")

    f.write("UCE provenance note:\n")
    f.write("The processed vector is the UCE 33L gene encoder output from all_tokens.torch.\n")
    f.write("This uses encoder.0 Linear, ReLU, and encoder.2 LayerNorm to project 5120-dim raw tokens to 1280 dims.\n")
    f.write("The contextual transformer layers are not used because they produce cell-context-dependent embeddings, not fixed gene embeddings.\n\n")

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
