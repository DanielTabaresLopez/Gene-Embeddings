from pathlib import Path
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


EMBEDDING_NAME = "scPRINT-medium-v1.5-gene-v1.1"
OUT_STEM = "scprint_medium_v1_5_gene_v1_1"
SOURCE_EMBEDDING = "official_scPRINT_medium_v1_5_gene_encoder_embedding"
SOURCE_IDENTIFIER_TYPE = "scPRINT checkpoint hyper_parameters genes Ensembl gene ID"
MODALITY = "single_cell_expression_foundation_model / learned gene-token embedding"
WEIGHT_KEY = "gene_encoder.embedding.weight"

home = Path.home()

master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"
raw_dir = home / "data/raw_embeddings/scprint_official"
checkpoint_path = raw_dir / "hf_checkpoint/medium-v1.5.ckpt"

out_dir = home / "data/processed_embeddings" / OUT_STEM
report_dir = home / "reports/mapping_reports"
out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

out_npz = out_dir / f"{OUT_STEM}_embeddings.npz"
out_genes = out_dir / f"{OUT_STEM}_genes.tsv"
out_meta = out_dir / f"{OUT_STEM}_metadata.json"

report_txt = report_dir / f"{OUT_STEM}_mapping_report.txt"
unmapped_tsv = report_dir / f"{OUT_STEM}_unmapped_source_identifiers.tsv"
duplicates_tsv = report_dir / f"{OUT_STEM}_duplicate_final_mappings.tsv"
excluded_nonhuman_tsv = report_dir / f"{OUT_STEM}_excluded_nonhuman_or_non_ensg_rows.tsv"


def clean_ensembl(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    return s.split(".")[0]


def first_existing(row, cols):
    for c in cols:
        if c in row.index and str(row[c]).strip() and str(row[c]).lower() != "nan":
            return str(row[c]).strip()
    return ""


print("Loading scPRINT checkpoint...")
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

hp = ckpt.get("hyper_parameters", {})
genes = hp.get("genes")
if genes is None:
    raise KeyError("Missing hyper_parameters['genes'].")

sd = ckpt.get("state_dict", {})
if WEIGHT_KEY not in sd:
    raise KeyError(f"Missing {WEIGHT_KEY} in state_dict.")

X_all = sd[WEIGHT_KEY].detach().cpu().float().numpy().astype(np.float32)

print(f"genes length: {len(genes):,}")
print(f"{WEIGHT_KEY} shape: {X_all.shape}")

if len(genes) != X_all.shape[0]:
    raise ValueError("genes list length does not match embedding rows.")

source_rows = []
excluded_rows = []

for i, g in enumerate(genes):
    gid = clean_ensembl(g)
    if gid.startswith("ENSG"):
        source_rows.append({
            "source_row_index": i,
            "source_identifier": gid,
        })
    else:
        excluded_rows.append({
            "source_row_index": i,
            "source_identifier": gid,
            "reason": "not_human_ENSG_identifier",
        })

source_df = pd.DataFrame(source_rows)
excluded_df = pd.DataFrame(excluded_rows)
excluded_df.to_csv(excluded_nonhuman_tsv, sep="\t", index=False)

X_source = X_all[source_df["source_row_index"].to_numpy()].astype(np.float32)

print(f"Human ENSG source rows: {len(source_df):,}")
print(f"Excluded non-human/non-ENSG rows: {len(excluded_df):,}")
print(f"Source matrix: {X_source.shape[0]:,} genes x {X_source.shape[1]:,} dims")

del ckpt, sd, X_all

print("Loading master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")
if "ensembl_gene_id" not in master.columns:
    raise ValueError("Missing ensembl_gene_id in master table.")

symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

master_map = defaultdict(list)
for i, gid in enumerate(master["ensembl_gene_id"].astype(str)):
    master_map[clean_ensembl(gid)].append(i)

unique_master_map = {
    gid: rows[0]
    for gid, rows in master_map.items()
    if gid and len(rows) == 1
}

ambiguous_master_map = {
    gid: rows
    for gid, rows in master_map.items()
    if gid and len(rows) > 1
}

mapped_records = []
unmapped_records = []
ambiguous_records = []

for src_i, row in source_df.iterrows():
    gid = row["source_identifier"]
    source_row_index = int(row["source_row_index"])

    if gid in unique_master_map:
        mapped_records.append({
            "source_matrix_row": src_i,
            "source_row_index": source_row_index,
            "source_identifier": gid,
            "master_row_index": unique_master_map[gid],
            "mapping_method": "ensembl_gene_id_exact_unique",
        })
    elif gid in ambiguous_master_map:
        rows = ambiguous_master_map[gid]
        ambiguous_records.append({
            "source_row_index": source_row_index,
            "source_identifier": gid,
            "reason": "ensembl_gene_id_maps_to_multiple_master_rows",
            "master_row_indices": ";".join(map(str, rows)),
            "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
        })
    else:
        unmapped_records.append({
            "source_row_index": source_row_index,
            "source_identifier": gid,
            "reason": "human_ENSG_not_in_master_gene_table",
        })

if ambiguous_records:
    ambiguous_tsv = report_dir / f"{OUT_STEM}_ambiguous_source_identifiers.tsv"
    pd.DataFrame(ambiguous_records).to_csv(ambiguous_tsv, sep="\t", index=False)
else:
    ambiguous_tsv = report_dir / f"{OUT_STEM}_ambiguous_source_identifiers.tsv"
    pd.DataFrame(columns=[
        "source_row_index",
        "source_identifier",
        "reason",
        "master_row_indices",
        "master_ensembl_gene_ids",
    ]).to_csv(ambiguous_tsv, sep="\t", index=False)

print(f"Mapped source rows before duplicate collapse: {len(mapped_records):,}")
print(f"Unmapped source rows: {len(unmapped_records):,}")
print(f"Ambiguous source rows: {len(ambiguous_records):,}")

master_to_source = defaultdict(list)
for rec in mapped_records:
    master_to_source[rec["master_row_index"]].append(rec)

final_master_indices = sorted(master_to_source.keys())

X_final = []
gene_rows = []
duplicate_rows = []

for master_i in final_master_indices:
    source_items = master_to_source[master_i]
    source_matrix_rows = [r["source_matrix_row"] for r in source_items]
    source_row_indices = [r["source_row_index"] for r in source_items]
    source_keys = [r["source_identifier"] for r in source_items]
    methods = sorted(set(r["mapping_method"] for r in source_items))

    row = master.loc[master_i]
    X_final.append(X_source[source_matrix_rows].mean(axis=0).astype(np.float32))

    if len(source_matrix_rows) > 1:
        duplicate_rows.append({
            "ensembl_gene_id": row["ensembl_gene_id"],
            "gene_symbol": row.get(symbol_col, ""),
            "source_key_count": len(source_matrix_rows),
            "source_keys": ";".join(source_keys),
            "source_row_indices": ";".join(map(str, source_row_indices)),
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
        "source_key_count": len(source_matrix_rows),
        "source_keys_examples": ";".join(source_keys[:10]),
        "source_row_indices_examples": ";".join(map(str, source_row_indices[:10])),
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
pd.DataFrame(duplicate_rows).to_csv(duplicates_tsv, sep="\t", index=False)

mapping_method_counts = genes_df["mapping_methods"].value_counts().to_dict()

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "source_embedding": SOURCE_EMBEDDING,
    "modality": MODALITY,
    "source_identifier_type": SOURCE_IDENTIFIER_TYPE,
    "download_source": "Official scPRINT HuggingFace repository jkobject/scPRINT",
    "original_model_or_method": "scPRINT",
    "embedding_generated_by": "Extracted locally from official scPRINT medium-v1.5 checkpoint",
    "fixed_embedding_definition": "Rows of state_dict['gene_encoder.embedding.weight'] matched to hyper_parameters['genes']; only human Ensembl gene IDs starting with ENSG were retained.",
    "source_files": {
        "raw_dir": str(raw_dir),
        "checkpoint": str(checkpoint_path),
        "weight_key": WEIGHT_KEY,
        "gene_order_source": "checkpoint hyper_parameters['genes']",
    },
    "checkpoint_metadata": {
        "d_model": hp.get("d_model", ""),
        "nhead": hp.get("nhead", ""),
        "nlayers": hp.get("nlayers", ""),
        "organisms": hp.get("organisms", ""),
        "precpt_gene_emb": hp.get("precpt_gene_emb", ""),
        "gene_pos_enc": "None" if hp.get("gene_pos_enc", None) is None else "present",
    },
    "sequence_and_reference_provenance": {
        "input_sequence_type": "not sequence-derived during local processing",
        "input_sequence_source": "not directly applicable; fixed gene-token embedding from checkpoint",
        "input_sequence_release": "not applicable",
        "species": "human rows only, selected by ENSG prefix from mixed human/mouse checkpoint vocabulary",
        "gene_identifier_source": "checkpoint hyper_parameters['genes']",
        "model_checkpoint": "medium-v1.5.ckpt",
        "pooling_strategy": "fixed checkpoint embedding table; no cell/context pooling",
        "sequence_reference_status": "not applicable for sequence versioning; checkpoint gene vocabulary recorded",
    },
    "master_table": str(master_path),
    "mapping_strategy": "Exact Ensembl gene ID mapping to master_gene_table_v1_1_enriched.csv. Only unique master mappings are accepted. Duplicate source rows mapping to the same final master gene are averaged.",
    "counts": {
        "checkpoint_gene_list_rows": int(len(genes)),
        "checkpoint_embedding_rows": int(len(genes)),
        "checkpoint_embedding_dimensions": int(X_source.shape[1]),
        "human_ENSG_source_rows": int(len(source_df)),
        "excluded_nonhuman_or_non_ENSG_rows": int(len(excluded_df)),
        "mapped_source_rows_before_duplicate_collapse": int(len(mapped_records)),
        "unmapped_source_rows": int(len(unmapped_records)),
        "ambiguous_source_rows": int(len(ambiguous_records)),
        "final_mapped_genes": int(X_final.shape[0]),
        "final_dimensions": int(X_final.shape[1]),
        "master_rows": int(len(master)),
        "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
    },
    "coverage": {
        "human_source_row_mapping_fraction": float(len(mapped_records) / len(source_df)) if len(source_df) else 0.0,
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
        "excluded_nonhuman_or_non_ENSG_rows_tsv": str(excluded_nonhuman_tsv),
    },
}

with open(out_meta, "w") as f:
    json.dump(metadata, f, indent=2)

with open(report_txt, "w") as f:
    f.write(f"{EMBEDDING_NAME} mapping report\n")
    f.write("=" * (len(EMBEDDING_NAME) + 15) + "\n\n")
    f.write(f"Raw directory:      {raw_dir}\n")
    f.write(f"Checkpoint:         {checkpoint_path}\n")
    f.write(f"Weight key:         {WEIGHT_KEY}\n")
    f.write("Gene order source:  checkpoint hyper_parameters['genes']\n")
    f.write(f"Master table:       {master_path}\n\n")

    f.write("scPRINT provenance note:\n")
    f.write("The processed vector is the fixed learned gene-token embedding table from the scPRINT medium-v1.5 checkpoint.\n")
    f.write("Rows are matched using checkpoint hyper_parameters['genes']; only human Ensembl gene IDs starting with ENSG are retained.\n")
    f.write("This is not a context-dependent cell-specific output.\n\n")

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
