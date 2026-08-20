from pathlib import Path
import json
import re
import gc
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd


home = Path.home()

MASTER_PATH = home / "metadata/master_gene_table_v1_1_enriched.csv"
RAW_DIR = home / "data/raw_embeddings/newt_official"
ZENODO_DIR = RAW_DIR / "zenodo"
REPO_DIR = RAW_DIR / "repo"

OUT_BASE = home / "data/processed_embeddings"
REPORT_DIR = home / "reports/mapping_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NEWT_FILES = OrderedDict([
    ("newt_go_256_v1_1", {
        "display_name": "NEWT-GO-256-v1.1",
        "filename": "gene_vec_go_256.csv",
        "component": "Gene Ontology functional annotation embedding",
    }),
    ("newt_archs4_256_v1_1", {
        "display_name": "NEWT-ARCHS4-256-v1.1",
        "filename": "gene_vec_archs4_256.csv",
        "component": "ARCHS4 co-expression embedding",
    }),
    ("newt_go_graph_v1_1", {
        "display_name": "NEWT-GO-graph-v1.1",
        "filename": "learned_gene_embeddings_go_graph.csv",
        "component": "GO graph-derived learned gene embedding",
    }),
    ("newt_msigdb_bundle_v1_1", {
        "display_name": "NEWT-MSigDB-bundle-v1.1",
        "filename": "msigdb_bundle_embeddings_entrez.csv",
        "component": "MSigDB gene-set bundle embedding",
    }),
    ("newt_cellnet_v1_1", {
        "display_name": "NEWT-CellNet-v1.1",
        "filename": "cellnet_filtered_entrez_embeddings.csv",
        "component": "CellNet transcription-factor network embedding",
    }),
    ("newt_dorothea_v1_1", {
        "display_name": "NEWT-DoRothEA-v1.1",
        "filename": "dorothea_embeddings_entrez_embeddings.csv",
        "component": "DoRothEA transcription-factor regulon embedding",
    }),
    ("newt_collectri_v1_1", {
        "display_name": "NEWT-CollecTRI-v1.1",
        "filename": "collectri_embeddings_entrez_embeddings.csv",
        "component": "CollecTRI transcription-factor regulon embedding",
    }),
])


def split_values(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return []
    return [t.strip() for t in re.split(r"[|,;\s]+", s) if t.strip()]


def clean_entrez(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    if s.lower().startswith("inhib_"):
        s = s[6:]
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    if re.fullmatch(r"\d+", s):
        return s
    return ""


def normalize_source_token(x):
    raw = str(x).strip()
    if raw.lower().startswith("inhib_"):
        base = clean_entrez(raw)
        return raw, base, "entrez_id_exact_unique_from_inhibitory_augmented_token"
    base = clean_entrez(raw)
    return raw, base, "entrez_id_exact_unique"


def first_existing(row, cols):
    for c in cols:
        if c in row.index and str(row[c]).strip() and str(row[c]).lower() != "nan":
            return str(row[c]).strip()
    return ""


def build_entrez_mapping(master):
    mapping = defaultdict(set)
    candidate_cols = [
        "entrez_id",
        "hgnc_entrez_id",
        "map_entrez_ids_all",
    ]

    for i, row in master.iterrows():
        for c in candidate_cols:
            if c not in master.columns:
                continue
            for token in split_values(row[c]):
                eid = clean_entrez(token)
                if eid:
                    mapping[eid].add(i)

    unique = {eid: next(iter(rows)) for eid, rows in mapping.items() if len(rows) == 1}
    ambiguous = {eid: sorted(rows) for eid, rows in mapping.items() if len(rows) > 1}

    return unique, ambiguous


def file_has_header(path):
    first_line = path.open().readline().strip()
    first_cell = first_line.split(",", 1)[0].strip().lower()
    return first_cell in {"gene", "token", "entrez", "entrez_id", "gene_id", "id"}


def load_newt_csv(path):
    has_header = file_has_header(path)
    if has_header:
        df = pd.read_csv(path, low_memory=False)
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "source_token"})
    else:
        df = pd.read_csv(path, header=None, low_memory=False)
        df.columns = ["source_token"] + [f"v{i}" for i in range(df.shape[1] - 1)]

    source_tokens = df["source_token"].astype(str).tolist()
    X = df.drop(columns=["source_token"]).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    bad_rows = np.where(~np.isfinite(X).all(axis=1))[0]
    if len(bad_rows):
        raise ValueError(f"{path} contains non-finite numeric values in rows: {bad_rows[:20]}")

    return source_tokens, X, has_header


def process_one(out_stem, info, master, unique_map, ambiguous_map):
    display_name = info["display_name"]
    filename = info["filename"]
    component = info["component"]

    path = ZENODO_DIR / filename
    out_dir = OUT_BASE / out_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{out_stem}_embeddings.npz"
    out_genes = out_dir / f"{out_stem}_genes.tsv"
    out_meta = out_dir / f"{out_stem}_metadata.json"

    report_txt = REPORT_DIR / f"{out_stem}_mapping_report.txt"
    unmapped_tsv = REPORT_DIR / f"{out_stem}_unmapped_source_identifiers.tsv"
    ambiguous_tsv = REPORT_DIR / f"{out_stem}_ambiguous_source_identifiers.tsv"
    duplicates_tsv = REPORT_DIR / f"{out_stem}_duplicate_final_mappings.tsv"
    excluded_tsv = REPORT_DIR / f"{out_stem}_excluded_non_entrez_source_tokens.tsv"

    print("\n" + "=" * 100)
    print("Processing:", display_name)
    print("File:", path)

    source_tokens, X_source, has_header = load_newt_csv(path)

    print("CSV had header:", has_header)
    print("Source rows:", len(source_tokens))
    print("Source dimensions:", X_source.shape[1])

    mapped_records = []
    unmapped_records = []
    ambiguous_records = []
    excluded_records = []

    for src_i, token in enumerate(source_tokens):
        raw_token, eid, method = normalize_source_token(token)

        if not eid:
            excluded_records.append({
                "source_row_index": src_i,
                "source_identifier": raw_token,
                "reason": "not_parseable_as_entrez_id",
            })
            continue

        if eid in unique_map:
            mapped_records.append({
                "source_row_index": src_i,
                "source_identifier": raw_token,
                "normalized_entrez_id": eid,
                "master_row_index": unique_map[eid],
                "mapping_method": method,
            })
        elif eid in ambiguous_map:
            rows = ambiguous_map[eid]
            ambiguous_records.append({
                "source_row_index": src_i,
                "source_identifier": raw_token,
                "normalized_entrez_id": eid,
                "reason": "entrez_id_maps_to_multiple_master_rows",
                "master_row_indices": ";".join(map(str, rows)),
                "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
            })
        else:
            unmapped_records.append({
                "source_row_index": src_i,
                "source_identifier": raw_token,
                "normalized_entrez_id": eid,
                "reason": "entrez_id_not_in_master_gene_table",
            })

    print("Mapped source rows before duplicate collapse:", len(mapped_records))
    print("Unmapped source rows:", len(unmapped_records))
    print("Ambiguous source rows:", len(ambiguous_records))
    print("Excluded non-Entrez tokens:", len(excluded_records))

    master_to_source = defaultdict(list)
    for rec in mapped_records:
        master_to_source[rec["master_row_index"]].append(rec)

    symbol_col = "gene_symbol" if "gene_symbol" in master.columns else "hgnc_approved_symbol"

    final_master_indices = sorted(master_to_source.keys())
    X_final = []
    gene_rows = []
    duplicate_rows = []

    for master_i in final_master_indices:
        source_items = master_to_source[master_i]
        source_row_indices = [r["source_row_index"] for r in source_items]
        source_keys = [r["source_identifier"] for r in source_items]
        normalized_ids = [r["normalized_entrez_id"] for r in source_items]
        methods = sorted(set(r["mapping_method"] for r in source_items))

        row = master.loc[master_i]
        X_final.append(X_source[source_row_indices].mean(axis=0).astype(np.float32))

        if len(source_row_indices) > 1:
            duplicate_rows.append({
                "ensembl_gene_id": row["ensembl_gene_id"],
                "gene_symbol": row.get(symbol_col, ""),
                "source_key_count": len(source_row_indices),
                "source_keys": ";".join(source_keys),
                "normalized_entrez_ids": ";".join(normalized_ids),
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
            "source_embedding": display_name,
            "source_identifier_type": "Entrez Gene ID from NEWT component embedding CSV",
            "source_key_count": len(source_row_indices),
            "source_keys_examples": ";".join(source_keys[:10]),
            "source_row_indices_examples": ";".join(map(str, source_row_indices[:10])),
            "mapping_methods": ";".join(methods),
        })

    X_final = np.vstack(X_final).astype(np.float32)
    genes_df = pd.DataFrame(gene_rows)

    print("Final mapped genes:", X_final.shape[0])
    print("Final dimensions:", X_final.shape[1])

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
    pd.DataFrame(excluded_records).to_csv(excluded_tsv, sep="\t", index=False)

    mapping_method_counts = genes_df["mapping_methods"].value_counts().to_dict()

    metadata = {
        "embedding_name": display_name,
        "source_embedding": display_name,
        "modality": "multimodal functional/regulatory/co-expression/pathway gene embedding component from NEWT",
        "newt_component": component,
        "source_identifier_type": "Entrez Gene ID",
        "download_source": "Official NEWT Zenodo dataset release, DOI 10.5281/zenodo.19340318",
        "original_model_or_method": "NEWT component embedding",
        "embedding_generated_by": "NEWT authors; processed locally into the project standard format",
        "source_files": {
            "raw_dir": str(RAW_DIR),
            "zenodo_dir": str(ZENODO_DIR),
            "repo_dir": str(REPO_DIR),
            "source_csv": str(path),
        },
        "reference_note": "The source CSVs use Entrez identifiers. Most files are headerless; this script detects headers and preserves the first data row.",
        "sequence_and_reference_provenance": {
            "input_sequence_type": "not sequence-derived during local processing",
            "input_sequence_source": "not applicable",
            "input_sequence_release": "not applicable",
            "species": "human",
            "gene_identifier_source": "Entrez IDs in NEWT component CSV",
            "model_checkpoint": "not applicable",
            "pooling_strategy": "precomputed component embedding table",
            "sequence_reference_status": "not applicable for sequence versioning; source component provenance recorded",
        },
        "master_table": str(MASTER_PATH),
        "mapping_strategy": "Exact Entrez ID mapping to master_gene_table_v1_1_enriched.csv using entrez_id, hgnc_entrez_id, and map_entrez_ids_all. Only unique mappings are accepted. Duplicate source rows mapping to the same final master gene are averaged.",
        "counts": {
            "source_rows": int(len(source_tokens)),
            "source_dimensions": int(X_source.shape[1]),
            "mapped_source_rows_before_duplicate_collapse": int(len(mapped_records)),
            "unmapped_source_rows": int(len(unmapped_records)),
            "ambiguous_source_rows": int(len(ambiguous_records)),
            "excluded_non_entrez_source_tokens": int(len(excluded_records)),
            "final_mapped_genes": int(X_final.shape[0]),
            "final_dimensions": int(X_final.shape[1]),
            "master_rows": int(len(master)),
            "duplicate_final_mappings_averaged": int(len(duplicate_rows)),
        },
        "coverage": {
            "source_row_mapping_fraction": float(len(mapped_records) / len(source_tokens)) if source_tokens else 0.0,
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
            "excluded_non_entrez_tsv": str(excluded_tsv),
        },
    }

    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(report_txt, "w") as f:
        f.write(f"{display_name} mapping report\n")
        f.write("=" * (len(display_name) + 15) + "\n\n")
        f.write(f"NEWT component: {component}\n")
        f.write(f"Source CSV:     {path}\n")
        f.write(f"Master table:   {MASTER_PATH}\n")
        f.write(f"CSV had header: {has_header}\n\n")

        f.write("NEWT provenance note:\n")
        f.write("This vector is a precomputed NEWT component embedding from the official NEWT Zenodo dataset release.\n")
        f.write("The source identifiers are Entrez Gene IDs. Headerless CSVs were read with header=None to preserve the first source row.\n")
        f.write("This is a component embedding, not a fused NEWT model output.\n\n")

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

    del X_source, X_final, genes_df
    gc.collect()

    return {
        "embedding": out_stem,
        "display_name": display_name,
        "file": filename,
        **metadata["counts"],
        **metadata["coverage"],
    }


print("Loading master table...")
master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
unique_map, ambiguous_map = build_entrez_mapping(master)

print("Unique Entrez mappings:", len(unique_map))
print("Ambiguous Entrez mappings:", len(ambiguous_map))

summary = []
for out_stem, info in NEWT_FILES.items():
    summary.append(process_one(out_stem, info, master, unique_map, ambiguous_map))

summary_df = pd.DataFrame(summary)
summary_path = REPORT_DIR / "newt_components_v1_1_summary.tsv"
summary_df.to_csv(summary_path, sep="\t", index=False)

print("\n" + "=" * 100)
print("NEWT component batch complete")
print(summary_df)
print("Summary:", summary_path)
