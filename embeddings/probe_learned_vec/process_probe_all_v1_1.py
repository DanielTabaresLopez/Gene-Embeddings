from pathlib import Path
import json
import re
import gc
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd


home = Path.home()

MASTER_PATH = home / "metadata/master_gene_table_v1_1_enriched.csv"
RAW_DIR = home / "data/raw_embeddings/probe_official"
PROBE_DIR = RAW_DIR / "human_representation_vectors/revision-1"

OUT_BASE = home / "data/processed_embeddings"
REPORT_DIR = home / "reports/mapping_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Complete official PROBE human representation archive: 20 CSV files.
PROBE_FILES = OrderedDict([
    ("probe_aac_v1_1", "AAC_UNIPROT_HUMAN.csv"),
    ("probe_albert_v1_1", "ALBERT_UNIPROT_HUMAN.csv"),
    ("probe_apaac_v1_1", "APAAC_UNIPROT_HUMAN.csv"),
    ("probe_bert_bfd_v1_1", "BERT-BFD_UNIPROT_HUMAN.csv"),
    ("probe_bert_pfam_v1_1", "BERT-PFAM_UNIPROT_HUMAN.csv"),
    ("probe_blast_v1_1", "BLAST_UNIPROT_HUMAN.csv"),
    ("probe_cpc_prot_v1_1", "CPC-PROT_UNIPROT_HUMAN.csv"),
    ("probe_esmb1_v1_1", "ESMB1_UNIPROT_HUMAN.csv"),
    ("probe_gene2vec_uniprot_v1_1", "GENE2VEC_UNIPROT_HUMAN.csv"),
    ("probe_hmmer_v1_1", "HMMER_UNIPROT_HUMAN.csv"),
    ("probe_ksep_v1_1", "KSEP_UNIPROT_HUMAN.csv"),
    ("probe_learned_vec_v1_1", "LEARNED-VEC_UNIPROT_HUMAN.csv"),
    ("probe_mut2vec_v1_1", "MUT2VEC_UNIPROT_HUMAN.csv"),
    ("probe_pfam_v1_1", "PFAM_UNIPROT_HUMAN.csv"),
    ("probe_protvec_v1_1", "PROTVEC_UNIPROT_HUMAN.csv"),
    ("probe_seqvec_v1_1", "SEQVEC_UNIPROT_HUMAN.csv"),
    ("probe_t5_v1_1", "T5_UNIPROT_HUMAN.csv"),
    ("probe_tcga_embedding_v1_1", "TCGA-EMBEDDING_UNIPROT_HUMAN.csv"),
    ("probe_unirep_v1_1", "UNIREP_UNIPROT_HUMAN.csv"),
    ("probe_xlnet_v1_1", "XLNET_UNIPROT_HUMAN.csv"),
])

SUMMARY_PATH = REPORT_DIR / "probe_all_v1_1_summary.tsv"


def split_values(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return []
    return [t.strip() for t in re.split(r"[|,;\s]+", s) if t.strip()]


def clean_uniprot(x):
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return ""
    # Use canonical accession base for isoform-like IDs.
    return s.split("-")[0].upper()


def first_existing(row, cols):
    for c in cols:
        if c in row.index and str(row[c]).strip() and str(row[c]).lower() != "nan":
            return str(row[c]).strip()
    return ""


def build_uniprot_mapping(master):
    mapping = defaultdict(set)

    candidate_cols = [
        "uniprot_id",
        "hgnc_uniprot_ids",
        "map_uniprot_ids_all",
    ]

    for i, row in master.iterrows():
        for c in candidate_cols:
            if c not in master.columns:
                continue
            for token in split_values(row[c]):
                u = clean_uniprot(token)
                if u:
                    mapping[u].add(i)

    unique = {u: next(iter(rows)) for u, rows in mapping.items() if len(rows) == 1}
    ambiguous = {u: sorted(rows) for u, rows in mapping.items() if len(rows) > 1}
    return unique, ambiguous


def load_probe_csv(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if not header or header[0] != "Entry":
        raise ValueError(f"{path} does not have Entry as first column.")

    dtype = {"Entry": str}
    for c in header[1:]:
        dtype[c] = np.float32

    return pd.read_csv(path, dtype=dtype, low_memory=False)


def process_one(out_stem, filename, master, unique_map, ambiguous_map):
    path = PROBE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing PROBE CSV: {path}")

    out_dir = OUT_BASE / out_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{out_stem}_embeddings.npz"
    out_genes = out_dir / f"{out_stem}_genes.tsv"
    out_meta = out_dir / f"{out_stem}_metadata.json"

    report_txt = REPORT_DIR / f"{out_stem}_mapping_report.txt"
    unmapped_tsv = REPORT_DIR / f"{out_stem}_unmapped_source_identifiers.tsv"
    ambiguous_tsv = REPORT_DIR / f"{out_stem}_ambiguous_source_identifiers.tsv"
    duplicates_tsv = REPORT_DIR / f"{out_stem}_duplicate_final_mappings.tsv"

    print("\n" + "=" * 100)
    print("Processing", filename)

    df = load_probe_csv(path)
    print("Raw shape:", df.shape)

    source_entries = df["Entry"].astype(str).map(clean_uniprot).tolist()
    X_source = df.drop(columns=["Entry"]).to_numpy(dtype=np.float32, copy=True)
    del df
    gc.collect()

    mapped_records = []
    unmapped_records = []
    ambiguous_records = []

    for src_i, entry in enumerate(source_entries):
        if entry in unique_map:
            mapped_records.append({
                "source_row_index": src_i,
                "source_identifier": entry,
                "master_row_index": unique_map[entry],
                "mapping_method": "uniprot_accession_exact_unique",
            })
        elif entry in ambiguous_map:
            rows = ambiguous_map[entry]
            ambiguous_records.append({
                "source_row_index": src_i,
                "source_identifier": entry,
                "reason": "uniprot_accession_maps_to_multiple_master_rows",
                "master_row_indices": ";".join(map(str, rows)),
                "master_ensembl_gene_ids": ";".join(master.loc[rows, "ensembl_gene_id"].astype(str)),
            })
        else:
            unmapped_records.append({
                "source_row_index": src_i,
                "source_identifier": entry,
                "reason": "uniprot_accession_not_in_master_gene_table",
            })

    print("Mapped source rows before duplicate collapse:", len(mapped_records))
    print("Unmapped source rows:", len(unmapped_records))
    print("Ambiguous source rows:", len(ambiguous_records))

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
        methods = sorted(set(r["mapping_method"] for r in source_items))

        row = master.loc[master_i]
        X_final.append(X_source[source_row_indices].mean(axis=0).astype(np.float32))

        if len(source_row_indices) > 1:
            duplicate_rows.append({
                "ensembl_gene_id": row["ensembl_gene_id"],
                "gene_symbol": row.get(symbol_col, ""),
                "source_key_count": len(source_row_indices),
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
            "source_embedding": f"PROBE_{filename.replace('_UNIPROT_HUMAN.csv', '')}",
            "source_identifier_type": "UniProt accession from PROBE Entry column",
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

    mapping_method_counts = genes_df["mapping_methods"].value_counts().to_dict()

    metadata = {
        "embedding_name": out_stem,
        "source_embedding": f"PROBE_{filename.replace('_UNIPROT_HUMAN.csv', '')}",
        "modality": "protein sequence / protein representation from PROBE human UniProt archive",
        "source_identifier_type": "UniProt accession",
        "download_source": "Official PROBE human_representation_vectors.tar.gz archive from Zenodo record 5795850 plus official HUBioDataLab/PROBE repository",
        "original_model_or_method": filename.replace("_UNIPROT_HUMAN.csv", ""),
        "embedding_generated_by": "PROBE authors / original PROBE human representation archive",
        "source_files": {
            "raw_dir": str(RAW_DIR),
            "probe_csv": str(path),
            "probe_repo": str(RAW_DIR / "repo"),
            "archive": str(RAW_DIR / "human_representation_vectors.tar.gz"),
        },
        "sequence_and_reference_provenance": {
            "input_sequence_type": "protein sequence-derived representation or protein feature vector",
            "input_sequence_source": "PROBE human UniProt representation archive",
            "source_identifier": "UniProt accession in Entry column",
            "species": "human",
            "sequence_reference_status": "PROBE archive provenance recorded; exact UniProt release to verify from original PROBE documentation if needed",
        },
        "master_table": str(MASTER_PATH),
        "mapping_strategy": "Exact UniProt accession mapping to master_gene_table_v1_1_enriched.csv using uniprot_id, hgnc_uniprot_ids, and map_uniprot_ids_all. Only unique mappings are accepted. Duplicate source rows mapping to the same final master gene are averaged.",
        "counts": {
            "source_rows": int(len(source_entries)),
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
            "source_row_mapping_fraction": float(len(mapped_records) / len(source_entries)) if source_entries else 0.0,
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
        f.write(f"{out_stem} mapping report\n")
        f.write("=" * (len(out_stem) + 15) + "\n\n")
        f.write(f"PROBE CSV:    {path}\n")
        f.write(f"Master table: {MASTER_PATH}\n\n")

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

    result = {
        "embedding": out_stem,
        "file": filename,
        **metadata["counts"],
        **metadata["coverage"],
    }

    del X_source, X_final, genes_df
    gc.collect()
    return result


def main():
    print("Loading master table...")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    unique_map, ambiguous_map = build_uniprot_mapping(master)

    print("Unique UniProt mappings:", len(unique_map))
    print("Ambiguous UniProt mappings:", len(ambiguous_map))
    print("PROBE files to process:", len(PROBE_FILES))

    summary = []
    for out_stem, filename in PROBE_FILES.items():
        summary.append(process_one(out_stem, filename, master, unique_map, ambiguous_map))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(SUMMARY_PATH, sep="\t", index=False)

    print("\n" + "=" * 100)
    print("Complete PROBE batch finished")
    print(summary_df)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
