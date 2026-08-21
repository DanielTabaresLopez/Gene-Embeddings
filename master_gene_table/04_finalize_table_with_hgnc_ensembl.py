#!/usr/bin/env python3
"""
Finalize the master gene table by adding HGNC annotations, Ensembl
GRCh38 and GRCh37 transcript/protein identifiers, and consolidated
identifier-mapping fields.

Input:
    ~/metadata/master_gene_table_v1_crosschecked.csv

Output:
    ~/metadata/master_gene_table_v1_1_enriched.csv
"""

from pathlib import Path
from collections import defaultdict
import gzip
import json
import re

import pandas as pd


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s in {"", "nan", "NaN", "None", "NONE"}:
        return ""
    return s


def strip_version(x):
    s = clean(x)
    if not s:
        return ""
    return s.split(".")[0]


def norm_symbol(x):
    s = clean(x).upper()
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
    if not s:
        return []
    out = []
    for part in str(s).split("|"):
        part = clean(part)
        if part:
            out.append(part)
    return out


def split_symbol_pipe(x):
    return [norm_symbol(v) for v in split_pipe(x) if norm_symbol(v)]


def join_unique(values):
    vals = []
    seen = set()
    for v in values:
        v = clean(v)
        if not v:
            continue
        if v not in seen:
            vals.append(v)
            seen.add(v)
    return "|".join(sorted(vals))


def join_unique_symbols(values):
    return join_unique([norm_symbol(v) for v in values if norm_symbol(v)])


def parse_gtf_attrs(attr_text):
    return dict(re.findall(r'(\S+) "([^"]*)"', attr_text))


def versioned_id(base, version):
    base = clean(base)
    version = clean(version)
    if not base:
        return ""
    if version:
        return f"{base}.{version}"
    return base


def unique_index_map(df, col, normalizer=lambda x: clean(x)):
    d = defaultdict(set)
    for idx, value in df[col].items():
        key = normalizer(value)
        if key:
            d[key].add(idx)

    unique = {}
    ambiguous = set()
    for key, idxs in d.items():
        if len(idxs) == 1:
            unique[key] = next(iter(idxs))
        else:
            ambiguous.add(key)

    return unique, ambiguous


def parse_gtf(gtf_path, master_gene_map):
    transcripts = defaultdict(set)
    transcripts_versioned = defaultdict(set)
    proteins = defaultdict(set)
    proteins_versioned = defaultdict(set)
    gene_names = defaultdict(set)

    n_lines = 0
    n_feature_lines = 0
    n_mapped_gene_lines = 0
    n_transcript_ids = 0
    n_protein_ids = 0

    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            n_lines += 1
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            n_feature_lines += 1
            attrs = parse_gtf_attrs(parts[8])

            gene_id = strip_version(attrs.get("gene_id", ""))
            if not gene_id or gene_id not in master_gene_map:
                continue

            idx = master_gene_map[gene_id]
            n_mapped_gene_lines += 1

            gene_name = norm_symbol(attrs.get("gene_name", ""))
            if gene_name:
                gene_names[idx].add(gene_name)

            transcript_id = strip_version(attrs.get("transcript_id", ""))
            transcript_ver = clean(attrs.get("transcript_version", ""))
            if transcript_id:
                transcripts[idx].add(transcript_id)
                transcripts_versioned[idx].add(versioned_id(transcript_id, transcript_ver))
                n_transcript_ids += 1

            protein_id = strip_version(attrs.get("protein_id", ""))
            protein_ver = clean(attrs.get("protein_version", ""))
            if protein_id:
                proteins[idx].add(protein_id)
                proteins_versioned[idx].add(versioned_id(protein_id, protein_ver))
                n_protein_ids += 1

    stats = {
        "gtf_path": str(gtf_path),
        "n_lines": n_lines,
        "n_feature_lines": n_feature_lines,
        "n_mapped_gene_lines": n_mapped_gene_lines,
        "n_transcript_id_observations": n_transcript_ids,
        "n_protein_id_observations": n_protein_ids,
        "n_master_genes_with_transcripts": len(transcripts),
        "n_master_genes_with_proteins": len(proteins),
        "n_master_genes_with_gene_names": len(gene_names),
    }

    return {
        "transcripts": transcripts,
        "transcripts_versioned": transcripts_versioned,
        "proteins": proteins,
        "proteins_versioned": proteins_versioned,
        "gene_names": gene_names,
        "stats": stats,
    }


home = Path.home()

master_path = home / "metadata/master_gene_table_v1_crosschecked.csv"
ref_dir = home / "metadata/reference"

hgnc_path = ref_dir / "hgnc_complete_set.txt"
gtf_current_path = ref_dir / "Homo_sapiens.GRCh38.current.gtf.gz"
gtf_grch37_path = ref_dir / "Homo_sapiens.GRCh37.87.gtf.gz"

out_path = home / "metadata/master_gene_table_v1_1_enriched.csv"
report_path = home / "metadata/reports/master_gene_table_v1_1_enrichment_report.txt"
ambiguous_path = home / "metadata/reports/master_gene_table_v1_1_ambiguous_identifiers.tsv"
hgnc_conflicts_path = home / "metadata/reports/master_gene_table_v1_1_hgnc_match_conflicts.tsv"

for p in [master_path, hgnc_path, gtf_current_path, gtf_grch37_path]:
    assert p.exists(), f"Missing file: {p}"

print("Loading master table...")
master = pd.read_csv(master_path, dtype=str).fillna("")

original_columns = list(master.columns)
n_original_rows = len(master)

print(f"Master rows: {n_original_rows:,}")

# Normalized helper columns for internal matching.
master["_ensembl_gene_id_norm"] = master["ensembl_gene_id"].map(strip_version)
master["_gene_symbol_norm"] = master["gene_symbol"].map(norm_symbol)
master["_entrez_id_norm"] = master["entrez_id"].map(norm_entrez)

master_gene_map, ambiguous_gene_ids = unique_index_map(
    master, "_ensembl_gene_id_norm", lambda x: clean(x)
)
master_symbol_map, ambiguous_symbols = unique_index_map(
    master, "_gene_symbol_norm", lambda x: clean(x)
)
master_entrez_map, ambiguous_entrez = unique_index_map(
    master, "_entrez_id_norm", lambda x: clean(x)
)

# UniProt map from all master UniProt-like columns we already have.
uniprot_source_cols = [
    c for c in ["uniprot_id", "uniprot_id_hgnc", "uniprot_id_biomart"]
    if c in master.columns
]

uniprot_to_indices = defaultdict(set)
for idx, row in master.iterrows():
    for col in uniprot_source_cols:
        for u in split_pipe(row[col]):
            uniprot_to_indices[clean(u)].add(idx)

master_uniprot_map = {}
ambiguous_uniprot = set()
for u, idxs in uniprot_to_indices.items():
    if len(idxs) == 1:
        master_uniprot_map[u] = next(iter(idxs))
    elif len(idxs) > 1:
        ambiguous_uniprot.add(u)

# Add new enrichment columns.
new_cols = [
    "hgnc_id",
    "hgnc_approved_symbol",
    "hgnc_name",
    "hgnc_status",
    "hgnc_locus_group",
    "hgnc_locus_type",
    "hgnc_previous_symbols",
    "hgnc_alias_symbols",
    "hgnc_previous_names",
    "hgnc_alias_names",
    "hgnc_gene_group",
    "hgnc_gene_group_id",
    "hgnc_ensembl_gene_id",
    "hgnc_entrez_id",
    "hgnc_uniprot_ids",
    "hgnc_refseq_accessions",
    "hgnc_ccds_ids",
    "hgnc_ucsc_id",
    "hgnc_mane_select",
    "hgnc_omim_ids",
    "hgnc_orphanet_ids",
    "hgnc_rnacentral_ids",
    "all_ensembl_transcript_ids_grch38_current",
    "all_ensembl_transcript_ids_grch38_current_versioned",
    "all_ensembl_protein_ids_grch38_current",
    "all_ensembl_protein_ids_grch38_current_versioned",
    "gene_names_grch38_current",
    "all_ensembl_transcript_ids_grch37",
    "all_ensembl_transcript_ids_grch37_versioned",
    "all_ensembl_protein_ids_grch37",
    "all_ensembl_protein_ids_grch37_versioned",
    "gene_names_grch37",
    "map_symbols_all",
    "map_ensembl_gene_ids_all",
    "map_entrez_ids_all",
    "map_uniprot_ids_all",
    "map_refseq_ids_all",
    "map_ccds_ids_all",
    "map_ensembl_transcript_ids_all",
    "map_ensembl_transcript_ids_all_versioned",
    "map_ensembl_protein_ids_all",
    "map_ensembl_protein_ids_all_versioned",
    "map_string_protein_ids_all",
    "identifier_enrichment_sources",
]

for col in new_cols:
    master[col] = ""

print("Loading HGNC...")
hgnc = pd.read_csv(hgnc_path, sep="\t", dtype=str).fillna("")

hgnc_conflicts = []
hgnc_matched = 0

def hgnc_get(row, col):
    return row[col] if col in row.index else ""

for _, row in hgnc.iterrows():
    candidates = set()

    ens = strip_version(hgnc_get(row, "ensembl_gene_id"))
    if ens and ens in master_gene_map:
        candidates.add(master_gene_map[ens])

    sym = norm_symbol(hgnc_get(row, "symbol"))
    if sym and sym in master_symbol_map:
        candidates.add(master_symbol_map[sym])

    entrez = norm_entrez(hgnc_get(row, "entrez_id"))
    if entrez and entrez in master_entrez_map:
        candidates.add(master_entrez_map[entrez])

    for u in split_pipe(hgnc_get(row, "uniprot_ids")):
        u = clean(u)
        if u and u in master_uniprot_map:
            candidates.add(master_uniprot_map[u])

    if len(candidates) != 1:
        if len(candidates) > 1:
            hgnc_conflicts.append({
                "hgnc_id": hgnc_get(row, "hgnc_id"),
                "symbol": hgnc_get(row, "symbol"),
                "ensembl_gene_id": hgnc_get(row, "ensembl_gene_id"),
                "entrez_id": hgnc_get(row, "entrez_id"),
                "uniprot_ids": hgnc_get(row, "uniprot_ids"),
                "candidate_master_indices": "|".join(map(str, sorted(candidates))),
                "candidate_master_genes": "|".join(master.loc[sorted(candidates), "ensembl_gene_id"].astype(str)),
            })
        continue

    idx = next(iter(candidates))
    hgnc_matched += 1

    master.at[idx, "hgnc_id"] = hgnc_get(row, "hgnc_id")
    master.at[idx, "hgnc_approved_symbol"] = hgnc_get(row, "symbol")
    master.at[idx, "hgnc_name"] = hgnc_get(row, "name")
    master.at[idx, "hgnc_status"] = hgnc_get(row, "status")
    master.at[idx, "hgnc_locus_group"] = hgnc_get(row, "locus_group")
    master.at[idx, "hgnc_locus_type"] = hgnc_get(row, "locus_type")
    master.at[idx, "hgnc_previous_symbols"] = join_unique_symbols(split_symbol_pipe(hgnc_get(row, "prev_symbol")))
    master.at[idx, "hgnc_alias_symbols"] = join_unique_symbols(split_symbol_pipe(hgnc_get(row, "alias_symbol")))
    master.at[idx, "hgnc_previous_names"] = join_unique(split_pipe(hgnc_get(row, "prev_name")))
    master.at[idx, "hgnc_alias_names"] = join_unique(split_pipe(hgnc_get(row, "alias_name")))
    master.at[idx, "hgnc_gene_group"] = hgnc_get(row, "gene_group")
    master.at[idx, "hgnc_gene_group_id"] = hgnc_get(row, "gene_group_id")
    master.at[idx, "hgnc_ensembl_gene_id"] = strip_version(hgnc_get(row, "ensembl_gene_id"))
    master.at[idx, "hgnc_entrez_id"] = norm_entrez(hgnc_get(row, "entrez_id"))
    master.at[idx, "hgnc_uniprot_ids"] = join_unique(split_pipe(hgnc_get(row, "uniprot_ids")))
    master.at[idx, "hgnc_refseq_accessions"] = join_unique(split_pipe(hgnc_get(row, "refseq_accession")))
    master.at[idx, "hgnc_ccds_ids"] = join_unique(split_pipe(hgnc_get(row, "ccds_id")))
    master.at[idx, "hgnc_ucsc_id"] = hgnc_get(row, "ucsc_id")
    master.at[idx, "hgnc_mane_select"] = hgnc_get(row, "mane_select")
    master.at[idx, "hgnc_omim_ids"] = join_unique(split_pipe(hgnc_get(row, "omim_id")))
    master.at[idx, "hgnc_orphanet_ids"] = join_unique(split_pipe(hgnc_get(row, "orphanet")))
    master.at[idx, "hgnc_rnacentral_ids"] = join_unique(split_pipe(hgnc_get(row, "rna_central_ids")))

print(f"HGNC rows matched to master: {hgnc_matched:,}")

print("Parsing Ensembl GRCh38 current GTF...")
gtf_current = parse_gtf(gtf_current_path, master_gene_map)

print("Parsing Ensembl GRCh37 GTF...")
gtf_grch37 = parse_gtf(gtf_grch37_path, master_gene_map)

for idx in master.index:
    # GTF-derived IDs.
    master.at[idx, "all_ensembl_transcript_ids_grch38_current"] = join_unique(gtf_current["transcripts"].get(idx, []))
    master.at[idx, "all_ensembl_transcript_ids_grch38_current_versioned"] = join_unique(gtf_current["transcripts_versioned"].get(idx, []))
    master.at[idx, "all_ensembl_protein_ids_grch38_current"] = join_unique(gtf_current["proteins"].get(idx, []))
    master.at[idx, "all_ensembl_protein_ids_grch38_current_versioned"] = join_unique(gtf_current["proteins_versioned"].get(idx, []))
    master.at[idx, "gene_names_grch38_current"] = join_unique_symbols(gtf_current["gene_names"].get(idx, []))

    master.at[idx, "all_ensembl_transcript_ids_grch37"] = join_unique(gtf_grch37["transcripts"].get(idx, []))
    master.at[idx, "all_ensembl_transcript_ids_grch37_versioned"] = join_unique(gtf_grch37["transcripts_versioned"].get(idx, []))
    master.at[idx, "all_ensembl_protein_ids_grch37"] = join_unique(gtf_grch37["proteins"].get(idx, []))
    master.at[idx, "all_ensembl_protein_ids_grch37_versioned"] = join_unique(gtf_grch37["proteins_versioned"].get(idx, []))
    master.at[idx, "gene_names_grch37"] = join_unique_symbols(gtf_grch37["gene_names"].get(idx, []))

    # Combined mapping columns for future scripts.
    symbols_all = []
    symbols_all.append(master.at[idx, "gene_symbol"])
    symbols_all.append(master.at[idx, "hgnc_approved_symbol"])
    symbols_all += split_symbol_pipe(master.at[idx, "hgnc_previous_symbols"])
    symbols_all += split_symbol_pipe(master.at[idx, "hgnc_alias_symbols"])
    symbols_all += split_symbol_pipe(master.at[idx, "gene_names_grch38_current"])
    symbols_all += split_symbol_pipe(master.at[idx, "gene_names_grch37"])
    master.at[idx, "map_symbols_all"] = join_unique_symbols(symbols_all)

    ensembl_gene_ids = [
        strip_version(master.at[idx, "ensembl_gene_id"]),
        strip_version(master.at[idx, "ensembl_gene_id_versioned"]),
        strip_version(master.at[idx, "hgnc_ensembl_gene_id"]),
    ]
    master.at[idx, "map_ensembl_gene_ids_all"] = join_unique(ensembl_gene_ids)

    entrez_ids = [
        norm_entrez(master.at[idx, "entrez_id"]),
        norm_entrez(master.at[idx, "entrez_id_hgnc"]) if "entrez_id_hgnc" in master.columns else "",
        norm_entrez(master.at[idx, "entrez_id_biomart"]) if "entrez_id_biomart" in master.columns else "",
        norm_entrez(master.at[idx, "hgnc_entrez_id"]),
    ]
    master.at[idx, "map_entrez_ids_all"] = join_unique(entrez_ids)

    uniprot_ids = []
    for col in ["uniprot_id", "uniprot_id_hgnc", "uniprot_id_biomart", "hgnc_uniprot_ids"]:
        if col in master.columns:
            uniprot_ids += split_pipe(master.at[idx, col])
    master.at[idx, "map_uniprot_ids_all"] = join_unique(uniprot_ids)

    refseq_ids = []
    refseq_ids += split_pipe(master.at[idx, "hgnc_refseq_accessions"])
    # MANE can contain RefSeq/Ensembl IDs separated in HGNC; keep whole tokens.
    refseq_ids += [v for v in re.split(r'[|,;\s]+', clean(master.at[idx, "hgnc_mane_select"])) if v.startswith(("NM_", "NR_", "XM_", "XR_"))]
    master.at[idx, "map_refseq_ids_all"] = join_unique(refseq_ids)

    ccds_ids = []
    if "ccdsid" in master.columns:
        ccds_ids += split_pipe(master.at[idx, "ccdsid"])
    ccds_ids += split_pipe(master.at[idx, "hgnc_ccds_ids"])
    master.at[idx, "map_ccds_ids_all"] = join_unique(ccds_ids)

    transcript_ids = []
    transcript_ids += [strip_version(master.at[idx, "canonical_transcript_id"])]
    transcript_ids += split_pipe(master.at[idx, "all_ensembl_transcript_ids_grch38_current"])
    transcript_ids += split_pipe(master.at[idx, "all_ensembl_transcript_ids_grch37"])
    transcript_ids += [strip_version(v) for v in re.split(r'[|,;\s]+', clean(master.at[idx, "hgnc_mane_select"])) if v.startswith("ENST")]
    master.at[idx, "map_ensembl_transcript_ids_all"] = join_unique(transcript_ids)

    transcript_ids_versioned = []
    transcript_ids_versioned += [clean(master.at[idx, "canonical_transcript_id_versioned"])]
    transcript_ids_versioned += split_pipe(master.at[idx, "all_ensembl_transcript_ids_grch38_current_versioned"])
    transcript_ids_versioned += split_pipe(master.at[idx, "all_ensembl_transcript_ids_grch37_versioned"])
    master.at[idx, "map_ensembl_transcript_ids_all_versioned"] = join_unique(transcript_ids_versioned)

    protein_ids = []
    protein_ids += [strip_version(master.at[idx, "canonical_protein_id"])]
    protein_ids += split_pipe(master.at[idx, "all_ensembl_protein_ids_grch38_current"])
    protein_ids += split_pipe(master.at[idx, "all_ensembl_protein_ids_grch37"])
    master.at[idx, "map_ensembl_protein_ids_all"] = join_unique(protein_ids)

    protein_ids_versioned = []
    protein_ids_versioned += [clean(master.at[idx, "canonical_protein_id_versioned"])]
    protein_ids_versioned += split_pipe(master.at[idx, "all_ensembl_protein_ids_grch38_current_versioned"])
    protein_ids_versioned += split_pipe(master.at[idx, "all_ensembl_protein_ids_grch37_versioned"])
    master.at[idx, "map_ensembl_protein_ids_all_versioned"] = join_unique(protein_ids_versioned)

    string_ids = []
    for p in split_pipe(master.at[idx, "map_ensembl_protein_ids_all"]):
        if p.startswith("ENSP"):
            string_ids.append(f"9606.{p}")
    master.at[idx, "map_string_protein_ids_all"] = join_unique(string_ids)

    sources = []
    if master.at[idx, "hgnc_id"]:
        sources.append("HGNC_complete_set")
    if master.at[idx, "all_ensembl_transcript_ids_grch38_current"] or master.at[idx, "all_ensembl_protein_ids_grch38_current"]:
        sources.append("Ensembl_GRCh38_current_GTF")
    if master.at[idx, "all_ensembl_transcript_ids_grch37"] or master.at[idx, "all_ensembl_protein_ids_grch37"]:
        sources.append("Ensembl_GRCh37_87_GTF")
    master.at[idx, "identifier_enrichment_sources"] = "|".join(sources)

# Ambiguity report for mapping columns.
mapping_cols = [
    "map_symbols_all",
    "map_ensembl_gene_ids_all",
    "map_entrez_ids_all",
    "map_uniprot_ids_all",
    "map_refseq_ids_all",
    "map_ccds_ids_all",
    "map_ensembl_transcript_ids_all",
    "map_ensembl_protein_ids_all",
    "map_string_protein_ids_all",
]

ambiguous_rows = []
for col in mapping_cols:
    id_to_genes = defaultdict(set)
    for _, row in master.iterrows():
        gene = row["ensembl_gene_id"]
        for identifier in split_pipe(row[col]):
            id_to_genes[identifier].add(gene)

    for identifier, genes in id_to_genes.items():
        if len(genes) > 1:
            ambiguous_rows.append({
                "mapping_column": col,
                "identifier": identifier,
                "n_master_genes": len(genes),
                "ensembl_gene_ids": "|".join(sorted(genes)),
            })

ambiguous_df = pd.DataFrame(ambiguous_rows)
ambiguous_df.to_csv(ambiguous_path, sep="\t", index=False)

pd.DataFrame(hgnc_conflicts).to_csv(hgnc_conflicts_path, sep="\t", index=False)

# Remove temporary helper columns before saving.
master = master.drop(columns=[c for c in master.columns if c.startswith("_")])

# Preserve original columns first, then new columns.
master = master[original_columns + new_cols]

master.to_csv(out_path, index=False)

report = {
    "input_master": str(master_path),
    "output_master": str(out_path),
    "n_original_rows": int(n_original_rows),
    "n_output_rows": int(len(master)),
    "n_original_columns": int(len(original_columns)),
    "n_added_columns": int(len(new_cols)),
    "n_output_columns": int(master.shape[1]),
    "hgnc_file": str(hgnc_path),
    "n_hgnc_rows": int(len(hgnc)),
    "n_hgnc_rows_matched_to_master": int(hgnc_matched),
    "n_hgnc_match_conflicts": int(len(hgnc_conflicts)),
    "grch38_current_gtf_stats": gtf_current["stats"],
    "grch37_gtf_stats": gtf_grch37["stats"],
    "n_ambiguous_identifier_rows_in_report": int(len(ambiguous_rows)),
    "ambiguous_identifier_report": str(ambiguous_path),
    "hgnc_conflicts_report": str(hgnc_conflicts_path),
    "added_columns": new_cols,
}

with open(report_path, "w") as f:
    f.write("Master gene table v1.1 enrichment report\n")
    f.write("========================================\n\n")
    for k, v in report.items():
        if isinstance(v, (dict, list)):
            f.write(f"{k}: {json.dumps(v, indent=2)}\n")
        else:
            f.write(f"{k}: {v}\n")

print("\nDONE")
print(f"Saved enriched master table: {out_path}")
print(f"Saved report: {report_path}")
print(f"Saved ambiguous identifier report: {ambiguous_path}")
print(f"Saved HGNC conflict report: {hgnc_conflicts_path}")
print("\nSummary:")
for k, v in report.items():
    if k != "added_columns":
        print(f"{k}: {v}")
