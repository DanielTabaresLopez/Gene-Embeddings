from pathlib import Path
import gzip
import re
import pandas as pd
from Bio import SeqIO

home = Path.home()

master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"

raw_dir = home / "data/raw_embeddings/helix_mrna_full_transcript_v1_1"
seq_dir = raw_dir / "source_sequences"
cdna_fasta_path = seq_dir / "Homo_sapiens.GRCh38.cdna.all.fa.gz"

# Reuse the already verified canonical CDS sequences from NT-v2 preparation.
nt_cds_fasta_path = home / "data/raw_embeddings/nt_v2_500m_multispecies_cds_v1_1/source_sequences/nt_v2_canonical_cds_sequences.fa"

out_report = home / "reports/mapping_reports/helix_mrna_full_transcript_v1_1_sequence_mapping_report.tsv"
out_helix_fasta = seq_dir / "helix_mrna_canonical_full_transcript_sequences.fa"
out_raw_rna_fasta = seq_dir / "helix_mrna_canonical_full_transcript_raw_rna_sequences.fa"
out_mapped_tsv = seq_dir / "helix_mrna_canonical_full_transcript_sequences.tsv"

def clean_dna(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return re.sub(r"[^ACGTN]", "N", seq)

def dna_to_rna(seq: str) -> str:
    return clean_dna(seq).replace("T", "U")

def add_codon_markers_to_cds(cds_rna: str) -> str:
    parts = []
    for i in range(0, len(cds_rna), 3):
        codon = cds_rna[i:i+3]
        if codon:
            parts.append("E" + codon)
    return "".join(parts)

print("Loading master table:", master_path)
master = pd.read_csv(master_path, dtype=str).fillna("")
print("Master rows:", len(master))

required_cols = [
    "ensembl_gene_id",
    "gene_symbol",
    "canonical_transcript_id",
    "canonical_transcript_id_versioned",
]
missing_cols = [c for c in required_cols if c not in master.columns]
if missing_cols:
    raise ValueError(f"Missing required master-table columns: {missing_cols}")

print("Indexing Ensembl cDNA FASTA:", cdna_fasta_path)

cdna_by_versioned = {}
cdna_by_unversioned = {}

with gzip.open(cdna_fasta_path, "rt") as handle:
    for rec in SeqIO.parse(handle, "fasta"):
        tid_versioned = rec.id.split()[0]
        tid_unversioned = tid_versioned.split(".")[0]
        seq = clean_dna(str(rec.seq))

        cdna_by_versioned.setdefault(tid_versioned, seq)
        cdna_by_unversioned.setdefault(tid_unversioned, seq)

print("cDNA transcript IDs versioned:", len(cdna_by_versioned))
print("cDNA transcript IDs unversioned:", len(cdna_by_unversioned))

print("Indexing canonical CDS FASTA from NT-v2 preparation:", nt_cds_fasta_path)

cds_by_ensg = {}
with open(nt_cds_fasta_path, "rt") as handle:
    for rec in SeqIO.parse(handle, "fasta"):
        ensg = rec.id.split("|")[0]
        cds_by_ensg[ensg] = clean_dna(str(rec.seq))

print("Canonical CDS genes indexed:", len(cds_by_ensg))

records = []
helix_fasta_records = []
raw_rna_fasta_records = []

for idx, row in master.iterrows():
    ensg = row.get("ensembl_gene_id", "")
    symbol = row.get("gene_symbol", "")
    entrez = row.get("entrez_id", "")
    uniprot = row.get("uniprot_id", "")
    tid = row.get("canonical_transcript_id", "").strip()
    tidv = row.get("canonical_transcript_id_versioned", "").strip()

    cdna = None
    cdna_method = ""
    source_tid = ""

    if tidv and tidv in cdna_by_versioned:
        cdna = cdna_by_versioned[tidv]
        cdna_method = "canonical_transcript_id_versioned_exact"
        source_tid = tidv
    elif tid and tid in cdna_by_unversioned:
        cdna = cdna_by_unversioned[tid]
        cdna_method = "canonical_transcript_id_unversioned_exact"
        source_tid = tid
    elif tid and tid in cdna_by_versioned:
        cdna = cdna_by_versioned[tid]
        cdna_method = "canonical_transcript_id_exact_in_versioned_index"
        source_tid = tid

    cds = cds_by_ensg.get(ensg, "")

    status = ""
    cds_start_0based = -1
    cds_end_0based_exclusive = -1
    cds_length_nt = len(cds) if cds else 0
    mrna_length_nt = len(cdna) if cdna else 0
    helix_input_length = 0
    mapping_method = ""

    if cdna is None:
        status = "missing_cdna_sequence"
    elif not cds:
        status = "missing_cds_sequence"
    else:
        hit = cdna.find(cds)
        second_hit = cdna.find(cds, hit + 1) if hit >= 0 else -1

        if hit < 0:
            status = "cds_not_found_inside_cdna"
        else:
            cds_start_0based = hit
            cds_end_0based_exclusive = hit + len(cds)

            raw_rna = dna_to_rna(cdna)
            cds_rna = dna_to_rna(cds)

            utr5 = raw_rna[:cds_start_0based]
            utr3 = raw_rna[cds_end_0based_exclusive:]
            marked_cds = add_codon_markers_to_cds(cds_rna)

            helix_input = utr5 + marked_cds + utr3
            helix_input_length = len(helix_input)

            if second_hit >= 0:
                status = "mapped_cds_multiple_hits_used_first"
            else:
                status = "mapped"

            mapping_method = cdna_method + "__cds_substring_match"

            header = (
                f"{ensg}|{symbol}|{source_tid}|"
                f"mrna_length_nt={mrna_length_nt}|"
                f"cds_start_0based={cds_start_0based}|"
                f"cds_length_nt={cds_length_nt}|"
                f"helix_input_length={helix_input_length}"
            )
            helix_fasta_records.append((header, helix_input))
            raw_rna_fasta_records.append((header, raw_rna))

    records.append({
        "master_row_index": idx,
        "ensembl_gene_id": ensg,
        "gene_symbol": symbol,
        "entrez_id": entrez,
        "uniprot_id": uniprot,
        "canonical_transcript_id": tid,
        "canonical_transcript_id_versioned": tidv,
        "source_transcript_id_used": source_tid,
        "mrna_length_nt": mrna_length_nt,
        "cds_length_nt": cds_length_nt,
        "cds_start_0based": cds_start_0based,
        "cds_end_0based_exclusive": cds_end_0based_exclusive,
        "helix_input_length": helix_input_length,
        "status": status,
        "mapping_method": mapping_method,
    })

report = pd.DataFrame(records)
report.to_csv(out_report, sep="\t", index=False)

mapped = report[report["status"].isin(["mapped", "mapped_cds_multiple_hits_used_first"])].copy()
mapped.to_csv(out_mapped_tsv, sep="\t", index=False)

with open(out_helix_fasta, "w") as f:
    for header, seq in helix_fasta_records:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

with open(out_raw_rna_fasta, "w") as f:
    for header, seq in raw_rna_fasta_records:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

print("\nSummary")
print("-------")
print("Mapped mature mRNA genes:", len(mapped))
print("Missing/failed genes:", len(report) - len(mapped))
print("Coverage:", round(len(mapped) / len(master) * 100, 4), "%")

if len(mapped):
    print("Min mRNA length nt:", int(mapped["mrna_length_nt"].min()))
    print("Median mRNA length nt:", float(mapped["mrna_length_nt"].median()))
    print("Max mRNA length nt:", int(mapped["mrna_length_nt"].max()))
    print("Min Helix input length:", int(mapped["helix_input_length"].min()))
    print("Median Helix input length:", float(mapped["helix_input_length"].median()))
    print("Max Helix input length:", int(mapped["helix_input_length"].max()))

print("\nStatus counts:")
print(report["status"].value_counts().to_string())

print("\nWrote:")
print(out_report)
print(out_mapped_tsv)
print(out_helix_fasta)
print(out_raw_rna_fasta)
