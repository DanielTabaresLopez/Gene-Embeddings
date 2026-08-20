from pathlib import Path
import gzip
import re
import pandas as pd
from Bio import SeqIO

home = Path.home()

master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"
raw_dir = home / "data/raw_embeddings/nt_v2_500m_multispecies_cds_v1_1"
seq_dir = raw_dir / "source_sequences"
fasta_path = seq_dir / "Homo_sapiens.GRCh38.cds.all.fa.gz"

out_tsv = seq_dir / "nt_v2_canonical_cds_sequences.tsv"
out_fasta = seq_dir / "nt_v2_canonical_cds_sequences.fa"
report_path = home / "reports/mapping_reports/nt_v2_500m_multispecies_cds_v1_1_sequence_mapping_report.tsv"

def clean_seq(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return re.sub(r"[^ACGTN]", "N", seq)

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

print("Indexing CDS FASTA:", fasta_path)

seq_by_versioned = {}
seq_by_unversioned = {}

with gzip.open(fasta_path, "rt") as handle:
    for rec in SeqIO.parse(handle, "fasta"):
        tid_versioned = rec.id.split()[0]
        tid_unversioned = tid_versioned.split(".")[0]
        seq = clean_seq(str(rec.seq))

        seq_by_versioned.setdefault(tid_versioned, seq)
        seq_by_unversioned.setdefault(tid_unversioned, seq)

print("FASTA transcript IDs versioned:", len(seq_by_versioned))
print("FASTA transcript IDs unversioned:", len(seq_by_unversioned))

records = []
fasta_records = []

for idx, row in master.iterrows():
    ensg = row.get("ensembl_gene_id", "")
    symbol = row.get("gene_symbol", "")
    entrez = row.get("entrez_id", "")
    uniprot = row.get("uniprot_id", "")
    tid = row.get("canonical_transcript_id", "").strip()
    tidv = row.get("canonical_transcript_id_versioned", "").strip()

    seq = None
    source_tid = ""
    method = ""
    status = ""

    if tidv and tidv in seq_by_versioned:
        seq = seq_by_versioned[tidv]
        source_tid = tidv
        method = "canonical_transcript_id_versioned_exact"
        status = "mapped"
    elif tid and tid in seq_by_unversioned:
        seq = seq_by_unversioned[tid]
        source_tid = tid
        method = "canonical_transcript_id_unversioned_exact"
        status = "mapped"
    elif tid and tid in seq_by_versioned:
        seq = seq_by_versioned[tid]
        source_tid = tid
        method = "canonical_transcript_id_exact_in_versioned_index"
        status = "mapped"
    else:
        status = "missing_cds_sequence"

    cds_len = len(seq) if seq else 0

    rec = {
        "master_row_index": idx,
        "ensembl_gene_id": ensg,
        "gene_symbol": symbol,
        "entrez_id": entrez,
        "uniprot_id": uniprot,
        "canonical_transcript_id": tid,
        "canonical_transcript_id_versioned": tidv,
        "source_transcript_id_used": source_tid,
        "cds_length_nt": cds_len,
        "status": status,
        "mapping_method": method,
    }
    records.append(rec)

    if seq:
        fasta_header = f"{ensg}|{symbol}|{source_tid}|cds_length_nt={cds_len}"
        fasta_records.append((fasta_header, seq))

report = pd.DataFrame(records)
report.to_csv(report_path, sep="\t", index=False)

mapped = report[report["status"] == "mapped"].copy()
mapped.to_csv(out_tsv, sep="\t", index=False)

with open(out_fasta, "w") as f:
    for header, seq in fasta_records:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

print("\nSummary")
print("-------")
print("Mapped CDS genes:", len(mapped))
print("Missing CDS genes:", int((report["status"] != "mapped").sum()))
print("Coverage:", round(len(mapped) / len(master) * 100, 4), "%")

if len(mapped):
    print("Min CDS length:", int(mapped["cds_length_nt"].min()))
    print("Median CDS length:", float(mapped["cds_length_nt"].median()))
    print("Max CDS length:", int(mapped["cds_length_nt"].max()))

print("\nWrote:")
print(out_tsv)
print(out_fasta)
print(report_path)
