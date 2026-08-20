#!/usr/bin/env python3

from pathlib import Path
import re
import pandas as pd

HOME = Path.home()

MASTER_PATH = HOME / "metadata/master_gene_table_v1_1_enriched.csv"
HELIX_RAW_DIR = HOME / "data/raw_embeddings/helix_mrna_full_transcript_v1_1"

OUT_DIR = HOME / "data/raw_embeddings/rnafm_full_transcript_v1_1/source_sequences"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TSV = OUT_DIR / "rnafm_full_transcript_sequences.tsv"
OUT_FASTA = OUT_DIR / "rnafm_full_transcript_sequences.fa"
OUT_REPORT = HOME / "reports/mapping_reports/rnafm_full_transcript_v1_1_sequence_mapping_report.txt"
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


def clean_seq(seq: str) -> str:
    seq = str(seq).strip().upper()
    seq = seq.replace(" ", "").replace("\n", "").replace("\r", "")
    seq = seq.replace("T", "U")
    seq = seq.replace("E", "")
    return seq


def valid_rna(seq: str) -> bool:
    return bool(seq) and re.fullmatch(r"[ACGUN]+", seq) is not None


def extract_gene_id(text: str):
    m = re.search(r"(ENSG\d+)(?:\.\d+)?", str(text))
    return m.group(1) if m else ""


def extract_transcript_id(text: str):
    m = re.search(r"(ENST\d+)(?:\.\d+)?", str(text))
    return m.group(1) if m else ""


def read_fasta(path: Path):
    records = []
    header = None
    seq_parts = []

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.strip())

    if header is not None:
        records.append((header, "".join(seq_parts)))

    return records


def fasta_candidates():
    candidates = []

    for path in sorted(HELIX_RAW_DIR.rglob("*")):
        if path.suffix.lower() not in [".fa", ".fasta"]:
            continue

        try:
            records = read_fasta(path)
        except Exception:
            continue

        good = 0
        for header, seq in records[:5000]:
            gene = extract_gene_id(header)
            cleaned = clean_seq(seq)
            if gene and valid_rna(cleaned):
                good += 1

        if good:
            candidates.append((good, path))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates


def tsv_candidates():
    candidates = []

    for path in sorted(HELIX_RAW_DIR.rglob("*.tsv")):
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, nrows=200).fillna("")
        except Exception:
            continue

        cols = list(df.columns)

        gene_cols = []
        seq_cols = []

        for c in cols:
            vals = df[c].astype(str).head(200).tolist()

            gene_hits = sum(1 for v in vals if extract_gene_id(v))
            if gene_hits > 0:
                gene_cols.append((gene_hits, c))

            seq_hits = 0
            for v in vals:
                s = clean_seq(v)
                if len(s) >= 20 and valid_rna(s):
                    seq_hits += 1
            if seq_hits > 0:
                seq_cols.append((seq_hits, c))

        for gh, gc in gene_cols:
            for sh, sc in seq_cols:
                if gc != sc:
                    candidates.append((gh + sh, path, gc, sc))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates


def load_from_fasta(path: Path):
    rows = []
    invalid = 0

    for header, raw_seq in read_fasta(path):
        gene = extract_gene_id(header)
        transcript = extract_transcript_id(header)
        seq = clean_seq(raw_seq)

        if not gene:
            continue

        if not valid_rna(seq):
            invalid += 1
            continue

        rows.append({
            "ensembl_gene_id": gene,
            "transcript_id": transcript,
            "rnafm_sequence": seq,
            "rnafm_sequence_length": len(seq),
            "source_type": "fasta",
            "source_file": str(path),
            "helix_E_markers_removed": True,
            "T_converted_to_U": True,
        })

    return pd.DataFrame(rows), invalid


def load_from_tsv(path: Path, gene_col: str, seq_col: str):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rows = []
    invalid = 0

    transcript_col = None
    for c in df.columns:
        if df[c].astype(str).head(200).map(extract_transcript_id).astype(bool).sum() > 0:
            transcript_col = c
            break

    for _, row in df.iterrows():
        gene = extract_gene_id(row[gene_col])
        transcript = extract_transcript_id(row[transcript_col]) if transcript_col else ""
        seq = clean_seq(row[seq_col])

        if not gene:
            continue

        if not valid_rna(seq):
            invalid += 1
            continue

        rows.append({
            "ensembl_gene_id": gene,
            "transcript_id": transcript,
            "rnafm_sequence": seq,
            "rnafm_sequence_length": len(seq),
            "source_type": "tsv",
            "source_file": str(path),
            "source_gene_column": gene_col,
            "source_sequence_column": seq_col,
            "helix_E_markers_removed": True,
            "T_converted_to_U": True,
        })

    return pd.DataFrame(rows), invalid


def main():
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    master_ids = set(master["ensembl_gene_id"])

    print("Searching Helix FASTA candidates...")
    f_cands = fasta_candidates()
    for score, path in f_cands[:10]:
        print("FASTA candidate:", score, path)

    print("\nSearching Helix TSV candidates...")
    t_cands = tsv_candidates()
    for score, path, gene_col, seq_col in t_cands[:10]:
        print("TSV candidate:", score, path, "gene_col=", gene_col, "seq_col=", seq_col)

    if f_cands:
        score, path = f_cands[0]
        print("\nSelected FASTA:", path)
        out, invalid = load_from_fasta(path)
        selected = str(path)
        selected_type = "fasta"
    elif t_cands:
        score, path, gene_col, seq_col = t_cands[0]
        print("\nSelected TSV:", path)
        print("Gene column:", gene_col)
        print("Sequence column:", seq_col)
        out, invalid = load_from_tsv(path, gene_col, seq_col)
        selected = str(path)
        selected_type = "tsv"
    else:
        raise RuntimeError(
            "No usable FASTA or TSV sequence file found under "
            f"{HELIX_RAW_DIR}. Please run: find {HELIX_RAW_DIR} -type f"
        )

    before_master_filter = len(out)

    out = out[out["ensembl_gene_id"].isin(master_ids)].copy()
    after_master_filter = len(out)

    out = out.sort_values(
        ["ensembl_gene_id", "rnafm_sequence_length"],
        ascending=[True, False],
    )

    before_dup = len(out)
    out = out.drop_duplicates("ensembl_gene_id", keep="first").reset_index(drop=True)
    duplicates_removed = before_dup - len(out)

    mapped = set(out["ensembl_gene_id"])
    missing = [g for g in master["ensembl_gene_id"] if g not in mapped]

    out.to_csv(OUT_TSV, sep="\t", index=False)

    with open(OUT_FASTA, "w") as f:
        for _, row in out.iterrows():
            header = row["ensembl_gene_id"]
            if row.get("transcript_id", ""):
                header += "|" + row["transcript_id"]
            f.write(f">{header}\n")
            seq = row["rnafm_sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

    report = []
    report.append("Sequence mapping report for rnafm_full_transcript_v1_1")
    report.append("=" * 58)
    report.append("")
    report.append(f"Master table: {MASTER_PATH}")
    report.append(f"Selected source type: {selected_type}")
    report.append(f"Selected source file: {selected}")
    report.append("")
    report.append(f"Rows before master filter: {before_master_filter}")
    report.append(f"Rows after master filter: {after_master_filter}")
    report.append(f"Mapped mature mRNA genes: {len(out)}")
    report.append(f"Missing master genes: {len(missing)}")
    report.append(f"Coverage of master table: {100 * len(out) / len(master):.4f}%")
    report.append(f"Duplicated gene rows removed: {duplicates_removed}")
    report.append(f"Invalid sequence rows skipped: {invalid}")
    report.append("")
    report.append(f"Min sequence length: {int(out['rnafm_sequence_length'].min())}")
    report.append(f"Median sequence length: {float(out['rnafm_sequence_length'].median())}")
    report.append(f"Max sequence length: {int(out['rnafm_sequence_length'].max())}")
    report.append("")
    report.append(f"Output TSV: {OUT_TSV}")
    report.append(f"Output FASTA: {OUT_FASTA}")

    OUT_REPORT.write_text("\n".join(report) + "\n")

    print("\n" + "\n".join(report))


if __name__ == "__main__":
    main()
