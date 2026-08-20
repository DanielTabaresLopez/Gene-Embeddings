from pathlib import Path
import gzip
import json
import re
import time
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from Bio import SeqIO
import esm


EMBEDDING_NAME = "MSATransformer-depth1-UniProt-human-v1.1"
MODEL_ID = "esm_msa1b_t12_100M_UR50S"

MASTER_PATH = Path.home() / "metadata/master_gene_table_v1_1_enriched.csv"
RAW_DIR = Path.home() / "data/raw_embeddings/msa_transformer_uniprot_human_v1_1"
FASTA_PATH = RAW_DIR / "human_UP000005640_uniprot_proteome.fasta.gz"

OUT_DIR = Path.home() / "data/processed_embeddings/msa_transformer_depth1_uniprot_human_v1_1"
REPORT_DIR = Path.home() / "reports/mapping_reports"
SAFE = "msa_transformer_depth1_uniprot_human_v1_1"

# MSA Transformer has learned positional embeddings; keep windows safely below 1024 aa.
MAX_AA_PER_WINDOW = 1022


def split_ids(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [p.strip() for p in re.split(r"[|,; ]+", str(x)) if p.strip()]


def base_uniprot_id(x):
    x = str(x).strip()
    return x.split("-")[0] if "-" in x else x


def parse_uniprot_accession(record_id):
    # UniProt FASTA commonly: sp|P12345|NAME or tr|A0A...|NAME
    parts = record_id.split("|")
    if len(parts) >= 2:
        return parts[1].strip()
    return record_id.split()[0].strip()


def load_fasta_index(fasta_path):
    """
    Robust FASTA parser for UniProt streams.
    Skips blank lines and comment/header metadata lines before and between records.
    """
    seqs = {}

    opener = gzip.open if str(fasta_path).endswith(".gz") else open

    current_header = None
    current_seq = []

    def flush_record(header, seq_parts):
        if header is None:
            return

        record_id = header.split()[0]
        acc = parse_uniprot_accession(record_id)
        seq = "".join(seq_parts).replace("*", "").upper()

        if acc and seq:
            if acc not in seqs:
                seqs[acc] = seq
            b = base_uniprot_id(acc)
            if b and b not in seqs:
                seqs[b] = seq

    with opener(fasta_path, "rt") as handle:
        for raw in handle:
            line = raw.strip()

            if not line:
                continue

            if line.startswith(("#", "!", ";")):
                continue

            if line.startswith(">"):
                flush_record(current_header, current_seq)
                current_header = line[1:].strip()
                current_seq = []
            else:
                if current_header is None:
                    # Ignore any unexpected metadata before the first sequence.
                    continue
                current_seq.append(line)

        flush_record(current_header, current_seq)

    return seqs


def clean_sequence(seq):
    valid = set("ACDEFGHIKLMNPQRSTVWYX")
    return "".join([aa if aa in valid else "X" for aa in str(seq).upper()])


def collect_uniprot_candidates(row):
    cols = [c for c in row.index if "uniprot" in c.lower()]
    preferred = [c for c in ["uniprot_id", "uniprot", "uniprot_accession"] if c in row.index]
    ordered_cols = preferred + [c for c in cols if c not in preferred]

    candidates = []
    seen = set()

    for col in ordered_cols:
        for x in split_ids(row.get(col, "")):
            for y in [x, base_uniprot_id(x)]:
                if y and y not in seen:
                    candidates.append(y)
                    seen.add(y)

    return candidates


def build_gene_jobs(master, seqs):
    rows = []

    for _, r in master.iterrows():
        candidates = collect_uniprot_candidates(r)

        matched_acc = None
        matched_seq = None

        for acc in candidates:
            if acc in seqs:
                matched_acc = acc
                matched_seq = clean_sequence(seqs[acc])
                break

        if matched_acc is None or not matched_seq:
            continue

        rows.append({
            "ensembl_gene_id": r.get("ensembl_gene_id", ""),
            "gene_symbol": r.get("gene_symbol", ""),
            "hgnc_approved_symbol": r.get("hgnc_approved_symbol", ""),
            "entrez_id": r.get("entrez_id", ""),
            "uniprot_id": r.get("uniprot_id", ""),
            "matched_uniprot_accession": matched_acc,
            "sequence_length": len(matched_seq),
            "sequence": matched_seq,
        })

    return pd.DataFrame(rows)


def sequence_windows(seq, max_len=MAX_AA_PER_WINDOW):
    if len(seq) <= max_len:
        return [(0, len(seq), seq)]

    windows = []
    start = 0
    while start < len(seq):
        end = min(start + max_len, len(seq))
        windows.append((start, end, seq[start:end]))
        start = end

    return windows


def embed_sequence_depth1(seq, label, model, batch_converter, device):
    windows = sequence_windows(seq)

    pieces = []
    weights = []

    for start, end, subseq in windows:
        # One-row MSA: [(sequence_id, aligned_sequence)]
        msa = [(f"{label}_{start}_{end}", subseq)]
        batch_labels, batch_strs, batch_tokens = batch_converter([msa])

        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            out = model(batch_tokens, repr_layers=[12], return_contacts=False)

        reps = out["representations"][12]  # shape: batch x msa_depth x tokens x dim

        # Remove beginning/end special tokens. For this model, residue tokens are 1 : L+1.
        residue_reps = reps[0, 0, 1:len(subseq) + 1, :]
        pooled = residue_reps.mean(dim=0).detach().cpu().numpy().astype("float32")

        pieces.append(pooled)
        weights.append(len(subseq))

        del batch_tokens, out, reps, residue_reps
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pieces = np.vstack(pieces).astype("float32")
    weights = np.array(weights, dtype="float32")
    weights = weights / weights.sum()

    return (pieces * weights[:, None]).sum(axis=0).astype("float32"), len(windows)


def load_model(device):
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)
    return model, batch_converter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not FASTA_PATH.exists():
        raise FileNotFoundError(f"Missing FASTA: {FASTA_PATH}")

    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    seqs = load_fasta_index(FASTA_PATH)
    jobs = build_gene_jobs(master, seqs)

    if args.limit is not None:
        jobs = jobs.iloc[:args.limit].copy()

    if jobs.empty:
        raise RuntimeError("No genes mapped to UniProt protein sequences.")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print(f"Embedding: {EMBEDDING_NAME}")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Master rows: {len(master)}")
    print(f"FASTA indexed accessions: {len(seqs)}")
    print(f"Mapped genes to process: {len(jobs)}")
    print(f"Coverage before embedding: {len(jobs) / len(master):.4%}")

    model, batch_converter = load_model(device)

    embeddings = []
    gene_rows = []
    failures = []

    for _, row in tqdm(jobs.iterrows(), total=len(jobs), desc=EMBEDDING_NAME):
        try:
            emb, n_windows = embed_sequence_depth1(
                row["sequence"],
                row["ensembl_gene_id"],
                model,
                batch_converter,
                device,
            )
            embeddings.append(emb)
            gene_rows.append({
                "row_index": len(gene_rows),
                "ensembl_gene_id": row["ensembl_gene_id"],
                "gene_symbol": row["gene_symbol"],
                "hgnc_approved_symbol": row["hgnc_approved_symbol"],
                "entrez_id": row["entrez_id"],
                "uniprot_id": row["uniprot_id"],
                "matched_uniprot_accession": row["matched_uniprot_accession"],
                "sequence_length": int(row["sequence_length"]),
                "n_windows": int(n_windows),
                "msa_depth": 1,
            })
        except RuntimeError as e:
            # If a single giant protein causes CUDA memory problems, record and continue.
            failures.append({
                "ensembl_gene_id": row["ensembl_gene_id"],
                "gene_symbol": row["gene_symbol"],
                "matched_uniprot_accession": row["matched_uniprot_accession"],
                "sequence_length": int(row["sequence_length"]),
                "error": repr(e),
            })
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            failures.append({
                "ensembl_gene_id": row["ensembl_gene_id"],
                "gene_symbol": row["gene_symbol"],
                "matched_uniprot_accession": row["matched_uniprot_accession"],
                "sequence_length": int(row["sequence_length"]),
                "error": repr(e),
            })

    if not embeddings:
        raise RuntimeError("No embeddings were successfully generated.")

    X = np.vstack(embeddings).astype("float32")
    genes = pd.DataFrame(gene_rows)
    failures_df = pd.DataFrame(failures)

    if X.shape[0] != len(genes):
        raise RuntimeError("Embedding rows do not match gene metadata rows.")
    if not np.isfinite(X).all():
        raise RuntimeError("NaN or Inf found in embeddings.")

    np.savez_compressed(OUT_DIR / f"{SAFE}_embeddings.npz", embeddings=X)
    genes.to_csv(OUT_DIR / f"{SAFE}_genes.tsv", sep="\t", index=False)
    failures_df.to_csv(OUT_DIR / f"{SAFE}_failed_sequences.tsv", sep="\t", index=False)

    zero_vectors = int((np.linalg.norm(X, axis=1) == 0).sum())

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "modality": "protein_sequence_msa_transformer_depth1",
        "important_note": "This is MSA Transformer with MSA depth 1: one reference UniProt sequence per gene, not a homologous multi-sequence MSA.",
        "original_model_or_method": "MSA Transformer",
        "model_checkpoint": MODEL_ID,
        "model_source": "fair-esm / Meta FAIR ESM",
        "embedding_generated_by": "Daniel locally",
        "input_sequence_source": "UniProt human proteome UP000005640 FASTA stream",
        "input_sequence_type": "protein",
        "pooling_strategy": "mean residue representation from layer 12; long proteins split into non-overlapping windows and length-weighted mean pooled",
        "master_table": str(MASTER_PATH),
        "fasta_path": str(FASTA_PATH),
        "master_rows": int(len(master)),
        "fasta_indexed_accessions": int(len(seqs)),
        "mapped_jobs_attempted": int(len(jobs)),
        "mapped_genes_successful": int(len(genes)),
        "master_coverage_successful": float(len(genes) / len(master)),
        "embedding_dim": int(X.shape[1]),
        "matrix_shape": list(X.shape),
        "dtype": str(X.dtype),
        "device": str(device),
        "max_aa_per_window": int(MAX_AA_PER_WINDOW),
        "msa_depth": 1,
        "zero_vectors": zero_vectors,
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "failed_sequence_count": int(len(failures_df)),
        "runtime_minutes": float((time.time() - t0) / 60),
    }

    with open(OUT_DIR / f"{SAFE}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    report = []
    report.append(f"Mapping/QC report for {EMBEDDING_NAME}")
    report.append("=" * (22 + len(EMBEDDING_NAME)))
    report.append("")
    report.append("Status: processed_qc_passed")
    report.append("Important note: MSA depth = 1; this is not a true homologous multi-sequence MSA run.")
    report.append(f"Model checkpoint: {MODEL_ID}")
    report.append(f"Master table: {MASTER_PATH}")
    report.append(f"FASTA: {FASTA_PATH}")
    report.append(f"Output directory: {OUT_DIR}")
    report.append("")
    report.append(f"Master rows: {len(master)}")
    report.append(f"FASTA indexed accessions: {len(seqs)}")
    report.append(f"Mapped jobs attempted: {len(jobs)}")
    report.append(f"Successful mapped genes: {len(genes)}")
    report.append(f"Master coverage successful: {len(genes) / len(master):.4%}")
    report.append(f"Embedding shape: {X.shape[0]} x {X.shape[1]}")
    report.append(f"NaN count: {int(np.isnan(X).sum())}")
    report.append(f"Inf count: {int(np.isinf(X).sum())}")
    report.append(f"Zero vectors: {zero_vectors}")
    report.append(f"Duplicated Ensembl IDs: {int(genes['ensembl_gene_id'].duplicated().sum())}")
    report.append(f"Failed sequence count: {len(failures_df)}")
    report.append(f"Runtime minutes: {(time.time() - t0) / 60:.2f}")
    report.append("")
    report.append("Sequence length summary for successful genes:")
    report.append(genes["sequence_length"].describe().to_string())

    report_text = "\n".join(report) + "\n"

    (OUT_DIR / f"{SAFE}_mapping_report.txt").write_text(report_text)
    (REPORT_DIR / f"{SAFE}_mapping_report.txt").write_text(report_text)

    print(report_text)


if __name__ == "__main__":
    main()
