from pathlib import Path
from datetime import datetime
import argparse
import json
import re
import time
import gc
import logging

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm

from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig

logging.getLogger('helical').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)


EMBEDDING_NAME = "helix_mrna_full_transcript_v1_1"
DIM = 256


def clean_helix_seq(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    return re.sub(r"[^ACGUNEN]", "N", seq)


def chunk_sequence(seq: str, chunk_len: int):
    seq = clean_helix_seq(seq)
    for i in range(0, len(seq), chunk_len):
        chunk = seq[i:i + chunk_len]
        if chunk:
            yield chunk


def real_token_embeddings(chunk_embedding: np.ndarray, chunk: str) -> np.ndarray:
    """
    Helix wrapper returns token-level embeddings.
    Diagnostic tokenizer test showed that input "EAUGEAUG" becomes:
    [E, A, U, G, E, A, U, G, PAD].
    Therefore, for an unpadded single-sequence call, the real sequence tokens are
    the first len(chunk) embeddings and the trailing extra token is padding.
    """
    chunk_embedding = np.asarray(chunk_embedding, dtype=np.float32)
    n = len(chunk)

    if chunk_embedding.ndim != 2:
        raise RuntimeError(f"Expected 2D chunk embedding, got shape {chunk_embedding.shape}")

    if chunk_embedding.shape[1] != DIM:
        raise RuntimeError(f"Unexpected dimension: {chunk_embedding.shape[1]} != {DIM}")

    if chunk_embedding.shape[0] >= n:
        return chunk_embedding[:n, :]

    raise RuntimeError(
        f"Embedding length {chunk_embedding.shape[0]} is smaller than input length {n}"
    )


def embed_sequence(seq: str, model, chunk_len: int):
    total = np.zeros(DIM, dtype=np.float64)
    total_tokens = 0
    chunks_used = 0

    for chunk in chunk_sequence(seq, chunk_len):
        processed = model.process_data([chunk])
        emb = model.get_embeddings(processed)

        arr = np.asarray(emb)

        if arr.ndim != 3 or arr.shape[0] != 1:
            raise RuntimeError(f"Unexpected Helix output shape for one chunk: {arr.shape}")

        token_emb = real_token_embeddings(arr[0], chunk)

        total += token_emb.astype(np.float64).sum(axis=0)
        total_tokens += token_emb.shape[0]
        chunks_used += 1

        del processed, emb, arr, token_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if total_tokens == 0:
        raise RuntimeError("No tokens embedded for sequence.")

    vec = (total / total_tokens).astype(np.float32)

    if vec.shape[0] != DIM:
        raise RuntimeError(f"Unexpected final vector shape: {vec.shape}")

    return vec, total_tokens, chunks_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process first N mapped genes only.")
    parser.add_argument("--chunk-len", type=int, default=12000, help="Max Helix input characters per chunk.")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    home = Path.home()

    raw_dir = home / "data/raw_embeddings/helix_mrna_full_transcript_v1_1"
    seq_dir = raw_dir / "source_sequences"
    fasta_path = seq_dir / "helix_mrna_canonical_full_transcript_sequences.fa"
    sequence_report_path = home / "reports/mapping_reports/helix_mrna_full_transcript_v1_1_sequence_mapping_report.tsv"
    master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"

    if args.limit is None:
        out_dir = home / "data/processed_embeddings" / EMBEDDING_NAME
        out_prefix = EMBEDDING_NAME
    else:
        out_dir = home / "data/processed_embeddings" / f"{EMBEDDING_NAME}_TEST"
        out_prefix = f"{EMBEDDING_NAME}_TEST"

    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{out_prefix}_embeddings.npz"
    out_genes = out_dir / f"{out_prefix}_genes.tsv"
    out_metadata = out_dir / f"{out_prefix}_metadata.json"
    out_report = home / "reports/mapping_reports" / f"{out_prefix}_mapping_report.txt"
    partial_path = out_dir / f"{out_prefix}_partial_progress.npz"

    print("Embedding name:", EMBEDDING_NAME)
    print("Output directory:", out_dir)
    print("FASTA:", fasta_path)
    print("Sequence report:", sequence_report_path)

    report = pd.read_csv(sequence_report_path, sep="\t", dtype=str).fillna("")
    mapped = report[
        report["status"].isin(["mapped", "mapped_cds_multiple_hits_used_first"])
    ].copy()

    mapped["mrna_length_nt"] = mapped["mrna_length_nt"].astype(int)
    mapped["helix_input_length"] = mapped["helix_input_length"].astype(int)

    if args.limit is not None:
        mapped = mapped.head(args.limit).copy()

    print("Mapped rows to embed:", len(mapped))

    print("\nLoading FASTA sequences...")
    seq_by_ensg = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        ensg = rec.id.split("|")[0]
        seq_by_ensg[ensg] = clean_helix_seq(str(rec.seq))

    missing = [x for x in mapped["ensembl_gene_id"] if x not in seq_by_ensg]
    if missing:
        raise RuntimeError(f"Mapped genes missing from FASTA: {len(missing)} examples={missing[:5]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nDevice:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nInitializing official Helix-mRNA wrapper...")
    config = HelixmRNAConfig(
        batch_size=1,
        device=device,
        max_length=12288,
    )
    model = HelixmRNA(configurer=config)

    n = len(mapped)

    if args.resume and partial_path.exists():
        print("Resuming from:", partial_path)
        partial = np.load(partial_path, allow_pickle=False)
        embeddings = partial["embeddings"]
        done_mask = partial["done_mask"].astype(bool)
        token_counts = partial["token_counts"]
        chunk_counts = partial["chunk_counts"]
    else:
        embeddings = np.zeros((n, DIM), dtype=np.float32)
        done_mask = np.zeros(n, dtype=bool)
        token_counts = np.zeros(n, dtype=np.int32)
        chunk_counts = np.zeros(n, dtype=np.int32)

    start = time.time()

    for i, row in tqdm(list(mapped.reset_index(drop=True).iterrows()), total=n):
        if done_mask[i]:
            continue

        ensg = row["ensembl_gene_id"]
        seq = seq_by_ensg[ensg]

        try:
            vec, ntokens, nchunks = embed_sequence(seq, model, args.chunk_len)
        except Exception as e:
            print(
                f"\nFAILED at index={i}, gene={ensg}, "
                f"symbol={row.get('gene_symbol', '')}, "
                f"helix_input_length={len(seq)}"
            )
            raise e

        embeddings[i, :] = vec
        done_mask[i] = True
        token_counts[i] = ntokens
        chunk_counts[i] = nchunks

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % args.checkpoint_every == 0:
            np.savez_compressed(
                partial_path,
                embeddings=embeddings,
                done_mask=done_mask,
                token_counts=token_counts,
                chunk_counts=chunk_counts,
            )
            print(f"\nCheckpoint saved after {i + 1} rows -> {partial_path}", flush=True)

    elapsed = time.time() - start

    if not done_mask.all():
        raise RuntimeError("Not all rows completed.")

    genes = mapped[
        [
            "master_row_index",
            "ensembl_gene_id",
            "gene_symbol",
            "entrez_id",
            "uniprot_id",
            "canonical_transcript_id",
            "canonical_transcript_id_versioned",
            "source_transcript_id_used",
            "mrna_length_nt",
            "cds_length_nt",
            "cds_start_0based",
            "cds_end_0based_exclusive",
            "helix_input_length",
            "mapping_method",
        ]
    ].copy()

    genes["helix_token_count_used_for_mean_pooling"] = token_counts
    genes["helix_chunk_count"] = chunk_counts

    np.savez_compressed(out_npz, embeddings=embeddings)
    genes.to_csv(out_genes, sep="\t", index=False)

    norms = np.linalg.norm(embeddings, axis=1)
    zero_vectors = int((norms == 0).sum())
    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    duplicated_ensembl = int(genes["ensembl_gene_id"].duplicated().sum())

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": "processed_qc_passed" if zero_vectors == 0 and nan_count == 0 and inf_count == 0 and duplicated_ensembl == 0 else "processed_qc_check_needed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modality": "mRNA language model / mature transcript sequence",
        "species": "human",
        "master_table": str(master_path),
        "raw_model_source": "official HuggingFace repository helical-ai/helix-mRNA, loaded through official helical wrapper",
        "raw_sequence_fasta": str(fasta_path),
        "sequence_mapping_report": str(sequence_report_path),
        "download_source": "official HuggingFace model repository",
        "model_checkpoint": "helical-ai/helix-mRNA",
        "original_model_or_method": "Helix-mRNA",
        "embedding_generated_by": "Daniel locally by model inference",
        "input_sequence_type": "canonical mature mRNA/cDNA transcript converted to RNA, with E inserted before CDS codons",
        "input_sequence_source": "Ensembl cDNA release 114 plus canonical CDS from Ensembl CDS release 114",
        "input_sequence_release": "Ensembl release 114",
        "genome_build": "GRCh38",
        "transcript_choice": "canonical transcript from master_gene_table_v1_1_enriched.csv",
        "pooling_strategy": "final token embeddings from official Helix wrapper, mean pooled over non-special sequence tokens; long transcripts chunked and token-count weighted",
        "embedding_dimension": DIM,
        "dtype": "float32",
        "chunk_len_requested": args.chunk_len,
        "raw_master_rows": int(len(pd.read_csv(master_path, dtype=str))),
        "mapped_mrna_genes_available": int(len(report[report["status"].isin(["mapped", "mapped_cds_multiple_hits_used_first"])])),
        "processed_gene_count": int(len(genes)),
        "missing_or_failed_sequence_genes": int(len(report) - len(report[report["status"].isin(["mapped", "mapped_cds_multiple_hits_used_first"])])),
        "coverage_of_master_table_percent": round(len(report[report["status"].isin(["mapped", "mapped_cds_multiple_hits_used_first"])]) / len(report) * 100, 4),
        "matrix_shape": list(embeddings.shape),
        "qc": {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_vectors": zero_vectors,
            "duplicated_ensembl_ids": duplicated_ensembl,
            "rows_match_genes_tsv": bool(embeddings.shape[0] == len(genes)),
            "min_norm": float(norms.min()),
            "median_norm": float(np.median(norms)),
            "max_norm": float(norms.max()),
        },
        "runtime_seconds": round(elapsed, 2),
        "notes": [
            "No model training or fine-tuning was performed.",
            "Direct Transformers loading was rejected because it randomly initialized multiple Mamba2 weights.",
            "The official helical HelixmRNA wrapper was used instead.",
            "121 master-table genes did not have a clean canonical cDNA/CDS match and were excluded from the processed matrix.",
            "Fast Mamba kernels were unavailable; the official wrapper used the naive implementation.",
        ],
    }

    with open(out_metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(out_report, "w") as f:
        f.write(f"Mapping/QC report for {out_prefix}\n")
        f.write("=" * (22 + len(out_prefix)) + "\n\n")
        f.write(f"Status: {metadata['status']}\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"Embeddings: {out_npz}\n")
        f.write(f"Genes: {out_genes}\n")
        f.write(f"Metadata: {out_metadata}\n\n")
        f.write(f"Matrix shape: {embeddings.shape}\n")
        f.write(f"Rows match genes.tsv: {embeddings.shape[0] == len(genes)}\n")
        f.write(f"NaN count: {nan_count}\n")
        f.write(f"Inf count: {inf_count}\n")
        f.write(f"Zero vectors: {zero_vectors}\n")
        f.write(f"Duplicated Ensembl IDs: {duplicated_ensembl}\n")
        f.write(f"Processed genes: {len(genes)}\n")
        f.write(f"Missing/failed sequence genes: {metadata['missing_or_failed_sequence_genes']}\n")
        f.write(f"Coverage of master table: {metadata['coverage_of_master_table_percent']}%\n")
        f.write(f"Runtime seconds: {metadata['runtime_seconds']}\n")

    print("\nDone")
    print("----")
    print("Status:", metadata["status"])
    print("Matrix shape:", embeddings.shape)
    print("Rows match genes.tsv:", embeddings.shape[0] == len(genes))
    print("NaN count:", nan_count)
    print("Inf count:", inf_count)
    print("Zero vectors:", zero_vectors)
    print("Duplicated Ensembl IDs:", duplicated_ensembl)
    print("Output:", out_dir)
    print("Report:", out_report)


if __name__ == "__main__":
    main()
