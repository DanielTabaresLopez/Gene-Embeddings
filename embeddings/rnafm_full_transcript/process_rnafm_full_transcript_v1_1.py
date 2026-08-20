#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import fm


EMBEDDING_NAME = "rnafm_full_transcript_v1_1"

HOME = Path.home()
RAW_DIR = HOME / "data/raw_embeddings" / EMBEDDING_NAME
HF_MODEL_DIR = RAW_DIR / "hf_model"
SEQ_TSV = RAW_DIR / "source_sequences/rnafm_full_transcript_sequences.tsv"
MASTER_PATH = HOME / "metadata/master_gene_table_v1_1_enriched.csv"

OUT_BASE = HOME / "data/processed_embeddings" / EMBEDDING_NAME
REPORT_PATH = HOME / "reports/mapping_reports" / f"{EMBEDDING_NAME}_mapping_report.txt"

MODEL_CKPT = HF_MODEL_DIR / "RNA-FM_pretrained.pth"
CACHE_CKPT = HOME / ".cache/torch/hub/checkpoints/RNA-FM_pretrained.pth"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_cached_checkpoint():
    CACHE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_CKPT.exists():
        if not MODEL_CKPT.exists():
            raise FileNotFoundError(
                f"Missing RNA-FM checkpoint at {MODEL_CKPT}. "
                "Expected RNA-FM_pretrained.pth in the HF snapshot."
            )
        import shutil
        shutil.copy2(MODEL_CKPT, CACHE_CKPT)


def split_sequence(seq: str, chunk_len: int, min_chunk_len: int):
    chunks = [seq[i:i + chunk_len] for i in range(0, len(seq), chunk_len)]
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk_len:
        chunks[-2] += chunks[-1]
        chunks = chunks[:-1]
    return chunks


def pool_chunks(model, alphabet, chunks, device, batch_size: int):
    batch_converter = alphabet.get_batch_converter()

    chunk_vectors = []
    chunk_lengths = []

    for i in range(0, len(chunks), batch_size):
        sub = chunks[i:i + batch_size]
        data = [(f"chunk_{i+j}", s) for j, s in enumerate(sub)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            out = model(tokens, repr_layers=[12])

        rep = out["representations"][12]

        for j, seq in enumerate(sub):
            n = len(seq)
            # position 0 = BOS, position 1..n = real RNA tokens, final token = EOS
            token_rep = rep[j, 1:1 + n, :]

            if token_rep.shape[0] != n:
                raise RuntimeError(
                    f"Token/sequence length mismatch: got {token_rep.shape[0]}, expected {n}"
                )

            vec = token_rep.mean(dim=0).detach().cpu().numpy().astype(np.float32)
            chunk_vectors.append(vec)
            chunk_lengths.append(n)

        del tokens, out, rep
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    weights = np.asarray(chunk_lengths, dtype=np.float32)
    mat = np.vstack(chunk_vectors).astype(np.float32)

    return np.average(mat, axis=0, weights=weights).astype(np.float32)


def write_outputs(out_dir, genes_df, matrix, metadata, report_text):
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{EMBEDDING_NAME}_embeddings.npz"
    genes_path = out_dir / f"{EMBEDDING_NAME}_genes.tsv"
    meta_path = out_dir / f"{EMBEDDING_NAME}_metadata.json"

    np.savez_compressed(npz_path, embeddings=matrix)
    genes_df.to_csv(genes_path, sep="\t", index=False)
    meta_path.write_text(json.dumps(metadata, indent=2))
    REPORT_PATH.write_text(report_text)

    return npz_path, genes_path, meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=1000)
    parser.add_argument("--min-chunk-len", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    start = time.time()

    if args.chunk_len > 1000:
        raise ValueError(
            "RNA-FM uses ESM-style positional embeddings. Keep --chunk-len <= 1000 "
            "so tokens plus BOS/EOS fit safely."
        )

    out_dir = OUT_BASE if args.limit is None else OUT_BASE.with_name(OUT_BASE.name + "_TEST")
    partial_path = out_dir / f"{EMBEDDING_NAME}_partial_progress.npz"

    seq_df = pd.read_csv(SEQ_TSV, sep="\t", dtype=str).fillna("")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")

    seq_df["rnafm_sequence_length"] = seq_df["rnafm_sequence"].str.len()

    if args.limit is not None:
        seq_df = seq_df.head(args.limit).copy()

    ensure_cached_checkpoint()

    print("Embedding:", EMBEDDING_NAME)
    print("Input sequences:", SEQ_TSV)
    print("Genes to process:", len(seq_df))
    print("Device:", args.device)
    print("Chunk length:", args.chunk_len)
    print("Batch size:", args.batch_size)
    print("Output:", out_dir)

    print("Loading RNA-FM...")
    model, alphabet = fm.pretrained.rna_fm_t12()
    model = model.to(args.device)
    model.eval()

    embeddings = []
    gene_rows = []
    start_idx = 0

    if args.resume and partial_path.exists():
        partial = np.load(partial_path, allow_pickle=True)
        embeddings = list(partial["embeddings"])
        saved_gene_ids = list(partial["ensembl_gene_id"])
        start_idx = len(saved_gene_ids)
        print(f"Resuming from {start_idx} rows")

    for idx in tqdm(range(start_idx, len(seq_df))):
        row = seq_df.iloc[idx]

        gene_id = row["ensembl_gene_id"]
        seq = row["rnafm_sequence"]

        chunks = split_sequence(seq, args.chunk_len, args.min_chunk_len)
        vec = pool_chunks(model, alphabet, chunks, args.device, args.batch_size)

        embeddings.append(vec)

        gene_row = {
            "ensembl_gene_id": gene_id,
            "rnafm_sequence_length": len(seq),
            "num_chunks": len(chunks),
        }

        if "transcript_id" in seq_df.columns:
            gene_row["transcript_id"] = row.get("transcript_id", "")

        gene_rows.append(gene_row)

        done = idx + 1
        if args.checkpoint_every and done % args.checkpoint_every == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                partial_path,
                embeddings=np.vstack(embeddings).astype(np.float32),
                ensembl_gene_id=np.asarray([g["ensembl_gene_id"] for g in gene_rows], dtype=object),
            )
            print(f"\nCheckpoint saved after {done} rows -> {partial_path}")

    matrix = np.vstack(embeddings).astype(np.float32)
    genes_df = pd.DataFrame(gene_rows)

    rows_match = matrix.shape[0] == len(genes_df)
    nan_count = int(np.isnan(matrix).sum())
    inf_count = int(np.isinf(matrix).sum())
    zero_vectors = int((np.linalg.norm(matrix, axis=1) == 0).sum())
    duplicated_genes = int(genes_df["ensembl_gene_id"].duplicated().sum())

    status = "processed_qc_passed"
    if not rows_match or nan_count or inf_count or zero_vectors or duplicated_genes:
        status = "processed_qc_failed"

    runtime = time.time() - start

    ckpt_sha = sha256_file(MODEL_CKPT) if MODEL_CKPT.exists() else None

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_dimension": int(matrix.shape[1]),
        "matrix_shape": list(matrix.shape),
        "model": "RNA-FM / rna_fm_t12",
        "model_checkpoint": str(MODEL_CKPT),
        "model_checkpoint_sha256": ckpt_sha,
        "model_source_snapshot": str(HF_MODEL_DIR),
        "software": {
            "python": "3.10",
            "torch": torch.__version__,
            "fm_module_file": getattr(fm, "__file__", "unknown"),
        },
        "input_sequence_type": "mature mRNA full transcript",
        "input_sequence_source": "Derived from Helix mature mRNA preparation based on Ensembl release 114 GRCh38 cDNA/canonical transcript mapping",
        "input_sequence_file": str(SEQ_TSV),
        "sequence_processing": "Helix E codon-start markers removed; T converted to U; sequences restricted to A/C/G/U/N",
        "pooling_strategy": "Final layer 12 mean token pooling over real RNA tokens only; BOS/EOS excluded; long transcripts split into chunks and chunk means length-weighted",
        "chunk_len": args.chunk_len,
        "min_chunk_len": args.min_chunk_len,
        "batch_size": args.batch_size,
        "master_gene_table": str(MASTER_PATH),
        "runtime_seconds": round(runtime, 2),
    }

    report_lines = [
        f"Mapping/QC report for {EMBEDDING_NAME}",
        "=" * (22 + len(EMBEDDING_NAME)),
        "",
        f"Status: {status}",
        f"Output directory: {out_dir}",
        f"Embeddings: {out_dir / (EMBEDDING_NAME + '_embeddings.npz')}",
        f"Genes: {out_dir / (EMBEDDING_NAME + '_genes.tsv')}",
        f"Metadata: {out_dir / (EMBEDDING_NAME + '_metadata.json')}",
        "",
        f"Model: RNA-FM / rna_fm_t12",
        f"Model checkpoint: {MODEL_CKPT}",
        f"Model checkpoint SHA256: {ckpt_sha}",
        f"Input sequences: {SEQ_TSV}",
        "",
        f"Matrix shape: {matrix.shape}",
        f"Rows match genes.tsv: {rows_match}",
        f"NaN count: {nan_count}",
        f"Inf count: {inf_count}",
        f"Zero vectors: {zero_vectors}",
        f"Duplicated Ensembl IDs: {duplicated_genes}",
        f"Processed genes: {len(genes_df)}",
        f"Coverage of master table: {100 * len(genes_df) / len(master):.4f}%",
        f"Runtime seconds: {runtime:.2f}",
        "",
    ]

    report_text = "\n".join(report_lines)

    npz_path, genes_path, meta_path = write_outputs(
        out_dir=out_dir,
        genes_df=genes_df,
        matrix=matrix,
        metadata=metadata,
        report_text=report_text,
    )

    print("\nDone")
    print("----")
    print(report_text)


if __name__ == "__main__":
    main()
