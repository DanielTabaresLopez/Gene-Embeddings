#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import subprocess
import time
import gc

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel


EMBEDDING_NAME = "orthrus_base_4track_full_transcript_v1_1"

HOME = Path.home()

MODEL_DIR = HOME / "data/raw_embeddings/orthrus_base_4track_full_transcript_v1_1/hf_model"
RAW_REPO = HOME / "data/raw_embeddings/orthrus_base_4track_full_transcript_v1_1/official_repo"
SEQ_TSV = HOME / "data/raw_embeddings/rnafm_full_transcript_v1_1/source_sequences/rnafm_full_transcript_sequences.tsv"
MASTER_PATH = HOME / "metadata/master_gene_table_v1_1_enriched.csv"

OUT_BASE = HOME / "data/processed_embeddings" / EMBEDDING_NAME
REPORT_PATH = HOME / "reports/mapping_reports" / f"{EMBEDDING_NAME}_mapping_report.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: Path):
    if not repo.exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def find_model_weight_file(model_dir: Path):
    candidates = []
    for pattern in ["*.safetensors", "*.bin", "*.pt", "*.pth"]:
        candidates.extend(model_dir.rglob(pattern))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]


def split_sequence(seq: str, chunk_len: int):
    return [seq[i:i + chunk_len] for i in range(0, len(seq), chunk_len)]


def seq_to_tensor(model, seq: str, device: str):
    seq = seq.upper().replace("U", "T")

    oh = model.seq_to_oh(seq)

    if hasattr(oh, "detach"):
        oh = oh.detach().cpu().numpy()

    oh = np.asarray(oh, dtype=np.float32)

    # Expected from smoke test: (L, 4)
    if oh.shape[0] == len(seq) and oh.shape[1] == 4:
        x = torch.tensor(oh, dtype=torch.float32).unsqueeze(0).to(device)
    elif oh.shape[1] == len(seq) and oh.shape[0] == 4:
        x = torch.tensor(oh.T, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        raise RuntimeError(f"Unexpected seq_to_oh shape {oh.shape} for sequence length {len(seq)}")

    lengths = torch.tensor([x.shape[1]], dtype=torch.float32).to(device)
    return x, lengths


def embed_one_sequence(model, seq: str, device: str):
    x, lengths = seq_to_tensor(model, seq, device)

    with torch.no_grad():
        emb = model.representation(
            x,
            lengths,
            channel_last=True,
        )

    arr = emb.detach().cpu().numpy().astype(np.float32)

    del x, lengths, emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if arr.shape[0] != 1:
        raise RuntimeError(f"Unexpected embedding batch shape: {arr.shape}")

    return arr[0]


def embed_sequence(model, seq: str, device: str, max_full_len: int, fallback_chunk_len: int):
    seq = seq.upper().replace("U", "T")
    n = len(seq)

    if max_full_len <= 0 or n <= max_full_len:
        return embed_one_sequence(model, seq, device), 1, False

    chunks = split_sequence(seq, fallback_chunk_len)

    vecs = []
    weights = []

    for chunk in chunks:
        vec = embed_one_sequence(model, chunk, device)
        vecs.append(vec)
        weights.append(len(chunk))

    mat = np.vstack(vecs).astype(np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    pooled = np.average(mat, axis=0, weights=weights).astype(np.float32)

    return pooled, len(chunks), True


def save_partial(partial_path: Path, embeddings, gene_rows):
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        partial_path,
        embeddings=np.vstack(embeddings).astype(np.float32),
        gene_rows_json=json.dumps(gene_rows),
    )


def load_partial(partial_path: Path):
    data = np.load(partial_path, allow_pickle=True)
    embeddings = [x.astype(np.float32) for x in data["embeddings"]]
    gene_rows = json.loads(str(data["gene_rows_json"]))
    return embeddings, gene_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--longest", type=int, default=None, help="Process N longest transcripts only, for stress testing.")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-full-len", type=int, default=50000)
    parser.add_argument("--fallback-chunk-len", type=int, default=50000)
    args = parser.parse_args()

    start = time.time()

    out_dir = OUT_BASE
    if args.limit is not None or args.longest is not None:
        out_dir = OUT_BASE.with_name(OUT_BASE.name + "_TEST")

    partial_path = out_dir / f"{EMBEDDING_NAME}_partial_progress.npz"

    seq_df = pd.read_csv(SEQ_TSV, sep="\t", dtype=str).fillna("")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")

    seq_df["orthrus_sequence"] = seq_df["rnafm_sequence"].str.upper().str.replace("U", "T", regex=False)
    seq_df["orthrus_sequence_length"] = seq_df["orthrus_sequence"].str.len()

    if args.longest is not None:
        seq_df = seq_df.sort_values("orthrus_sequence_length", ascending=False).head(args.longest).copy()
    elif args.limit is not None:
        seq_df = seq_df.head(args.limit).copy()

    print("Embedding:", EMBEDDING_NAME)
    print("Model directory:", MODEL_DIR)
    print("Input sequences:", SEQ_TSV)
    print("Genes to process:", len(seq_df))
    print("Device:", args.device)
    print("Max full length:", args.max_full_len)
    print("Fallback chunk length:", args.fallback_chunk_len)
    print("Output:", out_dir)

    print("Loading Orthrus base 4-track...")
    model = AutoModel.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )
    model = model.to(args.device)
    model.eval()

    embeddings = []
    gene_rows = []
    start_idx = 0

    if args.resume and partial_path.exists():
        embeddings, gene_rows = load_partial(partial_path)
        start_idx = len(gene_rows)
        print(f"Resuming from {start_idx} rows")

    for idx in tqdm(range(start_idx, len(seq_df))):
        row = seq_df.iloc[idx]

        gene_id = row["ensembl_gene_id"]
        transcript_id = row.get("transcript_id", "")
        seq = row["orthrus_sequence"]

        vec, num_chunks, was_chunked = embed_sequence(
            model=model,
            seq=seq,
            device=args.device,
            max_full_len=args.max_full_len,
            fallback_chunk_len=args.fallback_chunk_len,
        )

        embeddings.append(vec)

        gene_rows.append({
            "ensembl_gene_id": gene_id,
            "transcript_id": transcript_id,
            "orthrus_sequence_length": len(seq),
            "num_chunks": num_chunks,
            "was_chunked": was_chunked,
        })

        done = idx + 1
        if args.checkpoint_every and done % args.checkpoint_every == 0:
            save_partial(partial_path, embeddings, gene_rows)
            print(f"\nCheckpoint saved after {done} rows -> {partial_path}")

        gc.collect()

    matrix = np.vstack(embeddings).astype(np.float32)
    genes_df = pd.DataFrame(gene_rows)

    rows_match = matrix.shape[0] == len(genes_df)
    nan_count = int(np.isnan(matrix).sum())
    inf_count = int(np.isinf(matrix).sum())
    zero_vectors = int((np.linalg.norm(matrix, axis=1) == 0).sum())
    duplicated_genes = int(genes_df["ensembl_gene_id"].duplicated().sum())
    chunked_genes = int(genes_df["was_chunked"].astype(bool).sum()) if "was_chunked" in genes_df.columns else 0

    status = "processed_qc_passed"
    if not rows_match or nan_count or inf_count or zero_vectors or duplicated_genes:
        status = "processed_qc_failed"

    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{EMBEDDING_NAME}_embeddings.npz"
    genes_path = out_dir / f"{EMBEDDING_NAME}_genes.tsv"
    meta_path = out_dir / f"{EMBEDDING_NAME}_metadata.json"

    np.savez_compressed(npz_path, embeddings=matrix)
    genes_df.to_csv(genes_path, sep="\t", index=False)

    weight_file = find_model_weight_file(MODEL_DIR)
    weight_sha256 = sha256_file(weight_file) if weight_file else None

    runtime = time.time() - start

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "Orthrus base 4-track",
        "model_source": "quietflamingo/orthrus-base-4-track",
        "model_directory": str(MODEL_DIR),
        "model_weight_file": str(weight_file) if weight_file else None,
        "model_weight_sha256": weight_sha256,
        "official_repo": str(RAW_REPO),
        "official_repo_commit": git_commit(RAW_REPO),
        "embedding_dimension": int(matrix.shape[1]),
        "matrix_shape": list(matrix.shape),
        "input_sequence_type": "mature mRNA full transcript converted U to T for Orthrus one-hot encoding",
        "input_sequence_file": str(SEQ_TSV),
        "pooling_strategy": "Orthrus representation() sequence-level embedding. For transcripts longer than max_full_len, length-weighted mean of chunk-level Orthrus embeddings.",
        "max_full_len": args.max_full_len,
        "fallback_chunk_len": args.fallback_chunk_len,
        "chunked_genes": chunked_genes,
        "master_gene_table": str(MASTER_PATH),
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "runtime_seconds": round(runtime, 2),
    }

    meta_path.write_text(json.dumps(metadata, indent=2))

    report = []
    report.append(f"Mapping/QC report for {EMBEDDING_NAME}")
    report.append("=" * (22 + len(EMBEDDING_NAME)))
    report.append("")
    report.append(f"Status: {status}")
    report.append(f"Output directory: {out_dir}")
    report.append(f"Embeddings: {npz_path}")
    report.append(f"Genes: {genes_path}")
    report.append(f"Metadata: {meta_path}")
    report.append("")
    report.append(f"Model: Orthrus base 4-track")
    report.append(f"Model directory: {MODEL_DIR}")
    report.append(f"Model weight file: {weight_file}")
    report.append(f"Model weight SHA256: {weight_sha256}")
    report.append(f"Input sequences: {SEQ_TSV}")
    report.append("")
    report.append(f"Matrix shape: {matrix.shape}")
    report.append(f"Rows match genes.tsv: {rows_match}")
    report.append(f"NaN count: {nan_count}")
    report.append(f"Inf count: {inf_count}")
    report.append(f"Zero vectors: {zero_vectors}")
    report.append(f"Duplicated Ensembl IDs: {duplicated_genes}")
    report.append(f"Processed genes: {len(genes_df)}")
    report.append(f"Coverage of master table: {100 * len(genes_df) / len(master):.4f}%")
    report.append(f"Chunked genes: {chunked_genes}")
    report.append(f"Runtime seconds: {runtime:.2f}")
    report.append("")

    report_text = "\n".join(report)
    REPORT_PATH.write_text(report_text + "\n")

    print("\nDone")
    print("----")
    print(report_text)


if __name__ == "__main__":
    main()
