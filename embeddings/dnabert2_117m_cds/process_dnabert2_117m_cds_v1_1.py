#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import argparse
import gc
import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel


EMBEDDING_NAME = "dnabert2_117m_cds_v1_1"

HOME = Path.home()

MODEL_DIR = HOME / "data/raw_embeddings/dnabert2_117m_cds_v1_1/hf_model"
SEQ_TSV = HOME / "data/raw_embeddings/dnabert2_117m_cds_v1_1/source_sequences/dnabert2_canonical_cds_sequences.tsv"
RAW_REPO = HOME / "data/raw_embeddings/dnabert2_117m_cds_v1_1/official_repo"
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


def detect_column(df, candidates, required=True):
    lower_to_col = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_col:
            return lower_to_col[cand.lower()]
    if required:
        raise RuntimeError(f"Could not detect column. Tried: {candidates}. Available: {list(df.columns)}")
    return None


def extract_hidden(out):
    if isinstance(out, tuple):
        for item in out:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
            if isinstance(item, (tuple, list)):
                for subitem in reversed(item):
                    if torch.is_tensor(subitem) and subitem.ndim == 3:
                        return subitem
        raise RuntimeError("Could not find 3D hidden-state tensor in tuple output.")
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[-1]
    raise RuntimeError("Could not find hidden states.")


def split_sequence(seq: str, chunk_nt: int):
    return [seq[i:i + chunk_nt] for i in range(0, len(seq), chunk_nt)]


def token_len(tokenizer, seq: str):
    return int(tokenizer(seq, return_tensors="pt", truncation=False)["input_ids"].shape[1])


def safe_chunks(tokenizer, seq: str, initial_chunk_nt: int, max_tokens: int):
    seq = seq.upper().replace("U", "T")
    raw_chunks = split_sequence(seq, initial_chunk_nt)

    safe = []
    for chunk in raw_chunks:
        current = chunk
        while token_len(tokenizer, current) > max_tokens:
            if len(current) <= 100:
                raise RuntimeError(
                    f"Could not make safe chunk. Current nt len={len(current)}, token len={token_len(tokenizer, current)}"
                )
            half = len(current) // 2
            left = current[:half]
            right = current[half:]
            raw_chunks.append(right)
            current = left
        safe.append(current)

    return safe


def embed_chunk(model, tokenizer, seq: str, device: str):
    tok = tokenizer(
        seq,
        return_tensors="pt",
        truncation=False,
    )
    tok = {k: v.to(device) for k, v in tok.items()}

    with torch.no_grad():
        out = model(**tok, output_hidden_states=True)

    h = extract_hidden(out)

    # Exclude special tokens using attention_mask and first/last token convention.
    # For DNABERT-2 tokenizer, smoke test produced 254 tokens for 1000 nt, including special tokens.
    attention = tok.get("attention_mask", None)
    if attention is not None:
        mask = attention[0].bool()
        idx = torch.where(mask)[0]
        if len(idx) > 2:
            real_h = h[0, idx[1:-1], :]
        else:
            real_h = h[0, idx, :]
    else:
        real_h = h[0, 1:-1, :] if h.shape[1] > 2 else h[0]

    vec = real_h.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    del tok, out, h, real_h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vec


def embed_sequence(model, tokenizer, seq: str, device: str, chunk_nt: int, max_tokens: int):
    seq = seq.upper().replace("U", "T")

    if token_len(tokenizer, seq) <= max_tokens:
        vec = embed_chunk(model, tokenizer, seq, device)
        return vec, 1, False

    chunks = safe_chunks(tokenizer, seq, chunk_nt, max_tokens)

    vecs = []
    weights = []

    for chunk in chunks:
        vecs.append(embed_chunk(model, tokenizer, chunk, device))
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
    parser.add_argument("--longest", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-nt", type=int, default=1600)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    start = time.time()

    out_dir = OUT_BASE
    if args.limit is not None or args.longest is not None:
        out_dir = OUT_BASE.with_name(OUT_BASE.name + "_TEST")

    partial_path = out_dir / f"{EMBEDDING_NAME}_partial_progress.npz"

    seq_df = pd.read_csv(SEQ_TSV, sep="\t", dtype=str).fillna("")
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")

    gene_col = detect_column(seq_df, ["ensembl_gene_id", "gene_id", "gene"])
    seq_col = detect_column(seq_df, ["cds_sequence", "sequence", "seq", "dna_sequence", "nt_sequence"])
    transcript_col = detect_column(seq_df, ["transcript_id", "canonical_transcript_id"], required=False)

    seq_df["dnabert2_sequence"] = seq_df[seq_col].str.upper().str.replace("U", "T", regex=False)
    seq_df["dnabert2_sequence_length"] = seq_df["dnabert2_sequence"].str.len()

    if args.longest is not None:
        seq_df = seq_df.sort_values("dnabert2_sequence_length", ascending=False).head(args.longest).copy()
    elif args.limit is not None:
        seq_df = seq_df.head(args.limit).copy()

    print("Embedding:", EMBEDDING_NAME)
    print("Model directory:", MODEL_DIR)
    print("Input sequences:", SEQ_TSV)
    print("Detected gene column:", gene_col)
    print("Detected sequence column:", seq_col)
    print("Detected transcript column:", transcript_col)
    print("Genes to process:", len(seq_df))
    print("Device:", args.device)
    print("Chunk nt:", args.chunk_nt)
    print("Max tokens:", args.max_tokens)
    print("Output:", out_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    print("Loading config...")
    config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    original_attention_dropout = getattr(config, "attention_probs_dropout_prob", None)
    config.attention_probs_dropout_prob = 0.1

    print("Original attention_probs_dropout_prob:", original_attention_dropout)
    print("Forced attention_probs_dropout_prob:", config.attention_probs_dropout_prob)

    print("Loading DNABERT-2 model with PyTorch-attention workaround...")
    model = AutoModel.from_pretrained(
        str(MODEL_DIR),
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(args.device)

    model.eval()

    for name, module in model.named_modules():
        if hasattr(module, "p_dropout"):
            print("First module with p_dropout:", name, module.p_dropout)
            break

    embeddings = []
    gene_rows = []
    start_idx = 0

    if args.resume and partial_path.exists():
        embeddings, gene_rows = load_partial(partial_path)
        start_idx = len(gene_rows)
        print(f"Resuming from {start_idx} rows")

    for idx in tqdm(range(start_idx, len(seq_df))):
        row = seq_df.iloc[idx]

        gene_id = row[gene_col]
        transcript_id = row[transcript_col] if transcript_col else ""
        seq = row["dnabert2_sequence"]

        vec, num_chunks, was_chunked = embed_sequence(
            model=model,
            tokenizer=tokenizer,
            seq=seq,
            device=args.device,
            chunk_nt=args.chunk_nt,
            max_tokens=args.max_tokens,
        )

        embeddings.append(vec)

        gene_rows.append({
            "ensembl_gene_id": gene_id,
            "transcript_id": transcript_id,
            "dnabert2_sequence_length": len(seq),
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
        "model": "DNABERT-2 117M",
        "model_source": "zhihan1996/DNABERT-2-117M",
        "model_directory": str(MODEL_DIR),
        "model_weight_file": str(weight_file) if weight_file else None,
        "model_weight_sha256": weight_sha256,
        "embedding_dimension": int(matrix.shape[1]),
        "matrix_shape": list(matrix.shape),
        "input_sequence_type": "canonical protein-coding CDS DNA sequence",
        "input_sequence_file": str(SEQ_TSV),
        "pooling_strategy": "Mean pooling over final-layer non-special token embeddings. Long CDS sequences split into nucleotide chunks, then length-weighted mean of chunk-level embeddings.",
        "chunk_nt": args.chunk_nt,
        "max_tokens": args.max_tokens,
        "chunked_genes": chunked_genes,
        "attention_workaround": {
            "reason": "Official DNABERT-2 custom Triton FlashAttention path hangs on this workstation. Setting attention_probs_dropout_prob=0.1 routes to the PyTorch attention implementation while model.eval() disables stochastic dropout.",
            "original_attention_probs_dropout_prob": original_attention_dropout,
            "forced_attention_probs_dropout_prob": 0.1,
        },
        "official_repo": str(RAW_REPO),
        "official_repo_commit": git_commit(RAW_REPO),
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
    report.append(f"Model: DNABERT-2 117M")
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
    report.append(f"Attention workaround: attention_probs_dropout_prob {original_attention_dropout} -> 0.1, model.eval()")
    report.append(f"Runtime seconds: {runtime:.2f}")
    report.append("")

    report_text = "\n".join(report)
    REPORT_PATH.write_text(report_text + "\n")

    print("\nDone")
    print("----")
    print(report_text)


if __name__ == "__main__":
    main()
