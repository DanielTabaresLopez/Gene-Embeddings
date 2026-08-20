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
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


EMBEDDING_NAME = "hyenadna_medium_160k_cds_v1_1"
DIM = 256


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def clean_dna(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in seq)


def get_special_token_ids(tokenizer):
    special_ids = set()
    # Do not remove unk_token_id, because ambiguous N may map to unknown-like tokens.
    for attr in [
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "cls_token_id",
        "sep_token_id",
        "mask_token_id",
    ]:
        value = getattr(tokenizer, attr, None)
        if value is not None:
            special_ids.add(int(value))
    return special_ids


def embed_chunk(seq, tokenizer, model, device, special_token_ids):
    seq = clean_dna(seq)

    tok = tokenizer(
        seq,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )

    input_ids = tok["input_ids"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        h = out.last_hidden_state
    elif hasattr(out, "hidden_states") and out.hidden_states is not None:
        h = out.hidden_states[-1]
    else:
        raise RuntimeError("Could not find hidden states in HyenaDNA output.")

    if h.ndim != 3 or h.shape[0] != 1:
        raise RuntimeError(f"Unexpected hidden-state shape: {tuple(h.shape)}")

    if h.shape[-1] != DIM:
        raise RuntimeError(f"Unexpected embedding dimension: {h.shape[-1]} != {DIM}")

    ids = input_ids[0].detach().cpu().numpy()
    mask = np.ones(ids.shape[0], dtype=bool)
    for sid in special_token_ids:
        mask &= ids != sid

    if mask.sum() == 0:
        raise RuntimeError("No non-special tokens found after tokenization.")

    # Align hidden length and input_ids length defensively.
    h_cpu = h[0].detach().cpu().numpy().astype(np.float32)
    if h_cpu.shape[0] != mask.shape[0]:
        min_len = min(h_cpu.shape[0], mask.shape[0])
        h_cpu = h_cpu[:min_len]
        mask = mask[:min_len]

    token_emb = h_cpu[mask, :]

    if token_emb.shape[0] == 0:
        raise RuntimeError("No token embeddings left after special-token filtering.")

    del tok, input_ids, out, h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return token_emb


def embed_piece_with_backoff(seq, tokenizer, model, device, special_token_ids, min_chunk_len):
    try:
        token_emb = embed_chunk(seq, tokenizer, model, device, special_token_ids)
        return token_emb.sum(axis=0).astype(np.float64), int(token_emb.shape[0]), 1
    except RuntimeError as e:
        msg = str(e).lower()
        is_oom = "out of memory" in msg or "cuda" in msg and "memory" in msg

        if is_oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        if is_oom and len(seq) > min_chunk_len:
            mid = len(seq) // 2
            left_sum, left_n, left_chunks = embed_piece_with_backoff(
                seq[:mid], tokenizer, model, device, special_token_ids, min_chunk_len
            )
            right_sum, right_n, right_chunks = embed_piece_with_backoff(
                seq[mid:], tokenizer, model, device, special_token_ids, min_chunk_len
            )
            return left_sum + right_sum, left_n + right_n, left_chunks + right_chunks

        raise


def embed_sequence(seq, tokenizer, model, device, special_token_ids, chunk_len, min_chunk_len):
    seq = clean_dna(seq)

    total = np.zeros(DIM, dtype=np.float64)
    total_tokens = 0
    total_chunks = 0

    for start in range(0, len(seq), chunk_len):
        piece = seq[start:start + chunk_len]
        piece_sum, piece_tokens, piece_chunks = embed_piece_with_backoff(
            piece,
            tokenizer,
            model,
            device,
            special_token_ids,
            min_chunk_len,
        )
        total += piece_sum
        total_tokens += piece_tokens
        total_chunks += piece_chunks

    if total_tokens == 0:
        raise RuntimeError("No valid tokens embedded for sequence.")

    return (total / total_tokens).astype(np.float32), total_tokens, total_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-len", type=int, default=12000)
    parser.add_argument("--min-chunk-len", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    home = Path.home()

    model_dir = home / "data/raw_embeddings/hyenadna_medium_160k_cds_v1_1/hf_model"
    fasta_path = home / "data/raw_embeddings/nt_v2_500m_multispecies_cds_v1_1/source_sequences/nt_v2_canonical_cds_sequences.fa"
    sequence_report_path = home / "reports/mapping_reports/nt_v2_500m_multispecies_cds_v1_1_sequence_mapping_report.tsv"
    master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"

    if args.limit is None:
        out_dir = home / "data/processed_embeddings" / EMBEDDING_NAME
        out_prefix = EMBEDDING_NAME
    else:
        out_dir = home / "data/processed_embeddings" / f"{EMBEDDING_NAME}_TEST"
        out_prefix = f"{EMBEDDING_NAME}_TEST"

    out_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = home / "reports/mapping_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{out_prefix}_embeddings.npz"
    out_genes = out_dir / f"{out_prefix}_genes.tsv"
    out_metadata = out_dir / f"{out_prefix}_metadata.json"
    out_report = reports_dir / f"{out_prefix}_mapping_report.txt"
    partial_path = out_dir / f"{out_prefix}_partial_progress.npz"

    print("Embedding name:", EMBEDDING_NAME)
    print("Model directory:", model_dir)
    print("Input FASTA:", fasta_path)
    print("Sequence report:", sequence_report_path)

    print("\nLoading NT CDS sequence report...")
    report = pd.read_csv(sequence_report_path, sep="\t", dtype=str).fillna("")
    mapped = report[report["status"].str.startswith("mapped")].copy()

    if args.limit is not None:
        mapped = mapped.head(args.limit).copy()

    print("Rows to embed:", len(mapped))

    print("\nLoading FASTA sequences...")
    seq_by_ensg = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        ensg = rec.id.split("|")[0]
        seq_by_ensg[ensg] = clean_dna(str(rec.seq))

    missing_fasta = [x for x in mapped["ensembl_gene_id"] if x not in seq_by_ensg]
    if missing_fasta:
        raise RuntimeError(f"Mapped genes missing from FASTA: {len(missing_fasta)} examples={missing_fasta[:5]}")

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda but torch.cuda.is_available() is False")
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nDevice:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading HyenaDNA tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    special_token_ids = get_special_token_ids(tokenizer)
    print("Special token IDs removed during pooling:", sorted(special_token_ids))
    print("Tokenizer vocab size:", getattr(tokenizer, "vocab_size", "unknown"))

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

    start_time = time.time()

    mapped_reset = mapped.reset_index(drop=True)

    for i, row in tqdm(list(mapped_reset.iterrows()), total=n):
        if done_mask[i]:
            continue

        ensg = row["ensembl_gene_id"]
        seq = seq_by_ensg[ensg]

        try:
            vec, ntokens, nchunks = embed_sequence(
                seq,
                tokenizer,
                model,
                device,
                special_token_ids,
                args.chunk_len,
                args.min_chunk_len,
            )
        except Exception as e:
            print(
                f"\nFAILED index={i}, ensembl_gene_id={ensg}, "
                f"gene_symbol={row.get('gene_symbol', '')}, cds_len={len(seq)}"
            )
            raise e

        embeddings[i, :] = vec
        done_mask[i] = True
        token_counts[i] = ntokens
        chunk_counts[i] = nchunks

        if (i + 1) % args.checkpoint_every == 0:
            np.savez_compressed(
                partial_path,
                embeddings=embeddings,
                done_mask=done_mask,
                token_counts=token_counts,
                chunk_counts=chunk_counts,
            )
            print(f"\nCheckpoint saved after {i + 1} rows -> {partial_path}", flush=True)

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - start_time

    if not done_mask.all():
        raise RuntimeError("Not all rows completed.")

    keep_cols = [
        "master_row_index",
        "ensembl_gene_id",
        "gene_symbol",
        "entrez_id",
        "uniprot_id",
        "canonical_transcript_id",
        "canonical_transcript_id_versioned",
        "source_transcript_id_used",
        "cds_length_nt",
        "mapping_method",
        "status",
    ]
    keep_cols = [c for c in keep_cols if c in mapped_reset.columns]

    genes = mapped_reset[keep_cols].copy()
    genes["hyenadna_token_count_used_for_mean_pooling"] = token_counts
    genes["hyenadna_chunk_count"] = chunk_counts

    np.savez_compressed(out_npz, embeddings=embeddings)
    genes.to_csv(out_genes, sep="\t", index=False)

    norms = np.linalg.norm(embeddings, axis=1)
    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    zero_vectors = int((norms == 0).sum())
    duplicated_ensembl = int(genes["ensembl_gene_id"].duplicated().sum())

    status = (
        "processed_qc_passed"
        if (
            embeddings.shape[0] == len(genes)
            and embeddings.shape[1] == DIM
            and nan_count == 0
            and inf_count == 0
            and zero_vectors == 0
            and duplicated_ensembl == 0
        )
        else "processed_qc_check_needed"
    )

    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_file = model_dir / "model.safetensors"

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modality": "DNA language model / canonical CDS sequence",
        "species": "human",
        "source_model": "LongSafari/hyenadna-medium-160k-seqlen-hf",
        "source_model_local_dir": str(model_dir),
        "model_file_sha256": sha256sum(model_file) if model_file.exists() else "",
        "model_config_path": str(config_path),
        "raw_sequence_fasta": str(fasta_path),
        "sequence_mapping_report": str(sequence_report_path),
        "master_table": str(master_path),
        "download_source": "official HuggingFace repository LongSafari/hyenadna-medium-160k-seqlen-hf",
        "original_model_or_method": "HyenaDNA medium 160k sequence-length checkpoint",
        "embedding_generated_by": "Daniel locally by model inference",
        "input_sequence_type": "canonical CDS DNA sequence",
        "input_sequence_source": "canonical CDS FASTA generated for NT-v2 workflow from Ensembl release 114 GRCh38 CDS",
        "input_sequence_release": "Ensembl release 114",
        "genome_build": "GRCh38",
        "transcript_choice": "canonical transcript from master_gene_table_v1_1_enriched.csv",
        "pooling_strategy": "final hidden-state mean pooling over non-special nucleotide tokens; long CDS split into chunks and token-count weighted",
        "chunk_len_requested": args.chunk_len,
        "min_chunk_len_for_oom_backoff": args.min_chunk_len,
        "embedding_dimension": DIM,
        "dtype": "float32",
        "model_config_summary": {
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "d_model": config.get("d_model"),
            "hidden_size": config.get("hidden_size"),
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "max_seq_len": config.get("max_seq_len"),
            "num_layers": config.get("num_layers"),
        },
        "notes": [
            "No model training or fine-tuning was performed.",
            "The same canonical CDS sequences prepared for NT-v2 were reused to ensure comparability.",
            "Special tokens were removed before mean pooling.",
            "The tokenizer may add one extra special token; this is excluded by token-id filtering.",
            "If CUDA out-of-memory occurs, chunks are recursively split until min_chunk_len.",
        ],
        "processed_gene_count": int(len(genes)),
        "matrix_shape": list(embeddings.shape),
        "runtime_seconds": round(elapsed, 2),
        "qc": {
            "rows_match_genes_tsv": bool(embeddings.shape[0] == len(genes)),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_vectors": zero_vectors,
            "duplicated_ensembl_ids": duplicated_ensembl,
            "min_norm": float(norms.min()),
            "median_norm": float(np.median(norms)),
            "max_norm": float(norms.max()),
        },
    }

    with open(out_metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(out_report, "w") as f:
        f.write(f"Mapping/QC report for {out_prefix}\n")
        f.write("=" * (22 + len(out_prefix)) + "\n\n")
        f.write(f"Status: {status}\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"Embeddings: {out_npz}\n")
        f.write(f"Genes: {out_genes}\n")
        f.write(f"Metadata: {out_metadata}\n\n")
        f.write(f"Model: LongSafari/hyenadna-medium-160k-seqlen-hf\n")
        f.write(f"Model file SHA256: {metadata['model_file_sha256']}\n")
        f.write(f"Input FASTA: {fasta_path}\n")
        f.write(f"Processed matrix shape: {embeddings.shape}\n")
        f.write(f"Rows match genes.tsv: {embeddings.shape[0] == len(genes)}\n")
        f.write(f"NaN count: {nan_count}\n")
        f.write(f"Inf count: {inf_count}\n")
        f.write(f"Zero vectors: {zero_vectors}\n")
        f.write(f"Duplicated Ensembl IDs: {duplicated_ensembl}\n")
        f.write(f"Processed genes: {len(genes)}\n")
        f.write(f"Runtime seconds: {metadata['runtime_seconds']}\n")

    print("\nDone")
    print("----")
    print("Status:", status)
    print("Matrix shape:", embeddings.shape)
    print("Rows match genes.tsv:", embeddings.shape[0] == len(genes))
    print("NaN count:", nan_count)
    print("Inf count:", inf_count)
    print("Zero vectors:", zero_vectors)
    print("Duplicated Ensembl IDs:", duplicated_ensembl)
    print("Processed genes:", len(genes))
    print("Output:", out_dir)
    print("Report:", out_report)


if __name__ == "__main__":
    main()
