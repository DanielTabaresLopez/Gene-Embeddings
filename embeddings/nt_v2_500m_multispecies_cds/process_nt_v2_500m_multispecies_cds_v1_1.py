from pathlib import Path
from datetime import datetime
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


EMBEDDING_NAME = "nt_v2_500m_multispecies_cds_v1_1"
DIM = 1024


def clean_seq(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return re.sub(r"[^ACGTN]", "N", seq)


def chunk_sequence(seq: str, chunk_nt: int):
    seq = clean_seq(seq)
    for i in range(0, len(seq), chunk_nt):
        chunk = seq[i:i + chunk_nt]
        if chunk:
            yield chunk


def embed_chunk(chunk, tokenizer, model, device):
    enc = tokenizer(
        chunk,
        return_tensors="pt",
    )

    model_inputs = {
        k: v.to(device)
        for k, v in enc.items()
        if k in ["input_ids", "attention_mask"]
    }

    with torch.no_grad():
        out = model(**model_inputs, output_hidden_states=True, return_dict=True)

    input_ids = model_inputs["input_ids"][0]
    attention_mask = model_inputs["attention_mask"][0].bool()
    last_hidden = out.hidden_states[-1][0]  # tokens x dim

    # Do not use tokenizer special_tokens_mask here, because this NT tokenizer
    # can return a mask with a different length than input_ids.
    valid_mask = attention_mask.clone()

    special_ids = set()
    for attr in [
        "pad_token_id",
        "cls_token_id",
        "sep_token_id",
        "bos_token_id",
        "eos_token_id",
        "mask_token_id",
    ]:
        sid = getattr(tokenizer, attr, None)
        if sid is not None:
            special_ids.add(int(sid))

    for sid in special_ids:
        valid_mask = valid_mask & (input_ids != sid)

    valid_hidden = last_hidden[valid_mask]

    if valid_hidden.shape[0] == 0:
        raise RuntimeError("No valid tokens found for a chunk.")

    summed = valid_hidden.float().sum(dim=0)
    count = int(valid_hidden.shape[0])

    return summed, count

def embed_sequence(seq, tokenizer, model, device, chunk_nt):
    total = None
    total_count = 0

    for chunk in chunk_sequence(seq, chunk_nt):
        chunk_sum, chunk_count = embed_chunk(chunk, tokenizer, model, device)
        if total is None:
            total = chunk_sum
        else:
            total += chunk_sum
        total_count += chunk_count

    if total is None or total_count == 0:
        raise RuntimeError("No valid tokens found for sequence.")

    vec = (total / total_count).detach().cpu().numpy().astype(np.float32)

    if vec.shape[0] != DIM:
        raise RuntimeError(f"Unexpected embedding dimension: {vec.shape[0]} != {DIM}")

    return vec, total_count


def embed_sequence_with_backoff(seq, tokenizer, model, device, chunk_nt):
    current_chunk_nt = chunk_nt

    while True:
        try:
            return embed_sequence(seq, tokenizer, model, device, current_chunk_nt), current_chunk_nt
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_chunk_nt <= 750:
                    raise
                current_chunk_nt = current_chunk_nt // 2
                print(f"CUDA memory issue. Retrying with chunk_nt={current_chunk_nt}", flush=True)
            else:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process only first N mapped genes for testing.")
    parser.add_argument("--chunk-nt", type=int, default=6000, help="Nucleotides per chunk before tokenization.")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    home = Path.home()
    raw_dir = home / "data/raw_embeddings/nt_v2_500m_multispecies_cds_v1_1"
    model_dir = raw_dir / "hf_model"
    seq_dir = raw_dir / "source_sequences"
    fasta_path = seq_dir / "nt_v2_canonical_cds_sequences.fa"
    sequence_report_path = home / "reports/mapping_reports/nt_v2_500m_multispecies_cds_v1_1_sequence_mapping_report.tsv"
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
    print("Model directory:", model_dir)
    print("FASTA:", fasta_path)
    print("Sequence report:", sequence_report_path)

    print("\nLoading mapped sequence report...")
    report = pd.read_csv(sequence_report_path, sep="\t", dtype=str).fillna("")
    mapped = report[report["status"] == "mapped"].copy()
    mapped["cds_length_nt"] = mapped["cds_length_nt"].astype(int)

    if args.limit is not None:
        mapped = mapped.head(args.limit).copy()

    print("Mapped rows to embed:", len(mapped))

    print("\nLoading FASTA sequences...")
    seq_by_ensg = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        parts = rec.id.split("|")
        ensg = parts[0]
        seq_by_ensg[ensg] = clean_seq(str(rec.seq))

    missing_seq = [x for x in mapped["ensembl_gene_id"] if x not in seq_by_ensg]
    if missing_seq:
        raise RuntimeError(f"Mapped genes missing from FASTA: {len(missing_seq)} examples={missing_seq[:5]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nDevice:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Keep fp32. fp16 caused dtype mismatch in this custom NT model on this workstation.
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
    ).to(device)
    model = model.float()
    model.eval()

    n = len(mapped)

    if args.resume and partial_path.exists():
        print("Resuming from:", partial_path)
        partial = np.load(partial_path, allow_pickle=False)
        embeddings = partial["embeddings"]
        done_mask = partial["done_mask"].astype(bool)
        token_counts = partial["token_counts"]
        effective_chunk_nt = partial["effective_chunk_nt"]
    else:
        embeddings = np.zeros((n, DIM), dtype=np.float32)
        done_mask = np.zeros(n, dtype=bool)
        token_counts = np.zeros(n, dtype=np.int32)
        effective_chunk_nt = np.zeros(n, dtype=np.int32)

    start = time.time()

    for i, row in tqdm(list(mapped.reset_index(drop=True).iterrows()), total=n):
        if done_mask[i]:
            continue

        ensg = row["ensembl_gene_id"]
        seq = seq_by_ensg[ensg]

        try:
            (vec, ntokens), used_chunk_nt = embed_sequence_with_backoff(
                seq=seq,
                tokenizer=tokenizer,
                model=model,
                device=device,
                chunk_nt=args.chunk_nt,
            )
        except Exception as e:
            print(f"\nFAILED at index={i}, gene={ensg}, symbol={row.get('gene_symbol', '')}, len={len(seq)}")
            raise e

        embeddings[i, :] = vec
        done_mask[i] = True
        token_counts[i] = ntokens
        effective_chunk_nt[i] = used_chunk_nt

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % args.checkpoint_every == 0:
            np.savez_compressed(
                partial_path,
                embeddings=embeddings,
                done_mask=done_mask,
                token_counts=token_counts,
                effective_chunk_nt=effective_chunk_nt,
            )
            print(f"\nCheckpoint saved after {i + 1} rows -> {partial_path}", flush=True)

    elapsed = time.time() - start

    if not done_mask.all():
        raise RuntimeError("Not all rows were completed.")

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
            "cds_length_nt",
            "mapping_method",
        ]
    ].copy()

    genes["nt_v2_token_count_used_for_mean_pooling"] = token_counts
    genes["effective_chunk_nt"] = effective_chunk_nt

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
        "modality": "DNA language model / CDS sequence",
        "species": "human",
        "master_table": str(master_path),
        "raw_model_local_path": str(model_dir),
        "raw_sequence_fasta": str(fasta_path),
        "sequence_mapping_report": str(sequence_report_path),
        "download_source": "official HuggingFace model repository",
        "model_checkpoint": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "original_model_or_method": "Nucleotide Transformer v2",
        "embedding_generated_by": "Daniel locally by model inference",
        "input_sequence_type": "CDS DNA",
        "input_sequence_source": "Ensembl",
        "input_sequence_release": "Ensembl release 114",
        "genome_build": "GRCh38",
        "transcript_choice": "canonical transcript from master_gene_table_v1_1_enriched.csv",
        "pooling_strategy": "final hidden layer mean pooling over non-special, non-padding tokens; chunk-weighted by token count for long CDS",
        "sequence_reference_status": "verified against Ensembl release 114 CDS FASTA by canonical transcript ID",
        "embedding_dimension": DIM,
        "dtype": "float32",
        "chunk_nt_requested": args.chunk_nt,
        "raw_master_rows": int(len(pd.read_csv(master_path, dtype=str))),
        "mapped_cds_genes_available": int((report["status"] == "mapped").sum()),
        "processed_gene_count": int(len(genes)),
        "missing_cds_genes": int((report["status"] != "mapped").sum()),
        "coverage_of_master_table_percent": round(int((report["status"] == "mapped").sum()) / len(report) * 100, 4),
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
            "The official pretrained NT-v2 checkpoint was used only for inference.",
            "12 master-table genes lacked canonical CDS sequence in the selected Ensembl CDS FASTA and were excluded from the processed matrix.",
            "Model was run in fp32 because fp16 caused dtype mismatch in the custom NT model code on boevastudent03.",
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
        f.write(f"Missing CDS genes: {metadata['missing_cds_genes']}\n")
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
