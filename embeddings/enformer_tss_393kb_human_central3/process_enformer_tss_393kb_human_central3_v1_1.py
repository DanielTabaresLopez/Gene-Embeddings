#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from pyfaidx import Fasta
from tqdm import tqdm


EMBEDDING_NAME = "Enformer-TSS-393kb-human-central3-log1p-zscore-v1.1"
SEQ_LEN = 393_216
N_TRACKS_HUMAN = 5_313
TARGET_LENGTH = 896
CENTRAL_BIN = TARGET_LENGTH // 2
CENTRAL_SLICE = slice(CENTRAL_BIN - 1, CENTRAL_BIN + 2)


def one_hot_dna(seq: str) -> np.ndarray:
    seq = seq.upper().encode("ascii")
    arr = np.frombuffer(seq, dtype=np.uint8)
    x = np.zeros((len(arr), 4), dtype=np.float32)
    x[arr == ord("A"), 0] = 1.0
    x[arr == ord("C"), 1] = 1.0
    x[arr == ord("G"), 2] = 1.0
    x[arr == ord("T"), 3] = 1.0
    return x


def fetch_centered_sequence(fa: Fasta, chrom: str, center0: int, seq_len: int) -> tuple[str, int, int, int, int]:
    chrom_len = len(fa[chrom])
    start = center0 - seq_len // 2
    end = start + seq_len

    clipped_start = max(0, start)
    clipped_end = min(chrom_len, end)

    pad_left = max(0, -start)
    pad_right = max(0, end - chrom_len)

    seq = ""
    if pad_left:
        seq += "N" * pad_left
    if clipped_end > clipped_start:
        seq += str(fa[chrom][clipped_start:clipped_end]).upper()
    if pad_right:
        seq += "N" * pad_right

    if len(seq) != seq_len:
        raise RuntimeError(f"Bad sequence length for {chrom}:{start}-{end}: got {len(seq)}")

    return seq, start, end, pad_left, pad_right


def zscore_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    z = (x - mean) / std_safe
    return z.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(Path.home() / "metadata/master_gene_table_v1_1_enriched.csv"))
    ap.add_argument(
        "--fasta",
        default=str(
            Path.home()
            / "data/raw_embeddings/deepsea_tss_1kb_v1_1/reference_genome/GRCh38.primary_assembly.genome.fa"
        ),
    )
    ap.add_argument(
        "--outdir",
        default=str(Path.home() / "data/processed_embeddings/enformer_tss_393kb_human_central3_v1_1"),
    )
    ap.add_argument(
        "--report_dir",
        default=str(Path.home() / "reports/mapping_reports/enformer_tss_393kb_human_central3_v1_1"),
    )
    ap.add_argument("--model_url", default="https://tfhub.dev/deepmind/enformer/1")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_genes", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    master_path = Path(args.master)
    fasta_path = Path(args.fasta)
    outdir = Path(args.outdir)
    report_dir = Path(args.report_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding: {EMBEDDING_NAME}")
    print(f"Master: {master_path}")
    print(f"FASTA: {fasta_path}")
    print(f"Output: {outdir}")
    print(f"Model: {args.model_url}")
    print(f"Batch size: {args.batch_size}")

    master = pd.read_csv(master_path, dtype=str).fillna("")
    required = [
        "ensembl_gene_id",
        "gene_symbol",
        "gene_type",
        "chromosome",
        "gene_start",
        "gene_end",
        "strand",
        "assembly",
        "gencode_release",
        "entrez_id",
        "uniprot_id",
    ]
    missing = [c for c in required if c not in master.columns]
    if missing:
        raise RuntimeError(f"Missing master columns: {missing}")

    if args.max_genes is not None:
        master = master.head(args.max_genes).copy()

    fa = Fasta(str(fasta_path))
    chroms = set(fa.keys())

    records = []
    skipped = []

    for _, row in master.iterrows():
        ensg = row["ensembl_gene_id"]
        symbol = row["gene_symbol"]
        chrom = row["chromosome"]
        strand = row["strand"]

        if chrom not in chroms:
            skipped.append((ensg, symbol, chrom, "chromosome_not_in_fasta"))
            continue

        try:
            gene_start = int(float(row["gene_start"]))
            gene_end = int(float(row["gene_end"]))
        except Exception:
            skipped.append((ensg, symbol, chrom, "invalid_coordinates"))
            continue

        if strand == "+":
            tss_1based = gene_start
        elif strand == "-":
            tss_1based = gene_end
        else:
            skipped.append((ensg, symbol, chrom, "invalid_strand"))
            continue

        tss0 = tss_1based - 1

        records.append(
            {
                "ensembl_gene_id": ensg,
                "gene_symbol": symbol,
                "entrez_id": row["entrez_id"],
                "uniprot_id": row["uniprot_id"],
                "gene_type": row["gene_type"],
                "chromosome": chrom,
                "gene_start": gene_start,
                "gene_end": gene_end,
                "strand": strand,
                "tss_1based": tss_1based,
                "tss_0based": tss0,
                "assembly": row["assembly"],
                "gencode_release": row["gencode_release"],
            }
        )

    genes = pd.DataFrame(records)
    skipped_df = pd.DataFrame(skipped, columns=["ensembl_gene_id", "gene_symbol", "chromosome", "reason"])

    if genes.empty:
        raise RuntimeError("No valid genes for Enformer.")

    genes.to_csv(outdir / "genes.tsv", sep="\t", index=False)
    skipped_df.to_csv(report_dir / "skipped_tss_windows.tsv", sep="\t", index=False)

    n = len(genes)
    raw_path = outdir / "raw_enformer_human_central3_mean_tracks.npy"
    log_path = outdir / "raw_enformer_human_central3_log1p_tracks.npy"
    done_path = outdir / "processed_mask.npy"

    if args.resume and raw_path.exists() and done_path.exists():
        raw = np.load(raw_path, mmap_mode="r+")
        done = np.load(done_path)
        if raw.shape != (n, N_TRACKS_HUMAN) or done.shape != (n,):
            raise RuntimeError("Existing resume files have incompatible shape.")
        print(f"Resuming existing matrix: {raw_path}")
    else:
        raw = np.lib.format.open_memmap(
            raw_path,
            mode="w+",
            dtype=np.float32,
            shape=(n, N_TRACKS_HUMAN),
        )
        done = np.zeros(n, dtype=bool)
        np.save(done_path, done)

    print("Loading Enformer...")
    model = hub.load(args.model_url).model
    print("Loaded Enformer.")

    todo = np.where(~done)[0]
    print(f"Genes to process: {len(todo)} / {n}")

    for batch_start in tqdm(range(0, len(todo), args.batch_size)):
        idxs = todo[batch_start : batch_start + args.batch_size]
        batch = []

        for idx in idxs:
            r = genes.iloc[int(idx)]
            seq, seq_start, seq_end, pad_left, pad_right = fetch_centered_sequence(
                fa=fa,
                chrom=r["chromosome"],
                center0=int(r["tss_0based"]),
                seq_len=SEQ_LEN,
            )
            batch.append(one_hot_dna(seq))

        x = tf.convert_to_tensor(np.stack(batch, axis=0), dtype=tf.float32)

        y = model.predict_on_batch(x)

        # We only keep the human head. The mouse head is ignored and never saved.
        human = y["human"].numpy().astype(np.float32)
        central3 = human[:, CENTRAL_SLICE, :].mean(axis=1)

        raw[idxs, :] = central3
        done[idxs] = True

        if done.sum() % args.save_every < args.batch_size:
            raw.flush()
            np.save(done_path, done)

    raw.flush()
    np.save(done_path, done)

    raw_arr = np.array(raw, dtype=np.float32)
    if not np.isfinite(raw_arr).all():
        raise RuntimeError("Raw Enformer matrix contains non-finite values.")

    log_arr = np.log1p(raw_arr).astype(np.float32)
    np.save(log_path, log_arr)

    emb_z, mean, std = zscore_columns(log_arr)
    np.save(outdir / "embeddings.npy", emb_z)
    np.save(outdir / "feature_means_log1p_tracks.npy", mean)
    np.save(outdir / "feature_stds_log1p_tracks.npy", std)

    pd.DataFrame(
        {
            "feature_index": np.arange(N_TRACKS_HUMAN),
            "feature_name": [f"enformer_human_track_{i}" for i in range(N_TRACKS_HUMAN)],
        }
    ).to_csv(outdir / "enformer_human_track_columns.tsv", sep="\t", index=False)

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": "processed_qc_passed",
        "modality": "regulatory DNA / long-range TSS sequence model",
        "model": "Enformer",
        "model_source": args.model_url,
        "kept_output_head": "human only",
        "discarded_output_head": "mouse",
        "human_output_shape_per_sequence": [TARGET_LENGTH, N_TRACKS_HUMAN],
        "pooled_representation": "mean of central 3 human output bins around TSS",
        "raw_output_file": "raw_enformer_human_central3_mean_tracks.npy",
        "transformation": "log1p raw human central3 mean tracks, then dimension-wise z-score across genes",
        "final_embedding_file": "embeddings.npy",
        "input_sequence_type": "GRCh38 genomic DNA, 393216 bp centered on gene TSS",
        "input_sequence_source": str(fasta_path),
        "annotation_source": str(master_path),
        "genome_build": "GRCh38",
        "sequence_length_bp": SEQ_LEN,
        "target_length_bins": TARGET_LENGTH,
        "central_bins_used_0based": [CENTRAL_BIN - 1, CENTRAL_BIN, CENTRAL_BIN + 1],
        "tss_rule": "plus strand: gene_start; minus strand: gene_end; coordinates from master_gene_table_v1_1_enriched.csv",
        "strand_handling": "genomic reference orientation; no gene-strand reverse-complementing",
        "boundary_handling": "N-padding if the 393216 bp window extends beyond chromosome boundaries",
        "n_master_rows_input": int(len(master)),
        "n_valid_tss_windows": int(len(genes)),
        "n_skipped_tss_windows": int(len(skipped_df)),
        "dimension": N_TRACKS_HUMAN,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    report = f"""Mapping/QC report for {EMBEDDING_NAME}
==========================================

Status: processed_qc_passed

Master table: {master_path}
Reference FASTA: {fasta_path}
Output directory: {outdir}

Input master rows: {len(master)}
Valid TSS windows: {len(genes)}
Skipped TSS windows: {len(skipped_df)}

Raw Enformer human central3 matrix:
shape: {raw_arr.shape}
dtype: {raw_arr.dtype}
min: {float(raw_arr.min())}
max: {float(raw_arr.max())}
finite: {np.isfinite(raw_arr).all()}

Log1p matrix:
shape: {log_arr.shape}
dtype: {log_arr.dtype}
finite: {np.isfinite(log_arr).all()}

Final embedding matrix:
shape: {emb_z.shape}
dtype: {emb_z.dtype}
finite: {np.isfinite(emb_z).all()}

Coordinate decision:
- TSS from master table GRCh38 gene coordinates.
- Plus strand TSS = gene_start.
- Minus strand TSS = gene_end.
- 393216 bp sequence centered on TSS.
- Sequences run in genomic reference orientation, not reverse-complemented by gene strand.
- Boundary windows are N-padded if needed.

Final embedding decision:
- Kept Enformer human head only: 5313 tracks.
- Mouse head was discarded.
- Averaged central 3 output bins around the TSS.
- Applied log1p to raw human track predictions.
- Final embeddings.npy contains dimension-wise z-scored log1p tracks across genes.
"""
    with open(outdir / "mapping_report.txt", "w") as f:
        f.write(report)

    print()
    print(report)


if __name__ == "__main__":
    main()
