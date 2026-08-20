#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta


EMBEDDING_NAME = "DeepSEA-TSS-1kb-epigenomic-probabilities-zscore-v1.1"


def zscore_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    z = (x - mean) / std_safe
    return z.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def read_predictions_tsv(path: Path, expected_rows: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    # Kipoi normally writes a headered TSV. This fallback also handles headerless TSV.
    pred = pd.read_csv(path, sep="\t")
    if len(pred) != expected_rows:
        pred = pd.read_csv(path, sep="\t", header=None)

    numeric = pred.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [c for c in numeric.columns if not numeric[c].isna().all()]
    numeric = numeric[numeric_cols]

    if numeric.shape[1] < 919:
        raise RuntimeError(
            f"Expected at least 919 numeric DeepSEA output columns, found {numeric.shape[1]} in {path}"
        )

    # In case Kipoi adds metadata columns, the 919 prediction columns are the final numeric block.
    numeric = numeric.iloc[:, -919:]
    x = numeric.to_numpy(dtype=np.float32)

    if x.shape != (expected_rows, 919):
        raise RuntimeError(f"Prediction matrix has shape {x.shape}, expected ({expected_rows}, 919)")

    labels = [str(c) for c in numeric.columns]
    return pred, x, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--master",
        default=str(Path.home() / "metadata/master_gene_table_v1_1_enriched.csv"),
    )
    ap.add_argument(
        "--fasta",
        default=str(
            Path.home()
            / "data/raw_embeddings/deepsea_tss_1kb_v1_1/reference_genome/GRCh38.primary_assembly.genome.fa"
        ),
    )
    ap.add_argument(
        "--outdir",
        default=str(Path.home() / "data/processed_embeddings/deepsea_tss_1kb_epigenomic_probs_v1_1"),
    )
    ap.add_argument(
        "--report_dir",
        default=str(Path.home() / "reports/mapping_reports/deepsea_tss_1kb_epigenomic_probs_v1_1"),
    )
    ap.add_argument("--max_genes", type=int, default=None, help="Optional smoke-test subset.")
    ap.add_argument("--reuse_predictions", action="store_true", help="Skip Kipoi if prediction TSV already exists.")
    args = ap.parse_args()

    master_path = Path(args.master)
    fasta_path = Path(args.fasta)
    outdir = Path(args.outdir)
    report_dir = Path(args.report_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    bed_path = outdir / "deepsea_tss_1kb_windows.bed"
    pred_tsv = outdir / "deepsea_tss_1kb_predictions.tsv"

    print(f"Embedding: {EMBEDDING_NAME}")
    print(f"Master: {master_path}")
    print(f"FASTA: {fasta_path}")
    print(f"Output: {outdir}")

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
        raise RuntimeError(f"Master table missing required columns: {missing}")

    fa = Fasta(str(fasta_path))
    chrom_lengths = {k: len(fa[k]) for k in fa.keys()}

    records = []
    skipped = []

    df = master.copy()
    if args.max_genes is not None:
        df = df.head(args.max_genes).copy()

    for i, row in df.iterrows():
        ensg = row["ensembl_gene_id"]
        symbol = row["gene_symbol"]
        chrom = row["chromosome"]
        strand = row["strand"]

        if chrom not in chrom_lengths:
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

        # BED is 0-based half-open. TSS is converted from 1-based to 0-based.
        tss0 = tss_1based - 1
        bed_start = tss0 - 500
        bed_end = tss0 + 500

        if bed_start < 0 or bed_end > chrom_lengths[chrom]:
            skipped.append((ensg, symbol, chrom, "window_out_of_bounds"))
            continue

        name = f"{ensg}|{symbol}"
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
                "bed_start_0based": bed_start,
                "bed_end_0based": bed_end,
                "bed_name": name,
                "assembly": row["assembly"],
                "gencode_release": row["gencode_release"],
            }
        )

    genes = pd.DataFrame(records)
    skipped_df = pd.DataFrame(skipped, columns=["ensembl_gene_id", "gene_symbol", "chromosome", "reason"])

    if genes.empty:
        raise RuntimeError("No valid TSS windows were generated.")

    # Kipoi/kipoiseq DeepSEA expects interval columns only when ignore_targets=True.
    # Any extra BED columns are interpreted as numeric labels, so we write BED3 here.
    # Gene identifiers and strand are preserved in genes.tsv in the same row order.
    with open(bed_path, "w") as f:
        for r in records:
            f.write(
                f'{r["chromosome"]}\t{r["bed_start_0based"]}\t{r["bed_end_0based"]}\n'
            )

    genes.to_csv(outdir / "genes.tsv", sep="\t", index=False)
    skipped_df.to_csv(report_dir / "skipped_tss_windows.tsv", sep="\t", index=False)

    print(f"Valid TSS windows: {len(genes)}")
    print(f"Skipped windows: {len(skipped_df)}")
    print(f"Wrote BED: {bed_path}")

    if (not pred_tsv.exists()) or (not args.reuse_predictions):
        dl_args = {
            "intervals_file": str(bed_path),
            "fasta_file": str(fasta_path),
            "ignore_targets": True,
            "use_strand": False,
        }

        cmd = [
            "kipoi",
            "predict",
            "DeepSEA/predict",
            "--source=kipoi",
            "--dataloader_args",
            json.dumps(dl_args),
            "-o",
            str(pred_tsv),
        ]

        print("Running Kipoi DeepSEA prediction:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"Reusing existing predictions: {pred_tsv}")

    pred_raw, probs, labels = read_predictions_tsv(pred_tsv, expected_rows=len(genes))

    if not np.isfinite(probs).all():
        raise RuntimeError("DeepSEA probability matrix contains non-finite values.")

    emb_z, mean, std = zscore_columns(probs)

    np.save(outdir / "raw_deepsea_probabilities.npy", probs.astype(np.float32))
    np.save(outdir / "embeddings.npy", emb_z)
    np.save(outdir / "feature_means_raw_probabilities.npy", mean)
    np.save(outdir / "feature_stds_raw_probabilities.npy", std)

    pd.DataFrame(
        {
            "feature_index": np.arange(919),
            "feature_name_from_kipoi_output": labels,
        }
    ).to_csv(outdir / "deepsea_feature_columns.tsv", sep="\t", index=False)

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": "processed_qc_passed",
        "modality": "regulatory DNA / promoter-TSS sequence model",
        "model": "DeepSEA/predict from Kipoi",
        "model_source": "Kipoi model zoo; DeepSEA converted PyTorch model",
        "raw_output": "919 DeepSEA epigenomic probabilities per TSS-centered 1 kb sequence",
        "final_embedding": "dimension-wise z-scored DeepSEA probabilities across mapped genes",
        "input_sequence_type": "genomic DNA TSS-centered 1 kb window",
        "input_sequence_source": str(fasta_path),
        "genome_build": "GRCh38",
        "annotation_source": str(master_path),
        "tss_rule": "plus strand: gene_start; minus strand: gene_end; coordinates from master_gene_table_v1_1_enriched.csv",
        "window_rule": "BED interval [TSS0 - 500, TSS0 + 500), where TSS0 is 0-based TSS",
        "strand_handling": "use_strand=False; sequences are genomic plus-reference orientation, not gene-oriented reverse-complemented",
        "dimension": 919,
        "n_master_rows_input": int(len(master)),
        "n_valid_tss_windows": int(len(genes)),
        "n_skipped_tss_windows": int(len(skipped_df)),
        "output_files": {
            "embeddings": "embeddings.npy",
            "raw_probabilities": "raw_deepsea_probabilities.npy",
            "genes": "genes.tsv",
            "bed": "deepsea_tss_1kb_windows.bed",
            "predictions_tsv": "deepsea_tss_1kb_predictions.tsv",
            "feature_columns": "deepsea_feature_columns.tsv",
            "skipped_windows": str(report_dir / "skipped_tss_windows.tsv"),
        },
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

Embedding matrix:
shape: {emb_z.shape}
dtype: {emb_z.dtype}
finite: {np.isfinite(emb_z).all()}

Raw DeepSEA probability matrix:
shape: {probs.shape}
min: {float(np.min(probs))}
max: {float(np.max(probs))}
finite: {np.isfinite(probs).all()}

Coordinate decision:
- TSS from master table GRCh38 gene coordinates.
- Plus strand TSS = gene_start.
- Minus strand TSS = gene_end.
- BED windows are 1 kb, centered on TSS.
- DeepSEA was run in genomic reference orientation, not reverse-complemented by gene strand.

Final embedding decision:
- Saved raw DeepSEA probabilities separately.
- Final embeddings.npy contains dimension-wise z-scored probabilities across genes.
"""
    with open(outdir / "mapping_report.txt", "w") as f:
        f.write(report)

    print()
    print(report)


if __name__ == "__main__":
    main()
