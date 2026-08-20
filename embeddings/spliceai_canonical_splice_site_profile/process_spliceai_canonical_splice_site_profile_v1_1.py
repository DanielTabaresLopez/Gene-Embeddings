#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from datetime import datetime
from pathlib import Path

# By default, do not use GPU. This lets Enformer keep using the GPU.
if os.environ.get("SPLICEAI_USE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm import tqdm

from pkg_resources import resource_filename
from tensorflow.keras.models import load_model


EMBEDDING_NAME = "SpliceAI-canonical-splice-site-profile-zscore-v1.1"

CONTEXT = 10_000
CONTEXT_FLANK = CONTEXT // 2
PROFILE_RADIUS = 50
PROFILE_WIDTH = 2 * PROFILE_RADIUS + 1

# Final vector:
# all_sites acceptor profile 101
# all_sites donor profile 101
# donor_sites acceptor profile 101
# donor_sites donor profile 101
# acceptor_sites acceptor profile 101
# acceptor_sites donor profile 101
N_FEATURES = 6 * PROFILE_WIDTH


def parse_attrs(attr_string: str) -> dict[str, str]:
    attrs = {}
    for key, val in re.findall(r'(\S+) "([^"]*)"', attr_string):
        attrs[key] = val
    return attrs


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def one_hot_dna(seq: str) -> np.ndarray:
    arr = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    x = np.zeros((len(arr), 4), dtype=np.float32)
    x[arr == ord("A"), 0] = 1.0
    x[arr == ord("C"), 1] = 1.0
    x[arr == ord("G"), 2] = 1.0
    x[arr == ord("T"), 3] = 1.0
    return x


def fetch_padded_sequence(fa: Fasta, chrom: str, start0: int, end0: int) -> tuple[str, int, int]:
    chrom_len = len(fa[chrom])

    pad_left = max(0, -start0)
    pad_right = max(0, end0 - chrom_len)

    s0 = max(0, start0)
    e0 = min(chrom_len, end0)

    seq = "N" * pad_left
    if e0 > s0:
        seq += str(fa[chrom][s0:e0]).upper()
    seq += "N" * pad_right

    expected = end0 - start0
    if len(seq) != expected:
        raise RuntimeError(f"Bad padded sequence length for {chrom}:{start0}-{end0}: {len(seq)} vs {expected}")

    return seq, pad_left, pad_right


def make_spliceai_input_sequence(fa: Fasta, chrom: str, site0: int, strand: str) -> str:
    # SpliceAI custom-sequence convention:
    # input = N/context_left + output_region + N/context_right
    # Here output_region is 101 bp centered on the splice-site coordinate.
    output_start0 = site0 - PROFILE_RADIUS
    output_end0 = site0 + PROFILE_RADIUS + 1

    input_start0 = output_start0 - CONTEXT_FLANK
    input_end0 = output_end0 + CONTEXT_FLANK

    seq, _, _ = fetch_padded_sequence(fa, chrom, input_start0, input_end0)

    if strand == "-":
        seq = reverse_complement(seq)

    expected = CONTEXT + PROFILE_WIDTH
    if len(seq) != expected:
        raise RuntimeError(f"Input sequence length {len(seq)} != {expected}")

    return seq


def parse_gtf_exons(gtf_path: Path, wanted_transcripts: set[str]) -> dict[str, list[dict]]:
    exons_by_tx: dict[str, list[dict]] = {}

    with gzip.open(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            chrom, source, feature, start, end, score, strand, frame, attrs_raw = fields
            if feature != "exon":
                continue

            attrs = parse_attrs(attrs_raw)
            tx_id = attrs.get("transcript_id", "")
            tx_id_unversioned = tx_id.split(".")[0]

            if tx_id not in wanted_transcripts and tx_id_unversioned not in wanted_transcripts:
                continue

            exon = {
                "chromosome": chrom,
                "start": int(start),
                "end": int(end),
                "strand": strand,
                "exon_number": int(attrs.get("exon_number", "0")),
                "transcript_id": tx_id,
                "transcript_id_unversioned": tx_id_unversioned,
            }

            # Store under both versioned and unversioned IDs for robust lookup.
            for key in {tx_id, tx_id_unversioned}:
                exons_by_tx.setdefault(key, []).append(exon)

    return exons_by_tx


def canonical_sites_for_gene(row: pd.Series, exons_by_tx: dict[str, list[dict]]) -> tuple[list[dict], str]:
    tx_v = str(row.get("canonical_transcript_id_versioned", "")).strip()
    tx = str(row.get("canonical_transcript_id", "")).strip()
    strand = row["strand"]
    chrom = row["chromosome"]

    exons = None
    if tx_v and tx_v in exons_by_tx:
        exons = exons_by_tx[tx_v]
    elif tx and tx in exons_by_tx:
        exons = exons_by_tx[tx]

    if not exons:
        return [], "canonical_transcript_exons_not_found"

    # Deduplicate possible duplicate storage/records.
    seen = set()
    uniq = []
    for e in exons:
        key = (e["chromosome"], e["start"], e["end"], e["strand"], e["exon_number"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)

    exons = [e for e in uniq if e["chromosome"] == chrom]
    if len(exons) < 2:
        return [], "single_exon_or_no_junctions"

    if strand == "+":
        exons = sorted(exons, key=lambda e: (e["start"], e["end"]))
        donors = [e["end"] for e in exons[:-1]]
        acceptors = [e["start"] for e in exons[1:]]
    elif strand == "-":
        # Transcript order is high-to-low genomic coordinate.
        exons = sorted(exons, key=lambda e: (e["start"], e["end"]), reverse=True)
        donors = [e["start"] for e in exons[:-1]]
        acceptors = [e["end"] for e in exons[1:]]
    else:
        return [], "invalid_strand"

    sites = []
    for pos1 in donors:
        sites.append({"site_type": "donor", "site_1based": int(pos1), "site_0based": int(pos1) - 1})
    for pos1 in acceptors:
        sites.append({"site_type": "acceptor", "site_1based": int(pos1), "site_0based": int(pos1) - 1})

    return sites, "ok"


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
        "--gtf",
        default=str(
            Path.home()
            / "data/raw_embeddings/spliceai_canonical_junction_v1_1/annotation/gencode.v49.annotation.gtf.gz"
        ),
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
        default=str(Path.home() / "data/processed_embeddings/spliceai_canonical_splice_site_profile_v1_1"),
    )
    ap.add_argument(
        "--report_dir",
        default=str(Path.home() / "reports/mapping_reports/spliceai_canonical_splice_site_profile_v1_1"),
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_genes", type=int, default=None)
    args = ap.parse_args()

    master_path = Path(args.master)
    gtf_path = Path(args.gtf)
    fasta_path = Path(args.fasta)
    outdir = Path(args.outdir)
    report_dir = Path(args.report_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding: {EMBEDDING_NAME}")
    print(f"Master: {master_path}")
    print(f"GTF: {gtf_path}")
    print(f"FASTA: {fasta_path}")
    print(f"Output: {outdir}")
    print(f"Batch size: {args.batch_size}")
    print("GPU hidden by default. Set SPLICEAI_USE_GPU=1 to allow GPU.")

    master = pd.read_csv(master_path, dtype=str).fillna("")
    if args.max_genes is not None:
        master = master.head(args.max_genes).copy()

    required = [
        "ensembl_gene_id",
        "gene_symbol",
        "gene_type",
        "chromosome",
        "strand",
        "canonical_transcript_id",
        "canonical_transcript_id_versioned",
        "entrez_id",
        "uniprot_id",
        "assembly",
        "gencode_release",
    ]
    missing = [c for c in required if c not in master.columns]
    if missing:
        raise RuntimeError(f"Master table missing required columns: {missing}")

    wanted = set()
    for c in ["canonical_transcript_id", "canonical_transcript_id_versioned"]:
        wanted |= {x for x in master[c].astype(str).tolist() if x and x != "nan"}
        wanted |= {x.split(".")[0] for x in master[c].astype(str).tolist() if x and x != "nan"}

    print(f"Wanted canonical transcript IDs: {len(wanted)}")
    print("Parsing GTF exons...")
    exons_by_tx = parse_gtf_exons(gtf_path, wanted)
    print(f"Transcripts with parsed exons: {len(exons_by_tx)}")

    fa = Fasta(str(fasta_path))
    fasta_chroms = set(fa.keys())

    gene_records = []
    site_records = []
    status_counts = {}

    for gene_index, (_, row) in enumerate(master.iterrows()):
        chrom = row["chromosome"]
        status = "ok"

        if chrom not in fasta_chroms:
            status = "chromosome_not_in_fasta"
            sites = []
        else:
            sites, status = canonical_sites_for_gene(row, exons_by_tx)

        status_counts[status] = status_counts.get(status, 0) + 1

        n_donor = sum(1 for s in sites if s["site_type"] == "donor")
        n_acceptor = sum(1 for s in sites if s["site_type"] == "acceptor")

        gene_records.append(
            {
                "ensembl_gene_id": row["ensembl_gene_id"],
                "gene_symbol": row["gene_symbol"],
                "entrez_id": row["entrez_id"],
                "uniprot_id": row["uniprot_id"],
                "gene_type": row["gene_type"],
                "chromosome": row["chromosome"],
                "strand": row["strand"],
                "canonical_transcript_id": row["canonical_transcript_id"],
                "canonical_transcript_id_versioned": row["canonical_transcript_id_versioned"],
                "assembly": row["assembly"],
                "gencode_release": row["gencode_release"],
                "spliceai_site_status": status,
                "n_splice_sites": len(sites),
                "n_donor_sites": n_donor,
                "n_acceptor_sites": n_acceptor,
            }
        )

        for s in sites:
            site_records.append(
                {
                    "gene_index": gene_index,
                    "ensembl_gene_id": row["ensembl_gene_id"],
                    "gene_symbol": row["gene_symbol"],
                    "chromosome": row["chromosome"],
                    "strand": row["strand"],
                    "site_type": s["site_type"],
                    "site_1based": s["site_1based"],
                    "site_0based": s["site_0based"],
                    "canonical_transcript_id": row["canonical_transcript_id"],
                    "canonical_transcript_id_versioned": row["canonical_transcript_id_versioned"],
                }
            )

    genes = pd.DataFrame(gene_records)
    sites = pd.DataFrame(site_records)

    genes.to_csv(outdir / "genes.tsv", sep="\t", index=False)
    sites.to_csv(outdir / "canonical_splice_sites.tsv.gz", sep="\t", index=False, compression="gzip")

    print(f"Genes: {len(genes)}")
    print(f"Canonical splice sites: {len(sites)}")
    print("Gene site status counts:")
    for k, v in sorted(status_counts.items()):
        print(f"  {k}: {v}")

    # Accumulators
    n_genes = len(genes)
    all_acc = np.zeros((n_genes, PROFILE_WIDTH), dtype=np.float32)
    all_don = np.zeros((n_genes, PROFILE_WIDTH), dtype=np.float32)
    donor_acc = np.zeros((n_genes, PROFILE_WIDTH), dtype=np.float32)
    donor_don = np.zeros((n_genes, PROFILE_WIDTH), dtype=np.float32)
    acceptor_acc = np.zeros((n_genes, PROFILE_WIDTH), dtype=np.float32)
    acceptor_don = np.zeros((n_genes, PROFILE_WIDTH), dtype=np.float32)

    n_sites = genes["n_splice_sites"].astype(int).to_numpy()
    n_donor = genes["n_donor_sites"].astype(int).to_numpy()
    n_acceptor = genes["n_acceptor_sites"].astype(int).to_numpy()

    print("Loading SpliceAI models...")
    model_paths = [resource_filename("spliceai", f"models/spliceai{i}.h5") for i in range(1, 6)]
    models = []
    for p in model_paths:
        print("  ", p)
        models.append(load_model(p, compile=False))
    print("Loaded models:", len(models))

    site_dicts = sites.to_dict("records")

    if len(site_dicts) > 0:
        for start in tqdm(range(0, len(site_dicts), args.batch_size)):
            batch_sites = site_dicts[start : start + args.batch_size]
            batch_x = []

            for s in batch_sites:
                seq = make_spliceai_input_sequence(
                    fa=fa,
                    chrom=s["chromosome"],
                    site0=int(s["site_0based"]),
                    strand=s["strand"],
                )
                batch_x.append(one_hot_dna(seq))

            x = np.stack(batch_x, axis=0).astype(np.float32)

            preds = []
            for model in models:
                y = model.predict_on_batch(x)
                if isinstance(y, list):
                    y = y[0]
                preds.append(y.astype(np.float32))

            y = np.mean(preds, axis=0).astype(np.float32)

            if y.shape[1] != PROFILE_WIDTH or y.shape[2] < 3:
                raise RuntimeError(f"Unexpected SpliceAI output shape: {y.shape}")

            acc_profile = y[:, :, 1]
            don_profile = y[:, :, 2]

            for j, s in enumerate(batch_sites):
                gi = int(s["gene_index"])

                all_acc[gi] += acc_profile[j]
                all_don[gi] += don_profile[j]

                if s["site_type"] == "donor":
                    donor_acc[gi] += acc_profile[j]
                    donor_don[gi] += don_profile[j]
                elif s["site_type"] == "acceptor":
                    acceptor_acc[gi] += acc_profile[j]
                    acceptor_don[gi] += don_profile[j]

    # Convert sums to means. Genes with no canonical splice sites remain zero.
    mask = n_sites > 0
    all_acc[mask] /= n_sites[mask, None]
    all_don[mask] /= n_sites[mask, None]

    mask_d = n_donor > 0
    donor_acc[mask_d] /= n_donor[mask_d, None]
    donor_don[mask_d] /= n_donor[mask_d, None]

    mask_a = n_acceptor > 0
    acceptor_acc[mask_a] /= n_acceptor[mask_a, None]
    acceptor_don[mask_a] /= n_acceptor[mask_a, None]

    raw = np.concatenate(
        [all_acc, all_don, donor_acc, donor_don, acceptor_acc, acceptor_don],
        axis=1,
    ).astype(np.float32)

    if raw.shape != (n_genes, N_FEATURES):
        raise RuntimeError(f"Unexpected raw matrix shape: {raw.shape}")
    if not np.isfinite(raw).all():
        raise RuntimeError("Raw SpliceAI matrix contains non-finite values.")

    emb_z, mean, std = zscore_columns(raw)

    np.save(outdir / "raw_spliceai_reference_profiles.npy", raw)
    np.save(outdir / "embeddings.npy", emb_z)
    np.save(outdir / "feature_means_raw_profiles.npy", mean)
    np.save(outdir / "feature_stds_raw_profiles.npy", std)

    feature_rows = []
    groups = [
        "all_sites_acceptor_profile",
        "all_sites_donor_profile",
        "donor_sites_acceptor_profile",
        "donor_sites_donor_profile",
        "acceptor_sites_acceptor_profile",
        "acceptor_sites_donor_profile",
    ]
    idx = 0
    for group in groups:
        for offset in range(-PROFILE_RADIUS, PROFILE_RADIUS + 1):
            feature_rows.append(
                {
                    "feature_index": idx,
                    "feature_name": f"{group}_offset_{offset:+d}",
                    "profile_group": group,
                    "offset_bp_transcript_oriented": offset,
                }
            )
            idx += 1

    pd.DataFrame(feature_rows).to_csv(outdir / "spliceai_feature_columns.tsv", sep="\t", index=False)

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": "processed_qc_passed",
        "modality": "splicing sequence model / canonical transcript splice-site profiles",
        "model": "SpliceAI",
        "model_files": model_paths,
        "model_ensemble": "mean of spliceai1.h5 to spliceai5.h5",
        "kept_output_channels": {
            "acceptor_probability": 1,
            "donor_probability": 2,
        },
        "input_sequence_type": "GRCh38 genomic DNA around canonical transcript splice sites",
        "input_sequence_source": str(fasta_path),
        "annotation_source": str(gtf_path),
        "master_table": str(master_path),
        "genome_build": "GRCh38",
        "gencode_release": "49",
        "splice_site_definition": "canonical transcript exons from GENCODE v49; donor/acceptor assigned by transcript strand",
        "strand_handling": "minus-strand splice-site windows reverse-complemented so profiles are transcript-oriented",
        "context_bp": CONTEXT,
        "profile_radius_bp": PROFILE_RADIUS,
        "profile_width_bp": PROFILE_WIDTH,
        "dimension": N_FEATURES,
        "n_genes": int(n_genes),
        "n_canonical_splice_sites": int(len(sites)),
        "gene_site_status_counts": status_counts,
        "final_embedding": "dimension-wise z-scored mean SpliceAI acceptor/donor profiles across canonical splice sites",
        "genes_without_canonical_splice_sites": int((n_sites == 0).sum()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    report = f"""Mapping/QC report for {EMBEDDING_NAME}
==========================================

Status: processed_qc_passed

Master table: {master_path}
GENCODE GTF: {gtf_path}
Reference FASTA: {fasta_path}
Output directory: {outdir}

Input master rows: {n_genes}
Canonical splice sites processed: {len(sites)}
Genes with at least one canonical splice site: {int((n_sites > 0).sum())}
Genes without canonical splice sites: {int((n_sites == 0).sum())}

Gene site status counts:
{json.dumps(status_counts, indent=2)}

Raw SpliceAI reference-profile matrix:
shape: {raw.shape}
dtype: {raw.dtype}
min: {float(raw.min())}
max: {float(raw.max())}
finite: {np.isfinite(raw).all()}

Final embedding matrix:
shape: {emb_z.shape}
dtype: {emb_z.dtype}
finite: {np.isfinite(emb_z).all()}

Design decisions:
- Used the five pretrained SpliceAI models bundled in the installed package.
- Used canonical transcript IDs from master_gene_table_v1_1_enriched.csv.
- Parsed exon coordinates from GENCODE v49, matching the master table.
- Generated canonical donor and acceptor splice sites by transcript strand.
- Used 10 kb SpliceAI context and a ±{PROFILE_RADIUS} bp output profile around each splice site.
- For minus-strand genes, input windows were reverse-complemented so output profiles are transcript-oriented.
- Averaged acceptor/donor profiles per gene.
- Genes with no canonical splice sites are retained with zero raw profiles and documented in genes.tsv.
- Final embeddings.npy contains dimension-wise z-scored raw profile features across genes.
"""

    with open(outdir / "mapping_report.txt", "w") as f:
        f.write(report)

    print()
    print(report)


if __name__ == "__main__":
    main()
