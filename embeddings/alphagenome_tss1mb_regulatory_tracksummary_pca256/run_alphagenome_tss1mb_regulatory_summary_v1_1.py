import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client


MASTER = Path.home() / "metadata/master_gene_table_v1_1_enriched.csv"
OUTDIR = Path.home() / "data/raw_embeddings/alphagenome_tss1mb_regulatory_v1_1"
PANEL = OUTDIR / "alphagenome_ontology_panel_v1_1.tsv"
CHUNKDIR = OUTDIR / "chunks_summary_raw"
LOGDIR = OUTDIR / "logs"

OUTPUT_TYPES = [
    "CAGE",
    "RNA_SEQ",
    "DNASE",
    "ATAC",
    "CHIP_HISTONE",
    "CHIP_TF",
    "PROCAP",
]

ATTR = {
    "CAGE": "cage",
    "RNA_SEQ": "rna_seq",
    "DNASE": "dnase",
    "ATAC": "atac",
    "CHIP_HISTONE": "chip_histone",
    "CHIP_TF": "chip_tf",
    "PROCAP": "procap",
}

WINDOWS = [
    ("central_1kb_mean", 1_000, "mean"),
    ("central_10kb_mean", 10_000, "mean"),
    ("central_100kb_mean", 100_000, "mean"),
    ("full_1mb_mean", None, "mean"),
    ("central_10kb_max", 10_000, "max"),
]

STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def clean_text(x):
    return str(x).replace("\t", " ").replace("\n", " ").replace("|", "/")


def signed_log1p(x):
    x = np.asarray(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


def tss_interval(row):
    chrom = row["chromosome"]
    strand = row["strand"]
    gene_start = int(row["gene_start"])
    gene_end = int(row["gene_end"])
    tss_1based = gene_start if strand == "+" else gene_end
    tss_0based = tss_1based - 1

    anchor = genome.Interval(
        chromosome=chrom,
        start=tss_0based,
        end=tss_0based + 1,
        strand=strand,
        name=row["gene_symbol"],
    )
    interval = anchor.resize(dna_client.SEQUENCE_LENGTH_1MB)
    return interval, tss_1based


def summarize_trackdata(tdata, output_type):
    arr = tdata.values.astype(np.float32, copy=False)
    n_bins = arr.shape[0]
    center = n_bins // 2
    resolution = int(tdata.resolution)

    values = []
    feature_meta = []

    md = tdata.metadata.reset_index(drop=True).copy()

    for stat_name, width_bp, op in WINDOWS:
        if width_bp is None:
            start, end = 0, n_bins
        else:
            half_bins = max(1, int(round((width_bp / 2) / resolution)))
            start = max(0, center - half_bins)
            end = min(n_bins, center + half_bins)

        block = arr[start:end, :]
        if op == "mean":
            vals = np.nanmean(block, axis=0)
        elif op == "max":
            vals = np.nanmax(block, axis=0)
        else:
            raise ValueError(op)

        vals = signed_log1p(vals)
        values.extend(vals.tolist())

        for j in range(arr.shape[1]):
            r = md.iloc[j]
            feature_name = "|".join([
                clean_text(output_type),
                clean_text(stat_name),
                clean_text(r.get("name", "")),
                clean_text(r.get("strand", "")),
                clean_text(r.get("ontology_curie", "")),
                clean_text(r.get("biosample_name", "")),
                clean_text(r.get("data_source", "")),
                f"track{j}",
            ])

            feature_meta.append({
                "feature_name": feature_name,
                "output_type": output_type,
                "summary_stat": stat_name,
                "window_bp": "full_1mb" if width_bp is None else width_bp,
                "operation": op,
                "resolution": resolution,
                "track_index_within_output": j,
                "track_name": r.get("name", ""),
                "track_strand": r.get("strand", ""),
                "assay_title": r.get("Assay title", ""),
                "ontology_curie": r.get("ontology_curie", ""),
                "biosample_name": r.get("biosample_name", ""),
                "biosample_type": r.get("biosample_type", ""),
                "data_source": r.get("data_source", ""),
                "nonzero_mean": r.get("nonzero_mean", ""),
            })

    return values, feature_meta


def existing_success_ids():
    ids = set()
    if not CHUNKDIR.exists():
        return ids

    for f in sorted(CHUNKDIR.glob("alphagenome_tss1mb_regulatory_summary_chunk_*.tsv.gz")):
        try:
            x = pd.read_csv(f, sep="\t", usecols=["ensembl_gene_id"], dtype=str)
            ids.update(x["ensembl_gene_id"].dropna().tolist())
        except Exception as e:
            print(f"WARNING: could not read existing chunk {f}: {e}", flush=True)
    return ids


def next_chunk_index():
    CHUNKDIR.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"chunk_(\d+)\.tsv\.gz$")
    found = []
    for f in CHUNKDIR.glob("alphagenome_tss1mb_regulatory_summary_chunk_*.tsv.gz"):
        m = pat.search(f.name)
        if m:
            found.append(int(m.group(1)))
    return max(found) + 1 if found else 0


def write_chunk(rows, chunk_idx):
    CHUNKDIR.mkdir(parents=True, exist_ok=True)
    out = CHUNKDIR / f"alphagenome_tss1mb_regulatory_summary_chunk_{chunk_idx:05d}.tsv.gz"
    tmp = CHUNKDIR / f".tmp_chunk_{chunk_idx:05d}.tsv.gz"

    pd.DataFrame(rows).to_csv(tmp, sep="\t", index=False, compression="gzip")
    tmp.replace(out)
    print(f"Saved chunk {chunk_idx:05d}: {out} ({len(rows)} rows)", flush=True)


def append_failures(failures):
    if not failures:
        return
    out = OUTDIR / "alphagenome_tss1mb_regulatory_failures.tsv"
    df = pd.DataFrame(failures)
    header = not out.exists()
    df.to_csv(out, sep="\t", index=False, mode="a", header=header)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_genes", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--sleep_sec", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry_sleep_sec", type=float, default=30.0)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    CHUNKDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("ALPHAGENOME_API_KEY"):
        raise RuntimeError("ALPHAGENOME_API_KEY is not set in this shell.")

    ontology_terms = pd.read_csv(PANEL, sep="\t", dtype=str)["ontology_curie"].dropna().tolist()
    requested_outputs = {getattr(dna_client.OutputType, x) for x in OUTPUT_TYPES}

    master = pd.read_csv(MASTER, dtype=str).fillna("")
    genes = master[
        (master["gene_type"] == "protein_coding")
        & (master["assembly"] == "GRCh38")
        & (master["chromosome"].isin(STANDARD_CHROMS))
    ].copy()

    genes = genes.sort_values("ensembl_gene_id").reset_index(drop=True)

    already = existing_success_ids()
    if already:
        genes = genes[~genes["ensembl_gene_id"].isin(already)].copy()

    if args.max_genes is not None:
        genes = genes.head(args.max_genes).copy()

    config = {
        "embedding_name": "AlphaGenome-TSS1Mb-regulatoryTrackSummary-PCA256-v1.1",
        "raw_summary_name": "alphagenome_tss1mb_regulatory_summary_v1_1",
        "master_table": str(MASTER),
        "sequence_length": "SEQUENCE_LENGTH_1MB",
        "anchor": "canonical gene TSS from master table; + strand uses gene_start, - strand uses gene_end",
        "coordinate_assumption": "master coordinates are 1-based; AlphaGenome interval anchor is converted to 0-based half-open",
        "requested_outputs": OUTPUT_TYPES,
        "ontology_terms": ontology_terms,
        "windows": WINDOWS,
        "standard_chromosomes": sorted(STANDARD_CHROMS),
        "raw_predictions_saved": False,
        "chunk_size": args.chunk_size,
        "sleep_sec": args.sleep_sec,
        "already_completed_before_this_run": len(already),
        "genes_scheduled_this_run": len(genes),
    }

    with open(OUTDIR / "alphagenome_tss1mb_regulatory_run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("AlphaGenome regulatory summary run")
    print("Genes already completed:", len(already))
    print("Genes scheduled now:", len(genes))
    print("Ontology terms:", ontology_terms)
    print("Output types:", OUTPUT_TYPES)
    print("Chunk size:", args.chunk_size)
    print("Raw predictions saved: False")
    print("", flush=True)

    dna_model = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])

    feature_meta_ref = None
    feature_meta_out = OUTDIR / "alphagenome_tss1mb_regulatory_feature_metadata.tsv"

    if feature_meta_out.exists():
        old_meta = pd.read_csv(feature_meta_out, sep="\t", dtype=str).fillna("")
        feature_meta_ref = old_meta.to_dict(orient="records")
        print(f"Loaded existing feature metadata: {feature_meta_out} ({len(feature_meta_ref)} features)", flush=True)

    chunk_idx = next_chunk_index()
    rows = []
    failures_buffer = []
    start_time = time.time()
    done = 0

    for _, row in genes.iterrows():
        gene_id = row["ensembl_gene_id"]
        symbol = row["gene_symbol"]
        gene_t0 = time.time()

        try:
            interval, tss_1based = tss_interval(row)
            print(f"[{done + 1}/{len(genes)}] {symbol} {gene_id} {interval}", flush=True)

            last_error = None
            pred = None

            for attempt in range(1, args.retries + 1):
                try:
                    pred = dna_model.predict_interval(
                        interval=interval,
                        requested_outputs=requested_outputs,
                        ontology_terms=ontology_terms,
                    )
                    break
                except Exception as e:
                    last_error = e
                    wait = args.retry_sleep_sec * attempt
                    print(f"  API attempt {attempt}/{args.retries} failed: {repr(e)}", flush=True)
                    if attempt < args.retries:
                        print(f"  sleeping {wait:.1f} sec before retry", flush=True)
                        time.sleep(wait)

            if pred is None:
                raise RuntimeError(f"All API attempts failed. Last error: {repr(last_error)}")

            row_values = []
            feature_meta_this = []

            for ot_name in OUTPUT_TYPES:
                tdata = getattr(pred, ATTR[ot_name])
                vals, fmeta = summarize_trackdata(tdata, ot_name)
                row_values.extend(vals)
                feature_meta_this.extend(fmeta)

            if feature_meta_ref is None:
                feature_meta_ref = feature_meta_this
                pd.DataFrame(feature_meta_ref).to_csv(feature_meta_out, sep="\t", index=False)
                print(f"  saved feature metadata: {feature_meta_out}", flush=True)
            else:
                old_names = [x["feature_name"] for x in feature_meta_ref]
                new_names = [x["feature_name"] for x in feature_meta_this]
                if old_names != new_names:
                    raise RuntimeError(f"Feature metadata/order changed for {gene_id}")

            out_row = {
                "ensembl_gene_id": gene_id,
                "gene_symbol": symbol,
                "chromosome": row["chromosome"],
                "strand": row["strand"],
                "gene_start": row["gene_start"],
                "gene_end": row["gene_end"],
                "tss_1based": tss_1based,
                "input_interval": str(interval),
            }

            for fmeta, value in zip(feature_meta_ref, row_values):
                out_row[fmeta["feature_name"]] = value

            rows.append(out_row)
            done += 1

            print(f"  done in {time.time() - gene_t0:.2f} sec", flush=True)

            del pred
            gc.collect()

            if len(rows) >= args.chunk_size:
                write_chunk(rows, chunk_idx)
                chunk_idx += 1
                rows = []

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

        except Exception as e:
            print(f"  FAILED {symbol} {gene_id}: {repr(e)}", flush=True)
            failures_buffer.append({
                "ensembl_gene_id": gene_id,
                "gene_symbol": symbol,
                "chromosome": row.get("chromosome", ""),
                "strand": row.get("strand", ""),
                "gene_start": row.get("gene_start", ""),
                "gene_end": row.get("gene_end", ""),
                "error": repr(e),
            })

            if len(failures_buffer) >= 20:
                append_failures(failures_buffer)
                failures_buffer = []

    if rows:
        write_chunk(rows, chunk_idx)

    append_failures(failures_buffer)

    elapsed = time.time() - start_time
    print("")
    print("Finished run.")
    print("New successful genes:", done)
    print("Elapsed hours:", round(elapsed / 3600, 3))
    print("Chunks directory:", CHUNKDIR)
    print("Failure file:", OUTDIR / "alphagenome_tss1mb_regulatory_failures.tsv")
    print("Feature metadata:", feature_meta_out)


if __name__ == "__main__":
    main()
