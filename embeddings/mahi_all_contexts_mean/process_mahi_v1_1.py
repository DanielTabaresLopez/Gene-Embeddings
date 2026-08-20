#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import argparse
import gc
import hashlib
import json
import re
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


HOME = Path.home()

PARQUET_PATH = HOME / "data/raw_embeddings/mahi_v1_1/hf_dataset/mahi_embeddings_float16.parquet"
MASTER_PATH = HOME / "metadata/master_gene_table_v1_1_enriched.csv"
RAW_REPO = HOME / "data/raw_embeddings/mahi_v1_1/official_repo"

REPORT_DIR = HOME / "reports/mapping_reports"
OUT_ROOT = HOME / "data/processed_embeddings"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


def extract_entrez_ids(value):
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "na"}:
        return []

    # Convert strings such as 1234.0 to 1234.
    s = re.sub(r"(\d+)\.0(?=$|[\s,;|])", r"\1", s)

    ids = re.findall(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])", s)

    out = []
    seen = set()
    for x in ids:
        if x and x != "0" and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def find_column(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def get_entrez_columns(columns):
    preferred = [
        "entrez_id",
        "entrez_gene_id",
        "ncbi_gene_id",
        "map_entrez_id",
        "map_entrez_ids_all",
        "entrez_ids_all",
    ]

    out = []

    lower = {c.lower(): c for c in columns}
    for p in preferred:
        if p.lower() in lower:
            out.append(lower[p.lower()])

    for c in columns:
        cl = c.lower()
        if ("entrez" in cl or "ncbi_gene" in cl) and c not in out:
            out.append(c)

    return out


def load_master_mapping():
    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    cols = list(master.columns)

    ensembl_col = find_column(cols, [
        "ensembl_gene_id",
        "gene_id",
        "master_ensembl_gene_id",
        "ensembl_id",
    ])

    if ensembl_col is None:
        raise RuntimeError(f"Could not find Ensembl gene ID column in {MASTER_PATH}")

    symbol_col = find_column(cols, [
        "hgnc_symbol",
        "gene_symbol",
        "symbol",
        "external_gene_name",
    ])

    entrez_cols = get_entrez_columns(cols)

    if not entrez_cols:
        raise RuntimeError(
            "Could not find any Entrez-like column in the master table. "
            "Columns are: " + ", ".join(cols)
        )

    print("Master Ensembl column:", ensembl_col)
    print("Master symbol column:", symbol_col or "none")
    print("Master Entrez columns:", entrez_cols)

    raw_entries = []
    entrez_to_ensgs = defaultdict(set)

    for _, row in master.iterrows():
        ensg = str(row[ensembl_col]).strip()
        if not ensg:
            continue

        symbol = str(row[symbol_col]).strip() if symbol_col else ""

        entrez_ids = []
        seen = set()

        for c in entrez_cols:
            for eid in extract_entrez_ids(row[c]):
                if eid not in seen:
                    entrez_ids.append(eid)
                    seen.add(eid)

        raw_entries.append({
            "ensembl_gene_id": ensg,
            "hgnc_symbol": symbol,
            "candidate_entrez_ids_raw": entrez_ids,
        })

        for eid in entrez_ids:
            entrez_to_ensgs[eid].add(ensg)

    ambiguous_entrez = {
        eid for eid, ensgs in entrez_to_ensgs.items()
        if len(ensgs) > 1
    }

    entries = []
    for e in raw_entries:
        clean_ids = [
            eid for eid in e["candidate_entrez_ids_raw"]
            if eid not in ambiguous_entrez
        ]
        e["candidate_entrez_ids"] = clean_ids
        entries.append(e)

    target_entrez = sorted(
        {eid for e in entries for eid in e["candidate_entrez_ids"]},
        key=lambda x: int(x) if x.isdigit() else x,
    )

    return master, entries, target_entrez, ambiguous_entrez, entrez_cols


def write_embedding(
    embedding_name,
    mode,
    entries,
    target_entrez,
    entrez_to_idx,
    global_mat,
    has_global,
    sum_mat,
    counts,
    context_counts,
    source_sha256,
    elapsed,
    suffix="",
):
    out_dir = OUT_ROOT / f"{embedding_name}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors = []
    gene_rows = []

    for e in entries:
        chosen_eid = None
        chosen_idx = None

        for eid in e["candidate_entrez_ids"]:
            idx = entrez_to_idx.get(eid)
            if idx is None:
                continue

            if mode == "global":
                if has_global[idx]:
                    chosen_eid = eid
                    chosen_idx = idx
                    break
            elif mode == "mean":
                if counts[idx] > 0:
                    chosen_eid = eid
                    chosen_idx = idx
                    break

        if chosen_idx is None:
            continue

        if mode == "global":
            vec = global_mat[chosen_idx]
            context_count = 1
        else:
            context_count = int(counts[chosen_idx])
            vec = sum_mat[chosen_idx] / np.float32(context_count)

        vectors.append(vec.astype(np.float32))

        row = {
            "ensembl_gene_id": e["ensembl_gene_id"],
            "entrez_id": chosen_eid,
            "mahi_context_count": context_count,
        }

        if e.get("hgnc_symbol"):
            row["hgnc_symbol"] = e["hgnc_symbol"]

        gene_rows.append(row)

    if not vectors:
        print(f"No vectors available for {embedding_name}; skipped.")
        return

    matrix = np.vstack(vectors).astype(np.float32)
    genes_df = pd.DataFrame(gene_rows)

    rows_match = matrix.shape[0] == len(genes_df)
    nan_count = int(np.isnan(matrix).sum())
    inf_count = int(np.isinf(matrix).sum())
    zero_vectors = int((np.linalg.norm(matrix, axis=1) == 0).sum())
    duplicated_genes = int(genes_df["ensembl_gene_id"].duplicated().sum())

    status = "processed_qc_passed"
    if not rows_match or nan_count or inf_count or zero_vectors or duplicated_genes:
        status = "processed_qc_failed"

    npz_path = out_dir / f"{embedding_name}_embeddings.npz"
    genes_path = out_dir / f"{embedding_name}_genes.tsv"
    meta_path = out_dir / f"{embedding_name}_metadata.json"
    report_path = REPORT_DIR / f"{embedding_name}_mapping_report.txt"

    np.savez_compressed(npz_path, embeddings=matrix)
    genes_df.to_csv(genes_path, sep="\t", index=False)

    metadata = {
        "embedding_name": embedding_name,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "Mahi precomputed gene embeddings",
        "source_parquet": str(PARQUET_PATH),
        "source_parquet_sha256": source_sha256,
        "official_repo": str(RAW_REPO),
        "official_repo_commit": git_commit(RAW_REPO),
        "hf_dataset_repo": "anushaggs/Mahi_gene_embeddings",
        "raw_embedding_precision": "float16",
        "processed_embedding_precision": "float32",
        "embedding_dimension": int(matrix.shape[1]),
        "matrix_shape": list(matrix.shape),
        "mode": mode,
        "context_summary": {
            "total_contexts_in_source": len(context_counts),
            "global_context_present": "global" in context_counts,
            "non_global_contexts_in_source": len([c for c in context_counts if c != "global"]),
        },
        "pooling_strategy": (
            "global context only"
            if mode == "global"
            else "mean across all non-global tissue/cell contexts per Entrez gene"
        ),
        "master_gene_table": str(MASTER_PATH),
        "runtime_seconds": round(elapsed, 2),
    }

    meta_path.write_text(json.dumps(metadata, indent=2))

    report = []
    report.append(f"Mapping/QC report for {embedding_name}")
    report.append("=" * (22 + len(embedding_name)))
    report.append("")
    report.append(f"Status: {status}")
    report.append(f"Output directory: {out_dir}")
    report.append(f"Embeddings: {npz_path}")
    report.append(f"Genes: {genes_path}")
    report.append(f"Metadata: {meta_path}")
    report.append("")
    report.append(f"Source parquet: {PARQUET_PATH}")
    report.append(f"Source parquet SHA256: {source_sha256}")
    report.append(f"Mode: {mode}")
    report.append("")
    report.append(f"Matrix shape: {matrix.shape}")
    report.append(f"Rows match genes.tsv: {rows_match}")
    report.append(f"NaN count: {nan_count}")
    report.append(f"Inf count: {inf_count}")
    report.append(f"Zero vectors: {zero_vectors}")
    report.append(f"Duplicated Ensembl IDs: {duplicated_genes}")
    report.append(f"Processed genes: {len(genes_df)}")
    report.append(f"Runtime seconds: {elapsed:.2f}")
    report.append("")
    report_path.write_text("\n".join(report) + "\n")

    print("\nDone:", embedding_name)
    print("\n".join(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-row-groups", type=int, default=None)
    parser.add_argument("--compute-sha256", action="store_true")
    args = parser.parse_args()

    start = time.time()

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(PARQUET_PATH)

    master, entries, target_entrez, ambiguous_entrez, entrez_cols = load_master_mapping()

    print("Master rows:", len(master))
    print("Target unambiguous Entrez IDs from master:", len(target_entrez))
    print("Ambiguous Entrez IDs excluded:", len(ambiguous_entrez))

    entrez_to_idx = {eid: i for i, eid in enumerate(target_entrez)}

    pf = pq.ParquetFile(PARQUET_PATH)

    sample = pf.read_row_group(0, columns=["embedding"]).column("embedding")[0].as_py()
    dim = len(sample)

    print("Mahi parquet rows:", pf.metadata.num_rows)
    print("Mahi row groups:", pf.num_row_groups)
    print("Mahi embedding dim:", dim)

    n_targets = len(target_entrez)

    sum_mat = np.zeros((n_targets, dim), dtype=np.float32)
    counts = np.zeros(n_targets, dtype=np.uint16)

    global_mat = np.zeros((n_targets, dim), dtype=np.float32)
    has_global = np.zeros(n_targets, dtype=bool)

    context_counts = Counter()

    n_row_groups = pf.num_row_groups
    if args.max_row_groups is not None:
        n_row_groups = min(n_row_groups, args.max_row_groups)

    print("Row groups to process:", n_row_groups)

    for rg_idx in tqdm(range(n_row_groups)):
        tbl = pf.read_row_group(rg_idx, columns=["entrez_id", "tissue", "embedding"])
        df = tbl.to_pandas()

        tissues = df["tissue"].astype(str).to_numpy()
        entrez_vals = df["entrez_id"].astype(str).to_numpy()

        context_counts.update(tissues.tolist())

        idxs = np.array([entrez_to_idx.get(eid, -1) for eid in entrez_vals], dtype=np.int32)
        keep = idxs >= 0

        if not keep.any():
            del tbl, df
            gc.collect()
            continue

        kept_positions = np.where(keep)[0]
        kept_idxs = idxs[kept_positions]
        kept_tissues = tissues[kept_positions]

        emb_series = df["embedding"].iloc[kept_positions]
        mat = np.stack(
            [np.asarray(x, dtype=np.float32) for x in emb_series],
            axis=0,
        )

        is_global = kept_tissues == "global"

        if is_global.any():
            global_mat[kept_idxs[is_global]] = mat[is_global]
            has_global[kept_idxs[is_global]] = True

        if (~is_global).any():
            idx_non = kept_idxs[~is_global]
            mat_non = mat[~is_global]

            # Rows should be unique within one context, but np.add.at is safer.
            np.add.at(sum_mat, idx_non, mat_non)
            np.add.at(counts, idx_non, 1)

        del tbl, df, mat, emb_series
        gc.collect()

    context_counts_path = REPORT_DIR / "mahi_v1_1_context_counts.tsv"
    pd.DataFrame(
        [{"context": c, "rows": n} for c, n in sorted(context_counts.items())]
    ).to_csv(context_counts_path, sep="\t", index=False)

    print("Context counts written to:", context_counts_path)
    print("Contexts seen:", len(context_counts))
    print("Global rows mapped:", int(has_global.sum()))
    print("Genes with non-global context counts:", int((counts > 0).sum()))
    print("Max non-global context count:", int(counts.max()))

    source_sha256 = sha256_file(PARQUET_PATH) if args.compute_sha256 else "not_computed"

    elapsed = time.time() - start

    suffix = "_TEST" if args.max_row_groups is not None else ""

    write_embedding(
        embedding_name="mahi_global_v1_1",
        mode="global",
        entries=entries,
        target_entrez=target_entrez,
        entrez_to_idx=entrez_to_idx,
        global_mat=global_mat,
        has_global=has_global,
        sum_mat=sum_mat,
        counts=counts,
        context_counts=context_counts,
        source_sha256=source_sha256,
        elapsed=elapsed,
        suffix=suffix,
    )

    write_embedding(
        embedding_name="mahi_all_contexts_mean_v1_1",
        mode="mean",
        entries=entries,
        target_entrez=target_entrez,
        entrez_to_idx=entrez_to_idx,
        global_mat=global_mat,
        has_global=has_global,
        sum_mat=sum_mat,
        counts=counts,
        context_counts=context_counts,
        source_sha256=source_sha256,
        elapsed=elapsed,
        suffix=suffix,
    )


if __name__ == "__main__":
    main()
