#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import subprocess
import time

import numpy as np
import pandas as pd
import torch


EMBEDDING_NAME = "scgpt_pancancer_v1_1"

HOME = Path.home()

MASTER_PATH = HOME / "metadata/master_gene_table_v1_1_enriched.csv"
RAW_BASE = HOME / "data/raw_embeddings/scgpt_pancancer_v1_1"
RAW_REPO = RAW_BASE / "official_repo"
CHECKPOINT_DIR = RAW_BASE / "checkpoint"

VOCAB_PATH = CHECKPOINT_DIR / "vocab.json"
ARGS_PATH = CHECKPOINT_DIR / "args.json"
MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"

OUT_DIR = HOME / "data/processed_embeddings" / EMBEDDING_NAME
REPORT_PATH = HOME / "reports/mapping_reports" / f"{EMBEDDING_NAME}_mapping_report.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def split_symbols(value):
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    parts = re.split(r"[;,\|]", s)
    return [p.strip() for p in parts if p.strip()]


def build_symbol_to_master(master: pd.DataFrame):
    symbol_cols_primary = [c for c in ["gene_symbol", "symbol", "hgnc_symbol"] if c in master.columns]

    alias_candidates = [
        "alias_symbol",
        "alias_symbols",
        "prev_symbol",
        "prev_symbols",
        "hgnc_alias_symbols",
        "hgnc_prev_symbols",
        "alias_symbols_hgnc",
        "prev_symbols_hgnc",
        "map_gene_symbols_all",
        "gene_symbols_all",
        "all_symbols",
    ]
    symbol_cols_alias = [c for c in alias_candidates if c in master.columns]

    pairs = []

    # Primary symbols.
    for _, row in master.iterrows():
        ensg = row["ensembl_gene_id"]
        for col in symbol_cols_primary:
            for sym in split_symbols(row[col]):
                pairs.append((sym.upper(), ensg, "primary", col))

    # Alias symbols, if available. These are useful but lower-confidence.
    for _, row in master.iterrows():
        ensg = row["ensembl_gene_id"]
        for col in symbol_cols_alias:
            for sym in split_symbols(row[col]):
                pairs.append((sym.upper(), ensg, "alias", col))

    pair_df = pd.DataFrame(pairs, columns=["symbol_upper", "ensembl_gene_id", "symbol_source", "symbol_column"]).drop_duplicates()

    # Prefer primary symbols when a token exists as a primary symbol.
    primary_symbols = set(pair_df.loc[pair_df["symbol_source"] == "primary", "symbol_upper"])
    pair_df = pair_df[
        (pair_df["symbol_source"] == "primary") |
        (~pair_df["symbol_upper"].isin(primary_symbols))
    ].copy()

    counts = pair_df.groupby("symbol_upper")["ensembl_gene_id"].nunique()
    ambiguous_symbols = set(counts[counts > 1].index)

    unambig = pair_df[~pair_df["symbol_upper"].isin(ambiguous_symbols)].copy()

    symbol_to_ensg = dict(zip(unambig["symbol_upper"], unambig["ensembl_gene_id"]))
    symbol_to_source = dict(zip(unambig["symbol_upper"], unambig["symbol_source"]))

    return symbol_to_ensg, symbol_to_source, ambiguous_symbols, symbol_cols_primary, symbol_cols_alias


def collapse_strings(values):
    vals = sorted({str(v) for v in values if str(v) and str(v) != "nan"})
    return ";".join(vals)


def main():
    start = time.time()

    print("Embedding:", EMBEDDING_NAME)
    print("Checkpoint:", MODEL_PATH)
    print("Vocab:", VOCAB_PATH)
    print("Master:", MASTER_PATH)

    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    symbol_to_ensg, symbol_to_source, ambiguous_symbols, primary_cols, alias_cols = build_symbol_to_master(master)

    vocab = json.loads(VOCAB_PATH.read_text())
    args = json.loads(ARGS_PATH.read_text()) if ARGS_PATH.exists() else {}

    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "encoder.embedding.weight" in state:
        emb = state["encoder.embedding.weight"]
    elif isinstance(state, dict) and "model_state_dict" in state and "encoder.embedding.weight" in state["model_state_dict"]:
        emb = state["model_state_dict"]["encoder.embedding.weight"]
    else:
        keys = list(state.keys()) if isinstance(state, dict) else []
        raise RuntimeError(f"Could not find encoder.embedding.weight. Top keys: {keys[:30]}")

    emb = emb.detach().cpu().numpy().astype(np.float32)

    print("Embedding tensor shape:", emb.shape)
    print("Vocab size:", len(vocab))

    rows = []
    vectors = []

    special_tokens = set()
    invalid_index_tokens = []
    unmapped_tokens = 0
    ambiguous_source_symbols_seen = 0

    for token, idx in vocab.items():
        token_str = str(token)

        if token_str.startswith("<") and token_str.endswith(">"):
            special_tokens.add(token_str)
            continue

        try:
            idx_int = int(idx)
        except Exception:
            invalid_index_tokens.append(token_str)
            continue

        if idx_int < 0 or idx_int >= emb.shape[0]:
            invalid_index_tokens.append(token_str)
            continue

        token_upper = token_str.upper()

        if token_upper in ambiguous_symbols:
            ambiguous_source_symbols_seen += 1
            continue

        ensg = symbol_to_ensg.get(token_upper)
        if ensg is None:
            unmapped_tokens += 1
            continue

        rows.append({
            "ensembl_gene_id": ensg,
            "source_gene_symbol": token_str,
            "source_vocab_id": idx_int,
            "symbol_mapping_source": symbol_to_source.get(token_upper, ""),
        })
        vectors.append(emb[idx_int])

    mapped = pd.DataFrame(rows)
    matrix_pre = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, emb.shape[1]), dtype=np.float32)

    print("Mapped source symbols before Ensembl collapse:", len(mapped))

    # Collapse multiple vocab symbols mapping to the same Ensembl gene.
    emb_cols = [f"dim_{i}" for i in range(matrix_pre.shape[1])]
    tmp = pd.concat(
        [mapped.reset_index(drop=True), pd.DataFrame(matrix_pre, columns=emb_cols)],
        axis=1,
    )

    duplicated_ensembl_before_collapse = int(tmp["ensembl_gene_id"].duplicated().sum())

    vec_by_ensg = tmp.groupby("ensembl_gene_id", as_index=False)[emb_cols].mean()
    symbol_by_ensg = tmp.groupby("ensembl_gene_id")["source_gene_symbol"].apply(collapse_strings).reset_index()
    vocabid_by_ensg = tmp.groupby("ensembl_gene_id")["source_vocab_id"].apply(lambda x: ";".join(map(str, sorted(set(map(int, x)))))).reset_index()
    source_by_ensg = tmp.groupby("ensembl_gene_id")["symbol_mapping_source"].apply(collapse_strings).reset_index()

    final = vec_by_ensg.merge(symbol_by_ensg, on="ensembl_gene_id", how="left")
    final = final.merge(vocabid_by_ensg, on="ensembl_gene_id", how="left")
    final = final.merge(source_by_ensg, on="ensembl_gene_id", how="left")

    master_small = master[
        ["ensembl_gene_id", "gene_symbol", "entrez_id", "uniprot_id"]
    ].drop_duplicates("ensembl_gene_id")

    final = final.merge(master_small, on="ensembl_gene_id", how="left")
    final = final.sort_values("ensembl_gene_id").reset_index(drop=True)

    matrix = final[emb_cols].to_numpy(dtype=np.float32)

    genes_df = final[
        [
            "ensembl_gene_id",
            "gene_symbol",
            "source_gene_symbol",
            "source_vocab_id",
            "symbol_mapping_source",
            "entrez_id",
            "uniprot_id",
        ]
    ].copy()

    rows_match = matrix.shape[0] == len(genes_df)
    nan_count = int(np.isnan(matrix).sum())
    inf_count = int(np.isinf(matrix).sum())
    zero_vectors = int((np.linalg.norm(matrix, axis=1) == 0).sum())
    duplicated_genes = int(genes_df["ensembl_gene_id"].duplicated().sum())

    status = "processed_qc_passed"
    if not rows_match or nan_count or inf_count or zero_vectors or duplicated_genes:
        status = "processed_qc_failed"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    npz_path = OUT_DIR / f"{EMBEDDING_NAME}_embeddings.npz"
    genes_path = OUT_DIR / f"{EMBEDDING_NAME}_genes.tsv"
    meta_path = OUT_DIR / f"{EMBEDDING_NAME}_metadata.json"

    np.savez_compressed(npz_path, embeddings=matrix)
    genes_df.to_csv(genes_path, sep="\t", index=False)

    runtime = time.time() - start

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "scGPT pan-cancer gene-token embedding",
        "model_source": "Official scGPT model zoo pan-cancer checkpoint",
        "official_repo": str(RAW_REPO),
        "official_repo_commit": git_commit(RAW_REPO),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "checkpoint_file": str(MODEL_PATH),
        "checkpoint_sha256": sha256_file(MODEL_PATH),
        "vocab_file": str(VOCAB_PATH),
        "vocab_sha256": sha256_file(VOCAB_PATH),
        "args_file": str(ARGS_PATH),
        "args_sha256": sha256_file(ARGS_PATH) if ARGS_PATH.exists() else None,
        "embedding_tensor_key": "encoder.embedding.weight",
        "raw_embedding_tensor_shape": list(emb.shape),
        "embedding_dimension": int(matrix.shape[1]),
        "matrix_shape": list(matrix.shape),
        "original_identifier_type": "gene symbol from scGPT vocabulary",
        "mapping_policy": (
            "scGPT vocab gene symbols were mapped to master Ensembl gene IDs through master gene symbols "
            "and available alias/previous-symbol columns if present. Primary symbols were preferred over aliases. "
            "Ambiguous source symbols mapping to multiple Ensembl genes were excluded. "
            "Multiple source symbols mapping to the same final Ensembl gene were collapsed by averaging vectors."
        ),
        "primary_symbol_columns_used": primary_cols,
        "alias_symbol_columns_used": alias_cols,
        "special_tokens_excluded": sorted(special_tokens),
        "unmapped_vocab_gene_tokens": int(unmapped_tokens),
        "ambiguous_source_symbols_excluded": int(ambiguous_source_symbols_seen),
        "invalid_index_tokens": invalid_index_tokens[:50],
        "duplicated_ensembl_rows_before_final_collapse": duplicated_ensembl_before_collapse,
        "training_args_subset": {
            "data_source": args.get("data_source"),
            "input_style": args.get("input_style"),
            "input_emb_style": args.get("input_emb_style"),
            "n_bins": args.get("n_bins"),
            "max_seq_len": args.get("max_seq_len"),
            "training_tasks": args.get("training_tasks"),
            "mask_ratio": args.get("mask_ratio"),
            "n_layers": args.get("n_layers"),
            "nlayers": args.get("nlayers"),
            "embsize": args.get("embsize"),
            "nhead": args.get("nhead"),
            "d_hid": args.get("d_hid"),
        },
        "master_gene_table": str(MASTER_PATH),
        "runtime_seconds": round(runtime, 2),
    }

    meta_path.write_text(json.dumps(metadata, indent=2))

    report = []
    report.append(f"Mapping/QC report for {EMBEDDING_NAME}")
    report.append("=" * (22 + len(EMBEDDING_NAME)))
    report.append("")
    report.append(f"Status: {status}")
    report.append(f"Output directory: {OUT_DIR}")
    report.append(f"Embeddings: {npz_path}")
    report.append(f"Genes: {genes_path}")
    report.append(f"Metadata: {meta_path}")
    report.append("")
    report.append(f"Checkpoint: {MODEL_PATH}")
    report.append(f"Checkpoint SHA256: {metadata['checkpoint_sha256']}")
    report.append(f"Vocab: {VOCAB_PATH}")
    report.append(f"Vocab SHA256: {metadata['vocab_sha256']}")
    report.append(f"Official repo commit: {metadata['official_repo_commit']}")
    report.append("")
    report.append(f"Embedding tensor key: encoder.embedding.weight")
    report.append(f"Raw embedding tensor shape: {emb.shape}")
    report.append(f"Vocab size: {len(vocab)}")
    report.append(f"Special tokens excluded: {sorted(special_tokens)}")
    report.append(f"Unmapped vocab gene tokens: {unmapped_tokens}")
    report.append(f"Ambiguous source symbols excluded: {ambiguous_source_symbols_seen}")
    report.append(f"Mapped source symbols before Ensembl collapse: {len(mapped)}")
    report.append(f"Duplicated Ensembl rows before final collapse: {duplicated_ensembl_before_collapse}")
    report.append("")
    report.append(f"Matrix shape: {matrix.shape}")
    report.append(f"Rows match genes.tsv: {rows_match}")
    report.append(f"NaN count: {nan_count}")
    report.append(f"Inf count: {inf_count}")
    report.append(f"Zero vectors: {zero_vectors}")
    report.append(f"Duplicated Ensembl IDs: {duplicated_genes}")
    report.append(f"Processed genes: {len(genes_df)}")
    report.append(f"Coverage of master table: {100 * len(genes_df) / len(master):.4f}%")
    report.append(f"Runtime seconds: {runtime:.2f}")

    report_text = "\n".join(report)
    REPORT_PATH.write_text(report_text + "\n")

    print("\nDone")
    print("----")
    print(report_text)


if __name__ == "__main__":
    main()
