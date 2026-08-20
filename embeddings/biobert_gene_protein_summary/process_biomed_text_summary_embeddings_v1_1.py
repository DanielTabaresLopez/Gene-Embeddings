from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def split_pipe(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [v.strip() for v in str(x).split("|") if v.strip()]


def load_summary_json(raw_dir: Path, json_name: str):
    hits = list(raw_dir.rglob(json_name))
    if not hits:
        raise FileNotFoundError(f"Could not find {json_name} under {raw_dir}")
    if len(hits) > 1:
        print(f"Multiple {json_name} files found. Using: {hits[0]}")
    with open(hits[0], "r") as f:
        data = json.load(f)
    return {str(k).upper(): str(v) for k, v in data.items() if str(v).strip()}


def build_mapping(master: pd.DataFrame, summaries: dict):
    rows = []
    used = set()

    for _, r in master.iterrows():
        candidates = []

        for col in ["gene_symbol", "hgnc_approved_symbol"]:
            val = str(r.get(col, "")).strip()
            if val:
                candidates.append((val.upper(), col))

        # Only fallback after exact/current approved symbols.
        for val in split_pipe(r.get("map_symbols_all", "")):
            candidates.append((val.upper(), "map_symbols_all_fallback"))

        chosen_key = None
        chosen_source = None
        for key, source in candidates:
            if key in summaries:
                chosen_key = key
                chosen_source = source
                break

        if chosen_key is None:
            continue

        rows.append({
            "ensembl_gene_id": r["ensembl_gene_id"],
            "gene_symbol": r.get("gene_symbol", ""),
            "hgnc_approved_symbol": r.get("hgnc_approved_symbol", ""),
            "entrez_id": r.get("entrez_id", ""),
            "uniprot_id": r.get("uniprot_id", ""),
            "summary_key": chosen_key,
            "summary_mapping_source": chosen_source,
            "summary_text": summaries[chosen_key],
        })
        used.add(chosen_key)

    mapped = pd.DataFrame(rows)
    return mapped, used


def mean_pool(last_hidden, attention_mask, special_tokens_mask):
    valid = attention_mask * (1 - special_tokens_mask)
    valid_sum = valid.sum(dim=1, keepdim=True).clamp(min=1)
    pooled = (last_hidden * valid.unsqueeze(-1)).sum(dim=1) / valid_sum
    return pooled


def embed_texts(texts, model_id, batch_size, max_length, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    model.to(device)

    outs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding with {model_id}"):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids", None),
            )
            pooled = mean_pool(
                out.last_hidden_state,
                enc["attention_mask"],
                enc["special_tokens_mask"],
            )
            outs.append(pooled.cpu().numpy().astype("float32"))

    return np.vstack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding-name", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--master", default=str(Path.home() / "metadata/master_gene_table_v1_1_enriched.csv"))
    ap.add_argument("--raw-dir", default=str(Path.home() / "data/raw_embeddings/genept_text_summaries_v1_1"))
    ap.add_argument("--summary-json", default="NCBI_UniProt_summary_of_genes.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    t0 = time.time()

    master_path = Path(args.master)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path.home() / "reports/mapping_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(master_path, dtype=str).fillna("")
    summaries = load_summary_json(raw_dir, args.summary_json)
    mapped, used = build_mapping(master, summaries)

    if mapped.empty:
        raise RuntimeError("No genes mapped to summary texts.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding name: {args.embedding_name}")
    print(f"Model: {args.model_id}")
    print(f"Device: {device}")
    print(f"Master rows: {len(master)}")
    print(f"Summary records: {len(summaries)}")
    print(f"Mapped genes: {len(mapped)}")
    print(f"Coverage: {len(mapped) / len(master):.4%}")

    X = embed_texts(
        mapped["summary_text"].tolist(),
        args.model_id,
        args.batch_size,
        args.max_length,
        device,
    )

    if X.shape[0] != len(mapped):
        raise RuntimeError("Embedding rows do not match mapped genes.")

    if not np.isfinite(X).all():
        raise RuntimeError("NaN or Inf found in embeddings.")

    zero_vectors = int((np.linalg.norm(X, axis=1) == 0).sum())

    safe = args.embedding_name.lower().replace("-", "_").replace(".", "_")
    np.savez_compressed(out_dir / f"{safe}_embeddings.npz", embeddings=X)

    genes = mapped.drop(columns=["summary_text"]).copy()
    genes.insert(0, "row_index", np.arange(len(genes)))
    genes.to_csv(out_dir / f"{safe}_genes.tsv", sep="\t", index=False)

    metadata = {
        "embedding_name": args.embedding_name,
        "modality": "literature_text_gene_summary",
        "download_source": "GenePT Zenodo v2 summaries, DOI 10.5281/zenodo.10833191",
        "original_text_source": "NCBI gene card summaries plus UniProt protein summaries when available",
        "embedding_generated_by": "Daniel locally",
        "model_checkpoint": args.model_id,
        "pooling_strategy": "final hidden layer mean pooling over valid non-special, non-padding tokens",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "master_table": str(master_path),
        "master_rows": int(len(master)),
        "mapped_genes": int(len(mapped)),
        "master_coverage": float(len(mapped) / len(master)),
        "embedding_dim": int(X.shape[1]),
        "matrix_shape": list(X.shape),
        "dtype": str(X.dtype),
        "device": device,
        "zero_vectors": zero_vectors,
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "created_unix_time": time.time(),
    }
    with open(out_dir / f"{safe}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    report = []
    report.append(f"Mapping/QC report for {args.embedding_name}")
    report.append("=" * (22 + len(args.embedding_name)))
    report.append("")
    report.append("Status: processed_qc_passed")
    report.append(f"Model checkpoint: {args.model_id}")
    report.append(f"Master table: {master_path}")
    report.append(f"Raw summary JSON: {args.summary_json}")
    report.append(f"Output directory: {out_dir}")
    report.append("")
    report.append(f"Master rows: {len(master)}")
    report.append(f"Summary records: {len(summaries)}")
    report.append(f"Mapped genes: {len(mapped)}")
    report.append(f"Master coverage: {len(mapped) / len(master):.4%}")
    report.append(f"Embedding shape: {X.shape[0]} x {X.shape[1]}")
    report.append(f"NaN count: {int(np.isnan(X).sum())}")
    report.append(f"Inf count: {int(np.isinf(X).sum())}")
    report.append(f"Zero vectors: {zero_vectors}")
    report.append(f"Duplicated Ensembl IDs: {int(mapped['ensembl_gene_id'].duplicated().sum())}")
    report.append(f"Runtime minutes: {(time.time() - t0) / 60:.2f}")
    report.append("")
    report.append("Mapping source counts:")
    report.append(mapped["summary_mapping_source"].value_counts().to_string())

    report_path = report_dir / f"{safe}_mapping_report.txt"
    report_path.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
