from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


EMBEDDING_NAME = "BioBERT-GeneProteinSummary-v1.1"
MODEL_ID = "dmis-lab/biobert-base-cased-v1.2"
SUMMARY_JSON_NAME = "NCBI_UniProt_summary_of_genes.json"

MASTER_PATH = Path.home() / "metadata/master_gene_table_v1_1_enriched.csv"
RAW_DIR = Path.home() / "data/raw_embeddings/genept_text_summaries_v1_1"
OUT_DIR = Path.home() / "data/processed_embeddings/biobert_gene_protein_summary_v1_1"
REPORT_DIR = Path.home() / "reports/mapping_reports"

BATCH_SIZE = 16
MAX_LENGTH = 512


def split_pipe(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [v.strip() for v in str(x).split("|") if v.strip()]


def normalize_text_value(v):
    if isinstance(v, str):
        return v.strip()
    return json.dumps(v, ensure_ascii=False)


def find_summary_json():
    hits = list(RAW_DIR.rglob(SUMMARY_JSON_NAME))
    if not hits:
        raise FileNotFoundError(f"Could not find {SUMMARY_JSON_NAME} under {RAW_DIR}")
    return hits[0]


def load_summaries():
    path = find_summary_json()
    with open(path, "r") as f:
        data = json.load(f)

    summaries = {}
    for k, v in data.items():
        key = str(k).strip().upper()
        text = normalize_text_value(v)
        if key and text:
            summaries[key] = text

    print(f"Using summary JSON: {path}")
    return summaries, path


def build_mapping(master, summaries):
    rows = []

    for _, r in master.iterrows():
        candidates = []

        for col in ["gene_symbol", "hgnc_approved_symbol"]:
            val = str(r.get(col, "")).strip()
            if val:
                candidates.append((val.upper(), col))

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
            "ensembl_gene_id": r.get("ensembl_gene_id", ""),
            "gene_symbol": r.get("gene_symbol", ""),
            "hgnc_approved_symbol": r.get("hgnc_approved_symbol", ""),
            "entrez_id": r.get("entrez_id", ""),
            "uniprot_id": r.get("uniprot_id", ""),
            "summary_key": chosen_key,
            "summary_mapping_source": chosen_source,
            "summary_text": summaries[chosen_key],
        })

    return pd.DataFrame(rows)


def mean_pool(last_hidden_state, attention_mask, special_tokens_mask):
    valid_mask = attention_mask * (1 - special_tokens_mask)
    denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
    pooled = (last_hidden_state * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


def embed_texts(texts, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    model.to(device)

    chunks = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=EMBEDDING_NAME):
            batch = texts[i:i + BATCH_SIZE]

            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            model_inputs = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }
            if "token_type_ids" in enc:
                model_inputs["token_type_ids"] = enc["token_type_ids"]

            out = model(**model_inputs)

            pooled = mean_pool(
                out.last_hidden_state,
                enc["attention_mask"],
                enc["special_tokens_mask"],
            )

            chunks.append(pooled.cpu().numpy().astype("float32"))

    return np.vstack(chunks)


def main():
    t0 = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER_PATH, dtype=str).fillna("")
    summaries, summary_path = load_summaries()

    mapped = build_mapping(master, summaries)

    if mapped.empty:
        raise RuntimeError("No genes mapped to summary texts.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Embedding: {EMBEDDING_NAME}")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Master rows: {len(master)}")
    print(f"Summary records: {len(summaries)}")
    print(f"Mapped genes: {len(mapped)}")
    print(f"Coverage: {len(mapped) / len(master):.4%}")

    X = embed_texts(mapped["summary_text"].tolist(), device)

    if X.shape[0] != len(mapped):
        raise RuntimeError("Embedding row count does not match mapped genes.")
    if not np.isfinite(X).all():
        raise RuntimeError("NaN or Inf found in embeddings.")

    safe = "biobert_gene_protein_summary_v1_1"

    np.savez_compressed(OUT_DIR / f"{safe}_embeddings.npz", embeddings=X)

    genes = mapped.drop(columns=["summary_text"]).copy()
    genes.insert(0, "row_index", np.arange(len(genes)))
    genes.to_csv(OUT_DIR / f"{safe}_genes.tsv", sep="\t", index=False)

    zero_vectors = int((np.linalg.norm(X, axis=1) == 0).sum())

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "modality": "literature_text_gene_summary",
        "download_source": "GenePT Zenodo v2 summaries",
        "original_text_source": "NCBI gene card summaries plus UniProt protein summaries when available",
        "embedding_generated_by": "Daniel locally",
        "model_checkpoint": MODEL_ID,
        "pooling_strategy": "final hidden layer mean pooling over valid non-special, non-padding tokens",
        "summary_json": str(summary_path),
        "master_table": str(MASTER_PATH),
        "master_rows": int(len(master)),
        "mapped_genes": int(len(mapped)),
        "master_coverage": float(len(mapped) / len(master)),
        "embedding_dim": int(X.shape[1]),
        "matrix_shape": list(X.shape),
        "dtype": str(X.dtype),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "device": device,
        "zero_vectors": zero_vectors,
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "runtime_minutes": float((time.time() - t0) / 60),
    }

    with open(OUT_DIR / f"{safe}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    report = []
    report.append(f"Mapping/QC report for {EMBEDDING_NAME}")
    report.append("=" * (22 + len(EMBEDDING_NAME)))
    report.append("")
    report.append("Status: processed_qc_passed")
    report.append(f"Model checkpoint: {MODEL_ID}")
    report.append(f"Master table: {MASTER_PATH}")
    report.append(f"Raw summary JSON: {summary_path}")
    report.append(f"Output directory: {OUT_DIR}")
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

    report_text = "\n".join(report) + "\n"

    (OUT_DIR / f"{safe}_mapping_report.txt").write_text(report_text)
    (REPORT_DIR / f"{safe}_mapping_report.txt").write_text(report_text)

    print(report_text)


if __name__ == "__main__":
    main()
