from pathlib import Path
from datetime import datetime
import hashlib
import json
import pickle
import subprocess

import numpy as np
import pandas as pd
import torch


EMBEDDING_NAME = "genecompass_base_gene_token_v1_1"
EMBEDDING_KEY = "bert.embeddings.word_embeddings.weight"
DIM = 768


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return ""


def main():
    home = Path.home()

    repo = home / "data/raw_embeddings/genecompass_v1_1/official_repo"
    ckpt_dir = repo / "pretrained_models/GeneCompass_Base"
    ckpt_path = ckpt_dir / "pytorch_model.bin"
    config_path = ckpt_dir / "config.json"
    token_path = repo / "prior_knowledge/human_mouse_tokens.pickle"
    master_path = home / "metadata/master_gene_table_v1_1_enriched.csv"

    out_dir = home / "data/processed_embeddings" / EMBEDDING_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = home / "reports/mapping_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{EMBEDDING_NAME}_embeddings.npz"
    out_genes = out_dir / f"{EMBEDDING_NAME}_genes.tsv"
    out_metadata = out_dir / f"{EMBEDDING_NAME}_metadata.json"
    out_row_report = reports_dir / f"{EMBEDDING_NAME}_row_mapping_report.tsv"
    out_report = reports_dir / f"{EMBEDDING_NAME}_mapping_report.txt"

    print("Embedding:", EMBEDDING_NAME)
    print("Repo:", repo)
    print("Checkpoint:", ckpt_path)
    print("Token dictionary:", token_path)
    print("Master table:", master_path)

    print("\nLoading master table...")
    master = pd.read_csv(master_path, dtype=str).fillna("")
    print("Master rows:", len(master))

    if "ensembl_gene_id" not in master.columns:
        raise RuntimeError("master table missing required column: ensembl_gene_id")

    print("\nLoading GeneCompass token dictionary...")
    with open(token_path, "rb") as f:
        token_dict = pickle.load(f)

    specials = {k: v for k, v in token_dict.items() if isinstance(k, str) and k.startswith("<")}
    print("Token dictionary entries:", len(token_dict))
    print("Special tokens:", specials)

    print("\nLoading checkpoint on CPU...")
    state = torch.load(ckpt_path, map_location="cpu")

    if EMBEDDING_KEY not in state:
        print("Available candidate keys:")
        for k, v in state.items():
            if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == len(token_dict):
                print(k, tuple(v.shape))
        raise RuntimeError(f"Could not find {EMBEDDING_KEY}")

    emb_table = state[EMBEDDING_KEY].detach().cpu().numpy().astype(np.float32)
    print("Raw embedding table shape:", emb_table.shape)

    if emb_table.shape[1] != DIM:
        raise RuntimeError(f"Unexpected embedding dim: {emb_table.shape[1]} != {DIM}")

    row_records = []
    mapped_indices = []
    token_ids = []

    for i, row in master.reset_index(drop=True).iterrows():
        ensg = row.get("ensembl_gene_id", "")
        ensg_stripped = ensg.split(".")[0] if ensg else ""

        status = "missing_genecompass_token"
        token_id = ""

        if ensg in token_dict:
            token_id = int(token_dict[ensg])
            status = "mapped_exact_ensembl_gene_id"
        elif ensg_stripped and ensg_stripped in token_dict:
            token_id = int(token_dict[ensg_stripped])
            status = "mapped_stripped_ensembl_gene_id"

        if token_id != "":
            if token_id < 0 or token_id >= emb_table.shape[0]:
                status = "invalid_token_id"
            elif token_id in set(specials.values()):
                status = "special_token_not_gene"
            else:
                mapped_indices.append(i)
                token_ids.append(token_id)

        row_records.append({
            "master_row_index": i,
            "ensembl_gene_id": ensg,
            "gene_symbol": row.get("gene_symbol", ""),
            "entrez_id": row.get("entrez_id", ""),
            "uniprot_id": row.get("uniprot_id", ""),
            "genecompass_token_id": token_id,
            "status": status,
        })

    row_report = pd.DataFrame(row_records)

    mapped = row_report[
        row_report["status"].isin([
            "mapped_exact_ensembl_gene_id",
            "mapped_stripped_ensembl_gene_id",
        ])
    ].copy()

    mapped_token_ids = mapped["genecompass_token_id"].astype(int).to_numpy()
    embeddings = emb_table[mapped_token_ids, :].astype(np.float32)

    genes_cols = [
        "master_row_index",
        "ensembl_gene_id",
        "gene_symbol",
        "entrez_id",
        "uniprot_id",
        "genecompass_token_id",
        "status",
    ]
    genes = mapped[genes_cols].copy()

    norms = np.linalg.norm(embeddings, axis=1)
    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    zero_vectors = int((norms == 0).sum())
    duplicated_ensembl = int(genes["ensembl_gene_id"].duplicated().sum())
    duplicated_tokens = int(genes["genecompass_token_id"].duplicated().sum())

    status = (
        "processed_qc_passed"
        if (
            embeddings.shape[0] == len(genes)
            and embeddings.shape[1] == DIM
            and nan_count == 0
            and inf_count == 0
            and zero_vectors == 0
            and duplicated_ensembl == 0
            and duplicated_tokens == 0
        )
        else "processed_qc_check_needed"
    )

    np.savez_compressed(out_npz, embeddings=embeddings)
    genes.to_csv(out_genes, sep="\t", index=False)
    row_report.to_csv(out_row_report, sep="\t", index=False)

    with open(config_path) as f:
        config = json.load(f)

    metadata = {
        "embedding_name": EMBEDDING_NAME,
        "status": status,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "modality": "gene expression single-cell foundation model / knowledge-informed gene-token embedding",
        "species": "human genes extracted from human-mouse GeneCompass vocabulary",
        "source_repo": str(repo),
        "source_repo_git_commit": safe_git_commit(repo),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_md5": md5sum(ckpt_path),
        "checkpoint_expected_md5_from_sciencedb": "77719f3af27b3d3359032bf4ae315652",
        "checkpoint_config_path": str(config_path),
        "token_dictionary_path": str(token_path),
        "master_table": str(master_path),
        "download_source": "official GeneCompass ScienceDB checkpoint linked from official GitHub README",
        "original_model_or_method": "GeneCompass Base, 12-layer knowledge-informed cross-species foundation model",
        "embedding_generated_by": "Daniel locally by extracting checkpoint embedding weights; no training or inference",
        "embedding_key_extracted": EMBEDDING_KEY,
        "embedding_dimension": DIM,
        "dtype": "float32",
        "pooling_strategy": "none; static learned token embedding weight extracted directly from checkpoint",
        "model_config_summary": {
            "architectures": config.get("architectures"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "intermediate_size": config.get("intermediate_size"),
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "use_promoter": config.get("use_promoter"),
            "use_co_exp": config.get("use_co_exp"),
            "use_gene_family": config.get("use_gene_family"),
            "use_peca_grn": config.get("use_peca_grn"),
            "use_values": config.get("use_values"),
        },
        "notes": [
            "The bundled GeneCompass-emb-0708.npy files were not used because they contain only 1,582 task-specific contextual gene embeddings.",
            "This processed embedding uses the full checkpoint gene-token embedding matrix bert.embeddings.word_embeddings.weight.",
            "Special tokens <pad> and <mask> were excluded.",
            "Rows are ordered according to the mapped subset of master_gene_table_v1_1_enriched.csv.",
            "No model training, fine-tuning, or downstream task data were used.",
        ],
        "raw_master_rows": int(len(master)),
        "raw_genecompass_vocab_size": int(len(token_dict)),
        "processed_gene_count": int(len(genes)),
        "missing_master_genes": int(len(master) - len(genes)),
        "coverage_of_master_table_percent": round(len(genes) / len(master) * 100, 4),
        "matrix_shape": list(embeddings.shape),
        "qc": {
            "rows_match_genes_tsv": bool(embeddings.shape[0] == len(genes)),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_vectors": zero_vectors,
            "duplicated_ensembl_ids": duplicated_ensembl,
            "duplicated_genecompass_token_ids": duplicated_tokens,
            "min_norm": float(norms.min()) if len(norms) else None,
            "median_norm": float(np.median(norms)) if len(norms) else None,
            "max_norm": float(norms.max()) if len(norms) else None,
        },
        "status_counts": row_report["status"].value_counts().to_dict(),
    }

    with open(out_metadata, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(out_report, "w") as f:
        f.write(f"Mapping/QC report for {EMBEDDING_NAME}\n")
        f.write("=" * (22 + len(EMBEDDING_NAME)) + "\n\n")
        f.write(f"Status: {status}\n")
        f.write(f"Output directory: {out_dir}\n")
        f.write(f"Embeddings: {out_npz}\n")
        f.write(f"Genes: {out_genes}\n")
        f.write(f"Metadata: {out_metadata}\n")
        f.write(f"Row mapping report: {out_row_report}\n\n")
        f.write(f"Checkpoint MD5: {metadata['checkpoint_md5']}\n")
        f.write(f"Embedding key: {EMBEDDING_KEY}\n")
        f.write(f"Raw embedding table shape: {emb_table.shape}\n")
        f.write(f"Processed matrix shape: {embeddings.shape}\n")
        f.write(f"Rows match genes.tsv: {embeddings.shape[0] == len(genes)}\n")
        f.write(f"NaN count: {nan_count}\n")
        f.write(f"Inf count: {inf_count}\n")
        f.write(f"Zero vectors: {zero_vectors}\n")
        f.write(f"Duplicated Ensembl IDs: {duplicated_ensembl}\n")
        f.write(f"Duplicated GeneCompass token IDs: {duplicated_tokens}\n")
        f.write(f"Processed genes: {len(genes)}\n")
        f.write(f"Missing master genes: {len(master) - len(genes)}\n")
        f.write(f"Coverage of master table: {metadata['coverage_of_master_table_percent']}%\n\n")
        f.write("Status counts:\n")
        for k, v in row_report["status"].value_counts().items():
            f.write(f"  {k}: {v}\n")

    print("\nDone")
    print("----")
    print("Status:", status)
    print("Raw embedding table shape:", emb_table.shape)
    print("Matrix shape:", embeddings.shape)
    print("Rows match genes.tsv:", embeddings.shape[0] == len(genes))
    print("NaN count:", nan_count)
    print("Inf count:", inf_count)
    print("Zero vectors:", zero_vectors)
    print("Duplicated Ensembl IDs:", duplicated_ensembl)
    print("Duplicated GeneCompass token IDs:", duplicated_tokens)
    print("Processed genes:", len(genes))
    print("Missing master genes:", len(master) - len(genes))
    print("Coverage:", metadata["coverage_of_master_table_percent"], "%")
    print("Output:", out_dir)
    print("Report:", out_report)


if __name__ == "__main__":
    main()
