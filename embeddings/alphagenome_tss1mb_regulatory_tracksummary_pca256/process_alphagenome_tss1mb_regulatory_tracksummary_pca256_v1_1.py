from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

RAW = Path.home() / "data/raw_embeddings/alphagenome_tss1mb_regulatory_v1_1"
CHUNKDIR = RAW / "chunks_summary_raw"
MASTER = Path.home() / "metadata/master_gene_table_v1_1_enriched.csv"

OUT = Path.home() / "data/processed_embeddings/alphagenome_tss1mb_regulatory_tracksummary_pca256_v1_1"
OUT.mkdir(parents=True, exist_ok=True)

EMBEDDING_NAME = "AlphaGenome-TSS1Mb-regulatoryTrackSummary-PCA256-v1.1"
N_COMPONENTS = 256

print("Reading chunks...")
files = sorted(CHUNKDIR.glob("alphagenome_tss1mb_regulatory_summary_chunk_*.tsv.gz"))
if not files:
    raise SystemExit("No chunk files found.")

dfs = [pd.read_csv(f, sep="\t", dtype=str) for f in files]
raw = pd.concat(dfs, ignore_index=True)

if raw["ensembl_gene_id"].duplicated().any():
    dup = raw.loc[raw["ensembl_gene_id"].duplicated(), "ensembl_gene_id"].head().tolist()
    raise SystemExit(f"Duplicated Ensembl IDs found, example: {dup}")

feature_meta = pd.read_csv(RAW / "alphagenome_tss1mb_regulatory_feature_metadata.tsv", sep="\t", dtype=str).fillna("")
feature_cols = feature_meta["feature_name"].tolist()

missing = [c for c in feature_cols if c not in raw.columns]
if missing:
    raise SystemExit(f"Missing feature columns, first examples: {missing[:5]}")

master = pd.read_csv(MASTER, dtype=str).fillna("")
standard_chroms = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

target = master[
    (master["gene_type"] == "protein_coding")
    & (master["assembly"] == "GRCh38")
    & (master["chromosome"].isin(standard_chroms))
].copy()

target = target[["ensembl_gene_id", "gene_symbol", "gene_type", "chromosome", "gene_start", "gene_end", "strand", "assembly"]]

merged = target.merge(
    raw[["ensembl_gene_id", "input_interval", "tss_1based"] + feature_cols],
    on="ensembl_gene_id",
    how="left",
    validate="one_to_one",
)

covered = merged[feature_cols].notna().all(axis=1)
print("Target genes:", len(target))
print("Covered genes:", int(covered.sum()))
print("Missing genes:", int((~covered).sum()))

if covered.sum() == 0:
    raise SystemExit("No covered genes after merge.")

gene_meta = merged.loc[covered, [
    "ensembl_gene_id", "gene_symbol", "gene_type", "chromosome",
    "gene_start", "gene_end", "strand", "assembly", "tss_1based", "input_interval"
]].copy()

gene_meta.insert(0, "embedding_row_index", range(len(gene_meta)))

print("Building raw matrix...")
X = merged.loc[covered, feature_cols].astype("float32").to_numpy()
X[~np.isfinite(X)] = np.nan

nan_count = int(np.isnan(X).sum())
if nan_count:
    print("NaN values found:", nan_count)
    col_means = np.nanmean(X, axis=0)
    col_means[~np.isfinite(col_means)] = 0.0
    inds = np.where(np.isnan(X))
    X[inds] = col_means[inds[1]]

mean = X.mean(axis=0, dtype=np.float64).astype("float32")
std = X.std(axis=0, dtype=np.float64).astype("float32")
zero_std = std == 0
std_safe = std.copy()
std_safe[zero_std] = 1.0

Xz = ((X - mean) / std_safe).astype("float32")

print("Fitting PCA...")
pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=0)
Z = pca.fit_transform(Xz).astype("float32")

print("Saving outputs...")
np.save(OUT / "embeddings.npy", Z)
np.save(OUT / "raw_summary_zscore_matrix.npy", Xz)
np.save(OUT / "pca_components.npy", pca.components_.astype("float32"))

gene_meta.to_csv(OUT / "gene_metadata.tsv", sep="\t", index=False)
feature_meta.to_csv(OUT / "raw_summary_feature_metadata.tsv", sep="\t", index=False)

pd.DataFrame({
    "feature_name": feature_cols,
    "mean": mean,
    "std": std,
    "zero_std": zero_std,
}).to_csv(OUT / "raw_summary_feature_standardization.tsv", sep="\t", index=False)

pd.DataFrame({
    "pc": [f"PC{i+1}" for i in range(N_COMPONENTS)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "explained_variance": pca.explained_variance_,
}).to_csv(OUT / "pca_explained_variance.tsv", sep="\t", index=False)

metadata = {
    "embedding_name": EMBEDDING_NAME,
    "status": "processed_qc_passed",
    "raw_source_directory": str(RAW),
    "master_table": str(MASTER),
    "n_chunk_files": len(files),
    "n_raw_rows": int(len(raw)),
    "n_unique_raw_genes": int(raw["ensembl_gene_id"].nunique()),
    "n_target_genes": int(len(target)),
    "n_covered_genes": int(covered.sum()),
    "n_missing_genes": int((~covered).sum()),
    "n_raw_summary_features": int(len(feature_cols)),
    "n_pca_components": int(N_COMPONENTS),
    "raw_outputs_saved": False,
    "input_window": "1 Mb centered on canonical TSS from master table",
    "coordinate_system": "GRCh38; master table 1-based coordinates converted to AlphaGenome 0-based half-open interval anchor",
    "requested_outputs": ["CAGE", "RNA_SEQ", "DNASE", "ATAC", "CHIP_HISTONE", "CHIP_TF", "PROCAP"],
    "summary_statistics": ["central_1kb_mean", "central_10kb_mean", "central_100kb_mean", "full_1mb_mean", "central_10kb_max"],
    "transform": "signed log1p per summary feature, z-score across genes, PCA to 256 dimensions",
}

with open(OUT / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

with open(OUT / "qc_report.txt", "w") as f:
    f.write(f"{EMBEDDING_NAME}\n")
    f.write("=" * len(EMBEDDING_NAME) + "\n\n")
    for k, v in metadata.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nembedding_shape: {Z.shape}\n")
    f.write(f"raw_zscore_shape: {Xz.shape}\n")
    f.write(f"total_pca_explained_variance_ratio: {float(pca.explained_variance_ratio_.sum()):.6f}\n")
    f.write(f"nan_values_imputed_before_zscore: {nan_count}\n")
    f.write(f"zero_std_features: {int(zero_std.sum())}\n")

# Copy provenance files.
for name in [
    "alphagenome_output_metadata_all_tracks.tsv",
    "alphagenome_ontology_panel_v1_1.tsv",
    "alphagenome_ontology_panel_v1_1_track_counts_by_output.tsv",
    "alphagenome_tss1mb_regulatory_run_config.json",
]:
    src = RAW / name
    if src.exists():
        shutil.copy2(src, OUT / name)

print("\nDone")
print("Output:", OUT)
print("Embedding shape:", Z.shape)
print("Raw z-scored summary shape:", Xz.shape)
print("PCA variance sum:", float(pca.explained_variance_ratio_.sum()))
print("QC report:", OUT / "qc_report.txt")
