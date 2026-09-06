#!/usr/bin/env python3
"""
pfi_04_plots.py
 
Step 4 of the Pairwise Full-Intersection (PFI) analysis: figures.
 
Reads the tables written by pfi_01 and pfi_03 and writes publication-oriented
figures. Purely a rendering step; it computes no new statistics beyond what is
needed to draw, and it never modifies the result tables.
 
Figures written to <analysis_output>/figures/:
 
  local_calibration.png    raw overlap vs chance vs calibrated effect
  heatmaps/*.png                67 x 67 manifest-selected similarity heatmaps
 
Usage:
    python scripts/pairwise_full_intersection/pfi_04_plots.py \
        --config config/pairwise_full_intersection_v1.yaml
"""
 
from __future__ import annotations
 
import argparse
import json
import sys
import warnings
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")  # headless server: no display needed
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ConstantInputWarning

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfi_manifest import (load_manifest, manifest_maps, panel_sha256,
                          resolve_path, sha256_file)
 
warnings.filterwarnings("ignore", category=ConstantInputWarning)
 
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})
 
INK = "#22303C"
ACCENT = "#B4573C"
MUTED = "#8A9AA6"
LIGHT = "#D8DEE3"
 
K_VALUES = [10, 25, 50, 100]
 
# ===================================================================
# Heatmap color scale and ordering, matching 03_common_panel_atlas.py
# ===================================================================
#
# Square-root color scaling: raw 0.20 is displayed at sqrt(0.20) = 0.447 on
# viridis, which is approximately blue-green rather than dark blue. Saved
# matrices and colorbar tick LABELS remain raw values; only the color position
# is transformed.
HEATMAP_GAMMA = 0.50
HEATMAP_RAW_TICKS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)
HEATMAP_CMAP = "viridis"
 
# Registered reference order updated for the manifest-frozen 67-embedding panel.
# Every heatmap uses this same order so that figures are directly comparable
# to each other and to the common-panel atlas. Heatmaps are NOT re-clustered
# per figure; doing so would break that comparability.
REFERENCE_HEATMAP_ORDER = (
    "uce_33l",
    "newt_go_graph",
    "frogs_archs4_256",
    "genecompass_base_gene_token",
    "geneformer_v2_316m",
    "scgpt_whole_human",
    "scgpt_pancancer",
    "scprint_medium_v1_5_gene",
    "gene2vec",
    "gtex_tissue_median_log1p_zscore",
    "probe_tcga_embedding",
    "mahi_all_contexts_mean",
    "mahi_global",
    "alphagenome_tss1mb_regulatory_tracksummary_pca256",
    "enformer_tss_393kb_human_central3",
    "basenji2_tss_131kb_human_central3",
    "deepsea_tss_1kb_epigenomic_probs",
    "hpa_normal_ihc_tissue_level_zscore",
    "newt_cellnet",
    "depmap_crispr_gene_effect_pca256",
    "helix_mrna_full_transcript",
    "rnafm_full_transcript",
    "hyenadna_medium_160k_cds",
    "dnabert2_117m_cds",
    "nt_v2_500m_multispecies_cds",
    "esm2",
    "probe_esmb1",
    "probe_albert",
    "probe_bert_bfd",
    "probe_t5",
    "orthrus_base_4track_full_transcript",
    "probe_bert_pfam",
    "msa_transformer_depth1_uniprot_human",
    "probe_seqvec",
    "probe_cpc_prot",
    "probe_protvec",
    "probe_aac",
    "probe_apaac",
    "probe_learned_vec",
    "probe_unirep",
    "probe_xlnet",
    "probe_blast",
    "probe_hmmer",
    "node2vec_consensus_ppi",
    "newt_go_256",
    "go_bp_uniprot_tfidf_svd256",
    "hetionet_nonppi_biomedical_context_tfidf_svd256",
    "primekg_nonppi_biomedical_context_tfidf_svd256",
    "drkg_transe",
    "deepnf_string_v12_ppmi_svd_fusion",
    "mashup_string",
    "string_gnn",
    "string_wavegc",
    "genept_ada",
    "genept_model3_gene_protein",
    "pubmedbert_gene_protein_summary",
    "biobert_gene_protein_summary",
    "opa2vec_goa_human",
    "bioconceptvec_fasttext",
    "bioconceptvec_skipgram",
    "bioconceptvec_cbow",
    "bioconceptvec_glove",
    "probe_mut2vec",
    "hpa_subcellular_location_multihot_zscore",
    "newt_msigdb_bundle",
    "spliceai_canonical_splice_site_profile",
    "scfoundation_gene_pos",
)
 
 
def resolve_heatmap_order(embedding_ids):
    """Filter the registered order to the current panel, order preserved."""
    available = set(embedding_ids)
    registered = set(REFERENCE_HEATMAP_ORDER)
    unregistered = sorted(available - registered)
    order = [e for e in REFERENCE_HEATMAP_ORDER if e in available]
    if unregistered:
        # Do not silently drop: append, warn, and keep the figure usable.
        print(f"  WARNING: {len(unregistered)} embeddings absent from the "
              f"registered order, appended alphabetically: {unregistered}")
        order = order + unregistered
    return order
 
 
def matrix_from_pairs(results, order, column):
    """Symmetric embedding x embedding matrix for one pair-level column."""
    index = {e: i for i, e in enumerate(order)}
    size = len(order)
    matrix = np.full((size, size), np.nan)
    for row in results.itertuples():
        value = getattr(row, column, np.nan)
        i = index.get(row.embedding_x)
        j = index.get(row.embedding_y)
        if i is None or j is None:
            continue
        matrix[i, j] = matrix[j, i] = value
    np.fill_diagonal(matrix, 1.0)
    return pd.DataFrame(matrix, index=order, columns=order)
 
 
def enlarge_fonts_for_pdf(fig, increment=1.5):
    """Increase all text sizes for the PDF-only rendering."""
    for text in fig.findobj(match=Text):
        text.set_fontsize(text.get_fontsize() + increment)


def render_heatmap(similarity, labels, name, title, colorbar_label, path, dpi=200):
    """
    Render one heatmap on the registered order and the square-root color
    scale. The non-linear mapping is applied explicitly rather than through a
    matplotlib norm object, matching the reference implementation and avoiding
    backend differences in how norms are forwarded.
    """
    raw = np.clip(similarity.to_numpy(dtype=float), 0.0, 1.0)
    display = np.power(raw, HEATMAP_GAMMA)
    tick_positions = np.power(np.asarray(HEATMAP_RAW_TICKS, dtype=float),
                              HEATMAP_GAMMA)
 
    order = list(similarity.index)
    size = max(14, 0.28 * len(order))
    fig, ax = plt.subplots(figsize=(size, size))
 
    image = ax.imshow(display, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0,
                      interpolation="nearest", aspect="equal")
 
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, ha="center", va="top", fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Embedding")
    ax.set_ylabel("Embedding")
    ax.set_title(f"{title} \u00b7 shared registered reference order \u00b7 "
                 "nonlinear square-root color scale", fontsize=14, pad=12)
 
    colorbar = fig.colorbar(image, ax=ax, fraction=0.030, pad=0.02,
                            ticks=tick_positions)
    colorbar.set_label(f"{colorbar_label}; displayed color = sqrt(raw value)")
    colorbar.set_ticklabels([f"{v:g}" for v in HEATMAP_RAW_TICKS])
    colorbar.ax.tick_params(labelsize=8)
 
    fig.text(0.5, 0.01,
             "Colors equal sqrt(raw value): raw 0.20 maps to color position "
             "0.447 (blue-green). Saved matrices and colorbar labels are raw.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=dpi)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    colorbar.set_label("")
    for figure_text in list(fig.texts):
        figure_text.remove()
    enlarge_fonts_for_pdf(fig)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
 
 
 
def shorten(name: str, width: int = 26) -> str:
    return name if len(name) <= width else name[: width - 1] + "\u2026"
 
 
# ----------------------------------------------------------------- figures
 
 
def fig01_intersection_sizes(pairs, strict_n, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
 
    ax = axes[0]
    ax.hist(pairs.n_intersection, bins=50, color=MUTED, edgecolor="white", linewidth=0.4)
    ax.axvline(strict_n, color=ACCENT, lw=1.6, ls="--")
    ax.annotate(f"strict panel-wide universe\n{strict_n:,} genes",
                xy=(strict_n, ax.get_ylim()[1] * 0.92), xytext=(8, 0),
                textcoords="offset points", color=ACCENT, fontsize=8, va="top")
    ax.set_xlabel("genes in pair intersection")
    ax.set_ylabel("pairs")
    ax.set_title("Per-pair intersection sizes")
 
    ax = axes[1]
    ratio = pairs.n_intersection / strict_n
    ax.hist(ratio, bins=50, color=MUTED, edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color=ACCENT, lw=1.6, ls="--")
    ax.axvline(ratio.median(), color=INK, lw=1.4)
    ax.annotate(f"median {ratio.median():.2f}x", xy=(ratio.median(), ax.get_ylim()[1] * 0.92),
                xytext=(8, 0), textcoords="offset points", color=INK, fontsize=8, va="top")
    ax.set_xlabel("intersection size / fixed universe")
    ax.set_ylabel("pairs")
    ax.set_title("Power gained per pair")
 
    fig.suptitle("What the fixed gene universe was costing", y=1.02, fontsize=11)
    fig.savefig(path)
    plt.close(fig)
 
 
def fig02_pfi_vs_atlas(results, atlas, comparison, atlas_n, path):
    metrics = comparison.metric.tolist()
    n = len(metrics)
    if n == 0:
        return False
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    axes = np.atleast_1d(axes)
 
    for ax, (_, row) in zip(axes, comparison.iterrows()):
        merged = results.merge(atlas, on="canonical_pair", suffixes=("_pfi", "_atl"))
        left = f"{row.metric}_pfi" if f"{row.metric}_pfi" in merged else row.metric
        right = (f"{row.atlas_column}_atl" if f"{row.atlas_column}_atl" in merged
                 else row.atlas_column)
        x = pd.to_numeric(merged[right], errors="coerce")
        y = pd.to_numeric(merged[left], errors="coerce")
        ok = x.notna() & y.notna()
 
        ax.scatter(x[ok], y[ok], s=4, alpha=0.25, color=INK, linewidths=0)
        lo = float(min(x[ok].min(), y[ok].min()))
        hi = float(max(x[ok].max(), y[ok].max()))
        ax.plot([lo, hi], [lo, hi], color=ACCENT, lw=1.0, ls="--")
        ax.set_xlabel(f"atlas (fixed {atlas_n:,} genes)")
        ax.set_ylabel("PFI (per-pair intersection)")
        ax.set_title(f"{shorten(row.metric, 30)}\nSpearman rho = {row.spearman_pfi_vs_atlas:.3f}")
 
    fig.suptitle("Pair rankings survive the change of gene set", y=1.03, fontsize=11)
    fig.savefig(path)
    plt.close(fig)
    return True
 
 
def fig03_delta_vs_size(results, atlas, comparison, path):
    metrics = comparison.metric.tolist()
    n = len(metrics)
    if n == 0:
        return False
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    axes = np.atleast_1d(axes)
    merged = results.merge(atlas, on="canonical_pair", suffixes=("_pfi", "_atl"))
 
    for ax, (_, row) in zip(axes, comparison.iterrows()):
        left = f"{row.metric}_pfi" if f"{row.metric}_pfi" in merged else row.metric
        right = (f"{row.atlas_column}_atl" if f"{row.atlas_column}_atl" in merged
                 else row.atlas_column)
        delta = pd.to_numeric(merged[left], errors="coerce") - pd.to_numeric(merged[right], errors="coerce")
        size = pd.to_numeric(merged.n_intersection, errors="coerce")
        ok = delta.notna() & size.notna()
 
        ax.scatter(size[ok], delta[ok], s=4, alpha=0.25, color=INK, linewidths=0)
        ax.axhline(0.0, color=MUTED, lw=0.8)
        if ok.sum() > 2:
            coefficients = np.polyfit(size[ok], delta[ok], 1)
            xs = np.linspace(size[ok].min(), size[ok].max(), 50)
            ax.plot(xs, np.polyval(coefficients, xs), color=ACCENT, lw=1.4)
        ax.set_xlabel("genes in pair intersection")
        ax.set_ylabel("PFI - atlas")
        ax.set_title(f"{shorten(row.metric, 30)}\nrho(delta, size) = {row.spearman_delta_vs_size:.3f}")
 
    fig.suptitle("Pairs with more shared genes gained more alignment", y=1.03, fontsize=11)
    fig.savefig(path)
    plt.close(fig)
    return True
 
 
def fig04_metric_vs_n(tiers, path):
    table = tiers.sort_values("abs_spearman_vs_size", ascending=True).tail(24)
    matched = table.get("n_fixed_across_pairs", pd.Series([False] * len(table)))
    colors = [ACCENT if bool(m) else MUTED for m in matched]
 
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.24 * len(table))))
    ax.barh(range(len(table)), table.abs_spearman_vs_size, color=colors, height=0.72)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([shorten(m, 34) for m in table.metric], fontsize=7)
    ax.set_xlabel("|Spearman| vs intersection size")
    ax.axvline(0.15, color=LIGHT, lw=1.0, ls=":")
    ax.axvline(0.35, color=LIGHT, lw=1.0, ls=":")
 
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    ax.legend(handles, ["fixed n (association is substantive)",
                        "variable n (may be partly arithmetic)"],
              loc="lower right", fontsize=7.5)
    ax.set_title("Dependence of each metric on intersection size")
    fig.savefig(path)
    plt.close(fig)
 
 
def fig05_local_calibration(results, atlas, strict_n, atlas_n, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
 
    raw = results[[f"local_overlap_raw_k{k}" for k in K_VALUES]].mean(axis=1)
    null = results[[f"local_null_mean_k{k}" for k in K_VALUES]].mean(axis=1)
    effect = results.local_effect_mean
    local_n = int(results.local_n_genes.iloc[0])
 
    ax = axes[0]
    parts = [raw.dropna(), null.dropna(), effect.dropna()]
    labels = [f"PFI raw\n(n={local_n:,})", "PFI chance\nbaseline", "PFI calibrated\neffect"]
    if atlas is not None and "local_effect_mean" in atlas.columns:
        parts.insert(0, pd.to_numeric(atlas.local_effect_mean, errors="coerce").dropna())
        labels.insert(0, f"atlas\n'local_effect_mean'\n(n={atlas_n:,})")
    box = ax.boxplot(parts, showfliers=False, patch_artist=True, widths=0.55)
    for index, patch in enumerate(box["boxes"]):
        is_atlas = (index == 0 and len(parts) == 4)
        patch.set_facecolor(ACCENT if is_atlas else MUTED)
        patch.set_alpha(0.55)
        patch.set_edgecolor(INK)
    for element in ("medians", "whiskers", "caps"):
        for artist in box[element]:
            artist.set_color(INK)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("neighbourhood overlap")
    ax.set_title("The atlas column is raw overlap, not an effect")
 
    ax = axes[1]
    for k in K_VALUES:
        observed = results[f"local_overlap_raw_k{k}"].median()
        baseline = results[f"local_null_mean_k{k}"].median()
        ax.plot([k, k], [baseline, observed], color=LIGHT, lw=4, solid_capstyle="round")
        ax.scatter([k], [baseline], color=ACCENT, s=22, zorder=3)
        ax.scatter([k], [observed], color=INK, s=22, zorder=3)
        ax.annotate(f"{observed / baseline:.1f}x", xy=(k, observed), xytext=(6, -2),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("k (neighbourhood size)")
    ax.set_ylabel("median overlap")
    ax.set_title("Observed vs chance, by k")
    ax.scatter([], [], color=INK, s=22, label="observed")
    ax.scatter([], [], color=ACCENT, s=22, label="permutation null")
    ax.legend(fontsize=7.5)
 
    fig.suptitle("Local overlap needs its baseline stated", y=1.03, fontsize=11)
    fig.savefig(path)
    if fig._suptitle is not None:
        fig._suptitle.remove()
    enlarge_fonts_for_pdf(fig)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
 
 
def write_heatmaps(results, panel, out_root, figure_dir):
    """
    Render only the manuscript-selected heatmaps: row-normalized linear CKA,
    row-normalized RBF CKA (bandwidth factor 1.0), and raw tie-aware
    k-nearest-neighbour overlap at each configured k. Each matrix is also
    saved as a raw-valued TSV.
    """
    order = resolve_heatmap_order(panel.embedding_id.tolist())
    display_map, modality_map = manifest_maps(panel)
 
    definitions = []
    if "unbiased_linear_cka_row_normalized" in results.columns:
        definitions.append((
            "linear_cka", "unbiased_linear_cka_row_normalized",
            "Absolute global representation alignment\nRow-normalized unbiased linear CKA",
            "Unbiased linear CKA (raw value)"))
    rbf_column = "unbiased_rbf_cka_row_normalized_bw_1p0"
    if rbf_column in results.columns:
        definitions.append((
            "rbf_cka_bw_1p0", rbf_column,
            "Absolute global representation alignment\nUnbiased RBF CKA, bandwidth factor 1.0",
            "Unbiased RBF CKA (raw value)"))
 
    for k in K_VALUES:
        raw_column = f"local_overlap_raw_k{k}"
        if raw_column in results.columns:
            definitions.append((
                f"local_k{k}", raw_column,
                f"Absolute local gene-neighbourhood alignment\nTie-aware neighbourhood overlap at k={k}",
                "Tie-aware overlap (raw proportion)"))
 
    matrix_dir = out_root / "matrices"
    heatmap_dir = figure_dir / "heatmaps"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    order_table = pd.DataFrame({
        "heatmap_position": range(1, len(order) + 1),
        "embedding_id": order,
        "display_name": [display_map[value] for value in order],
        "modality": [modality_map[value] for value in order],
    })
    order_table.to_csv(matrix_dir / "heatmap_embedding_order.tsv", sep="\t", index=False)
 
    written = []
    for name, column, title, colorbar_label in definitions:
        similarity = matrix_from_pairs(results, order, column)
        display_similarity = similarity.rename(
            index=display_map, columns=display_map
        ).rename_axis("embedding_display_name")
        display_similarity.to_csv(matrix_dir / f"matrix__{name}.tsv", sep="\t")
        labels = [display_map[value] for value in order]
        render_heatmap(similarity, labels, name, title, colorbar_label,
                       heatmap_dir / f"heatmap__{name}.png")
        written.append(name)
    return written
 
 
def fig07_tie_diagnostics(results, path):
    tie_x = results[[f"local_boundary_tie_rate_x_k{k}" for k in K_VALUES]].mean(axis=1)
    tie_y = results[[f"local_boundary_tie_rate_y_k{k}" for k in K_VALUES]].mean(axis=1)
    worst = np.maximum(tie_x, tie_y)
    both = np.minimum(tie_x, tie_y)
 
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
 
    ax = axes[0]
    ax.hist(worst, bins=50, color=MUTED, edgecolor="white", linewidth=0.4)
    ax.axvline(0.10, color=ACCENT, lw=1.4, ls="--")
    ax.set_xlabel("boundary-tie rate, worse side")
    ax.set_ylabel("pairs")
    ax.set_title(f"One-sided ties: {int((worst > 0.10).sum())} of {len(results):,} pairs > 10%")
 
    ax = axes[1]
    ax.scatter(both, results.local_effect_mean, s=5, alpha=0.3, color=INK, linewidths=0)
    ax.axvline(0.10, color=ACCENT, lw=1.4, ls="--")
    ax.set_xlabel("boundary-tie rate, better side")
    ax.set_ylabel("local_effect_mean")
    ax.set_title(f"Two-sided ties: {int((both > 0.10).sum())} pairs\n"
                 "(only these depend on the combination rule)")
 
    fig.suptitle("Tie handling affects a small, identifiable minority", y=1.03, fontsize=11)
    fig.savefig(path)
    plt.close(fig)
 
 
# -------------------------------------------------------------------- main
 
 
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--atlas-pairs", type=Path, default=None)
    args = parser.parse_args()
 
    config = yaml.safe_load(args.config.read_text())
    panel, manifest_path = load_manifest(config, args.config)
    inter = resolve_path(config["intersection_output"], args.config)
    out = resolve_path(config["analysis_output"], args.config)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    summary_metadata = json.loads((out / "summary_metadata.json").read_text())
    if summary_metadata.get("panel_sha256") != panel_sha256(panel):
        raise RuntimeError("summary outputs were produced from a different manifest")
    if summary_metadata.get("status") != "PASS":
        raise RuntimeError(
            "summary_metadata.json is not publication-complete; rerun steps 2-3 "
            "without --allow-incomplete"
        )
 
    print("=" * 68)
    print("PFI STEP 4: FIGURES")
    print("=" * 68)
 
    results = pd.read_csv(out / "pair_results.tsv", sep="\t")
    result_ids = set(results.embedding_x) | set(results.embedding_y)
    manifest_ids = set(panel.embedding_id)
    if result_ids != manifest_ids:
        raise RuntimeError(
            "pair_results.tsv does not contain the complete current manifest panel; "
            "rerun steps 1-3 before plotting"
        )
    pairs = pd.read_csv(inter / "pair_intersections.tsv", sep="\t")
    strict_n = int(json.loads((inter / "run_metadata.json").read_text())["strict_universal_genes"])
 
    tiers = pd.read_csv(out / "comparability_tiers.tsv", sep="\t")
    comparison_path = out / "comparison_vs_atlas.tsv"
    comparison = pd.read_csv(comparison_path, sep="\t") if comparison_path.is_file() else pd.DataFrame()
 
    atlas = None
    atlas_n = int(config.get("atlas_gene_count", 4605))
    atlas_path = args.atlas_pairs or config.get("atlas_pair_table")
    if atlas_path:
        atlas_file = resolve_path(str(atlas_path), args.config)
        if atlas_file.is_file():
            atlas = pd.read_csv(atlas_file, sep="\t")
            atlas["canonical_pair"] = ["__VS__".join(sorted((str(a), str(b))))
                                       for a, b in zip(atlas.embedding_x, atlas.embedding_y)]
 
    written = []
 
 
    fig05_local_calibration(
        results, atlas, strict_n, atlas_n, figures / "local_calibration.png")
    written.extend(["local_calibration.png", "local_calibration.pdf"])
 
    heatmap_names = write_heatmaps(results, panel, out, figures)
    for name in heatmap_names:
        written.extend([f"heatmaps/heatmap__{name}.png", f"heatmaps/heatmap__{name}.pdf"])
 
 
    for name in written:
        print(f"  wrote {name}")
    (figures / "figure_metadata.json").write_text(json.dumps({
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config),
        "manifest_file": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "panel_sha256": panel_sha256(panel),
        "n_embeddings": int(len(panel)),
        "n_pairs": int(len(results)),
        "heatmap_order_embedding_ids": resolve_heatmap_order(panel.embedding_id.tolist()),
        "written_figures": written,
        "status": "PASS",
    }, indent=2))
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(figures)}"
        for path in sorted(figures.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (figures / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    print(f"\nDONE -> {figures}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
