#!/usr/bin/env python3
"""Plot the saved calibrated CKA and k=100 overlap matrices in a fixed order.

Visualization only: no embedding loading, similarity calculation, clustering,
permutation testing, rank transformation, or consensus reconstruction is done.
Names are read directly from the supplied Excel manifest on every run.
Both views use the reference script's square-root viridis colour mapping;
the matrix files and colourbar labels retain calibrated similarity values.

Standalone bundle:
    python scripts/consensus_graph/cs_12_plot_component_heatmaps.py

Installed in the main project:
    python scripts/consensus_graph/cs_12_plot_component_heatmaps.py \
        --root "$CS_PROJECT_ROOT"
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEATMAP_GAMMA = 0.50
HEATMAP_VALUE_TICKS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)
HEATMAP_CMAP = "viridis"


VIEWS = {
    "platonic_linear_calibrated": (
        "Calibrated linear CKA", "Row-normalized unbiased linear CKA",
        "01_calibrated_cka_reference_order",
    ),
    "aristotelian_local_k100_calibrated": (
        "Calibrated neighbourhood overlap", "Tie-aware local overlap at k = 100",
        "02_calibrated_knn_k100_reference_order",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_order(path: Path) -> list[str]:
    order = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    duplicates = [item for item, count in Counter(order).items() if count > 1]
    if duplicates or not order:
        raise ValueError(f"Empty order or duplicate IDs in {path}: {duplicates}")
    return order


def read_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, sep="\t", index_col=0, float_precision="round_trip")
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError(f"Duplicate embedding IDs: {path}")
    if matrix.shape[0] != matrix.shape[1] or set(matrix.index) != set(matrix.columns):
        raise ValueError(f"Not a square matrix with matching IDs: {path}")
    matrix = matrix.loc[matrix.index, matrix.index].astype(float)
    values = matrix.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite similarity: {path}")
    if not np.array_equal(values, values.T):
        raise ValueError(f"Asymmetric similarity matrix: {path}")
    if not np.array_equal(np.diag(values), np.ones(len(values))):
        raise ValueError(f"Diagonal differs from one: {path}")
    return matrix


def read_labels(path: Path, ids: list[str]) -> pd.DataFrame:
    manifest = pd.read_excel(path, sheet_name=0, dtype=str, keep_default_na=False)
    columns = {"Files name": "embedding_id", "Embedding display name": "display_name",
               "Final modality": "modality"}
    missing = set(columns) - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest columns missing: {sorted(missing)}")
    labels = manifest[list(columns)].rename(columns=columns)
    for column in labels:
        labels[column] = labels[column].str.strip()
    labels = labels.loc[labels.embedding_id.ne("")].set_index("embedding_id")
    if labels.index.has_duplicates:
        raise ValueError("Duplicate embedding IDs in the manifest")
    missing = set(ids) - set(labels.index)
    if missing:
        raise ValueError(f"Panel IDs absent from the manifest: {sorted(missing)}")
    labels = labels.loc[ids].copy()
    if labels.display_name.eq("").any() or labels.display_name.duplicated().any():
        raise ValueError("Blank or duplicate display names in the selected panel")
    return labels


def plot_heatmap(matrix, labels, title, subtitle, genes, output, font_size, dpi):
    # Set the heatmap dimensions in inches so cells remain square. Reserve
    # label margins from measured text lengths; never stretch the raster.
    count = len(matrix)
    cell_inches = max(0.17, font_size * 1.3 / 72)
    side = count * cell_inches
    figure = plt.figure(figsize=(12, 12), dpi=120)
    probe = [figure.text(0, 0, label, fontsize=font_size) for label in labels]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    label_inches = max(t.get_window_extent(renderer).width for t in probe) / figure.dpi
    for artist in probe:
        artist.remove()
    left = bottom = label_inches + 0.24
    right, top = 1.32, 0.9
    width, height = left + side + right, bottom + side + top
    figure.set_size_inches(width, height)
    ax = figure.add_axes([left / width, bottom / height, side / width, side / height])
    # Match pfi_04_plots.py exactly: clip a DISPLAY COPY to [0, 1], then
    # apply gamma=0.5 explicitly. The input/exported matrices stay untouched.
    display = np.power(np.clip(matrix.to_numpy(dtype=float), 0.0, 1.0), HEATMAP_GAMMA)
    tick_positions = np.power(np.asarray(HEATMAP_VALUE_TICKS), HEATMAP_GAMMA)
    im = ax.imshow(display, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0,
                   interpolation="nearest", origin="upper", aspect="equal")
    ax.set_xticks(np.arange(count), labels, rotation=90,
                  ha="center", va="top", fontsize=font_size)
    ax.set_yticks(np.arange(count), labels, fontsize=font_size)
    ax.tick_params(length=2, width=0.4, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    center = (left + side / 2) / width
    figure.text(center, (bottom + side + 0.57) / height, title,
                ha="center", fontsize=16, fontweight="bold")
    figure.text(center, (bottom + side + 0.29) / height,
                f"{subtitle} | {count} embeddings | {genes:,} common genes",
                ha="center", fontsize=10)
    cax = figure.add_axes([(left + side + 0.22) / width,
                          (bottom + 0.05 * side) / height,
                          0.22 / width, 0.9 * side / height])
    colorbar = figure.colorbar(im, cax=cax, ticks=tick_positions)
    colorbar.set_ticklabels([f"{value:g}" for value in HEATMAP_VALUE_TICKS])
    colorbar.set_label("Calibrated similarity (square-root colour scale)",
                       fontsize=11, labelpad=10)
    colorbar.ax.tick_params(labelsize=10)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tick_boxes = [t.get_window_extent(renderer) for t in
                  ax.get_xticklabels() + ax.get_yticklabels()]
    for box in tick_boxes:
        if not figure.bbox.contains(box.x0, box.y0) or not figure.bbox.contains(box.x1, box.y1):
            raise ValueError("A heatmap label would be clipped; increase label margins")
    for group in (tick_boxes[:count], tick_boxes[count:]):
        for i, box in enumerate(group):
            if any(box.overlaps(other) for other in group[i + 1:]):
                raise ValueError("Overlapping tick labels; increase cell spacing")
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=dpi, facecolor="white")
    plt.close(figure)
    return {"figure_inches": [width, height], "cell_inches": cell_inches,
            "label_font_size_pt": font_size, "label_overlap_check": "PASS"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--order-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font-size", type=float, default=9.2)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.font_size <= 0 or args.dpi <= 0:
        parser.error("font size and dpi must be positive")
    root = args.root.resolve()
    results = args.results_dir or root / "results/consensus_graph_v2"
    manifest = args.manifest or root / "config/consensus_graph_embedding_manifest.xlsx"
    order_file = args.order_file or root / "config/consensus_graph_heatmap_order.txt"
    output = args.output_dir or results / "10_figures/component_heatmaps_reference_order"
    summary_path = results / "04_views/view_summary.json"
    consensus_path = results / "06_consensus/consensus_summary.json"
    summary, consensus = read_json(summary_path), read_json(consensus_path)
    if summary.get("status") != "PASS" or consensus.get("status") != "PASS":
        raise ValueError("Upstream metadata is not PASS")
    construction = consensus["construction"]
    if (construction.get("platonic_view") != "platonic_linear_calibrated" or
            construction.get("aristotelian_view") != "aristotelian_local_k100_calibrated"):
        raise ValueError("The recorded primary consensus is not calibrated CKA + k=100 overlap")
    panel = summary["primary_panel"]
    genes = int(panel["fixed_common_genes"])
    paths = {v: results / f"04_views/matrices_primary_fixed/{v}.tsv" for v in VIEWS}
    matrices = {v: read_matrix(path) for v, path in paths.items()}
    ids = set(next(iter(matrices.values())).index)
    if any(set(m.index) != ids for m in matrices.values()):
        raise ValueError("The two matrices do not contain the same embeddings")
    n = len(ids)
    if (n != panel["embeddings"] or n != consensus["n_embeddings"] or
            n * (n - 1) // 2 != panel["pairs"] or panel["pairs"] != consensus["n_pairs"]):
        raise ValueError("Matrix size disagrees with the recorded panel")
    reference_order = read_order(order_file)
    unknown = ids - set(reference_order)
    if unknown:
        raise ValueError(f"Add these retained IDs to the order file: {sorted(unknown)}")
    order = [item for item in reference_order if item in ids]
    excluded = [item for item in reference_order if item not in ids]
    labels = read_labels(manifest, order)
    labels.insert(0, "position", np.arange(1, n + 1))
    labels.insert(1, "reference_position", [reference_order.index(item) + 1 for item in order])
    output.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output / "heatmap_embedding_order.tsv", sep="\t", index=True)
    (output / "reference_ids_not_in_panel.txt").write_text(
        "".join(f"{item}\n" for item in excluded), encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none"})
    report = {
        "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_embeddings": n, "n_pairs": n * (n - 1) // 2, "fixed_common_genes": genes,
        "local_overlap_k": 100, "operation": "reindex_rows_and_columns_only",
        "row_and_column_order_identical": True, "both_heatmaps_same_order": True,
        "reference_ids_omitted": excluded, "names_source": str(manifest),
        "name_source_policy": "read directly from manifest; ignore frozen display names",
        "inputs": {str(p): sha256(p) for p in
                   [summary_path, consensus_path, manifest, order_file, *paths.values()]},
        "script_sha256": sha256(Path(__file__)),
        "colour_scale": {
            "type": "square_root", "cmap": HEATMAP_CMAP, "gamma": HEATMAP_GAMMA,
            "display_formula": "clip(calibrated_similarity, 0, 1) ** 0.5",
            "display_vmin": 0.0, "display_vmax": 1.0,
            "tick_labels_calibrated_values": list(HEATMAP_VALUE_TICKS),
            "tick_positions": np.power(np.asarray(HEATMAP_VALUE_TICKS), HEATMAP_GAMMA).tolist(),
            "same_mapping_for_both_views": True,
            "negative_value_policy": "zero colour only; original negative values retained in TSVs",
        },
        "software": {"numpy": np.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__}, "views": {},
    }
    for view, (title, subtitle, filename) in VIEWS.items():
        original = matrices[view]
        reordered = original.loc[order, order]
        restored = reordered.loc[original.index, original.columns]
        if not np.array_equal(restored.to_numpy(), original.to_numpy()):
            raise AssertionError("Reordering changed matrix values")
        tsv = output / f"{filename}.tsv"
        reordered.to_csv(tsv, sep="\t", float_format="%.17g", index_label="embedding_id")
        if not np.array_equal(read_matrix(tsv).to_numpy(), reordered.to_numpy()):
            raise AssertionError("Export changed matrix values")
        layout = plot_heatmap(reordered, labels.display_name.tolist(), title, subtitle,
                              genes, output / filename, args.font_size, args.dpi)
        report["views"][view] = {"values_unchanged_exactly": True,
                                  "negative_pairs_displayed_with_zero_colour": int(np.count_nonzero(
                                      original.to_numpy()[np.triu_indices(n, 1)] < 0)),
                                  "reordered_matrix_sha256": sha256(tsv), **layout}
        print(f"PASS {view}: {n} embeddings; values unchanged; PDF/PNG/SVG written")
    (output / "heatmap_reorder_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Display names: read directly from {manifest}")
    print(f"Reference IDs excluded from fixed panel: {len(excluded)}")
    print("Colour scale: shared viridis; gamma=0.5; colourbar labels are calibrated values")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
