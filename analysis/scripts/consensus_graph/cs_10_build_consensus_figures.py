#!/usr/bin/env python3
"""Build publication-ready figures for Workflow 33.

This stage visualizes results already established in cs_06 through cs_09. It
does not recompute similarities, communities, modality tests, or residuals.  The
network layout is a deterministic weighted force layout learned only from the
primary consensus graph; community and modality annotations are overlaid after
the coordinates are fixed.

Outputs comprise a graph map, a three-panel evidence figure, manuscript-ready
community heatmap panels, exact plotting tables, a caption, and a provenance
summary.  PNG, PDF, and SVG versions are written for each figure.  The figures
are descriptive displays of the formal and descriptive results recorded
upstream; no new p-values are calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


EXPECTED_EMBEDDINGS = 56
EXPECTED_PAIRS = 1_540
EXPECTED_MODALITIES = 11
SEED = 33_110
LAYOUT_ITERATIONS = 700
TOP_BRIDGE_LABELS = EXPECTED_EMBEDDINGS
GRAPH_LABEL_FONTSIZE = 8.2
GRAPH_LABEL_INITIAL_OFFSET = 0.035
GRAPH_NODE_CLEARANCE = 0.014

PRIMARY_SCORE = "consensus_similarity"
PLATONIC_SCORE = "platonic_percentile"
ARISTOTELIAN_SCORE = "aristotelian_k100_percentile"
RESIDUAL_SCORE = "modality_adjusted_consensus_residual"

MODALITY_ORDER = [
    "Biomedical text",
    "DNA sequence",
    "RNA sequence",
    "Functional",
    "Multimodal",
    "Knowledge graph",
    "Networks",
    "Protein sequence",
    "Regulatory genomics",
    "Expression",
    "Single-cell",
]
MODALITY_COLORS = {
    # Reference-inspired palette used in the manuscript graph.  DNA/RNA and
    # expression/single-cell were combined in the reference, so each pair is
    # represented here by related but distinguishable shades.
    "Biomedical text": "#B35C9C",
    "DNA sequence": "#2A9D8F",
    "RNA sequence": "#66C2BD",
    "Functional": "#F18DA5",
    "Multimodal": "#F2C247",
    "Knowledge graph": "#9C6B3E",
    "Networks": "#59A14F",
    "Protein sequence": "#F28E2B",
    "Regulatory genomics": "#4E79A7",
    "Expression": "#E15759",
    "Single-cell": "#C43C46",
}
COMMUNITY_MARKERS = {
    index: marker
    for index, marker in enumerate(("o", "s", "^", "D", "P", "X", "v", "<", ">", "h"), start=1)
}
COMMUNITY_COLORS = {
    index: color
    for index, color in enumerate(
        ("#3B5BA7", "#9C4F96", "#D07B20", "#3C8D70", "#A65E2E", "#6C6FB3", "#C04B67", "#567D46", "#7D5A9E", "#A07D32"),
        start=1,
    )
}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument(
        "--consensus-dir",
        type=Path,
        default=None,
        help="Input consensus directory (default: consensus_graph_v2/06_consensus).",
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Input graph directory (default: consensus_graph_v2/08_graph_structure).",
    )
    parser.add_argument(
        "--interpretation-dir",
        type=Path,
        default=None,
        help=(
            "Input interpretation directory "
            "(default: consensus_graph_v2/09_community_interpretation)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: consensus_graph_v2/10_figures).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"Layout seed (default: {SEED})."
    )
    parser.add_argument(
        "--layout-iterations",
        type=int,
        default=LAYOUT_ITERATIONS,
        help=f"Force-layout iterations (default: {LAYOUT_ITERATIONS}).",
    )
    parser.add_argument(
        "--top-bridge-labels",
        type=int,
        default=TOP_BRIDGE_LABELS,
        help=(
            "Number of nodes labeled on the graph map, selected by bridge rank "
            f"(default: all {TOP_BRIDGE_LABELS})."
        ),
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run a deterministic end-to-end test."
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def numeric(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"{label} is missing numeric column {column!r}")
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} column {column!r} contains non-finite values")
    return values


def parse_boolean(series: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    lowered = series.astype(str).str.strip().str.lower()
    if not lowered.isin(mapping).all():
        bad = sorted(set(lowered[~lowered.isin(mapping)]))
        raise ValueError(f"{label} contains invalid booleans: {bad}")
    return lowered.map(mapping).to_numpy(dtype=bool)


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(require_file(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def verify_recorded_hash(
    payload: dict[str, object], section: str, key: str, path: Path
) -> None:
    section_payload = payload.get(section, {})
    if not isinstance(section_payload, dict):
        raise ValueError(f"Summary section {section!r} is invalid")
    record = section_payload.get(key, {})
    if not isinstance(record, dict):
        raise ValueError(f"Summary record {section}.{key} is invalid")
    expected = record.get("sha256")
    if not isinstance(expected, str) or expected != sha256(path):
        raise ValueError(f"Upstream hash mismatch for {path}")


def canonicalize_pairs(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    require_columns(frame, ("embedding_x", "embedding_y"), label)
    result = frame.copy()
    result["embedding_x"] = result["embedding_x"].astype(str)
    result["embedding_y"] = result["embedding_y"].astype(str)
    if (result["embedding_x"] == result["embedding_y"]).any():
        raise ValueError(f"{label} contains self-pairs")
    low = np.minimum(result["embedding_x"], result["embedding_y"])
    high = np.maximum(result["embedding_x"], result["embedding_y"])
    result["embedding_x"] = low
    result["embedding_y"] = high
    if result.duplicated(["embedding_x", "embedding_y"]).any():
        raise ValueError(f"{label} contains duplicate pairs")
    return result.sort_values(["embedding_x", "embedding_y"]).reset_index(drop=True)


def validate_pair_coverage(frame: pd.DataFrame, ids: list[str]) -> None:
    expected = {tuple(sorted(pair)) for pair in combinations(ids, 2)}
    observed = set(zip(frame["embedding_x"], frame["embedding_y"]))
    if observed != expected:
        raise ValueError(
            "Consensus pair coverage changed: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )


def weighted_force_layout(
    adjacency: np.ndarray,
    seed: int,
    iterations: int,
) -> np.ndarray:
    """Deterministic Fruchterman-Reingold layout using graph weights only."""

    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("Adjacency matrix must be symmetric")
    if np.any(matrix < 0.0) or np.any(np.diag(matrix) != 0.0):
        raise ValueError("Adjacency matrix must have nonnegative off-diagonal weights")
    if iterations < 50:
        raise ValueError("At least 50 layout iterations are required")
    n = matrix.shape[0]
    if n < 3 or np.any(np.count_nonzero(matrix, axis=1) == 0):
        raise ValueError("Graph layout requires at least three non-isolated nodes")

    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles += rng.normal(0.0, 0.025, size=n)
    positions = np.column_stack((np.cos(angles), np.sin(angles)))
    positions += rng.normal(0.0, 0.015, size=(n, 2))

    positive = matrix[matrix > 0.0]
    scaled = np.zeros_like(matrix)
    if positive.max() > positive.min():
        scaled[matrix > 0.0] = 0.35 + 0.65 * (
            (positive - positive.min()) / (positive.max() - positive.min())
        )
    else:
        scaled[matrix > 0.0] = 1.0

    ideal = np.sqrt(4.0 / n)
    initial_temperature = 0.22
    for step in range(iterations):
        delta = positions[:, None, :] - positions[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        distance = np.clip(distance, 1e-4, None)
        np.fill_diagonal(distance, np.inf)

        repulsion = np.sum(
            delta * ((ideal * ideal) / (distance * distance))[:, :, None],
            axis=1,
        )
        finite_distance = np.where(np.isfinite(distance), distance, 0.0)
        attraction = np.sum(
            delta
            * (scaled * finite_distance / max(ideal, 1e-12))[:, :, None],
            axis=1,
        )
        displacement = repulsion - attraction
        norms = np.linalg.norm(displacement, axis=1)
        temperature = initial_temperature * (1.0 - step / iterations) ** 1.5
        movement = displacement / np.maximum(norms[:, None], 1e-12)
        movement *= np.minimum(norms, temperature)[:, None]
        positions += movement
        positions -= positions.mean(axis=0, keepdims=True)

    span = np.ptp(positions, axis=0)
    if np.any(span <= 0.0):
        raise AssertionError("Force layout collapsed")
    positions /= np.max(np.abs(positions))
    if not np.isfinite(positions).all():
        raise AssertionError("Force layout produced non-finite coordinates")
    return positions


def build_graph_data(
    annotations: pd.DataFrame,
    edges: pd.DataFrame,
    roles: pd.DataFrame,
    seed: int,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    ids = sorted(annotations["embedding_id"].astype(str))
    index = {embedding_id: position for position, embedding_id in enumerate(ids)}
    adjacency = np.zeros((len(ids), len(ids)), dtype=float)
    graph_edges = canonicalize_pairs(edges, "graph edges")
    weights = numeric(graph_edges, PRIMARY_SCORE, "graph edges")
    if np.any(weights <= 0.0):
        raise ValueError("Graph weights must be positive")
    for row, weight in zip(graph_edges.itertuples(index=False), weights):
        if row.embedding_x not in index or row.embedding_y not in index:
            raise ValueError("Graph edges contain an unexpected embedding ID")
        i, j = index[row.embedding_x], index[row.embedding_y]
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    if np.any(np.count_nonzero(adjacency, axis=1) == 0):
        raise ValueError("Graph contains an isolated embedding")

    positions = weighted_force_layout(adjacency, seed, iterations)
    node_data = annotations.set_index("embedding_id").loc[ids].reset_index()
    role_columns = [
        "embedding_id",
        "weighted_participation_coefficient",
        "cross_community_weight_fraction",
        "bridge_rank",
    ]
    require_columns(roles, role_columns, "embedding roles")
    node_data = node_data.merge(roles[role_columns], on="embedding_id", how="left")
    if node_data[role_columns[1:]].isna().any().any():
        raise ValueError("Graph annotations and embedding roles do not match")
    node_data["layout_x"] = positions[:, 0]
    node_data["layout_y"] = positions[:, 1]
    node_data = node_data.sort_values("embedding_id").reset_index(drop=True)

    edge_data = graph_edges.copy()
    x_pos = node_data.set_index("embedding_id")[["layout_x", "layout_y"]]
    edge_data["x1"] = edge_data["embedding_x"].map(x_pos["layout_x"])
    edge_data["y1"] = edge_data["embedding_x"].map(x_pos["layout_y"])
    edge_data["x2"] = edge_data["embedding_y"].map(x_pos["layout_x"])
    edge_data["y2"] = edge_data["embedding_y"].map(x_pos["layout_y"])
    edge_data = edge_data.sort_values(PRIMARY_SCORE).reset_index(drop=True)
    return node_data, edge_data, adjacency


def spearman_rho(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    if not np.isfinite(value):
        raise ValueError("Spearman correlation is undefined")
    return value


def component_figure_data(
    pairs: pd.DataFrame, annotations: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    require_columns(
        pairs,
        (
            "embedding_x",
            "embedding_y",
            PLATONIC_SCORE,
            ARISTOTELIAN_SCORE,
            PRIMARY_SCORE,
        ),
        "consensus pairs",
    )
    label_map = annotations.set_index("embedding_id")["coarse_modality"].to_dict()
    community_map = annotations.set_index("embedding_id")["community_id"].to_dict()
    result = pairs[
        [
            "embedding_x",
            "embedding_y",
            PLATONIC_SCORE,
            ARISTOTELIAN_SCORE,
            PRIMARY_SCORE,
        ]
    ].copy()
    result["modality_x"] = result["embedding_x"].map(label_map)
    result["modality_y"] = result["embedding_y"].map(label_map)
    result["community_x"] = result["embedding_x"].map(community_map)
    result["community_y"] = result["embedding_y"].map(community_map)
    if result[["modality_x", "modality_y", "community_x", "community_y"]].isna().any().any():
        raise ValueError("Consensus pairs could not be mapped to graph annotations")
    result["same_modality"] = result["modality_x"] == result["modality_y"]
    result["same_community"] = result["community_x"] == result["community_y"]
    result["absolute_component_difference"] = np.abs(
        numeric(result, PLATONIC_SCORE, "component figure data")
        - numeric(result, ARISTOTELIAN_SCORE, "component figure data")
    )
    rho = spearman_rho(
        numeric(result, PLATONIC_SCORE, "component figure data"),
        numeric(result, ARISTOTELIAN_SCORE, "component figure data"),
    )
    return result, rho


def enrichment_figure_data(enrichment: pd.DataFrame) -> pd.DataFrame:
    required = (
        "community_id",
        "coarse_modality",
        "observed_count",
        "expected_count_under_fixed_margins",
        "fold_enrichment",
        "standardized_count_residual",
        "holm_p_two_sided_across_cells",
        "cell_interpretation",
    )
    require_columns(enrichment, required, "community-modality enrichment")
    result = enrichment[list(required)].copy()
    result["community_id"] = pd.to_numeric(result["community_id"], errors="raise").astype(int)
    if set(result["coarse_modality"].astype(str)) != set(MODALITY_ORDER):
        raise ValueError("Enrichment modalities differ from the frozen manifest order")
    residual = pd.to_numeric(
        result["standardized_count_residual"], errors="raise"
    ).to_numpy(dtype=float)
    observed = pd.to_numeric(
        result["observed_count"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
        raise ValueError("Observed community-modality counts must be finite and non-negative")
    if not np.allclose(observed, np.rint(observed), rtol=0.0, atol=0.0):
        raise ValueError("Observed community-modality counts must be integers")
    adjusted_p = pd.to_numeric(
        result["holm_p_two_sided_across_cells"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(residual)):
        raise ValueError("Standardized count residuals contain non-finite values")
    if not np.all(np.isfinite(adjusted_p)) or np.any(
        (adjusted_p < 0.0) | (adjusted_p > 1.0)
    ):
        raise ValueError("Holm-adjusted p-values must lie in [0, 1]")

    interpretation = result["cell_interpretation"].astype(str)
    valid_interpretations = {
        "enriched",
        "depleted",
        "not_significant_after_holm",
    }
    if not set(interpretation).issubset(valid_interpretations):
        raise ValueError("Unexpected enrichment-cell interpretation")
    significant = adjusted_p <= 0.05
    interpreted_significant = interpretation.isin({"enriched", "depleted"}).to_numpy()
    if not np.array_equal(significant, interpreted_significant):
        raise ValueError(
            "Cell interpretations disagree with Holm-adjusted significance"
        )
    if np.any((interpretation == "enriched").to_numpy() & (residual <= 0.0)):
        raise ValueError("An enriched cell does not have a positive residual")
    if np.any((interpretation == "depleted").to_numpy() & (residual >= 0.0)):
        raise ValueError("A depleted cell does not have a negative residual")
    if np.any(significant & np.isclose(residual, 0.0, rtol=0.0, atol=1e-12)):
        raise ValueError("A Holm-significant cell has an exactly zero residual")

    def significance_stars(value: float) -> str:
        if value <= 0.001:
            return "***"
        if value <= 0.01:
            return "**"
        if value <= 0.05:
            return "*"
        return ""

    result["plot_symbol"] = [significance_stars(value) for value in adjusted_p]
    result["plot_annotation"] = [
        f"{int(round(value))}{stars}"
        for value, stars in zip(observed, result["plot_symbol"])
    ]
    return result.sort_values(["community_id", "coarse_modality"]).reset_index(drop=True)


def residual_figure_data(summary: pd.DataFrame) -> pd.DataFrame:
    required = (
        "community_x",
        "community_y",
        "n_pairs",
        "mean_modality_adjusted_residual",
    )
    require_columns(summary, required, "community-pair residual summary")
    result = summary[list(required)].copy()
    result["community_x"] = pd.to_numeric(result["community_x"], errors="raise").astype(int)
    result["community_y"] = pd.to_numeric(result["community_y"], errors="raise").astype(int)
    numeric(result, "mean_modality_adjusted_residual", "residual figure data")
    return result.sort_values(["community_x", "community_y"]).reset_index(drop=True)


def community_modality_matrices(
    enrichment: pd.DataFrame,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    community_order = sorted(int(value) for value in enrichment["community_id"].unique())
    matrix = np.zeros((len(community_order), len(MODALITY_ORDER)), dtype=float)
    annotations = np.empty_like(matrix, dtype=object)
    for i, community in enumerate(community_order):
        for j, modality in enumerate(MODALITY_ORDER):
            row = enrichment.loc[
                (enrichment["community_id"] == community)
                & (enrichment["coarse_modality"].astype(str) == modality)
            ]
            if len(row) != 1:
                raise ValueError("Community-modality figure grid is incomplete")
            matrix[i, j] = float(row["standardized_count_residual"].iloc[0])
            annotations[i, j] = str(row["plot_annotation"].iloc[0])
    return community_order, matrix, annotations


def community_residual_matrices(
    residuals: pd.DataFrame,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    communities = sorted(
        set(residuals["community_x"].astype(int))
        | set(residuals["community_y"].astype(int))
    )
    matrix = np.full((len(communities), len(communities)), np.nan, dtype=float)
    counts = np.zeros_like(matrix, dtype=int)
    position = {value: index for index, value in enumerate(communities)}
    for row in residuals.itertuples(index=False):
        i, j = position[int(row.community_x)], position[int(row.community_y)]
        value = float(row.mean_modality_adjusted_residual)
        count = int(row.n_pairs)
        matrix[i, j] = value
        matrix[j, i] = value
        counts[i, j] = count
        counts[j, i] = count
    if np.isnan(matrix).any():
        raise ValueError("Community residual matrix is incomplete")
    return communities, matrix, counts


def draw_community_modality_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    enrichment: pd.DataFrame,
    *,
    title: str,
    annotation_fontsize: float,
    tick_fontsize: float,
) -> None:
    community_order, matrix, annotations = community_modality_matrices(enrichment)
    limit = max(2.0, float(np.max(np.abs(matrix))))
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if abs(matrix[i, j]) > 0.56 * limit else "#202124"
            ax.text(
                j,
                i,
                annotations[i, j],
                ha="center",
                va="center",
                color=color,
                fontsize=annotation_fontsize,
                fontweight="semibold",
            )
    ax.set_xticks(
        range(len(MODALITY_ORDER)),
        MODALITY_ORDER,
        rotation=35,
        ha="right",
        fontsize=tick_fontsize,
    )
    ax.set_yticks(
        range(len(community_order)),
        [f"Community {value}" for value in community_order],
        fontsize=tick_fontsize,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(
        "Cell text: observed embedding count; *, **, *** = Holm-adjusted "
        "permutation P ≤ 0.05, 0.01, 0.001",
        fontsize=max(9.5, tick_fontsize),
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Standardized count residual", fontsize=max(9.5, tick_fontsize))
    colorbar.ax.tick_params(labelsize=max(8.5, tick_fontsize - 0.5))


def draw_community_residual_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    residuals: pd.DataFrame,
    *,
    title: str,
    annotation_fontsize: float,
    tick_fontsize: float,
) -> None:
    communities, matrix, counts = community_residual_matrices(residuals)
    limit = max(0.05, float(np.max(np.abs(matrix))))
    image = ax.imshow(
        matrix,
        cmap="PuOr_r",
        vmin=-limit,
        vmax=limit,
        aspect="equal",
    )
    for i in range(len(communities)):
        for j in range(len(communities)):
            color = "white" if abs(matrix[i, j]) > 0.58 * limit else "#202124"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.3f}\n(n={counts[i, j]})",
                ha="center",
                va="center",
                color=color,
                fontsize=annotation_fontsize,
                fontweight="semibold",
            )
    labels = [f"C{value}" for value in communities]
    ax.set_xticks(range(len(labels)), labels, fontsize=tick_fontsize)
    ax.set_yticks(range(len(labels)), labels, fontsize=tick_fontsize)
    ax.set_xlabel("Community", fontsize=max(9.5, tick_fontsize))
    ax.set_ylabel("Community", fontsize=max(9.5, tick_fontsize))
    ax.set_title(title, loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean descriptive residual", fontsize=max(9.5, tick_fontsize))
    colorbar.ax.tick_params(labelsize=max(8.5, tick_fontsize - 0.5))


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> dict[str, Path]:
    paths = {
        "png": base.with_suffix(".png"),
        "pdf": base.with_suffix(".pdf"),
        "svg": base.with_suffix(".svg"),
    }
    for extension, path in paths.items():
        metadata = {"Creator": "consensus graph cs_10"} if extension == "pdf" else None
        fig.savefig(
            path,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
    return paths


def separate_graph_labels(
    fig: plt.Figure,
    ax: plt.Axes,
    texts: list[plt.Text],
    target_x: np.ndarray,
    target_y: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
) -> dict[str, float | int]:
    """Deterministically remove residual text-text and text-node collisions."""
    if not texts:
        return {
            "label_label_overlaps": 0,
            "label_node_overlaps": 0,
            "leader_lines": 0,
            "maximum_leader_length_pixels": 0.0,
        }

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    positions = np.asarray(
        [ax.transData.transform(text.get_position()) for text in texts], dtype=float
    )
    target_pixels = ax.transData.transform(np.column_stack([target_x, target_y]))
    node_pixels = ax.transData.transform(np.column_stack([node_x, node_y]))
    widths: list[float] = []
    heights: list[float] = []
    for text in texts:
        patch = text.get_bbox_patch()
        extent = (
            patch.get_window_extent(renderer)
            if patch is not None
            else text.get_window_extent(renderer)
        )
        widths.append(float(extent.width))
        heights.append(float(extent.height))
    half_width = np.asarray(widths, dtype=float) / 2.0
    half_height = np.asarray(heights, dtype=float) / 2.0

    axes_extent = ax.get_window_extent(renderer)
    text_padding = 3.0
    node_clearance = 8.5

    def collision_counts(current: np.ndarray) -> tuple[int, int]:
        label_label = 0
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                overlap_x = (
                    half_width[left]
                    + half_width[right]
                    + text_padding
                    - abs(current[right, 0] - current[left, 0])
                )
                overlap_y = (
                    half_height[left]
                    + half_height[right]
                    + text_padding
                    - abs(current[right, 1] - current[left, 1])
                )
                if overlap_x > 0.0 and overlap_y > 0.0:
                    label_label += 1
        label_node = 0
        for index in range(len(current)):
            delta = node_pixels - current[index]
            overlap_x = half_width[index] + node_clearance - np.abs(delta[:, 0])
            overlap_y = half_height[index] + node_clearance - np.abs(delta[:, 1])
            label_node += int(np.count_nonzero((overlap_x > 0.0) & (overlap_y > 0.0)))
        return label_label, label_node

    placed: list[int] = []
    graph_centroid = node_pixels.mean(axis=0)
    distance_from_centroid = np.linalg.norm(target_pixels - graph_centroid, axis=1)
    placement_order = sorted(
        range(len(positions)),
        key=lambda index: (
            -float(distance_from_centroid[index]),
            -float(widths[index] * heights[index]),
            index,
        ),
    )

    def candidate_is_free(index: int, candidate: np.ndarray) -> bool:
        if candidate[0] - half_width[index] < axes_extent.x0 + 2.0:
            return False
        if candidate[0] + half_width[index] > axes_extent.x1 - 2.0:
            return False
        if candidate[1] - half_height[index] < axes_extent.y0 + 2.0:
            return False
        if candidate[1] + half_height[index] > axes_extent.y1 - 2.0:
            return False
        node_delta = np.abs(node_pixels - candidate)
        node_collision = (
            (node_delta[:, 0] < half_width[index] + node_clearance)
            & (node_delta[:, 1] < half_height[index] + node_clearance)
        )
        if bool(np.any(node_collision)):
            return False
        for other in placed:
            if (
                abs(candidate[0] - positions[other, 0])
                < half_width[index] + half_width[other] + text_padding
                and abs(candidate[1] - positions[other, 1])
                < half_height[index] + half_height[other] + text_padding
            ):
                return False
        return True

    angle_offsets = np.asarray(
        [
            0.0,
            np.pi / 12.0,
            -np.pi / 12.0,
            np.pi / 6.0,
            -np.pi / 6.0,
            np.pi / 4.0,
            -np.pi / 4.0,
            np.pi / 3.0,
            -np.pi / 3.0,
            np.pi / 2.0,
            -np.pi / 2.0,
            2.0 * np.pi / 3.0,
            -2.0 * np.pi / 3.0,
            3.0 * np.pi / 4.0,
            -3.0 * np.pi / 4.0,
            np.pi,
        ],
        dtype=float,
    )
    maximum_radius = float(
        np.hypot(axes_extent.width, axes_extent.height)
    )
    radii = np.arange(18.0, maximum_radius + 8.0, 8.0)

    for index in placement_order:
        original = positions[index].copy()
        if candidate_is_free(index, original):
            placed.append(index)
            continue
        outward_vector = target_pixels[index] - graph_centroid
        outward_angle = float(np.arctan2(outward_vector[1], outward_vector[0]))
        chosen: np.ndarray | None = None
        for radius in radii:
            for offset in angle_offsets:
                angle = outward_angle + float(offset)
                candidate = target_pixels[index] + radius * np.asarray(
                    [np.cos(angle), np.sin(angle)], dtype=float
                )
                if candidate_is_free(index, candidate):
                    chosen = candidate
                    break
            if chosen is not None:
                break
        if chosen is None:
            # Extremely dense layouts fall back to a deterministic coarse grid
            # over the available label area, preferring the closest free slot.
            x_candidates = np.arange(
                axes_extent.x0 + half_width[index] + 2.0,
                axes_extent.x1 - half_width[index] - 1.0,
                10.0,
            )
            y_candidates = np.arange(
                axes_extent.y0 + half_height[index] + 2.0,
                axes_extent.y1 - half_height[index] - 1.0,
                10.0,
            )
            grid = np.asarray(
                [(x_value, y_value) for x_value in x_candidates for y_value in y_candidates],
                dtype=float,
            )
            grid_order = np.argsort(
                np.linalg.norm(grid - target_pixels[index], axis=1), kind="stable"
            )
            for grid_index in grid_order:
                candidate = grid[int(grid_index)]
                if candidate_is_free(index, candidate):
                    chosen = candidate
                    break
        if chosen is None:
            raise RuntimeError(
                f"No collision-free graph-label position found for {texts[index].get_text()!r}"
            )
        positions[index] = chosen
        placed.append(index)

    final_label_label, final_label_node = collision_counts(positions)
    if final_label_label or final_label_node:
        raise RuntimeError(
            "Graph-label collision resolution did not converge: "
            f"label-label={final_label_label}, label-node={final_label_node}"
        )

    inverse = ax.transData.inverted()
    for text, pixel_position in zip(texts, positions):
        text.set_position(inverse.transform(pixel_position))

    leader_lengths = np.linalg.norm(positions - target_pixels, axis=1)
    leader_lines = 0
    for text, target, length in zip(texts, np.column_stack([target_x, target_y]), leader_lengths):
        # Nearby labels remain visually associated with their node without a
        # connector; only materially displaced labels receive leader lines.
        if float(length) < 26.0:
            continue
        leader_lines += 1
        ax.annotate(
            "",
            xy=(float(target[0]), float(target[1])),
            xytext=text.get_position(),
            arrowprops={
                "arrowstyle": "-",
                "color": "#929EAA",
                "linewidth": 0.55,
                "alpha": 0.90,
                "shrinkA": 3.0,
                "shrinkB": 3.0,
            },
            zorder=4,
        )

    return {
        "label_label_overlaps": int(final_label_label),
        "label_node_overlaps": int(final_label_node),
        "leader_lines": int(leader_lines),
        "maximum_leader_length_pixels": float(leader_lengths.max(initial=0.0)),
    }


def plot_graph_map(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    top_bridge_labels: int,
    output_dir: Path,
) -> tuple[dict[str, Path], dict[str, float | int]]:
    # A dedicated legend column keeps explanatory material out of the graph and
    # leaves enough horizontal room for every manifest display name.
    fig = plt.figure(figsize=(18.0, 10.5), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(0.22, 0.78),
        left=0.025,
        right=0.992,
        bottom=0.035,
        top=0.975,
        wspace=0.025,
    )
    legend_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])

    weights = numeric(edges, PRIMARY_SCORE, "graph figure edges")
    low, high = float(weights.min()), float(weights.max())
    normalized = np.ones_like(weights) if high == low else (weights - low) / (high - low)
    for row, scaled in zip(edges.itertuples(index=False), normalized):
        ax.plot(
            [row.x1, row.x2],
            [row.y1, row.y2],
            color="#7D8793",
            alpha=0.055 + 0.28 * float(scaled),
            linewidth=0.25 + 1.10 * float(scaled),
            zorder=1,
        )

    communities = sorted(int(value) for value in nodes["community_id"].unique())
    for community in communities:
        community_subset = nodes.loc[nodes["community_id"] == community]
        for modality in MODALITY_ORDER:
            subset = community_subset.loc[
                community_subset["coarse_modality"].astype(str) == modality
            ]
            if subset.empty:
                continue
            ax.scatter(
                subset["layout_x"],
                subset["layout_y"],
                s=54,
                marker=COMMUNITY_MARKERS.get(community, "o"),
                c=MODALITY_COLORS[modality],
                edgecolors="white",
                linewidths=0.75,
                zorder=3,
            )

    x_values = nodes["layout_x"].to_numpy(dtype=float)
    y_values = nodes["layout_y"].to_numpy(dtype=float)
    x_span = max(float(np.ptp(x_values)), 1e-6)
    y_span = max(float(np.ptp(y_values)), 1e-6)
    # The extra plotting margin is reserved for displaced labels.  It does not
    # change the force-directed node coordinates or any graph result.
    ax.set_xlim(float(x_values.min()) - 0.22 * x_span, float(x_values.max()) + 0.22 * x_span)
    ax.set_ylim(float(y_values.min()) - 0.19 * y_span, float(y_values.max()) + 0.19 * y_span)

    # Label all embeddings by default.  Starting each label just outside its
    # node and then applying deterministic collision avoidance produces short
    # leader lines while keeping text and nodes separate.
    labelled = nodes.nsmallest(top_bridge_labels, "bridge_rank").copy()
    centroid_x = float(x_values.mean())
    centroid_y = float(y_values.mean())
    label_texts: list[plt.Text] = []
    target_x: list[float] = []
    target_y: list[float] = []
    for row in labelled.itertuples(index=False):
        direction = np.array(
            [row.layout_x - centroid_x, row.layout_y - centroid_y], dtype=float
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        initial_x = float(
            row.layout_x + direction[0] * GRAPH_LABEL_INITIAL_OFFSET * x_span
        )
        initial_y = float(
            row.layout_y + direction[1] * GRAPH_LABEL_INITIAL_OFFSET * y_span
        )
        label_texts.append(
            ax.text(
                initial_x,
                initial_y,
                str(row.display_name),
                ha="center",
                va="center",
                fontsize=GRAPH_LABEL_FONTSIZE,
                color="#27313D",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "#D8DEE6",
                    "linewidth": 0.35,
                    "alpha": 0.94,
                },
                zorder=5,
            )
        )
        target_x.append(float(row.layout_x))
        target_y.append(float(row.layout_y))

    if label_texts:
        # adjustText is deterministic for this input once NumPy's seed is fixed.
        # Each graph node is surrounded by a small nine-point obstacle halo.
        # This makes labels avoid the visible node marker rather than merely its
        # centre.  Labels also repel one another; leader lines retain the exact
        # node-to-label correspondence when dense regions require displacement.
        halo_x = GRAPH_NODE_CLEARANCE * x_span
        halo_y = GRAPH_NODE_CLEARANCE * y_span
        obstacle_x = np.concatenate(
            [
                x_values + dx
                for dx in (-halo_x, 0.0, halo_x)
                for _dy in (-halo_y, 0.0, halo_y)
            ]
        )
        obstacle_y = np.concatenate(
            [
                y_values + dy
                for _dx in (-halo_x, 0.0, halo_x)
                for dy in (-halo_y, 0.0, halo_y)
            ]
        )
        random_state = np.random.get_state()
        np.random.seed(SEED)
        try:
            adjust_text(
                label_texts,
                x=obstacle_x,
                y=obstacle_y,
                target_x=np.asarray(target_x),
                target_y=np.asarray(target_y),
                ax=ax,
                expand=(1.12, 1.24),
                expand_axes=False,
                force_text=(0.42, 0.62),
                force_static=(0.32, 0.48),
                force_pull=(0.006, 0.010),
                force_explode=(0.50, 0.75),
                explode_radius=125,
                max_move=(24, 24),
                pull_threshold=14,
                ensure_inside_axes=True,
                prevent_crossings=True,
                iter_lim=3000,
            )
        finally:
            np.random.set_state(random_state)

    label_audit = separate_graph_labels(
        fig,
        ax,
        label_texts,
        np.asarray(target_x, dtype=float),
        np.asarray(target_y, dtype=float),
        x_values,
        y_values,
    )

    # Reference-style integrated legend with exact counts from the plotted
    # panel.  Color encodes the definitive manifest modality; shape encodes the
    # unsupervised graph community.
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    legend_ax.set_axis_off()
    heading_color = "#173A63"
    legend_ax.text(
        0.02,
        0.965,
        "Definitive modality",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=heading_color,
    )
    modality_counts = nodes["coarse_modality"].astype(str).value_counts()
    y_position = 0.925
    modality_step = 0.041
    for modality in MODALITY_ORDER:
        count = int(modality_counts.get(modality, 0))
        legend_ax.scatter(
            [0.035],
            [y_position],
            s=34,
            marker="o",
            c=MODALITY_COLORS[modality],
            edgecolors="white",
            linewidths=0.55,
            clip_on=False,
        )
        legend_ax.text(
            0.075,
            y_position,
            f"{modality} ({count})",
            ha="left",
            va="center",
            fontsize=8.7,
            color="#303B47",
        )
        y_position -= modality_step

    separator_y = y_position - 0.008
    legend_ax.plot(
        [0.02, 0.72],
        [separator_y, separator_y],
        color="#D8DEE6",
        linewidth=0.8,
    )
    community_heading_y = separator_y - 0.032
    legend_ax.text(
        0.02,
        community_heading_y,
        "Unsupervised community",
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=heading_color,
    )
    y_position = community_heading_y - 0.052
    for community in communities:
        legend_ax.scatter(
            [0.035],
            [y_position],
            s=42,
            marker=COMMUNITY_MARKERS.get(community, "o"),
            c="#7D8793",
            edgecolors="white",
            linewidths=0.55,
            clip_on=False,
        )
        legend_ax.text(
            0.075,
            y_position,
            f"Community {community}",
            ha="left",
            va="center",
            fontsize=8.7,
            color="#202124",
        )
        y_position -= 0.047

    ax.set_aspect("equal")
    ax.set_axis_off()
    paths = save_figure(fig, output_dir / "figure_1_consensus_graph_map")
    plt.close(fig)
    return paths, label_audit


def plot_evidence_figure(
    enrichment: pd.DataFrame,
    components: pd.DataFrame,
    component_rho: float,
    residuals: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    fig = plt.figure(figsize=(18.5, 6.3), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[2.45, 1.05, 0.95])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    draw_community_modality_heatmap(
        fig,
        ax_a,
        enrichment,
        title="A  Community-modality enrichment",
        annotation_fontsize=10.5,
        tick_fontsize=9.5,
    )

    same_community = parse_boolean(components["same_community"], "same_community")
    for mask, color, label, alpha, size in (
        (~same_community, "#B8BDC6", "Between communities", 0.38, 13),
        (same_community, "#476FB1", "Within community", 0.58, 16),
    ):
        subset = components.loc[mask]
        ax_b.scatter(
            subset[PLATONIC_SCORE],
            subset[ARISTOTELIAN_SCORE],
            s=size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            rasterized=True,
            label=label,
        )
    ax_b.plot([0, 1], [0, 1], color="#343A40", linewidth=0.9, linestyle="--", alpha=0.75)
    ax_b.set_xlim(-0.025, 1.025)
    ax_b.set_ylim(-0.025, 1.025)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("Platonic percentile (linear CKA)")
    ax_b.set_ylabel("Aristotelian percentile (local overlap k=100)")
    ax_b.set_title("B  Complementary primary views", loc="left", fontweight="bold")
    ax_b.text(
        0.04,
        0.96,
        f"Spearman rho = {component_rho:.3f}",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#C7CBD1"},
    )
    ax_b.legend(loc="lower right", frameon=True)
    ax_b.grid(color="#E5E7EB", linewidth=0.55, alpha=0.8)

    draw_community_residual_heatmap(
        fig,
        ax_c,
        residuals,
        title="C  Modality-adjusted residual affinity",
        annotation_fontsize=10.0,
        tick_fontsize=9.5,
    )

    fig.suptitle(
        "Evidence supporting the two-view consensus structure",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    paths = save_figure(fig, output_dir / "figure_2_consensus_evidence")
    plt.close(fig)
    return paths


def plot_manuscript_heatmap_panels(
    enrichment: pd.DataFrame,
    residuals: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write panels at their intrinsic data aspect ratios for manuscript use."""
    paths: dict[str, Path] = {}

    fig_b, ax_b = plt.subplots(figsize=(11.8, 4.7), constrained_layout=True)
    draw_community_modality_heatmap(
        fig_b,
        ax_b,
        enrichment,
        title="Community-modality enrichment",
        annotation_fontsize=12.5,
        tick_fontsize=11.0,
    )
    for key, path in save_figure(
        fig_b, output_dir / "manuscript_figure_2b_community_modality_heatmap"
    ).items():
        paths[f"manuscript_figure_2b_{key}"] = path
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(6.0, 5.2), constrained_layout=True)
    draw_community_residual_heatmap(
        fig_c,
        ax_c,
        residuals,
        title="Modality-adjusted residual affinity",
        annotation_fontsize=12.0,
        tick_fontsize=11.0,
    )
    for key, path in save_figure(
        fig_c, output_dir / "manuscript_figure_2c_community_residual_heatmap"
    ).items():
        paths[f"manuscript_figure_2c_{key}"] = path
    plt.close(fig_c)

    return paths


def atomic_dataframe(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, sep="\t", index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: dict[str, object], destination: Path) -> None:
    atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", destination)


def build_caption(
    n_edges: int,
    n_communities: int,
    top_bridge_labels: int,
    component_rho: float,
    same_residual: float,
    between_residual: float,
) -> str:
    label_sentence = (
        "Every node is labeled with the display name from the frozen embedding "
        "manifest; leader lines connect displaced labels to their nodes."
        if top_bridge_labels == EXPECTED_EMBEDDINGS
        else (
            f"Labels identify the {top_bridge_labels} embeddings with the highest "
            "weighted participation coefficient."
        )
    )
    return f"""# Workflow 33 consensus-space figures

## Figure 1 - Consensus affinity graph

Weighted, symmetrized 10-nearest-neighbour graph for {EXPECTED_EMBEDDINGS} gene embeddings ({n_edges} undirected edges). Node positions come from a deterministic force-directed layout using only graph edges and consensus weights. Node color denotes the definitive manifest modality and node shape denotes the unsupervised community. {label_sentence} Community and bridge status are descriptive and are not embedding-quality scores.

## Figure 2 - Evidence supporting the two-view consensus structure

**A.** Community-by-modality composition. Cell text gives the observed number of embeddings in that community-modality combination; *, ** and *** denote two-sided permutation P ≤ 0.05, 0.01 and 0.001, respectively, after Holm correction across all {n_communities * EXPECTED_MODALITIES} cells. Cell color gives the signed standardized count residual, with positive and negative values indicating enrichment and depletion. Consequently, a cell labeled `0*` contains zero embeddings and is significantly depleted; it does not report a residual of zero. **B.** Pairwise percentile ranks from the primary Platonic view (row-normalized unbiased linear CKA) and primary Aristotelian view (tie-aware local overlap at k=100). Their Spearman correlation is {component_rho:.3f}; pairwise ordinary p-values are not used. **C.** Mean consensus residual after subtracting the leave-one-pair-out mean for the exact modality-pair block. Mean residuals are {same_residual:+.3f} within communities and {between_residual:+.3f} between communities. Residual values are descriptive because the same affinities were used to learn the graph communities.

The files `manuscript_figure_2b_community_modality_heatmap` and `manuscript_figure_2c_community_residual_heatmap` provide the two heatmaps as separate vector panels with equal-aspect cells. They should be placed in the manuscript with aspect-ratio locking enabled; width and height must not be rescaled independently.

## Interpretation limits

- Community membership, graph centrality, and bridge status are not measures of embedding quality.
- Coarse modality labels represent dominant provenance, not exclusive biological content.
- RBF CKA remains a nonlinear sensitivity analysis and is not a third consensus component.
- The rank consensus is an affinity; the layout must not be read as a metric embedding with meaningful axis distances.
"""


def run_analysis(args: argparse.Namespace, quiet: bool = False) -> dict[str, object]:
    if args.layout_iterations < 50:
        raise ValueError("--layout-iterations must be at least 50")
    if not 0 <= args.top_bridge_labels <= EXPECTED_EMBEDDINGS:
        raise ValueError(f"--top-bridge-labels must be between 0 and {EXPECTED_EMBEDDINGS}")

    root = args.root.expanduser().resolve()
    workflow_root = root / "results/consensus_graph_v2"
    consensus_dir = (
        args.consensus_dir.expanduser().resolve()
        if args.consensus_dir is not None
        else workflow_root / "06_consensus"
    )
    graph_dir = (
        args.graph_dir.expanduser().resolve()
        if args.graph_dir is not None
        else workflow_root / "08_graph_structure"
    )
    interpretation_dir = (
        args.interpretation_dir.expanduser().resolve()
        if args.interpretation_dir is not None
        else workflow_root / "09_community_interpretation"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow_root / "10_figures"
    )

    consensus_summary_path = require_file(consensus_dir / "consensus_summary.json")
    pairs_path = require_file(consensus_dir / "consensus_pair_scores.tsv")
    graph_summary_path = require_file(graph_dir / "graph_structure_summary.json")
    annotations_path = require_file(graph_dir / "embedding_graph_annotations.tsv")
    edges_path = require_file(graph_dir / "consensus_graph_edges.tsv")
    interpretation_summary_path = require_file(
        interpretation_dir / "community_interpretation_summary.json"
    )
    enrichment_path = require_file(
        interpretation_dir / "community_modality_enrichment.tsv"
    )
    profiles_path = require_file(interpretation_dir / "community_profiles.tsv")
    roles_path = require_file(interpretation_dir / "embedding_graph_roles.tsv")
    residuals_path = require_file(
        interpretation_dir / "community_pair_residual_summary.tsv"
    )

    consensus_summary = load_json(consensus_summary_path)
    graph_summary = load_json(graph_summary_path)
    interpretation_summary = load_json(interpretation_summary_path)
    for label, payload in (
        ("consensus", consensus_summary),
        ("graph", graph_summary),
        ("community interpretation", interpretation_summary),
    ):
        if payload.get("status") != "PASS":
            raise ValueError(f"Upstream {label} stage did not PASS")
        if int(payload.get("n_embeddings", -1)) != EXPECTED_EMBEDDINGS:
            raise ValueError(f"Upstream {label} embedding count changed")
    if int(consensus_summary.get("n_pairs", -1)) != EXPECTED_PAIRS:
        raise ValueError("Upstream consensus pair count changed")
    if int(interpretation_summary.get("n_pairs", -1)) != EXPECTED_PAIRS:
        raise ValueError("Upstream interpretation pair count changed")
    accepted_recommendations = {
        "REPORT_MODALITY_ENRICHED_COMMUNITIES_WITH_RESIDUAL_CROSS_MODALITY_STRUCTURE",
        "REPORT_UNSUPERVISED_COMMUNITIES_WITHOUT_CELL_LEVEL_ENRICHMENT",
    }
    if interpretation_summary.get("recommendation") not in accepted_recommendations:
        raise ValueError("Unexpected upstream community-interpretation recommendation")

    verify_recorded_hash(graph_summary, "sources", "consensus_pairs", pairs_path)
    verify_recorded_hash(graph_summary, "outputs", "embedding_annotations", annotations_path)
    verify_recorded_hash(graph_summary, "outputs", "graph_edges", edges_path)
    verify_recorded_hash(
        interpretation_summary,
        "outputs",
        "community_modality_enrichment",
        enrichment_path,
    )
    verify_recorded_hash(interpretation_summary, "outputs", "community_profiles", profiles_path)
    verify_recorded_hash(interpretation_summary, "outputs", "embedding_roles", roles_path)
    verify_recorded_hash(
        interpretation_summary,
        "outputs",
        "community_pair_residual_summary",
        residuals_path,
    )

    pairs = canonicalize_pairs(pd.read_csv(pairs_path, sep="\t"), "consensus pairs")
    annotations = pd.read_csv(annotations_path, sep="\t")
    edges = pd.read_csv(edges_path, sep="\t")
    enrichment = pd.read_csv(enrichment_path, sep="\t")
    profiles = pd.read_csv(profiles_path, sep="\t")
    roles = pd.read_csv(roles_path, sep="\t")
    residual_summary = pd.read_csv(residuals_path, sep="\t")

    require_columns(
        annotations,
        ("embedding_id", "display_name", "coarse_modality", "community_id"),
        "graph annotations",
    )
    annotations["embedding_id"] = annotations["embedding_id"].astype(str)
    annotations["community_id"] = pd.to_numeric(
        annotations["community_id"], errors="raise"
    ).astype(int)
    if len(annotations) != EXPECTED_EMBEDDINGS or annotations["embedding_id"].duplicated().any():
        raise ValueError(
            f"Graph annotations must contain {EXPECTED_EMBEDDINGS} unique embeddings"
        )
    if set(annotations["coarse_modality"].astype(str)) != set(MODALITY_ORDER):
        raise ValueError("Graph annotations do not contain the frozen manifest modalities")
    community_order = sorted(annotations["community_id"].unique())
    if len(community_order) != int(interpretation_summary.get("n_communities", -1)):
        raise ValueError("Community count changed")
    ids = sorted(annotations["embedding_id"].astype(str))
    validate_pair_coverage(pairs, ids)
    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError("Consensus pair table changed")
    require_columns(
        pairs,
        (PRIMARY_SCORE, PLATONIC_SCORE, ARISTOTELIAN_SCORE),
        "consensus pairs",
    )
    for column in (PRIMARY_SCORE, PLATONIC_SCORE, ARISTOTELIAN_SCORE):
        values = numeric(pairs, column, "consensus pairs")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"Consensus pair column {column!r} lies outside [0, 1]")

    nodes, edge_data, _ = build_graph_data(
        annotations, edges, roles, args.seed, args.layout_iterations
    )
    expected_edges = int(graph_summary.get("graph", {}).get("n_edges", -1))
    if len(edge_data) != expected_edges:
        raise ValueError("Primary graph edge count changed")
    components, component_rho = component_figure_data(pairs, annotations)
    enrichment_data = enrichment_figure_data(enrichment)
    residual_data = residual_figure_data(residual_summary)

    recorded_rho = float(
        consensus_summary.get("consensus_to_component_spearman_rho", {}).get(
            "platonic", np.nan
        )
    )
    if not np.isfinite(recorded_rho):
        raise ValueError("Consensus summary is missing its component correlation")
    n_communities = len(community_order)
    profile_ids = sorted(pd.to_numeric(profiles["community_id"], errors="raise").astype(int))
    if profile_ids != community_order:
        raise ValueError("Community profiles and graph annotations disagree")

    same_residual = float(
        interpretation_summary.get("residual_structure", {}).get(
            "same_community_mean_modality_adjusted_residual", np.nan
        )
    )
    between_residual = float(
        interpretation_summary.get("residual_structure", {}).get(
            "between_community_mean_modality_adjusted_residual", np.nan
        )
    )
    if not np.isfinite(same_residual) or not np.isfinite(between_residual):
        raise ValueError("Interpretation summary is missing residual means")

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    graph_paths, graph_label_audit = plot_graph_map(
        nodes, edge_data, args.top_bridge_labels, output_dir
    )
    evidence_paths = plot_evidence_figure(
        enrichment_data,
        components,
        component_rho,
        residual_data,
        output_dir,
    )
    manuscript_panel_paths = plot_manuscript_heatmap_panels(
        enrichment_data,
        residual_data,
        output_dir,
    )

    data_paths = {
        "graph_layout_coordinates": output_dir / "graph_layout_coordinates.tsv",
        "graph_figure_edges": output_dir / "graph_figure_edges.tsv",
        "community_modality_figure_data": output_dir / "community_modality_figure_data.tsv",
        "component_scatter_figure_data": output_dir / "component_scatter_figure_data.tsv",
        "community_residual_figure_data": output_dir / "community_residual_figure_data.tsv",
    }
    atomic_dataframe(nodes, data_paths["graph_layout_coordinates"])
    atomic_dataframe(edge_data, data_paths["graph_figure_edges"])
    atomic_dataframe(enrichment_data, data_paths["community_modality_figure_data"])
    atomic_dataframe(components, data_paths["component_scatter_figure_data"])
    atomic_dataframe(residual_data, data_paths["community_residual_figure_data"])

    caption_path = output_dir / "figure_captions.md"
    atomic_text(
        build_caption(
            len(edge_data),
            n_communities,
            args.top_bridge_labels,
            component_rho,
            same_residual,
            between_residual,
        ),
        caption_path,
    )

    figure_paths = {
        **{f"graph_{key}": value for key, value in graph_paths.items()},
        **{f"evidence_{key}": value for key, value in evidence_paths.items()},
        **manuscript_panel_paths,
    }
    summary: dict[str, object] = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_pairs": EXPECTED_PAIRS,
        "n_graph_edges": int(len(edge_data)),
        "n_communities": int(n_communities),
        "n_modalities": EXPECTED_MODALITIES,
        "layout": {
            "method": "deterministic_weighted_force_directed",
            "inputs": "primary graph topology and edge weights only",
            "seed": int(args.seed),
            "iterations": int(args.layout_iterations),
            "community_or_modality_labels_used_for_coordinates": False,
            "node_label_font_size_points": GRAPH_LABEL_FONTSIZE,
            "node_label_collision_audit": graph_label_audit,
        },
        "displayed_results": {
            "platonic_aristotelian_spearman_rho": component_rho,
            "same_community_mean_modality_adjusted_residual": same_residual,
            "between_community_mean_modality_adjusted_residual": between_residual,
            "community_modality_inference": (
                "fixed-count node-label permutations with Holm correction, imported from cs_09"
            ),
            "community_modality_cell_text": "observed embedding count plus Holm significance stars",
            "community_modality_cell_color": "signed standardized count residual",
            "new_inferential_tests_performed": False,
        },
        "sources": {
            "consensus_summary": {
                "path": str(consensus_summary_path),
                "sha256": sha256(consensus_summary_path),
            },
            "consensus_pairs": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "graph_summary": {"path": str(graph_summary_path), "sha256": sha256(graph_summary_path)},
            "graph_annotations": {"path": str(annotations_path), "sha256": sha256(annotations_path)},
            "graph_edges": {"path": str(edges_path), "sha256": sha256(edges_path)},
            "interpretation_summary": {
                "path": str(interpretation_summary_path),
                "sha256": sha256(interpretation_summary_path),
            },
            "community_modality_enrichment": {
                "path": str(enrichment_path),
                "sha256": sha256(enrichment_path),
            },
            "community_profiles": {"path": str(profiles_path), "sha256": sha256(profiles_path)},
            "embedding_roles": {"path": str(roles_path), "sha256": sha256(roles_path)},
            "community_pair_residual_summary": {
                "path": str(residuals_path),
                "sha256": sha256(residuals_path),
            },
        },
        "outputs": {
            **{
                key: {"path": str(path), "sha256": sha256(path)}
                for key, path in figure_paths.items()
            },
            **{
                key: {"path": str(path), "sha256": sha256(path)}
                for key, path in data_paths.items()
            },
            "figure_captions": {
                "path": str(caption_path),
                "sha256": sha256(caption_path),
            },
        },
        "interpretation_limits": {
            "layout": "Graph axes and Euclidean node distances have no metric interpretation.",
            "node_roles": "Community, bridge, and centrality status are not embedding-quality scores.",
            "residuals": "Modality-adjusted pair residuals are descriptive, not independent test statistics.",
            "modalities": "Coarse labels denote dominant provenance, not exclusive biological content.",
        },
    }
    summary_path = output_dir / "figure_summary.json"
    atomic_json(summary, summary_path)

    if not quiet:
        print("===== CONSENSUS GRAPH FIGURES =====")
        print("status: PASS")
        print(f"embeddings: {EXPECTED_EMBEDDINGS}")
        print(f"pairs: {EXPECTED_PAIRS}")
        print(f"graph edges: {len(edge_data)}")
        print(f"communities: {n_communities}")
        print("figure 1: weighted consensus graph map")
        print("  coordinates use graph topology and weights only")
        print(f"  node labels: {args.top_bridge_labels}")
        print(f"  node-label font size: {GRAPH_LABEL_FONTSIZE:.1f} pt")
        print(
            "  residual label overlaps: "
            f"text-text={graph_label_audit['label_label_overlaps']}; "
            f"text-node={graph_label_audit['label_node_overlaps']}"
        )
        print(f"  leader lines: {graph_label_audit['leader_lines']}")
        print("figure 2: enrichment, view concordance, and residual structure")
        print("  manuscript-ready panels 2b and 2c use equal-aspect heatmap cells")
        print(f"  Platonic-Aristotelian Spearman rho: {component_rho:.6f}")
        print(f"  same-community mean residual: {same_residual:.6f}")
        print(f"  between-community mean residual: {between_residual:.6f}")
        print("formats: PNG, PDF, SVG")
        print(f"output: {output_dir}")
        print(
            "NOTE: this stage is visualization-only; it performs no new "
            "inferential tests or similarity calculations."
        )
    return summary


def write_fixture(base: Path) -> SimpleNamespace:
    workflow = base / "results/consensus_graph_v2"
    consensus_dir = workflow / "06_consensus"
    graph_dir = workflow / "08_graph_structure"
    interpretation_dir = workflow / "09_community_interpretation"
    output_dir = workflow / "10_figures"
    for path in (consensus_dir, graph_dir, interpretation_dir):
        path.mkdir(parents=True, exist_ok=True)

    ids = [f"embedding_{index:02d}" for index in range(EXPECTED_EMBEDDINGS)]
    modalities: list[str] = []
    modality_counts = [3, 5, 4, 3, 8, 2, 17, 3, 3, 5, 3]
    for modality, count in zip(MODALITY_ORDER, modality_counts):
        modalities.extend([modality] * count)
    communities = np.array(
        [
            1
            if modality in {"DNA sequence", "RNA sequence", "Regulatory genomics"}
            else 2
            if modality in {"Protein sequence", "Functional"}
            else 3
            for modality in modalities
        ],
        dtype=int,
    )
    annotations = pd.DataFrame(
        {
            "embedding_id": ids,
            "display_name": [f"Embedding {index:02d}" for index in range(EXPECTED_EMBEDDINGS)],
            "coarse_modality": modalities,
            "community_id": communities,
            "hybrid_or_multimodal": [
                modality == "Multimodal" for modality in modalities
            ],
        }
    )
    annotations_path = graph_dir / "embedding_graph_annotations.tsv"
    annotations.to_csv(annotations_path, sep="\t", index=False)

    pair_rows: list[dict[str, object]] = []
    for i, j in combinations(range(EXPECTED_EMBEDDINGS), 2):
        jitter = ((i * 17 + j * 29) % 97) / 300.0
        base_score = 0.62 if communities[i] == communities[j] else 0.10
        platonic = min(1.0, base_score + jitter)
        aristotelian = min(
            1.0,
            base_score + 0.82 * jitter + (((i * 7 + j * 5) % 19) - 9) / 90.0,
        )
        aristotelian = max(0.0, aristotelian)
        pair_rows.append(
            {
                "embedding_x": ids[i],
                "embedding_y": ids[j],
                PLATONIC_SCORE: platonic,
                ARISTOTELIAN_SCORE: aristotelian,
                PRIMARY_SCORE: 0.5 * (platonic + aristotelian),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    pairs_path = consensus_dir / "consensus_pair_scores.tsv"
    pairs.to_csv(pairs_path, sep="\t", index=False)
    consensus_rho = spearman_rho(
        pairs[PLATONIC_SCORE].to_numpy(dtype=float),
        pairs[ARISTOTELIAN_SCORE].to_numpy(dtype=float),
    )
    consensus_summary_path = consensus_dir / "consensus_summary.json"
    consensus_summary = {
        "status": "PASS",
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_pairs": EXPECTED_PAIRS,
        "consensus_to_component_spearman_rho": {
            "platonic": float(
                spearman_rho(
                    pairs[PRIMARY_SCORE].to_numpy(dtype=float),
                    pairs[PLATONIC_SCORE].to_numpy(dtype=float),
                )
            )
        },
    }
    consensus_summary_path.write_text(json.dumps(consensus_summary, indent=2) + "\n")

    edge_rows: list[dict[str, object]] = []
    for community in sorted(set(communities)):
        members = np.flatnonzero(communities == community)
        for offset, i in enumerate(members):
            for step in (1, 2, 3):
                j = members[(offset + step) % len(members)]
                x, y = sorted((ids[i], ids[j]))
                score = float(
                    pairs.loc[
                        (pairs["embedding_x"] == x) & (pairs["embedding_y"] == y),
                        PRIMARY_SCORE,
                    ].iloc[0]
                )
                edge_rows.append({"embedding_x": x, "embedding_y": y, PRIMARY_SCORE: score})
    for i, j in ((0, 11), (10, 28), (27, 47), (39, 54), (5, 45)):
        x, y = sorted((ids[i], ids[j]))
        score = float(
            pairs.loc[
                (pairs["embedding_x"] == x) & (pairs["embedding_y"] == y),
                PRIMARY_SCORE,
            ].iloc[0]
        )
        edge_rows.append({"embedding_x": x, "embedding_y": y, PRIMARY_SCORE: score})
    edges = pd.DataFrame(edge_rows).drop_duplicates(["embedding_x", "embedding_y"])
    edges_path = graph_dir / "consensus_graph_edges.tsv"
    edges.to_csv(edges_path, sep="\t", index=False)

    graph_summary_path = graph_dir / "graph_structure_summary.json"
    graph_summary = {
        "status": "PASS",
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_pairs": EXPECTED_PAIRS,
        "graph": {"n_edges": int(len(edges))},
        "sources": {"consensus_pairs": {"sha256": sha256(pairs_path)}},
        "outputs": {
            "embedding_annotations": {"sha256": sha256(annotations_path)},
            "graph_edges": {"sha256": sha256(edges_path)},
        },
    }
    graph_summary_path.write_text(json.dumps(graph_summary, indent=2) + "\n")

    counts = annotations.groupby(["community_id", "coarse_modality"]).size()
    modality_sizes = annotations["coarse_modality"].value_counts()
    enrichment_rows: list[dict[str, object]] = []
    for community in sorted(set(communities)):
        community_size = int(np.count_nonzero(communities == community))
        for modality in MODALITY_ORDER:
            observed = int(counts.get((community, modality), 0))
            expected = community_size * int(modality_sizes[modality]) / EXPECTED_EMBEDDINGS
            variance = max(expected * (1.0 - int(modality_sizes[modality]) / EXPECTED_EMBEDDINGS), 0.1)
            standardized = (observed - expected) / np.sqrt(variance)
            interpretation = (
                "enriched"
                if observed > expected and observed == max(annotations.loc[annotations["community_id"] == community, "coarse_modality"].value_counts())
                else "not_significant_after_holm"
            )
            enrichment_rows.append(
                {
                    "community_id": community,
                    "coarse_modality": modality,
                    "observed_count": observed,
                    "expected_count_under_fixed_margins": expected,
                    "fold_enrichment": (observed / community_size) / (int(modality_sizes[modality]) / EXPECTED_EMBEDDINGS),
                    "standardized_count_residual": standardized,
                    "holm_p_two_sided_across_cells": 0.01 if interpretation == "enriched" else 1.0,
                    "cell_interpretation": interpretation,
                }
            )
    enrichment = pd.DataFrame(enrichment_rows)
    enrichment_path = interpretation_dir / "community_modality_enrichment.tsv"
    enrichment.to_csv(enrichment_path, sep="\t", index=False)

    profiles = pd.DataFrame(
        {
            "community_id": [1, 2, 3],
            "dominant_modality": [
                "DNA sequence",
                "Protein sequence",
                "Biomedical text",
            ],
        }
    )
    profiles_path = interpretation_dir / "community_profiles.tsv"
    profiles.to_csv(profiles_path, sep="\t", index=False)

    degree = {embedding_id: 0.0 for embedding_id in ids}
    cross = {embedding_id: 0.0 for embedding_id in ids}
    community_map = annotations.set_index("embedding_id")["community_id"].to_dict()
    for row in edges.itertuples(index=False):
        weight = float(getattr(row, PRIMARY_SCORE))
        degree[row.embedding_x] += weight
        degree[row.embedding_y] += weight
        if community_map[row.embedding_x] != community_map[row.embedding_y]:
            cross[row.embedding_x] += weight
            cross[row.embedding_y] += weight
    role_rows = []
    for embedding_id in ids:
        fraction = cross[embedding_id] / degree[embedding_id] if degree[embedding_id] else 0.0
        role_rows.append(
            {
                "embedding_id": embedding_id,
                "weighted_participation_coefficient": fraction * (2.0 - fraction),
                "cross_community_weight_fraction": fraction,
            }
        )
    roles = pd.DataFrame(role_rows)
    roles["bridge_rank"] = roles["weighted_participation_coefficient"].rank(
        method="min", ascending=False
    ).astype(int)
    roles_path = interpretation_dir / "embedding_graph_roles.tsv"
    roles.to_csv(roles_path, sep="\t", index=False)

    residual_rows = []
    for left in (1, 2, 3):
        for right in range(left, 4):
            n_left = int(np.count_nonzero(communities == left))
            n_right = int(np.count_nonzero(communities == right))
            n_pairs = n_left * (n_left - 1) // 2 if left == right else n_left * n_right
            residual_rows.append(
                {
                    "community_x": left,
                    "community_y": right,
                    "n_pairs": n_pairs,
                    "mean_modality_adjusted_residual": 0.08 if left == right else -0.04,
                }
            )
    residual_summary = pd.DataFrame(residual_rows)
    residuals_path = interpretation_dir / "community_pair_residual_summary.tsv"
    residual_summary.to_csv(residuals_path, sep="\t", index=False)

    interpretation_summary_path = interpretation_dir / "community_interpretation_summary.json"
    interpretation_summary = {
        "status": "PASS",
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_pairs": EXPECTED_PAIRS,
        "n_communities": 3,
        "recommendation": "REPORT_MODALITY_ENRICHED_COMMUNITIES_WITH_RESIDUAL_CROSS_MODALITY_STRUCTURE",
        "residual_structure": {
            "same_community_mean_modality_adjusted_residual": 0.08,
            "between_community_mean_modality_adjusted_residual": -0.04,
        },
        "outputs": {
            "community_modality_enrichment": {"sha256": sha256(enrichment_path)},
            "community_profiles": {"sha256": sha256(profiles_path)},
            "embedding_roles": {"sha256": sha256(roles_path)},
            "community_pair_residual_summary": {"sha256": sha256(residuals_path)},
        },
    }
    interpretation_summary_path.write_text(
        json.dumps(interpretation_summary, indent=2) + "\n"
    )
    return SimpleNamespace(
        root=base,
        consensus_dir=consensus_dir,
        graph_dir=graph_dir,
        interpretation_dir=interpretation_dir,
        output_dir=output_dir,
        seed=SEED,
        layout_iterations=120,
        top_bridge_labels=TOP_BRIDGE_LABELS,
        fixture_component_rho=consensus_rho,
    )


def run_self_test() -> None:
    triangle = np.array(
        [
            [0.0, 0.8, 0.6],
            [0.8, 0.0, 0.7],
            [0.6, 0.7, 0.0],
        ]
    )
    first = weighted_force_layout(triangle, 7, 80)
    second = weighted_force_layout(triangle, 7, 80)
    if not np.allclose(first, second, rtol=0.0, atol=1e-12):
        raise AssertionError("Layout is not deterministic")
    with tempfile.TemporaryDirectory(prefix="cs_10_self_test_") as directory:
        args = write_fixture(Path(directory))
        summary = run_analysis(args, quiet=True)
        if summary.get("status") != "PASS":
            raise AssertionError("End-to-end figure self-test did not PASS")
        expected = {
            "figure_1_consensus_graph_map.png",
            "figure_1_consensus_graph_map.pdf",
            "figure_1_consensus_graph_map.svg",
            "figure_2_consensus_evidence.png",
            "figure_2_consensus_evidence.pdf",
            "figure_2_consensus_evidence.svg",
            "manuscript_figure_2b_community_modality_heatmap.png",
            "manuscript_figure_2b_community_modality_heatmap.pdf",
            "manuscript_figure_2b_community_modality_heatmap.svg",
            "manuscript_figure_2c_community_residual_heatmap.png",
            "manuscript_figure_2c_community_residual_heatmap.pdf",
            "manuscript_figure_2c_community_residual_heatmap.svg",
            "graph_layout_coordinates.tsv",
            "graph_figure_edges.tsv",
            "community_modality_figure_data.tsv",
            "component_scatter_figure_data.tsv",
            "community_residual_figure_data.tsv",
            "figure_captions.md",
            "figure_summary.json",
        }
        observed = {path.name for path in args.output_dir.iterdir()}
        if observed != expected:
            raise AssertionError(
                f"End-to-end figure output contract changed: {sorted(observed ^ expected)}"
            )
        enrichment_plot_data = pd.read_csv(
            args.output_dir / "community_modality_figure_data.tsv", sep="\t"
        )
        plot_annotations = enrichment_plot_data["plot_annotation"].astype(str)
        if plot_annotations.str.contains(r"[ED]$", regex=True).any():
            raise AssertionError("Legacy E/D heatmap annotations remain")
        if not plot_annotations.str.contains(r"\*", regex=True).any():
            raise AssertionError("Significance stars are missing from heatmap annotations")
        if not plot_annotations.str.match(r"^\d+\*{0,3}$").all():
            raise AssertionError("Heatmap count-and-star annotation format changed")
        for name in (
            "figure_1_consensus_graph_map.png",
            "figure_1_consensus_graph_map.pdf",
            "figure_1_consensus_graph_map.svg",
            "figure_2_consensus_evidence.png",
            "figure_2_consensus_evidence.pdf",
            "figure_2_consensus_evidence.svg",
            "manuscript_figure_2b_community_modality_heatmap.png",
            "manuscript_figure_2b_community_modality_heatmap.pdf",
            "manuscript_figure_2b_community_modality_heatmap.svg",
            "manuscript_figure_2c_community_residual_heatmap.png",
            "manuscript_figure_2c_community_residual_heatmap.pdf",
            "manuscript_figure_2c_community_residual_heatmap.svg",
        ):
            if (args.output_dir / name).stat().st_size < 1_000:
                raise AssertionError(f"Figure output {name} is unexpectedly small")
    print("SELF-TEST: PASS")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_analysis(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
