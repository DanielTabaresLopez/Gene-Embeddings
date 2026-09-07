#!/usr/bin/env python3
"""Create the two presentation-ready information/predictability scatter plots.

This is a plotting-only companion to dp_06_information_redundancy_plots.py.
It reads the already-verified ``information_redundancy_scores.tsv`` so the
coordinates are unchanged, while applying the following presentation edits:

* creates only the cross-domain primary view and all-domain sensitivity view;
* omits the modality-adjusted/residual plot;
* omits explanatory subtitles and guide-line captions;
* retains only the dashed y=x diagonal (no median guides);
* uses larger embedding and legend labels with collision adjustment;
* labels x as "Predictability" and y as "Predictive capacity".

Example
-------
python dp_07_information_predictability_plots.py \
  --scores-table results/.../information_redundancy_scores.tsv \
  --output-dir results/.../plots_information_predictability_presentation
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox


DOMAIN_ORDER = [
    "Biomedical text",
    "DNA sequence",
    "RNA sequence",
    "Expression",
    "Functional",
    "Knowledge graph",
    "Multimodal",
    "Networks",
    "Protein sequence",
    "Regulatory genomics",
    "Single-cell",
    # Legacy combined categories retained for old score tables and self-tests.
    "DNA / RNA sequence",
    "Functional / phenotypic",
    "Integrated / multimodal",
    "Knowledge graph context",
    "Networks / ontologies",
    "Transcriptomics / cell state",
]

DOMAIN_COLORS = {
    "Biomedical text": "#E64B35",
    "DNA sequence": "#B64C5C",
    "RNA sequence": "#B65F76",
    "Expression": "#D65DB1",
    "Functional": "#6C7680",
    "Knowledge graph": "#C25A9B",
    "Multimodal": "#6F57A5",
    "Networks": "#6B8E3D",
    "Single-cell": "#2C9A86",
    "DNA / RNA sequence": "#F39C34",
    "Functional / phenotypic": "#C9A227",
    "Integrated / multimodal": "#3BA55D",
    "Knowledge graph context": "#00A087",
    "Networks / ontologies": "#2F80C1",
    "Protein sequence": "#4C78D0",
    "Regulatory genomics": "#8C65D3",
    "Transcriptomics / cell state": "#D65DB1",
}

REQUIRED_COLUMNS = {
    "embedding",
    "domain",
    "information_cross_domain",
    "redundancy_cross_domain",
    "information_all_domains",
    "redundancy_all_domains",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replot verified information/predictability scores for presentation."
    )
    parser.add_argument("--scores-table", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label-mode", choices=("all", "extremes", "none"), default="all")
    parser.add_argument("--label-font-size", type=float, default=9.5)
    parser.add_argument("--legend-font-size", type=float, default=11.0)
    parser.add_argument("--point-size", type=float, default=64.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scores table not found: {path}")
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Scores table is missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["embedding"] = frame["embedding"].astype(str).str.strip()
    if "display_name" in frame.columns:
        frame["display_name"] = frame["display_name"].astype(str).str.strip()
        if frame.display_name.eq("").any() or frame.display_name.duplicated().any():
            raise ValueError("Display names must be nonblank and unique.")
    frame["domain"] = frame["domain"].astype(str).str.strip()
    if frame.embedding.eq("").any() or frame.domain.eq("").any():
        raise ValueError("Empty embedding or domain values were found.")
    if frame.embedding.duplicated().any():
        duplicated = sorted(frame.loc[frame.embedding.duplicated(False), "embedding"].unique())
        raise ValueError(f"Duplicate embeddings in score table: {duplicated}")

    numeric = sorted(REQUIRED_COLUMNS - {"embedding", "domain"})
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite plotted score values were found.")
    return frame


def expanded_square_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = max(hi - lo, 0.10)
    margin = 0.13 * span
    return lo - margin, hi + margin


def select_labels(frame: pd.DataFrame, x_col: str, y_col: str, mode: str) -> pd.DataFrame:
    if mode == "none":
        return frame.iloc[0:0]
    if mode == "all":
        return frame

    # Label extremes plus quadrant representatives for a clean fallback view.
    selected: set[str] = set()
    for column in (x_col, y_col):
        selected.update(frame.nlargest(8, column).embedding)
        selected.update(frame.nsmallest(5, column).embedding)
    x_mid = frame[x_col].median()
    y_mid = frame[y_col].median()
    for _, group in frame.groupby([frame[x_col].ge(x_mid), frame[y_col].ge(y_mid)]):
        distance = ((group[x_col] - x_mid) ** 2 + (group[y_col] - y_mid) ** 2)
        selected.update(group.loc[distance.nlargest(min(3, len(group))).index, "embedding"])
    return frame[frame.embedding.isin(selected)]


def _overlap(a: Bbox, b: Bbox, padding: float = 2.0) -> tuple[float, float]:
    overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0) + padding
    overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0) + padding
    return overlap_x, overlap_y


def fallback_repel(ax: plt.Axes, texts: list[plt.Text], anchors: np.ndarray) -> None:
    """Deterministic display-coordinate label repulsion when adjustText is absent."""
    if len(texts) < 2:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_box = ax.get_window_extent(renderer).expanded(0.985, 0.985)

    boxes = [text.get_window_extent(renderer) for text in texts]
    widths = np.array([box.width for box in boxes], float)
    heights = np.array([box.height for box in boxes], float)
    anchor_px = ax.transData.transform(anchors)

    # A deterministic spiral prevents identical starting positions.
    order = np.argsort([text.get_text() for text in texts])
    centers = anchor_px.copy()
    for rank, index in enumerate(order):
        angle = rank * 2.399963229728653
        radius = 7.0 + 2.2 * math.sqrt(rank)
        centers[index] += radius * np.array([math.cos(angle), math.sin(angle)])

    for _ in range(1200):
        movement = np.zeros_like(centers)
        overlaps = 0
        current = [
            Bbox.from_bounds(
                centers[i, 0] - widths[i] / 2,
                centers[i, 1] - heights[i] / 2,
                widths[i],
                heights[i],
            )
            for i in range(len(texts))
        ]
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                ox, oy = _overlap(current[i], current[j], padding=2.5)
                if ox <= 0 or oy <= 0:
                    continue
                overlaps += 1
                delta = centers[j] - centers[i]
                if ox < oy:
                    sign = 1.0 if delta[0] >= 0 else -1.0
                    push = np.array([(ox / 2 + 0.7) * sign, 0.0])
                else:
                    sign = 1.0 if delta[1] >= 0 else -1.0
                    push = np.array([0.0, (oy / 2 + 0.7) * sign])
                movement[i] -= push
                movement[j] += push

        # Keep text inside the plotting area.
        centers += np.clip(movement, -12.0, 12.0)
        centers[:, 0] = np.clip(
            centers[:, 0], axis_box.x0 + widths / 2, axis_box.x1 - widths / 2
        )
        centers[:, 1] = np.clip(
            centers[:, 1], axis_box.y0 + heights / 2, axis_box.y1 - heights / 2
        )
        if overlaps == 0:
            break

    data_positions = ax.transData.inverted().transform(centers)
    for text, position in zip(texts, data_positions):
        text.set_position(position)

    # Leader lines are added only after text placement, so they do not affect boxes.
    for anchor, position in zip(anchors, data_positions):
        if np.linalg.norm(ax.transData.transform(position) - ax.transData.transform(anchor)) > 8:
            ax.annotate(
                "",
                xy=anchor,
                xytext=position,
                arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.45, "alpha": 0.60},
                zorder=2,
            )


def place_labels(ax: plt.Axes, label_frame: pd.DataFrame, x_col: str, y_col: str,
                 font_size: float, seed: int) -> str:
    if label_frame.empty:
        return "none"

    np.random.seed(seed)
    anchors = label_frame[[x_col, y_col]].to_numpy(float)
    texts = [
        ax.text(
            getattr(row, x_col),
            getattr(row, y_col),
            str(getattr(row, "display_name", row.embedding)),
            fontsize=font_size,
            color="#242424",
            ha="center",
            va="center",
            zorder=4,
            clip_on=True,
        )
        for row in label_frame.itertuples(index=False)
    ]

    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            x=anchors[:, 0],
            y=anchors[:, 1],
            ax=ax,
            expand=(1.10, 1.20),
            force_text=(0.70, 0.95),
            force_static=(0.22, 0.30),
            force_pull=(0.015, 0.020),
            pull_threshold=12,
            max_move=(18, 18),
            explode_radius="auto",
            ensure_inside_axes=True,
            prevent_crossings=True,
            time_lim=30,
            arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.45, "alpha": 0.60},
        )
        return "adjustText"
    except ImportError:
        fallback_repel(ax, texts, anchors)
        return "built-in"


def plot_view(
    frame: pd.DataFrame,
    output_dir: Path,
    basename: str,
    title: str,
    x_col: str,
    y_col: str,
    args: argparse.Namespace,
) -> str:
    fig, ax = plt.subplots(figsize=(16.5, 12.5))

    known_domains = [domain for domain in DOMAIN_ORDER if domain in set(frame.domain)]
    extra_domains = sorted(set(frame.domain) - set(DOMAIN_ORDER))
    domains = known_domains + extra_domains
    fallback = plt.get_cmap("tab20")

    for index, domain in enumerate(domains):
        group = frame[frame.domain.eq(domain)]
        color = DOMAIN_COLORS.get(domain, fallback(index % 20))
        ax.scatter(
            group[x_col],
            group[y_col],
            s=args.point_size,
            c=[color],
            edgecolors="white",
            linewidths=0.75,
            alpha=0.94,
            label=domain,
            zorder=3,
        )

    low, high = expanded_square_limits(frame[x_col].to_numpy(), frame[y_col].to_numpy())
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")

    labels = select_labels(frame, x_col, y_col, args.label_mode)
    placement = place_labels(
        ax, labels, x_col, y_col, args.label_font_size, args.seed
    )

    # Keep only the requested dashed y=x reference; draw after label expansion.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    diagonal_low = max(xlim[0], ylim[0])
    diagonal_high = min(xlim[1], ylim[1])
    ax.plot(
        [diagonal_low, diagonal_high],
        [diagonal_low, diagonal_high],
        linestyle=(0, (6, 5)),
        color="#777777",
        linewidth=1.25,
        alpha=0.85,
        zorder=1,
    )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    ax.set_xlabel("Predictability", fontsize=15, labelpad=12)
    ax.set_ylabel("Predictive capacity", fontsize=15, labelpad=12)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.70)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.8)

    legend = ax.legend(
        title="Modality",
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        frameon=False,
        fontsize=args.legend_font_size,
        title_fontsize=args.legend_font_size + 0.5,
        borderaxespad=0.0,
        handletextpad=0.55,
        labelspacing=0.75,
        markerscale=1.15,
    )
    legend._legend_box.align = "left"

    fig.subplots_adjust(left=0.09, right=0.76, bottom=0.09, top=0.92)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{basename}.{extension}",
            dpi=args.dpi if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    return placement


def make_synthetic_scores(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(65):
        domain = DOMAIN_ORDER[index % len(DOMAIN_ORDER)]
        latent = rng.beta(2.2, 4.0)
        incoming = np.clip(0.04 + 0.45 * latent + rng.normal(0, 0.055), -0.05, 0.52)
        outgoing = np.clip(0.05 + 0.38 * latent + rng.normal(0, 0.060), -0.05, 0.52)
        rows.append(
            {
                "embedding": f"synthetic_embedding_{index:02d}",
                "domain": domain,
                "information_cross_domain": outgoing,
                "redundancy_cross_domain": incoming,
                "information_all_domains": outgoing + rng.normal(0.015, 0.012),
                "redundancy_all_domains": incoming + rng.normal(0.018, 0.012),
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    if args.self_test:
        root = Path(tempfile.mkdtemp(prefix="predictability_replot_selftest_"))
        scores_path = root / "information_redundancy_scores.tsv"
        output_dir = root / "plots"
        make_synthetic_scores(scores_path, args.seed)
    else:
        if args.scores_table is None or args.output_dir is None:
            raise SystemExit("--scores-table and --output-dir are required unless --self-test is used.")
        scores_path = args.scores_table
        output_dir = args.output_dir

    frame = read_scores(scores_path)
    placement_primary = plot_view(
        frame,
        output_dir,
        "information_predictability_cross_domain",
        "Cross-domain information versus redundancy",
        "redundancy_cross_domain",
        "information_cross_domain",
        args,
    )
    placement_sensitivity = plot_view(
        frame,
        output_dir,
        "information_predictability_all_domains_sensitivity",
        "All-domain information versus redundancy",
        "redundancy_all_domains",
        "information_all_domains",
        args,
    )

    expected = [
        output_dir / "information_predictability_cross_domain.png",
        output_dir / "information_predictability_cross_domain.pdf",
        output_dir / "information_predictability_all_domains_sensitivity.png",
        output_dir / "information_predictability_all_domains_sensitivity.pdf",
    ]
    missing = [str(path) for path in expected if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty outputs: {missing}")

    print(f"Scores table: {scores_path.resolve()}")
    print(f"Embeddings:   {len(frame):,}")
    print(f"Outputs:      {output_dir.resolve()}")
    print(f"Labels:       primary={placement_primary}; sensitivity={placement_sensitivity}")
    print("Generated exactly two plot families (PNG + PDF); no residual plot.")
    if args.self_test:
        print("SELF-TEST PASSED")


if __name__ == "__main__":
    run(parse_args())
