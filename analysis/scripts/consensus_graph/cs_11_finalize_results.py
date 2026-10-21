#!/usr/bin/env python3
"""Audit the completed consensus-graph workflow and write a final manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd


EXPECTED_EMBEDDINGS = 56
EXPECTED_GENES = 14_039
EXPECTED_PAIRS = 1_540
EXPECTED_GENE_SHA256 = "caa45ecca3ce0526d2dbb53e9dbf008016a1d6ba513488682a22ca69a04ee238"


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(require_file(path).read_text())
    if payload.get("status") != "PASS":
        raise ValueError(f"Upstream stage did not PASS: {path}")
    return payload


def canonical_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    left = frame["embedding_x"].astype(str)
    right = frame["embedding_y"].astype(str)
    if (left == right).any():
        raise ValueError("Pair table contains a self-pair")
    pairs = {(min(x, y), max(x, y)) for x, y in zip(left, right)}
    if len(pairs) != len(frame):
        raise ValueError("Pair table contains duplicate canonical pairs")
    return pairs


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    workflow = root / "results/consensus_graph_v2"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow / "11_final_audit"
    )

    panel_path = require_file(workflow / "01_fixed_panel/embedding_metadata.tsv")
    genes_path = require_file(workflow / "01_fixed_panel/common_genes.txt")
    atlas_path = require_file(workflow / "02_common_panel_atlas/atlas_pair_scores.tsv")
    consensus_pairs_path = require_file(workflow / "06_consensus/consensus_pair_scores.tsv")
    annotations_path = require_file(
        workflow / "08_graph_structure/embedding_graph_annotations.tsv"
    )
    figure_dir = workflow / "10_figures"

    summaries = {
        "views": workflow / "04_views/view_summary.json",
        "concordance": workflow / "05_view_concordance/concordance_summary.json",
        "consensus": workflow / "06_consensus/consensus_summary.json",
        "modalities": workflow / "07_modalities/modality_review_summary.json",
        "graph": workflow / "08_graph_structure/graph_structure_summary.json",
        "interpretation": workflow / "09_community_interpretation/community_interpretation_summary.json",
        "figures": figure_dir / "figure_summary.json",
    }
    payloads = {name: read_json(path) for name, path in summaries.items()}

    panel = pd.read_csv(panel_path, sep="\t", dtype=str)
    required_panel = {"embedding_id", "display_name", "modality"}
    if not required_panel.issubset(panel.columns):
        raise ValueError("Frozen panel metadata lacks publication labels")
    if len(panel) != EXPECTED_EMBEDDINGS or panel["embedding_id"].duplicated().any():
        raise ValueError("Frozen panel is not the expected 56-embedding panel")
    if panel["display_name"].duplicated().any():
        raise ValueError("Display names are not unique")
    ids = panel["embedding_id"].astype(str).tolist()
    expected_pairs = set(combinations(sorted(ids), 2))
    if len(expected_pairs) != EXPECTED_PAIRS:
        raise ValueError("Frozen panel does not imply 1,540 pairs")

    genes = [line.strip() for line in genes_path.read_text().splitlines() if line.strip()]
    if len(genes) != EXPECTED_GENES or len(set(genes)) != EXPECTED_GENES:
        raise ValueError("Frozen common-gene file is not the expected 14,039-gene set")
    if sha256(genes_path) != EXPECTED_GENE_SHA256:
        raise ValueError("Frozen common-gene SHA256 changed")

    atlas = pd.read_csv(atlas_path, sep="\t", low_memory=False)
    consensus_pairs = pd.read_csv(consensus_pairs_path, sep="\t", low_memory=False)
    for label, frame in (("atlas", atlas), ("consensus", consensus_pairs)):
        if canonical_pairs(frame) != expected_pairs:
            raise ValueError(f"{label} pair coverage differs from the frozen panel")
    if set(pd.to_numeric(atlas["gene_count"], errors="raise")) != {EXPECTED_GENES}:
        raise ValueError("Atlas rows do not all use the 14,039 fixed genes")

    annotations = pd.read_csv(annotations_path, sep="\t", dtype=str)
    if set(annotations["embedding_id"]) != set(ids):
        raise ValueError("Graph annotations differ from the frozen panel")
    manifest_display = panel.set_index("embedding_id")["display_name"].to_dict()
    manifest_path = require_file(args.manifest or root / "config/consensus_graph_embedding_manifest.xlsx")
    actual_manifest = pd.read_excel(manifest_path, dtype=str).rename(columns={
        "Files name": "embedding_id", "Embedding display name": "display_name",
        "Final modality": "modality",
    })
    for column in ("embedding_id", "display_name", "modality"):
        actual_manifest[column] = actual_manifest[column].str.strip()
    if actual_manifest.embedding_id.duplicated().any():
        raise ValueError("Duplicate embedding IDs in the publication manifest")
    current = actual_manifest.set_index("embedding_id").loc[ids]
    if current.display_name.to_dict() != manifest_display:
        raise ValueError("Frozen panel display names differ from the publication manifest")
    if current.modality.to_dict() != panel.set_index("embedding_id").modality.to_dict():
        raise ValueError("Frozen panel modalities differ from the publication manifest")
    observed_display = annotations.set_index("embedding_id")["display_name"].to_dict()
    if observed_display != manifest_display:
        raise ValueError("Graph display names differ from embedding_manifest.xlsx")

    for name, payload in payloads.items():
        if int(payload.get("n_embeddings", EXPECTED_EMBEDDINGS)) != EXPECTED_EMBEDDINGS:
            raise ValueError(f"{name} reports an unexpected embedding count")
        if int(payload.get("n_pairs", EXPECTED_PAIRS)) != EXPECTED_PAIRS:
            raise ValueError(f"{name} reports an unexpected pair count")

    figure_names = [
        f"figure_1_consensus_graph_map.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ] + [
        f"figure_2_consensus_evidence.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ] + [
        f"manuscript_figure_2b_community_modality_heatmap.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ] + [
        f"manuscript_figure_2c_community_residual_heatmap.{suffix}"
        for suffix in ("png", "pdf", "svg")
    ]
    figures = [require_file(figure_dir / name) for name in figure_names]

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_rows = []
    for stage, path in summaries.items():
        path = require_file(path)
        artifact_rows.append(
            {"artifact_type": "stage_summary", "stage": stage, "path": str(path), "sha256": sha256(path)}
        )
    for path in figures:
        artifact_rows.append(
            {"artifact_type": "figure", "stage": "figures", "path": str(path), "sha256": sha256(path)}
        )
    artifact_manifest = pd.DataFrame(artifact_rows)
    artifact_manifest.to_csv(output_dir / "final_artifact_manifest.tsv", sep="\t", index=False)

    final = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_embeddings": EXPECTED_EMBEDDINGS,
        "n_common_genes": EXPECTED_GENES,
        "n_pairs": EXPECTED_PAIRS,
        "common_genes_sha256": EXPECTED_GENE_SHA256,
        "manifest_display_names_verified": True,
        "complete_pair_coverage_verified": True,
        "figures_verified": figure_names,
        "stage_summaries": {
            name: {"path": str(path), "sha256": sha256(require_file(path))}
            for name, path in summaries.items()
        },
        "artifact_manifest": str(output_dir / "final_artifact_manifest.tsv"),
    }
    (output_dir / "final_audit.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n"
    )
    print("===== CONSENSUS GRAPH FINAL AUDIT =====")
    print("status: PASS")
    print(f"embeddings: {EXPECTED_EMBEDDINGS}")
    print(f"fixed common genes: {EXPECTED_GENES:,}")
    print(f"pairs: {EXPECTED_PAIRS:,}")
    print("manifest display names: verified")
    print("figure formats: PNG, PDF, SVG")
    print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
