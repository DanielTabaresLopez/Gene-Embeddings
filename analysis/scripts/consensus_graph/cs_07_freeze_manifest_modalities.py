#!/usr/bin/env python3
"""Freeze manifest display names and modalities for consensus-graph analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--panel-dir", type=Path, default=None)
    parser.add_argument("--consensus-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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


def atomic_json(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    workflow = root / "results/consensus_graph_v2"
    panel_dir = (
        args.panel_dir.expanduser().resolve()
        if args.panel_dir is not None
        else workflow / "01_fixed_panel"
    )
    consensus_dir = (
        args.consensus_dir.expanduser().resolve()
        if args.consensus_dir is not None
        else workflow / "06_consensus"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workflow / "07_modalities"
    )

    metadata_path = require_file(panel_dir / "embedding_metadata.tsv")
    consensus_summary_path = require_file(consensus_dir / "consensus_summary.json")
    metadata = pd.read_csv(metadata_path, sep="\t", dtype=str)
    required = {"embedding_id", "display_name", "modality"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Frozen panel metadata is missing {sorted(missing)}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("Frozen panel metadata contains missing labels")
    if metadata["embedding_id"].duplicated().any():
        raise ValueError("Frozen panel metadata contains duplicate embedding IDs")
    if metadata["display_name"].duplicated().any():
        raise ValueError("Frozen panel metadata contains duplicate display names")

    consensus_summary = json.loads(consensus_summary_path.read_text())
    n_embeddings = len(metadata)
    n_pairs = n_embeddings * (n_embeddings - 1) // 2
    if consensus_summary.get("status") != "PASS":
        raise ValueError("Upstream consensus summary does not have status PASS")
    if int(consensus_summary.get("n_embeddings", -1)) != n_embeddings:
        raise ValueError("Consensus and frozen-panel embedding counts differ")
    if int(consensus_summary.get("n_pairs", -1)) != n_pairs:
        raise ValueError("Consensus and frozen-panel pair counts differ")

    labels = metadata[["embedding_id", "display_name", "modality"]].copy()
    labels = labels.rename(columns={"modality": "coarse_modality"})
    labels["hybrid_or_multimodal"] = labels["coarse_modality"].eq("Multimodal")
    labels["classification_basis"] = "Final modality from embedding_manifest.xlsx"
    labels["review_status"] = "manifest_frozen"

    label_path = output_dir / "reviewed_coarse_modalities.tsv"
    atomic_dataframe(labels, label_path)
    modality_order = list(dict.fromkeys(labels["coarse_modality"].astype(str)))
    counts = labels["coarse_modality"].value_counts().reindex(modality_order).astype(int)
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "review_version": "embedding_manifest_final_modality_v2",
        "n_embeddings": n_embeddings,
        "n_pairs": n_pairs,
        "coarse_modalities": modality_order,
        "counts": {key: int(value) for key, value in counts.items()},
        "hybrid_rule": "hybrid_or_multimodal is true only for Final modality=Multimodal",
        "sources": {
            "panel_metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
            "consensus_summary": {
                "path": str(consensus_summary_path),
                "sha256": sha256(consensus_summary_path),
            },
        },
        "outputs": {
            "reviewed_labels": {"path": str(label_path), "sha256": sha256(label_path)}
        },
    }
    atomic_json(summary, output_dir / "modality_review_summary.json")

    print("===== MANIFEST MODALITIES =====")
    print("status: PASS")
    print(f"embeddings: {n_embeddings}")
    print(f"pairs: {n_pairs}")
    print(counts.to_string())
    print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
