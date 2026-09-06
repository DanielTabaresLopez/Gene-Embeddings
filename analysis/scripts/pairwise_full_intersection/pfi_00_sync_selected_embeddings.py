#!/usr/bin/env python3
"""Download or verify exactly the embedding packages selected by the manifest."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfi_manifest import (load_manifest, panel_sha256, resolve_path, sha256_file,
                          write_normalized_manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Do not contact Hugging Face; validate the existing local mirror.",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Ask huggingface_hub to refresh already cached selected files.",
    )
    args = parser.parse_args()

    started = time.time()
    config = yaml.safe_load(args.config.read_text())
    panel, manifest_path = load_manifest(config, args.config)
    library = resolve_path(config["library_root"], args.config)
    output = resolve_path(config["selection_output"], args.config)
    output.mkdir(parents=True, exist_ok=True)

    resolved_revision = None
    if not args.verify_only:
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as error:
            raise RuntimeError(
                "huggingface_hub is required for downloads; install requirements.txt "
                "or rerun with --verify-only"
            ) from error

        repo_id = str(config["hf_repo_id"])
        repo_type = str(config.get("hf_repo_type", "dataset"))
        revision = str(config.get("hf_revision", "main"))
        patterns = []
        for embedding_id in panel.embedding_id:
            base = f"data/embeddings/{embedding_id}"
            patterns.extend([
                f"{base}/embeddings.npz",
                f"{base}/genes.tsv",
                f"{base}/metadata.json",
                f"{base}/README.md",
            ])
        print(f"Downloading {len(panel)} selected packages from {repo_id}@{revision}")
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=library,
            allow_patterns=patterns,
            max_workers=int(config.get("hf_download_workers", 4)),
            force_download=args.force_download,
        )
        resolved_revision = HfApi().repo_info(
            repo_id=repo_id, repo_type=repo_type, revision=revision
        ).sha

    package_root = library / "data" / "embeddings"
    if not package_root.is_dir():
        raise FileNotFoundError(f"embedding package root not found: {package_root}")

    array_key = str(config.get("array_key", "embeddings"))
    id_column = str(config.get("id_column", "ensembl_gene_id"))
    records = []
    for position, row in enumerate(panel.itertuples(index=False), 1):
        package = package_root / row.embedding_id
        embeddings_file = package / "embeddings.npz"
        genes_file = package / "genes.tsv"
        missing = [str(path) for path in (embeddings_file, genes_file) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{row.embedding_id}: selected package is incomplete; missing {missing}"
            )
        genes = pd.read_csv(genes_file, sep="\t", dtype=str)
        if id_column not in genes.columns:
            raise ValueError(f"{row.embedding_id}: genes.tsv lacks {id_column!r}")
        with np.load(embeddings_file) as archive:
            if array_key not in archive:
                raise KeyError(
                    f"{row.embedding_id}: embeddings.npz lacks array key {array_key!r}"
                )
            shape = archive[array_key].shape
        if len(shape) != 2:
            raise ValueError(f"{row.embedding_id}: embedding matrix is not 2D: {shape}")
        if shape[0] != len(genes):
            raise ValueError(
                f"{row.embedding_id}: genes.tsv has {len(genes)} rows but matrix has {shape[0]}"
            )
        records.append({
            "manifest_order": int(row.manifest_order),
            "embedding_id": row.embedding_id,
            "display_name": row.display_name,
            "modality": row.modality,
            "n_rows": int(shape[0]),
            "n_dimensions": int(shape[1]),
            "embeddings_size_bytes": embeddings_file.stat().st_size,
        })
        print(
            f"  [{position:>2}/{len(panel)}] {row.embedding_id:<52} "
            f"rows={shape[0]:>6,} dim={shape[1]:>6,}"
        )

    selected = set(panel.embedding_id)
    local_packages = {
        path.name for path in package_root.iterdir()
        if path.is_dir() and (path / "embeddings.npz").is_file()
    }
    ignored = sorted(local_packages - selected)

    table = pd.DataFrame(records)
    table.to_csv(output / "selected_embeddings.tsv", sep="\t", index=False)
    write_normalized_manifest(panel, output / "normalized_manifest.tsv")

    elapsed = time.time() - started
    metadata = {
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config),
        "manifest_file": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "panel_sha256": panel_sha256(panel),
        "n_embeddings": int(len(panel)),
        "n_pairs": int(len(panel) * (len(panel) - 1) // 2),
        "embedding_ids": panel.embedding_id.tolist(),
        "ignored_local_packages": ignored,
        "verify_only": bool(args.verify_only),
        "hf_repo_id": config.get("hf_repo_id"),
        "hf_requested_revision": config.get("hf_revision"),
        "hf_resolved_revision": resolved_revision,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(elapsed, 2),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "status": "PASS",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    report = [
        "# Selected embedding panel", "",
        f"- Selected by manifest: {len(panel)} embeddings ({metadata['n_pairs']:,} pairs)",
        f"- Present locally but ignored: {len(ignored)}", "",
        "The manifest is the sole source of panel membership. Local packages",
        "absent from the manifest are intentionally excluded from every downstream step.", "",
    ]
    if ignored:
        report += ["## Ignored local packages", "", *[f"- `{value}`" for value in ignored], ""]
    (output / "report.md").write_text("\n".join(report))

    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n")

    print(f"\nValidated {len(panel)} selected embeddings.")
    print(f"Ignored {len(ignored)} unlisted local packages.")
    print(f"DONE in {elapsed:.1f}s -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
