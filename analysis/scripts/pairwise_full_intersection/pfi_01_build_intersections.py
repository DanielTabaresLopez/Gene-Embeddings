#!/usr/bin/env python3
"""
pfi_01_build_intersections.py

Step 1 of 4 of the Pairwise Full-Intersection (PFI) analysis, after optional
download/verification step 0.

Determines each manifest-selected embedding's valid gene set and every pair's
full shared gene set. Packages absent from the manifest are ignored, even when
they are present in the local Hugging Face mirror.
"""

from __future__ import annotations

import argparse, json, os, platform, sys, time
from itertools import combinations
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfi_manifest import (annotate_pairs, dataframe_to_markdown, load_manifest,
                          panel_sha256, resolve_path, sha256_file,
                          write_normalized_manifest)


def valid_mask(gene_ids, matrix):
    """Finite, non-zero-norm, named, non-duplicate. Never zero-fill."""
    finite = np.isfinite(matrix).all(axis=1)
    nonzero = np.linalg.norm(matrix, axis=1) > 0.0
    ids = pd.Series(gene_ids)
    named = ids.notna().to_numpy() & (ids.fillna("").str.strip() != "").to_numpy()
    unique = ~ids.duplicated(keep=False).to_numpy()
    return finite & nonzero & named & unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    started = time.time()
    config = yaml.safe_load(args.config.read_text())

    panel, manifest_path = load_manifest(config, args.config)
    library = resolve_path(config["library_root"], args.config)
    out = resolve_path(config["intersection_output"], args.config)
    out.mkdir(parents=True, exist_ok=True)
    array_key = config.get("array_key", "embeddings")
    id_column = config.get("id_column", "ensembl_gene_id")

    print("=" * 68)
    print("PFI STEP 1: PER-PAIR FULL INTERSECTIONS")
    print("=" * 68)
    print(f"library : {library}")
    print(f"output  : {out}")

    pkg_root = library / "data" / "embeddings"
    if not pkg_root.is_dir():
        raise FileNotFoundError(f"not found: {pkg_root}")

    ids = panel.embedding_id.tolist()
    selected = set(ids)
    local_ids = {
        path.name for path in pkg_root.iterdir()
        if path.is_dir() and (path / "embeddings.npz").is_file()
    }
    missing = [embedding_id for embedding_id in ids if embedding_id not in local_ids]
    if missing:
        raise FileNotFoundError(
            "manifest-selected embedding packages are missing locally: "
            + ", ".join(missing)
            + ". Run pfi_00_sync_selected_embeddings.py first."
        )
    ignored = sorted(local_ids - selected)
    print(f"panel   : {len(ids)} manifest-selected embeddings")
    print(f"ignored : {len(ignored)} local packages absent from the manifest\n")

    display_map = dict(zip(panel.embedding_id, panel.display_name))
    modality_map = dict(zip(panel.embedding_id, panel.modality))
    order_map = dict(zip(panel.embedding_id, panel.manifest_order))

    valid, rows = {}, []
    for position, eid in enumerate(ids, 1):
        genes = pd.read_csv(pkg_root / eid / "genes.tsv", sep="\t", dtype=str)
        if id_column not in genes.columns:
            raise ValueError(f"{eid}: genes.tsv lacks required column {id_column!r}")
        with np.load(pkg_root / eid / "embeddings.npz") as npz:
            if array_key not in npz:
                raise KeyError(f"{eid}: embeddings.npz lacks array key {array_key!r}")
            matrix = np.asarray(npz[array_key], dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"{eid}: embedding matrix must be 2D, found {matrix.shape}")
        if matrix.shape[0] != len(genes):
            raise ValueError(f"{eid}: genes.tsv has {len(genes)} rows, matrix has {matrix.shape[0]}")
        mask = valid_mask(genes[id_column].to_numpy(), matrix)
        valid[eid] = genes[id_column].to_numpy()[mask]
        rows.append({"manifest_order": int(order_map[eid]),
                     "embedding_id": eid,
                     "display_name": display_map[eid],
                     "modality": modality_map[eid],
                     "n_rows": int(matrix.shape[0]),
                     "n_dimensions": int(matrix.shape[1]),
                     "n_valid_genes": int(mask.sum()),
                     "n_dropped": int((~mask).sum())})
        print(f"  [{position:>3}/{len(ids)}] {eid:<50} "
              f"valid={int(mask.sum()):>6} dim={matrix.shape[1]:>5}")
        del matrix

    pd.DataFrame(rows).to_csv(out / "embedding_valid_genes.tsv", sep="\t", index=False)
    write_normalized_manifest(panel, out / "normalized_manifest.tsv")

    universe = sorted(set().union(*[set(v.tolist()) for v in valid.values()]))
    index = {g: i for i, g in enumerate(universe)}
    validity = np.zeros((len(universe), len(ids)), dtype=bool)
    for column, eid in enumerate(ids):
        validity[[index[g] for g in valid[eid]], column] = True

    strict = int(validity.all(axis=1).sum())
    print(f"\n  gene union      : {len(universe):,}")
    print(f"  strict universal: {strict:,}  (current manifest panel)")

    np.savez_compressed(out / "validity_matrix.npz", validity=validity,
                        embedding_ids=np.array(ids, dtype=object),
                        gene_ids=np.array(universe, dtype=object))

    counts = validity.astype(np.int32).T @ validity.astype(np.int32)
    pair_rows = []
    for i, j in combinations(range(len(ids)), 2):
        a, b = ids[i], ids[j]
        shared = int(counts[i, j])
        pair_rows.append({"pair_id": f"{a}__VS__{b}", "embedding_x": a, "embedding_y": b,
                          "index_x": i, "index_y": j,
                          "n_valid_x": int(counts[i, i]), "n_valid_y": int(counts[j, j]),
                          "n_intersection": shared,
                          "jaccard": round(shared / (counts[i, i] + counts[j, j] - shared), 6),
                          "ratio_over_strict_universal": round(shared / strict, 3) if strict else None})

    pairs = annotate_pairs(pd.DataFrame(pair_rows), panel)
    pairs = pairs.sort_values("n_intersection", ascending=False)
    pairs.to_csv(out / "pair_intersections.tsv", sep="\t", index=False)

    n_int = pairs.n_intersection
    print(f"\n  pairs           : {len(pairs):,}")
    print(f"  intersection    : min={n_int.min():,}  q25={int(n_int.quantile(.25)):,}"
          f"  median={int(n_int.median()):,}  q75={int(n_int.quantile(.75)):,}  max={n_int.max():,}")
    if strict:
        print(f"  median gain over the current panel-wide strict universe: "
              f"{n_int.median() / strict:.2f}x")
    else:
        print("  current panel-wide strict universe is empty; gain ratio is undefined")

    elapsed = time.time() - started
    report = [
        "# PFI step 1 - per-pair full intersections", "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}    Runtime: {elapsed:.1f}s", "",
        "## Panel", "",
        f"- Embeddings: {len(ids)}",
        f"- Manifest: `{manifest_path.name}`",
        f"- Local packages ignored because absent from manifest: {len(ignored)}",
        f"- Pairs: {len(pairs):,}",
        f"- Gene union: {len(universe):,}",
        f"- Strict universal for the current {len(ids)}-embedding panel: {strict:,}", "",
        "## Intersection size distribution", "",
        "| statistic | genes | ratio over strict universal |",
        "|---|---|---|",
        f"| min | {n_int.min():,} | {n_int.min()/strict:.2f}x |" if strict else f"| min | {n_int.min():,} | n/a |",
        f"| q25 | {int(n_int.quantile(.25)):,} | {n_int.quantile(.25)/strict:.2f}x |" if strict else f"| q25 | {int(n_int.quantile(.25)):,} | n/a |",
        f"| median | {int(n_int.median()):,} | {n_int.median()/strict:.2f}x |" if strict else f"| median | {int(n_int.median()):,} | n/a |",
        f"| q75 | {int(n_int.quantile(.75)):,} | {n_int.quantile(.75)/strict:.2f}x |" if strict else f"| q75 | {int(n_int.quantile(.75)):,} | n/a |",
        f"| max | {n_int.max():,} | {n_int.max()/strict:.2f}x |" if strict else f"| max | {n_int.max():,} | n/a |", "",
        "## What to check before step 2", "",
        "1. Confirm that all manifest rows are present and that the valid-gene",
        "   counts are plausible. The strict-universal count need not reproduce",
        "   an older atlas after the embedding roster changes.",
        "2. The spread of intersection sizes is the whole reason for this",
        "   analysis. It is also the confound: intersection size is not random,",
        "   because low-coverage embeddings cover the well-studied core of the",
        "   genome. Step 3 tests that association explicitly.", "",
        "## Smallest 15 intersections", "",
        dataframe_to_markdown(
            pairs.nsmallest(15, "n_intersection")[["pair_id", "n_intersection"]]
        ), "",
        "## Largest 15 intersections", "",
        dataframe_to_markdown(
            pairs.nlargest(15, "n_intersection")[["pair_id", "n_intersection"]]
        ), "",
    ]
    (out / "report.md").write_text("\n".join(report))

    (out / "run_metadata.json").write_text(json.dumps({
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config),
        "manifest_file": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "panel_sha256": panel_sha256(panel),
        "config": config, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(elapsed, 2), "python_version": sys.version,
        "platform": platform.platform(), "numpy_version": np.__version__,
        "n_embeddings": len(ids), "n_pairs": int(len(pairs)),
        "embedding_ids": ids, "gene_union": len(universe),
        "ignored_local_packages": ignored,
        "strict_universal_genes": strict, "status": "PASS",
    }, indent=2))

    lines = [f"{sha256_file(p)}  {p.name}"
             for p in sorted(out.iterdir()) if p.is_file() and p.name != "checksums.sha256"]
    (out / "checksums.sha256").write_text("\n".join(lines) + "\n")

    print(f"\nDONE in {elapsed:.1f}s -> {out}")
    print("Read report.md, then run step 2 with --estimate first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
