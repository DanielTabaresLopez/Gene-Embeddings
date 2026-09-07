#!/usr/bin/env python3
"""
dp_02_build_pca_cache.py
 
Step 2 of the Directional Predictability track.
 
Reduces every embedding to a PCA basis fitted on TRAINING GENES ONLY, and
caches the projected coordinates for all of that embedding's valid genes.
 
Why PCA at all
--------------
1. Tractability. probe_blast and probe_hmmer are 20,421-dimensional. An MLP
   with a 20,421-unit output layer for 4,160 models is not sensible, and the
   information in those spaces is far lower-dimensional than the raw width.
2. Noise. Raw embedding dimensions include many low-variance directions that
   are close to unpredictable. Including them deflates R-squared without
   telling you anything about shared structure.
3. Comparability. Reducing every target to a comparable number of components
   makes R-squared comparable across pairs of very different width.
4. Interpretability. Components are ordered by variance, so per-component
   R-squared becomes a spectrum: how far into the target's structure the
   prediction reaches. That is the nonlinear analogue of a CCA spectrum, and
   it is much more informative than a single scalar.
 
The target is NOT whitened. Keeping the natural variance ordering means the
reported R-squared is the standard variance-weighted quantity, dominated by
directions that actually carry the signal. Whitening would up-weight the
noisiest retained directions.
 
Leakage control: the PCA mean and components come from training genes only.
Validation and test genes are projected with that fixed basis.
 
Outputs, per embedding, under <pca_output>/<embedding_id>/:
    coordinates.npy      float32 (n_valid_genes, n_components)
    genes.tsv            gene ids, row-aligned with coordinates.npy
    basis.npz            mean, components, explained_variance
    meta.json
"""
 
from __future__ import annotations
 
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
 
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
 
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.decomposition import PCA

from dp_manifest import id_aliases, load_manifest, panel_sha256
 
 
def valid_mask(gene_ids, matrix):
    finite = np.isfinite(matrix).all(axis=1)
    nonzero = np.linalg.norm(matrix, axis=1) > 0.0
    series = pd.Series(gene_ids)
    named = series.notna().to_numpy() & (series.fillna("").str.strip() != "").to_numpy()
    unique = ~series.duplicated(keep=False).to_numpy()
    return finite & nonzero & named & unique


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_pca_root(config: dict, split_column: str) -> Path:
    """Give every split its own cache; sharing a cache leaks held-out genes."""
    raw = os.path.expanduser(str(config["pca_output"]))
    if "{split_column}" in raw:
        raw = raw.format(split_column=split_column)
        return Path(raw)
    return Path(raw) / split_column


def source_fingerprint(*paths: Path) -> dict:
    """Cheap provenance for large source packages (hashing multi-GB NPZs is wasteful)."""
    return {
        str(path.name): {
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for path in paths
    }


def report_table(frame: pd.DataFrame) -> str:
    """Render a readable report even when optional ``tabulate`` is absent."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def existing_cache_dir(root: Path, embedding_id: str, aliases: dict[str, str]) -> Path:
    """Find a canonical cache or a cache written under a documented legacy ID."""
    canonical = root / embedding_id
    if canonical.is_dir():
        return canonical
    legacy = sorted(old for old, new in aliases.items() if new == embedding_id)
    candidates = [root / old for old in legacy if (root / old).is_dir()]
    if len(candidates) > 1:
        raise RuntimeError(f"multiple legacy PCA caches found for {embedding_id}: {candidates}")
    return candidates[0] if candidates else canonical


def validate_reused_cache(
    directory: Path,
    embedding_id: str,
    split_column: str,
    split_sha256: str,
) -> dict:
    files = [
        directory / "coordinates.npy", directory / "genes.tsv",
        directory / "basis.npz", directory / "meta.json",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{embedding_id}: existing PCA cache is required for an unselected "
            f"embedding but files are missing: {missing}"
        )
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("split_column") != split_column:
        raise RuntimeError(
            f"{embedding_id}: cached split {meta.get('split_column')!r} does not "
            f"match {split_column!r}"
        )
    if meta.get("split_table_sha256") != split_sha256:
        raise RuntimeError(
            f"{embedding_id}: cached PCA was fitted against a different gene split"
        )
    if not meta.get("cache_signature"):
        raise RuntimeError(f"{embedding_id}: cached PCA metadata lacks cache_signature")
    result = dict(meta)
    result["embedding_id"] = embedding_id
    if directory.name != embedding_id:
        result["legacy_cache_embedding_id"] = directory.name
    return result
 
 
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-column", default=None,
                        help="Override split column, e.g. split_random.")
    parser.add_argument(
        "--include-embedding", action="append", default=[],
        help=(
            "Build/revalidate only this embedding (repeatable), while retaining "
            "the other manifest-selected caches in pca_summary.tsv."
        ),
    )
    parser.add_argument(
        "--panel-extension", action="store_true",
        help="Use panel_extension_new_embeddings from the YAML.",
    )
    args = parser.parse_args()
 
    started = time.time()
    config = yaml.safe_load(args.config.read_text())
    panel, manifest_path = load_manifest(config, args.config)
    aliases = id_aliases(config)
 
    library = Path(os.path.expanduser(config["library_root"]))
    splits_dir = Path(os.path.expanduser(config["splits_output"]))
    split_column = args.split_column or config.get("split_column", "split_family")
    out = resolve_pca_root(config, split_column)
    out.mkdir(parents=True, exist_ok=True)
 
    array_key = config.get("array_key", "embeddings")
    id_column = config.get("id_column", "ensembl_gene_id")
    max_components = int(config["pca_max_components"])
    variance_target = float(config["pca_variance_target"])
    seed = int(config["seed"])
 
    print("=" * 68)
    print("DP STEP 2: PCA CACHE (fitted on training genes only)")
    print("=" * 68)
    print(f"split column   : {split_column}")
    print(f"max components : {max_components}")
    print(f"variance target: {variance_target:.0%}\n")
 
    split_path = splits_dir / "gene_splits.tsv"
    splits = pd.read_csv(split_path, sep="\t", dtype={"gene_id": str})
    if split_column not in splits.columns:
        raise ValueError(f"{split_path}: missing split column {split_column!r}")
    split_sha256 = sha256_file(split_path)
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    train_genes = set(splits.loc[splits[split_column] == "train", "gene_id"])
    print(f"training genes : {len(train_genes):,}\n")
 
    pkg_root = library / "data" / "embeddings"
    ids = panel.embedding_id.tolist()
    local_ids = {
        d.name for d in pkg_root.iterdir()
        if d.is_dir() and (d / "embeddings.npz").is_file()
    }
    missing_packages = sorted(set(ids) - local_ids)
    if missing_packages:
        raise FileNotFoundError(
            f"manifest embeddings absent from local library: {missing_packages}"
        )
    requested = set(args.include_embedding)
    if args.panel_extension:
        requested.update(config.get("panel_extension_new_embeddings") or [])
        if not requested:
            raise ValueError(
                "--panel-extension was requested but panel_extension_new_embeddings is empty"
            )
    unknown = sorted(requested - set(ids))
    if unknown:
        raise ValueError(f"unknown --include-embedding values: {unknown}")
    selected = requested or set(ids)
    print(f"manifest        : {manifest_path}")
    print(f"panel           : {len(ids)} embeddings")
    if requested:
        print(f"incremental set : {sorted(requested)}\n")

    rows = []
    n_reused = 0
    n_computed = 0
    for position, embedding_id in enumerate(ids, 1):
        target_dir = out / embedding_id
        cache_dir = existing_cache_dir(out, embedding_id, aliases)
        package = pkg_root / embedding_id
        source_paths = (package / "genes.tsv", package / "embeddings.npz")
        fingerprint = source_fingerprint(*source_paths)
        if embedding_id not in selected:
            existing = validate_reused_cache(
                cache_dir, embedding_id, split_column, split_sha256
            )
            rows.append(existing)
            n_reused += 1
            print(f"  [{position:>3}/{len(ids)}] {embedding_id:<50} retained")
            continue
        signature_payload = {
            "script_sha256": script_sha256,
            "config_sha256": config_sha256,
            "split_table_sha256": split_sha256,
            "split_column": split_column,
            "embedding_id": embedding_id,
            "source_fingerprint": fingerprint,
            "pca_max_components": max_components,
            "pca_variance_target": variance_target,
            "pca_min_components": int(config.get("pca_min_components", 8)),
            "seed": seed,
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        }
        cache_signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode()
        ).hexdigest()
        cache_files = [
            cache_dir / "coordinates.npy", cache_dir / "genes.tsv",
            cache_dir / "basis.npz", cache_dir / "meta.json",
        ]
        if all(path.is_file() for path in cache_files):
            existing = json.loads((cache_dir / "meta.json").read_text())
            if existing.get("cache_signature") == cache_signature:
                existing = dict(existing)
                existing["embedding_id"] = embedding_id
                rows.append(existing)
                n_reused += 1
                print(f"  [{position:>3}/{len(ids)}] {embedding_id:<50} cached")
                continue

        genes = pd.read_csv(package / "genes.tsv", sep="\t", dtype=str)
        with np.load(package / "embeddings.npz") as npz:
            matrix = np.asarray(npz[array_key], dtype=np.float32)
 
        gene_array = genes[id_column].to_numpy()
        mask = valid_mask(gene_array, matrix)
        valid_genes = gene_array[mask]
        valid_matrix = matrix[mask]
        del matrix
 
        is_train = np.array([g in train_genes for g in valid_genes])
        n_train = int(is_train.sum())
        if n_train < 100:
            raise RuntimeError(f"{embedding_id}: only {n_train} training genes")
 
        # Randomized SVD keeps very wide packages (d ~ 20,000) affordable.
        n_components = int(min(max_components, n_train - 1, valid_matrix.shape[1]))
        solver = "randomized" if valid_matrix.shape[1] > 1000 else "full"
        pca = PCA(n_components=n_components, svd_solver=solver, random_state=seed)
        pca.fit(valid_matrix[is_train])
 
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        keep = int(np.searchsorted(cumulative, variance_target) + 1)
        keep = int(min(max(keep, config.get("pca_min_components", 8)), n_components))
 
        coordinates = ((valid_matrix - pca.mean_) @ pca.components_[:keep].T).astype(np.float32)
 
        target_dir.mkdir(parents=True, exist_ok=True)
        np.save(target_dir / "coordinates.npy", coordinates)
        pd.DataFrame({"gene_id": valid_genes}).to_csv(
            target_dir / "genes.tsv", sep="\t", index=False)
        np.savez_compressed(
            target_dir / "basis.npz",
            mean=pca.mean_.astype(np.float32),
            components=pca.components_[:keep].astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_[:keep].astype(np.float32),
        )
        meta = {
            "embedding_id": embedding_id,
            "split_column": split_column,
            "n_valid_genes": int(len(valid_genes)),
            "n_train_genes": n_train,
            "original_dimension": int(valid_matrix.shape[1]),
            "n_components": keep,
            "variance_retained": float(cumulative[keep - 1]),
            "solver": solver,
            "cache_signature": cache_signature,
            "script_sha256": script_sha256,
            "config_sha256": config_sha256,
            "split_table_sha256": split_sha256,
            "source_fingerprint": fingerprint,
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
        }
        (target_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        rows.append(meta)
        n_computed += 1
 
        print(f"  [{position:>3}/{len(ids)}] {embedding_id:<50} "
              f"d={valid_matrix.shape[1]:>5} -> {keep:>4} comps "
              f"({cumulative[keep - 1]:.1%} var)")
        del valid_matrix
 
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "pca_summary.tsv", sep="\t", index=False)
    pca_run_signature = hashlib.sha256(json.dumps(
        sorted(str(row["cache_signature"]) for row in rows)
    ).encode()).hexdigest()
 
    elapsed = time.time() - started
    report = [
        "# DP step 2 — PCA cache", "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}    Runtime: {elapsed:.1f}s", "",
        f"- Embeddings: {len(summary)}",
        f"- Split column: `{split_column}`",
        f"- Variance target: {variance_target:.0%}, capped at {max_components} components",
        f"- Components retained: min {summary.n_components.min()}, "
        f"median {int(summary.n_components.median())}, max {summary.n_components.max()}",
        f"- Variance retained: min {summary.variance_retained.min():.1%}, "
        f"median {summary.variance_retained.median():.1%}", "",
        "## Leakage control", "",
        "The PCA mean and components are fitted on training genes only.",
        "Validation and test genes are projected with that fixed basis and",
        "never influence it.", "",
        "## Per embedding", "",
        report_table(summary[[
            "embedding_id", "original_dimension", "n_components",
            "variance_retained", "n_valid_genes",
        ]]), "",
    ]
    (out / "report.md").write_text("\n".join(report))
 
    (out / "run_metadata.json").write_text(json.dumps({
        "script": Path(__file__).name,
        "script_sha256": script_sha256,
        "config_sha256": config_sha256,
        "split_table_sha256": split_sha256,
        "pca_run_signature": pca_run_signature,
        "manifest_file": str(manifest_path),
        "panel_sha256": panel_sha256(panel),
        "embedding_ids": ids,
        "incremental_requested_embeddings": sorted(requested),
        "n_reused_caches": n_reused,
        "n_computed_caches": n_computed,
        "config": config, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(elapsed, 2), "python_version": sys.version,
        "platform": platform.platform(), "split_column": split_column,
        "sklearn_version": sklearn.__version__,
        "n_embeddings": len(summary), "status": "PASS",
    }, indent=2))
 
    print(f"\nDONE in {elapsed:.1f}s -> {out}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 
