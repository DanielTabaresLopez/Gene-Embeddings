#!/usr/bin/env python3
"""
pfi_02_run_pair_metrics.py

Step 2 of 4 of the Pairwise Full-Intersection (PFI) analysis.

Each pair is evaluated on its ENTIRE shared valid gene set (local overlap
capped at local_max_n; RBF capped at rbf_max_n; linear CKA left uncapped,
because using the full intersection for the symmetric global metric is the
point of this analysis).

COST WARNING, based on real panel data: linear CKA cost scales as n*d*e and
local overlap cost scales as n^2*d. A handful of very high-dimensional
embeddings (for example PROBE-BLAST and PROBE-HMMER style profile embeddings
at d ~ 20,000) can make individual pairs far more expensive than pairs of
similar n but modest d. --estimate calibrates against your REAL panel,
including its dimension extremes, rather than assuming benchmark values from
a different collection.

Resumable: one JSON per pair. Existing files are skipped.

    python .../pfi_02_run_pair_metrics.py --config ... --estimate
    python .../pfi_02_run_pair_metrics.py --config ... --limit 5
    bash scripts/launch_nohup.sh .../pfi_02_run_pair_metrics.py pfi_v1 --config ...
"""

from __future__ import annotations

import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path

# Threads: default 1 for server etiquette. Linear CKA's matmul-dominated cost
# parallelizes well through BLAS; raise with PFI_THREADS=4 for a large run on
# an idle machine. Must happen before numpy is imported.
_threads = os.environ.get("PFI_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _threads)

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfi_metrics import (analytic_local_baseline, deterministic_subsample,
                         l2_row_normalize, local_overlap, local_permutation_null,
                         preprocess, unbiased_linear_cka, unbiased_rbf_cka)
from pfi_manifest import (load_manifest, manifest_maps, panel_sha256,
                          resolve_path, sha256_file)


class Cache:
    """Holds a few embedding matrices resident; pairs are ordered to exploit it."""

    def __init__(self, library, array_key, id_column, capacity=3):
        self.library, self.array_key, self.id_column = library, array_key, id_column
        self.capacity, self._store, self._order = capacity, {}, []

    def get(self, eid):
        if eid in self._store:
            self._order.remove(eid); self._order.append(eid)
            return self._store[eid]
        pkg = self.library / "data" / "embeddings" / eid
        genes = pd.read_csv(pkg / "genes.tsv", sep="\t", dtype=str)
        with np.load(pkg / "embeddings.npz") as npz:
            matrix = np.asarray(npz[self.array_key], dtype=np.float32)
        lookup = {g: i for i, g in enumerate(genes[self.id_column].to_numpy())}
        self._store[eid] = (lookup, matrix); self._order.append(eid)
        while len(self._order) > self.capacity:
            del self._store[self._order.pop(0)]
        return self._store[eid]


def load_pair_arrays(cache: Cache, validity, gene_ids, row):
    i, j = int(row.index_x), int(row.index_y)
    shared = gene_ids[validity[:, i] & validity[:, j]]
    lx, mx = cache.get(row.embedding_x)
    ly, my = cache.get(row.embedding_y)
    rows_x = np.fromiter((lx[g] for g in shared), dtype=np.int64, count=shared.size)
    rows_y = np.fromiter((ly[g] for g in shared), dtype=np.int64, count=shared.size)
    return mx[rows_x], my[rows_y], shared


def run_pair(X_raw, Y_raw, genes, config, seed):
    """All metrics for one pair on its full intersection."""
    n_full = X_raw.shape[0]
    result = {"n_genes_full": n_full}

    # ---- global: unbiased linear CKA on the FULL intersection ----------
    for mode in config["preprocessing_modes"]:
        result[f"unbiased_linear_cka_{mode}"] = unbiased_linear_cka(
            preprocess(X_raw, mode), preprocess(Y_raw, mode))

    # ---- global: unbiased RBF CKA, capped -------------------------------
    rbf_cap = config.get("rbf_max_n")
    rbf_n = min(n_full, rbf_cap) if rbf_cap else n_full
    result["rbf_n_genes"] = rbf_n
    result["rbf_subsampled"] = bool(rbf_n < n_full)
    if rbf_n >= config.get("min_genes", 100):
        if rbf_n < n_full:
            keep = np.isin(genes, deterministic_subsample(genes, rbf_n, seed))
            Xr, Yr = X_raw[keep], Y_raw[keep]
        else:
            Xr, Yr = X_raw, Y_raw
        for factor in config["rbf_bandwidth_factors"]:
            label = str(factor).replace(".", "p")
            result[f"unbiased_rbf_cka_row_normalized_bw_{label}"] = unbiased_rbf_cka(
                preprocess(Xr, "row_normalized"), preprocess(Yr, "row_normalized"),
                bandwidth_factor=float(factor),
                block_size=int(config["kernel_block_size"]), seed=seed)
        del Xr, Yr

    # ---- local: tie-aware overlap, capped -------------------------------
    local_cap = config.get("local_max_n")
    local_n = min(n_full, local_cap) if local_cap else n_full
    result["local_n_genes"] = local_n
    result["local_subsampled"] = bool(local_n < n_full)
    if local_n < n_full:
        keep = np.isin(genes, deterministic_subsample(genes, local_n, seed))
        Xl, Yl = X_raw[keep], Y_raw[keep]
    else:
        Xl, Yl = X_raw, Y_raw

    Xn = np.ascontiguousarray(l2_row_normalize(Xl.astype(np.float64)), dtype=np.float32)
    Yn = np.ascontiguousarray(l2_row_normalize(Yl.astype(np.float64)), dtype=np.float32)

    ks = [k for k in config["local_k_values"] if k < local_n]
    n_perm = int(config["local_permutations"])
    local = local_overlap(Xn, Yn, ks, block_size=int(config["local_block_size"]),
                          mode=config.get("tie_combination_mode", "min"),
                          collect_topk=n_perm > 0)

    effects, normalized = [], []
    for k in ks:
        entry = local[k]
        observed = entry["overlap"]
        result[f"local_overlap_raw_k{k}"] = observed
        result[f"local_baseline_analytic_k{k}"] = analytic_local_baseline(local_n, k)
        result[f"local_boundary_tie_rate_x_k{k}"] = entry["boundary_tie_rate_x"]
        result[f"local_boundary_tie_rate_y_k{k}"] = entry["boundary_tie_rate_y"]

        if n_perm > 0:
            null_mean, null_std = local_permutation_null(
                entry["topk_x"], entry["topk_y"], n_perm, seed + k)
            effect = observed - null_mean
            result[f"local_null_mean_k{k}"] = null_mean
            result[f"local_null_std_k{k}"] = null_std
            result[f"local_effect_k{k}"] = effect
            result[f"local_z_k{k}"] = (effect / null_std) if null_std > 0 else float("nan")
            norm = effect / (1.0 - null_mean) if null_mean < 1.0 else float("nan")
            result[f"local_normalized_effect_k{k}"] = norm
            normalized.append(norm)
        else:
            effect = observed - analytic_local_baseline(local_n, k)
            result[f"local_effect_k{k}"] = effect
        effects.append(effect)

    if effects:
        result["local_effect_mean"] = float(np.nanmean(effects))
    if normalized:
        result["local_normalized_effect_mean"] = float(np.nanmean(normalized))

    return result


# ======================================================================
# Dimension-aware cost estimation
# ======================================================================


def cost_proxies(pairs: pd.DataFrame, dims: pd.Series, config: dict) -> pd.DataFrame:
    """
    Two engineering proxies, not exact FLOP counts, used purely to rank pairs
    and calibrate a linear cost model against real timings:

      linear_proxy  ~ n * (dx*dy + dx^2 + dy^2)   [linear CKA, uncapped in n]
      quad_proxy    ~ n_eff^2 * mean(dx, dy)      [RBF + local, capped in n]

    where n_eff uses the larger of the two caps, since that upper-bounds
    both capped terms together.
    """
    out = pairs.copy()
    out["dim_x"] = out.embedding_x.map(dims)
    out["dim_y"] = out.embedding_y.map(dims)

    rbf_cap = config.get("rbf_max_n") or float("inf")
    local_cap = config.get("local_max_n") or float("inf")
    finite_caps = [c for c in (rbf_cap, local_cap) if np.isfinite(c)]
    cap = max(finite_caps) if finite_caps else None

    n_eff = out.n_intersection if cap is None else np.minimum(out.n_intersection, cap)

    out["linear_proxy"] = out.n_intersection * (
        out.dim_x * out.dim_y + out.dim_x**2 + out.dim_y**2
    )
    out["quad_proxy"] = (n_eff.astype(float) ** 2) * ((out.dim_x + out.dim_y) / 2.0)
    return out


def pick_calibration_pairs(scored: pd.DataFrame, n_points: int = 10) -> pd.DataFrame:
    """
    Span both cost regimes: percentiles of the uncapped linear-CKA proxy
    (which is where dimension outliers bite) plus a few points from the
    capped quadratic proxy, so both terms in the cost model get real data.
    """
    quantiles = [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]
    by_linear = scored.sort_values("linear_proxy")
    idx = (np.array(quantiles) * (len(by_linear) - 1)).round().astype(int)
    chosen = by_linear.iloc[idx]

    by_quad = scored.sort_values("quad_proxy")
    extra_idx = (np.array([0.5, 0.9, 1.0]) * (len(by_quad) - 1)).round().astype(int)
    chosen = pd.concat([chosen, by_quad.iloc[extra_idx]]).drop_duplicates("pair_id")

    return chosen.head(n_points + 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--estimate", action="store_true",
        help="Time real representative pairs spanning size AND dimension, "
             "fit a cost model, and project every pair. Slower than a naive "
             "estimate but honest about your actual panel.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    started = time.time()
    config = yaml.safe_load(args.config.read_text())

    panel, manifest_path = load_manifest(config, args.config)
    display_map, modality_map = manifest_maps(panel)
    library = resolve_path(config["library_root"], args.config)
    inter = resolve_path(config["intersection_output"], args.config)
    out = resolve_path(config["analysis_output"], args.config)
    pair_dir = out / "pair_results"
    pair_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("PFI STEP 2: FULL-INTERSECTION PAIR METRICS")
    print("=" * 68)
    print(f"threads : {os.environ.get('OMP_NUM_THREADS')}"
          f"{'  (set PFI_THREADS=4 before running for a large speedup)' if os.environ.get('OMP_NUM_THREADS') == '1' else ''}")

    with np.load(inter / "validity_matrix.npz", allow_pickle=True) as npz:
        validity = npz["validity"]
        embedding_ids = [str(x) for x in npz["embedding_ids"]]
        gene_ids = np.array([str(x) for x in npz["gene_ids"]])
    expected_ids = panel.embedding_id.tolist()
    if embedding_ids != expected_ids:
        raise RuntimeError(
            "the intersection files were built from a different manifest/order; "
            "rerun pfi_01_build_intersections.py into a new output directory"
        )
    step1_metadata = json.loads((inter / "run_metadata.json").read_text())
    current_panel_hash = panel_sha256(panel)
    if step1_metadata.get("panel_sha256") != current_panel_hash:
        raise RuntimeError(
            "the manifest content changed after step 1; rebuild intersections "
            "before computing metrics"
        )

    fingerprint_payload = {
        "config_sha256": sha256_file(args.config),
        "manifest_file_sha256": sha256_file(manifest_path),
        "panel_sha256": current_panel_hash,
        "step1_script_sha256": step1_metadata.get("script_sha256"),
        "metrics_module_sha256": sha256_file(Path(__file__).parent / "pfi_metrics.py"),
    }
    analysis_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    pairs = pd.read_csv(inter / "pair_intersections.tsv", sep="\t")
    dims = pd.read_csv(inter / "embedding_valid_genes.tsv", sep="\t").set_index(
        "embedding_id")["n_dimensions"]

    shortlist_path = config.get("pair_shortlist")
    if shortlist_path:
        f = resolve_path(shortlist_path, args.config)
        if f.is_file():
            table = pd.read_csv(f, sep="\t")
            wanted = {tuple(sorted((str(a), str(b))))
                      for a, b in zip(table.iloc[:, 0], table.iloc[:, 1])}
            keep = [tuple(sorted((a, b))) in wanted
                    for a, b in zip(pairs.embedding_x, pairs.embedding_y)]
            pairs = pairs[keep].reset_index(drop=True)
            print(f"shortlist: restricted to {len(pairs):,} pairs")
        else:
            print(f"  WARNING: shortlist not found at {f}; using all pairs")

    min_genes = int(config.get("min_genes", 100))
    skipped = int((pairs.n_intersection < min_genes).sum())
    pairs = pairs[pairs.n_intersection >= min_genes].reset_index(drop=True)
    if skipped:
        print(f"  skipped {skipped} pairs below min_genes={min_genes}")

    done, incompatible = set(), []
    for path in pair_dir.glob("*.json"):
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if existing.get("status") != "PASS":
            continue
        if existing.get("analysis_fingerprint") != analysis_fingerprint:
            incompatible.append(path.name)
        else:
            done.add(path.stem)
    if incompatible:
        preview = ", ".join(sorted(incompatible)[:5])
        raise RuntimeError(
            f"{len(incompatible)} completed pair files were produced by a different "
            f"configuration/manifest (for example: {preview}). Use a new "
            "analysis_output directory rather than mixing runs."
        )
    print(f"pairs   : {len(pairs):,}   already complete: {len(done):,}")

    cache = Cache(library, config.get("array_key", "embeddings"),
                  config.get("id_column", "ensembl_gene_id"),
                  int(config.get("cache_capacity", 3)))
    base_seed = int(config["seed"])

    # ---------------- estimate mode ----------------
    if args.estimate:
        try:
            from scipy.optimize import nnls
        except ImportError:
            print("scipy is required for --estimate (pip install scipy).")
            return 1

        scored = cost_proxies(pairs, dims, config)
        calibration = pick_calibration_pairs(scored)

        print(f"\nTiming {len(calibration)} real pairs spanning your panel's "
              f"size AND dimension range.")
        print("This runs the exact metric code, including your most "
              "extreme-dimension embeddings, so it may take a while for the "
              "largest ones. That cost is paid once.\n")
        print(f"{'n':>8} {'dim_x':>7} {'dim_y':>7} {'seconds':>9}  pair")

        timed = []
        for _, row in calibration.iterrows():
            t0 = time.time()
            X_raw, Y_raw, shared = load_pair_arrays(cache, validity, gene_ids, row)
            seed = base_seed + 1_000_003 * int(row.index_x) + 10_007 * int(row.index_y)
            run_pair(X_raw, Y_raw, shared, config, seed)
            dt = time.time() - t0
            timed.append({"linear_proxy": row.linear_proxy, "quad_proxy": row.quad_proxy,
                          "seconds": dt})
            print(f"{row.n_intersection:>8,} {row.dim_x:>7} {row.dim_y:>7} "
                  f"{dt:>9.1f}  {row.pair_id[:70]}", flush=True)

        calib = pd.DataFrame(timed)
        A = calib[["linear_proxy", "quad_proxy"]].to_numpy()
        A = A / A.max(axis=0)  # normalize columns so nnls conditions well
        scale = calib[["linear_proxy", "quad_proxy"]].max(axis=0).to_numpy()
        b = calib["seconds"].to_numpy()
        coefficients, residual = nnls(A, b)
        coefficients = coefficients / scale  # undo normalization

        predicted_calib = A @ (coefficients * scale)
        ss_res = float(((b - predicted_calib) ** 2).sum())
        ss_tot = float(((b - b.mean()) ** 2).sum()) if b.std() > 0 else 1.0
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        print(f"\nfit: seconds ~ {coefficients[0]:.3e}*linear_proxy + "
              f"{coefficients[1]:.3e}*quad_proxy   (R^2={r_squared:.3f} on "
              f"{len(calib)} calibration points)")
        if r_squared < 0.7:
            print("  NOTE: fit is weak. Treat the projection below as a rough")
            print("  order of magnitude, not a precise estimate.")

        scored["predicted_seconds"] = (
            coefficients[0] * scored.linear_proxy + coefficients[1] * scored.quad_proxy
        )
        total_seconds = float(scored.predicted_seconds.sum())

        worst = scored.sort_values("predicted_seconds", ascending=False).head(20)
        print(f"\nworst 20 pairs by projected cost:")
        print(f"{'n':>8} {'dim_x':>7} {'dim_y':>7} {'proj_sec':>10}  pair")
        for _, row in worst.iterrows():
            print(f"{row.n_intersection:>8,} {row.dim_x:>7} {row.dim_y:>7} "
                  f"{row.predicted_seconds:>10.1f}  {row.pair_id[:70]}")

        projection_path = out / "cost_projection.tsv"
        out.mkdir(parents=True, exist_ok=True)
        scored.sort_values("predicted_seconds", ascending=False).to_csv(
            projection_path, sep="\t", index=False)

        current_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        print(f"\nprojected total: {total_seconds / 3600:.1f} h at "
              f"{current_threads} thread(s)")
        for t in (2, 4, 8):
            if t != current_threads:
                print(f"                 {total_seconds / 3600 / t:.1f} h "
                      f"if run with PFI_THREADS={t}")
        print(f"\nFull per-pair projection saved to: {projection_path}")
        print("\nIf the total is too long:")
        print("  - PFI_THREADS=4 (or 8) before the real run: linear CKA's")
        print("    matmul-dominated cost parallelizes well through BLAS.")
        print("  - Set pair_shortlist to the 174-pair geometry shortlist for")
        print("    a first pass.")
        print("  - The worst offenders above are usually driven by a handful")
        print("    of very high-dimensional embeddings. Consider whether any")
        print("    should be handled separately (documented, not silent).")
        return 0

    # ---------------- full run ----------------
    order = pairs.sort_values(["index_x", "index_y"])
    n_done = n_fail = 0
    last = time.time()

    for position, row in enumerate(order.itertuples(), 1):
        stem = row.pair_id
        if len(stem) > 200:
            stem = hashlib.sha256(stem.encode()).hexdigest()[:40]
        target = pair_dir / f"{stem}.json"
        if stem in done or target.is_file():
            continue
        try:
            t0 = time.time()
            X_raw, Y_raw, shared = load_pair_arrays(cache, validity, gene_ids, row)
            seed = base_seed + 1_000_003 * int(row.index_x) + 10_007 * int(row.index_y)
            metrics = run_pair(X_raw, Y_raw, shared, config, seed)
            target.write_text(json.dumps({
                "pair_id": row.pair_id, "embedding_x": row.embedding_x,
                "embedding_y": row.embedding_y,
                "embedding_x_display": display_map[row.embedding_x],
                "embedding_y_display": display_map[row.embedding_y],
                "modality_x": modality_map[row.embedding_x],
                "modality_y": modality_map[row.embedding_y],
                "n_intersection": int(shared.size),
                "seed": seed, "runtime_seconds": round(time.time() - t0, 2),
                "analysis_fingerprint": analysis_fingerprint,
                "status": "PASS", "metrics": metrics,
            }, indent=2))
            n_done += 1
        except Exception as error:  # noqa: BLE001
            n_fail += 1
            target.write_text(json.dumps({
                "pair_id": row.pair_id, "embedding_x": row.embedding_x,
                "embedding_y": row.embedding_y,
                "embedding_x_display": display_map[row.embedding_x],
                "embedding_y_display": display_map[row.embedding_y],
                "modality_x": modality_map[row.embedding_x],
                "modality_y": modality_map[row.embedding_y],
                "analysis_fingerprint": analysis_fingerprint,
                "status": "FAIL",
                "error_type": type(error).__name__, "error": str(error)}, indent=2))

        if time.time() - last > 120:
            elapsed = time.time() - started
            rate = n_done / elapsed if elapsed else 0
            eta = (len(order) - position) / rate / 3600 if rate else float("inf")
            print(f"  [{position:>5,}/{len(order):,}] done={n_done:,} fail={n_fail:,} "
                  f"eta={eta:.1f}h", flush=True)
            last = time.time()

        if args.limit and n_done >= args.limit:
            print(f"\n--limit {args.limit} reached.")
            break

    elapsed = time.time() - started
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_metadata.json").write_text(json.dumps({
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "metrics_module_sha256": sha256_file(
            Path(__file__).parent / "pfi_metrics.py"),
        "config_sha256": sha256_file(args.config),
        "manifest_file": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "panel_sha256": current_panel_hash,
        "analysis_fingerprint": analysis_fingerprint,
        "fingerprint_components": fingerprint_payload,
        "config": config, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(elapsed, 2), "python_version": sys.version,
        "platform": platform.platform(), "numpy_version": np.__version__,
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "n_pairs_targeted": int(len(order)),
        "n_pairs_computed_this_run": n_done, "n_pairs_failed_this_run": n_fail,
        "status": "PASS" if n_fail == 0 else "PASS_WITH_FAILURES",
    }, indent=2))

    print(f"\nDONE in {elapsed / 3600:.2f}h  computed={n_done:,}  failed={n_fail:,}")
    print(f"Results: {pair_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
