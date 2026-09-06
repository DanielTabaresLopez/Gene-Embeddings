#!/usr/bin/env python3
"""
pfi_03_summarize.py

Step 3 of 4 of the Pairwise Full-Intersection (PFI) analysis.

Does three things:

1. CONSOLIDATES the per-task JSON files into pair-level tables.

2. Runs the diagnostic this design makes mandatory: because n differs between
   pairs, any metric that depends on n will correlate with intersection size
   for purely arithmetic reasons. This script quantifies that for every
   metric and sorts each into a safe / caution / not-comparable tier for
   cross-pair reading.

3. Optionally compares against the atlas on the pairs both cover.
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ConstantInputWarning, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pfi_manifest import (annotate_pairs, dataframe_to_markdown, load_manifest,
                          panel_sha256, resolve_path, sha256_file)

PAIR_COLUMNS = [("embedding_x", "embedding_y"), ("embedding_a", "embedding_b"),
                ("embedding_1", "embedding_2"), ("source_embedding", "target_embedding")]

# PFI writes bandwidth labels as "1p0"; the atlas wrote "1". Same quantity,
# different spelling. Aliases apply ONLY to the atlas comparison; PFI's own
# outputs keep PFI naming.
METRIC_ALIASES = {
    "unbiased_rbf_cka_row_normalized_bw_1p0": "unbiased_rbf_cka_row_normalized_bw_1",
    "unbiased_rbf_cka_row_normalized_bw_0p5": "unbiased_rbf_cka_row_normalized_bw_0_5",
    "unbiased_rbf_cka_row_normalized_bw_2p0": "unbiased_rbf_cka_row_normalized_bw_2",
}

# Metrics whose sample size is fixed by a config cap rather than by the
# intersection. Used to label comparability honestly.
CAP_COLUMN_FOR_METRIC = [
    ("local_", "local_n_genes"),
    ("unbiased_rbf_", "rbf_n_genes"),
    ("unbiased_linear_", "n_intersection"),
]

warnings.filterwarnings("ignore", category=ConstantInputWarning)


def canonical(a, b):
    return "__VS__".join(sorted((str(a), str(b))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--atlas-pairs", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Allow partial/failed pair sets for debugging. Never use for paper results.",
    )
    args = parser.parse_args()

    started = time.time()
    config = yaml.safe_load(args.config.read_text())
    panel, manifest_path = load_manifest(config, args.config)
    inter = resolve_path(config["intersection_output"], args.config)
    out = resolve_path(config["analysis_output"], args.config)
    pair_dir = out / "pair_results"

    step1_metadata = json.loads((inter / "run_metadata.json").read_text())
    current_panel_hash = panel_sha256(panel)
    if step1_metadata.get("panel_sha256") != current_panel_hash:
        raise RuntimeError("the manifest differs from the one used for step 1")

    print("=" * 68)
    print("PFI STEP 3: CONSOLIDATION AND CONFOUND DIAGNOSTICS")
    print("=" * 68)

    # ---------------- consolidate ----------------
    records, failures = [], []
    files = sorted(pair_dir.glob("*.json"))
    print(f"\n[1] reading {len(files):,} pair files")
    observed_pair_ids, observed_fingerprints = set(), set()
    for path in files:
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"    WARNING unreadable: {path.name}"); continue
        if record.get("pair_id"):
            observed_pair_ids.add(str(record["pair_id"]))
        observed_fingerprints.add(record.get("analysis_fingerprint"))
        if record.get("status") != "PASS":
            failures.append(record); continue
        flat = {k: v for k, v in record.items() if k != "metrics"}
        flat.update(record["metrics"])
        flat["canonical_pair"] = canonical(record["embedding_x"], record["embedding_y"])
        records.append(flat)

    expected = pd.read_csv(inter / "pair_intersections.tsv", sep="\t")
    shortlist_path = config.get("pair_shortlist")
    if shortlist_path:
        shortlist_file = resolve_path(shortlist_path, args.config)
        if shortlist_file.is_file():
            shortlist = pd.read_csv(shortlist_file, sep="\t")
            wanted = {
                tuple(sorted((str(a), str(b))))
                for a, b in zip(shortlist.iloc[:, 0], shortlist.iloc[:, 1])
            }
            expected = expected[
                [tuple(sorted((a, b))) in wanted
                 for a, b in zip(expected.embedding_x, expected.embedding_y)]
            ]
    expected = expected[
        expected.n_intersection >= int(config.get("min_genes", 100))
    ]
    expected_pair_ids = set(expected.pair_id.astype(str))
    missing_pair_ids = sorted(expected_pair_ids - observed_pair_ids)
    extra_pair_ids = sorted(observed_pair_ids - expected_pair_ids)
    step2_metadata_path = out / "run_metadata.json"
    if not step2_metadata_path.is_file():
        raise RuntimeError("step 2 run_metadata.json is missing; the metric run did not finish")
    step2_metadata = json.loads(step2_metadata_path.read_text())
    expected_fingerprint = step2_metadata.get("analysis_fingerprint")
    if not expected_fingerprint or observed_fingerprints != {expected_fingerprint}:
        raise RuntimeError(
            "pair JSON provenance fingerprints are missing or mixed; use a clean "
            "analysis_output and rerun step 2"
        )
    if (missing_pair_ids or extra_pair_ids or failures) and not args.allow_incomplete:
        raise RuntimeError(
            "pair result set is not publication-complete: "
            f"missing={len(missing_pair_ids)}, failed={len(failures)}, "
            f"unexpected={len(extra_pair_ids)}. Rerun step 2, or use "
            "--allow-incomplete only for debugging."
        )

    if not records:
        raise RuntimeError(f"no successful pair results in {pair_dir}")

    results = annotate_pairs(pd.DataFrame(records), panel)
    print(f"    passed={len(results):,}  failed={len(failures):,}")
    if failures:
        pd.DataFrame(failures).to_csv(out / "failed_pairs.tsv", sep="\t", index=False)
        print("    WARNING: see failed_pairs.tsv")

    dims = pd.read_csv(inter / "embedding_valid_genes.tsv", sep="\t")
    dim_map = dict(zip(dims.embedding_id, dims.n_dimensions))
    results["dim_x"] = results.embedding_x.map(dim_map)
    results["dim_y"] = results.embedding_y.map(dim_map)
    results["log_geometric_mean_dimension"] = np.log(
        np.sqrt(results.dim_x.astype(float) * results.dim_y.astype(float)))

    results = results.sort_values("n_intersection", ascending=False)
    results.to_csv(out / "pair_results.tsv", sep="\t", index=False)
    print(f"    pair_results.tsv: {len(results):,} rows")

    metric_columns = [c for c in results.columns if c.startswith(
        ("unbiased_", "local_overlap", "local_effect", "local_z",
         "local_normalized", "local_null", "local_baseline"))]

    # ---------------- confound diagnostics ----------------
    print("\n[2] dependence of each metric on intersection size")
    print("    (the price of variable n; near zero = safe to compare across pairs)")
    rows = []
    for metric in metric_columns:
        values = pd.to_numeric(results[metric], errors="coerce")
        size = pd.to_numeric(results.n_intersection, errors="coerce")
        ok = values.notna() & size.notna()
        if ok.sum() < 10:
            continue
        rho_n, p_n = spearmanr(size[ok], values[ok])
        dimension = pd.to_numeric(results.log_geometric_mean_dimension, errors="coerce")
        ok_d = ok & dimension.notna()
        rho_d = spearmanr(dimension[ok_d], values[ok_d])[0] if ok_d.sum() >= 10 else np.nan
        rows.append({"metric": metric, "n_pairs": int(ok.sum()),
                     "median": round(float(values[ok].median()), 5),
                     "spearman_vs_intersection_size": round(float(rho_n), 4),
                     "p_value_vs_size": float(p_n),
                     "spearman_vs_log_dimension": round(float(rho_d), 4) if np.isfinite(rho_d) else None})

    confounds = pd.DataFrame(rows).sort_values(
        "spearman_vs_intersection_size", key=lambda s: s.abs(), ascending=False)
    confounds.to_csv(out / "confound_diagnostics.tsv", sep="\t", index=False)

    headline = [m for m in ("unbiased_linear_cka_row_normalized", "local_overlap_raw_k50",
                            "local_effect_k50", "local_z_k50",
                            "local_normalized_effect_k50")
                if m in set(confounds.metric)]
    for metric in headline:
        row = confounds[confounds.metric == metric].iloc[0]
        print(f"    {metric:<40} rho_vs_n={row.spearman_vs_intersection_size:>7.3f}")

    # ---------------- comparability tiers ----------------
    # Baselines and nulls are functions of n by construction; they are
    # diagnostics, not comparison metrics, so they are excluded from the tiers.
    diagnostic = ("local_baseline_analytic", "local_null_mean", "local_null_std")

    def sample_size_column(metric_name):
        for prefix, column in CAP_COLUMN_FOR_METRIC:
            if metric_name.startswith(prefix):
                return column
        return "n_intersection"

    tiers = []
    for _, row in confounds.iterrows():
        if row.metric.startswith(diagnostic):
            continue
        rho = abs(row.spearman_vs_intersection_size)
        column = sample_size_column(row.metric)
        n_values = pd.to_numeric(results.get(column), errors="coerce")
        matched = n_values is not None and n_values.nunique(dropna=True) == 1
        actual_n = int(n_values.iloc[0]) if matched else None

        if matched:
            # n was identical for every pair, so no part of this correlation
            # can be arithmetic. Whatever remains is substantive.
            tier = "matched-n (comparable)"
            interpretation = ("n fixed by cap; correlation with intersection "
                              "size is substantive, not arithmetic")
        else:
            tier = "safe" if rho < 0.15 else ("caution" if rho < 0.35 else "not comparable")
            interpretation = "n varies across pairs; correlation may be partly arithmetic"

        tiers.append({"metric": row.metric,
                      "sample_size_column": column,
                      "n_fixed_across_pairs": bool(matched),
                      "n_used": actual_n,
                      "abs_spearman_vs_size": round(rho, 4),
                      "cross_pair_comparability": tier,
                      "interpretation": interpretation})
    pd.DataFrame(tiers).to_csv(out / "comparability_tiers.tsv", sep="\t", index=False)

    n_matched = sum(1 for t in tiers if t["n_fixed_across_pairs"])
    if n_matched:
        print(f"    NOTE: {n_matched} metrics were computed at a FIXED n for every")
        print("    pair (config cap below the smallest intersection). For those,")
        print("    the correlation with intersection size cannot be arithmetic.")

    # ---------------- optional atlas comparison ----------------
    comparison = pd.DataFrame()
    note = "Atlas table not supplied; comparison skipped."
    atlas_path = args.atlas_pairs or config.get("atlas_pair_table")
    if atlas_path:
        atlas_file = resolve_path(str(atlas_path), args.config)
        if atlas_file.is_file():
            print("\n[3] comparison against the fixed-universe atlas")
            atlas = pd.read_csv(atlas_file, sep="\t")
            columns = next((c for c in PAIR_COLUMNS if set(c).issubset(atlas.columns)), None)
            if columns is None:
                note = f"No pair columns found in {atlas_file.name}."
            else:
                atlas["canonical_pair"] = [canonical(a, b)
                                           for a, b in zip(atlas[columns[0]], atlas[columns[1]])]
                merged = results.merge(atlas, on="canonical_pair", how="inner",
                                       suffixes=("_pfi", "_atlas"))
                print(f"    matched pairs: {len(merged):,}")
                rows = []
                for metric in metric_columns:
                    atlas_name = METRIC_ALIASES.get(metric, metric)
                    left = f"{metric}_pfi" if f"{metric}_pfi" in merged.columns else metric
                    if f"{atlas_name}_atlas" in merged.columns:
                        right = f"{atlas_name}_atlas"
                    elif atlas_name in atlas.columns and atlas_name in merged.columns:
                        right = atlas_name
                    else:
                        right = None
                    if right is None or left not in merged.columns or left == right:
                        continue
                    a = pd.to_numeric(merged[left], errors="coerce")
                    b = pd.to_numeric(merged[right], errors="coerce")
                    ok = a.notna() & b.notna()
                    if ok.sum() < 10:
                        continue
                    delta = a[ok] - b[ok]
                    rho_size = spearmanr(
                        pd.to_numeric(merged.n_intersection, errors="coerce")[ok], delta)[0]
                    rows.append({"metric": metric,
                                 "atlas_column": atlas_name,
                                 "n_pairs": int(ok.sum()),
                                 "spearman_pfi_vs_atlas": round(float(spearmanr(a[ok], b[ok])[0]), 4),
                                 "median_pfi": round(float(a[ok].median()), 5),
                                 "median_atlas": round(float(b[ok].median()), 5),
                                 "median_delta": round(float(delta.median()), 5),
                                 "spearman_delta_vs_size": round(float(rho_size), 4)})
                comparison = pd.DataFrame(rows)
                if not comparison.empty:
                    comparison.to_csv(out / "comparison_vs_atlas.tsv", sep="\t", index=False)
                    print(comparison.to_string(index=False))
                    note = ""
                    weak = comparison[comparison.spearman_pfi_vs_atlas < 0.90]
                    if not weak.empty:
                        note = ("Below rho 0.90 against the atlas: "
                                + ", ".join(weak.metric.tolist())
                                + ". Check the convention assumptions in pfi_metrics.py "
                                  "before interpreting these.")
        else:
            note = f"Atlas table not found at {atlas_file}."

    # ---------------- report ----------------
    elapsed = time.time() - started
    n_int = pd.to_numeric(results.n_intersection, errors="coerce")
    report = [
        "# PFI step 3 - summary", "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}    Runtime: {elapsed:.1f}s", "",
        "## Coverage", "",
        f"- Pairs analysed: {len(results):,}",
        f"- Pairs failed: {len(failures):,}",
        f"- Intersection size: min={int(n_int.min()):,} median={int(n_int.median()):,} "
        f"max={int(n_int.max()):,}", "",
        "## How to read these results", "",
        "Every pair was evaluated on its own full intersection, so `n` differs",
        "between rows. That is intended: it gives each pair its maximum-power,",
        "most biologically representative estimate. It also means that any",
        "metric whose value depends on `n` will correlate with intersection",
        "size for arithmetic rather than biological reasons.",
        "",
        "`comparability_tiers.tsv` sorts every metric into three tiers by the",
        "strength of that dependence:",
        "",
        "- **safe** (|rho| < 0.15) - compare freely across pairs.",
        "- **caution** (0.15 to 0.35) - compare only with intersection size shown.",
        "- **not comparable** (> 0.35) - read within a pair only.",
        "",
        "Expected pattern: unbiased CKA lands in `safe`, because the unbiased",
        "estimator has expectation zero under independence at any `n`. Raw local",
        "overlap drifts with `n`, because its chance baseline is about `k/n`.",
        "The null-corrected columns (`local_effect_*`, `local_z_*`,",
        "`local_normalized_effect_*`) reduce that drift but do not remove it,",
        "which is why the dependence is measured here rather than assumed.",
        "",
        "**This diagnostic is descriptive, not causal.** A metric can correlate",
        "with intersection size for two quite different reasons, and this table",
        "cannot separate them:",
        "",
        "1. Arithmetic - the statistic's own baseline moves with `n`.",
        "2. Biology and provenance - intersection size is not random. Pairs with",
        "   small intersections are those involving low-coverage embeddings (GO,",
        "   knowledge graphs, literature), which cover the well-studied core of",
        "   the genome and belong to particular modalities. Those pairs may",
        "   genuinely differ in alignment.",
        "",
        "Treat a strong correlation as a flag to investigate, not as proof of an",
        "artefact. Separating the two requires the matched-n comparison or the",
        "study-bias stratification in the roadmap.",
        "",
        "## Metric dependence on intersection size", "",
        dataframe_to_markdown(confounds), "",
        "## Comparison against the fixed-universe atlas", "",
    ]
    report += ([dataframe_to_markdown(comparison), ""] if not comparison.empty else ["_Not run._", ""])
    if note:
        report += [f"- {note}", ""]
    report += [
        "## Guardrails", "",
        "- Do not build a heatmap from any column in the `not comparable` tier.",
        "- Do not read a raw overlap difference between two pairs with different",
        "  intersection sizes as a biological difference.",
        "- Intersection size is not random: low-coverage embeddings cover the",
        "  well-studied core of the genome. A metric correlating with size may",
        "  be reporting annotation density, not geometry.",
        "- Unbiased CKA can be negative. That is the estimator working, not a bug.",
        "",
    ]
    report[4:4] = [
        "## Provenance", "",
        f"- Manifest: `{manifest_path.name}` ({len(panel)} embeddings)",
        f"- Expected manifest pairs above min_genes: {len(expected_pair_ids):,}",
        "",
    ]
    (out / "report.md").write_text("\n".join(report))

    (out / "summary_metadata.json").write_text(json.dumps({
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.config),
        "manifest_file": str(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "panel_sha256": current_panel_hash,
        "analysis_fingerprint": expected_fingerprint,
        "n_embeddings": int(len(panel)),
        "n_expected_pairs": int(len(expected_pair_ids)),
        "n_successful_pairs": int(len(results)),
        "n_failed_pairs": int(len(failures)),
        "n_missing_pairs": int(len(missing_pair_ids)),
        "allow_incomplete": bool(args.allow_incomplete),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(elapsed, 2),
        "status": "PASS" if not failures and not missing_pair_ids else "PASS_INCOMPLETE",
    }, indent=2))

    lines = [f"{sha256_file(p)}  {p.name}"
             for p in sorted(out.iterdir()) if p.is_file() and p.name != "checksums.sha256"]
    (out / "checksums.sha256").write_text("\n".join(lines) + "\n")

    print(f"\nDONE in {elapsed:.1f}s -> {out}")
    print("Read comparability_tiers.tsv before plotting anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
