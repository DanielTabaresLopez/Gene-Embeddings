#!/usr/bin/env python3
"""Leakage-safe summary of the Directional Predictability depth sweep.

Depth is selected with validation R-squared. Test R-squared is read exactly
once for the selected model. The script rejects incomplete results and permits
the documented legacy-plus-extension signature mixture only when enabled in
the manifest-frozen configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from dp_manifest import (
    add_direction_labels,
    canonicalize_id,
    id_aliases,
    load_manifest,
    panel_sha256,
)


KEY = ["pair_id", "source", "target", "split_column", "input_mode"]


def canonical_pair(left: str, right: str) -> str:
    return "__VS__".join(sorted((left, right)))


def load_directory(
    directory: Path,
    expected_pair_ids: set[str],
    aliases: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    rows, pair_ids, signatures, failed = [], set(), set(), []
    extra_pair_ids: set[str] = set()
    duplicate_pair_ids: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            failed.append(f"{path.name}: unreadable ({error})")
            continue
        embedding_x = canonicalize_id(record.get("embedding_x", ""), aliases)
        embedding_y = canonicalize_id(record.get("embedding_y", ""), aliases)
        pair_id = canonical_pair(embedding_x, embedding_y)
        if pair_id not in expected_pair_ids:
            extra_pair_ids.add(pair_id)
            continue
        if record.get("status") != "PASS":
            failed.append(f"{path.name}: status={record.get('status')}")
            continue
        if pair_id in pair_ids:
            duplicate_pair_ids.add(pair_id)
            continue
        pair_ids.add(pair_id)
        signatures.add(record.get("run_signature"))
        for direction, source, target, variance_key in (
            ("x_to_y", embedding_x, embedding_y, "variance_retained_y"),
            ("y_to_x", embedding_y, embedding_x, "variance_retained_x"),
        ):
            block = record.get(direction, {})
            base = {
                "pair_id": pair_id,
                "source": source,
                "target": target,
                "split_column": record["split_column"],
                "input_mode": record["input_mode"],
                "run_signature": record.get("run_signature"),
                "hidden_width": record.get("hidden_width"),
                "n_test": record.get("n_test"),
                "target_variance_retained": record.get(variance_key),
            }
            for model, metrics in block.items():
                if not isinstance(metrics, dict) or "r2" not in metrics:
                    continue
                rows.append({
                    **base,
                    "model": model,
                    # Primary metric: exact performance against the full raw
                    # target for a predictor that sets omitted PCs to their
                    # training mean. Retained-subspace R2 remains available as
                    # a diagnostic and for cap-sensitivity comparisons.
                    "r2": metrics.get("r2_full_target_at_k", metrics.get("r2")),
                    "val_r2": metrics.get(
                        "val_r2_full_target_at_k", metrics.get("val_r2")
                    ),
                    "r2_retained_subspace": metrics.get(
                        "r2_retained_subspace", metrics.get("r2")
                    ),
                    "val_r2_retained_subspace": metrics.get(
                        "val_r2_retained_subspace", metrics.get("val_r2")
                    ),
                    "target_variance_fraction_test": metrics.get(
                        "target_variance_fraction_test"
                    ),
                    "fit_status": metrics.get("fit_status", "PASS"),
                    "retrieval_top1": metrics.get("retrieval_top1"),
                    "n_parameters": metrics.get("n_parameters"),
                    "epochs_run": metrics.get("epochs_run"),
                })
    audit = {
        "directory": directory.name,
        "n_pairs": len(pair_ids),
        "pair_ids": pair_ids,
        "signatures": signatures,
        "failed": failed,
        "extra_pair_ids": extra_pair_ids,
        "duplicate_pair_ids": duplicate_pair_ids,
    }
    return pd.DataFrame(rows), audit


def select_by_validation(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One row per direction; tie-break toward the shallower model."""
    block = frame[
        frame.model.isin(models)
        & frame.fit_status.eq("PASS")
        & np.isfinite(frame.val_r2)
        & np.isfinite(frame.r2)
    ].copy()
    if block.empty:
        return block
    order = {model: position for position, model in enumerate(models)}
    block["model_order"] = block.model.map(order)
    block = block.sort_values(
        KEY + ["val_r2", "model_order"],
        ascending=[True] * len(KEY) + [False, True],
    )
    selected = block.drop_duplicates(KEY, keep="first").copy()
    return selected.rename(columns={
        "model": "selected_model",
        "val_r2": "selected_val_r2",
        "r2": "selected_test_r2",
        "retrieval_top1": "selected_retrieval_top1",
    })


def paired_summary(values: pd.Series) -> dict:
    values = values[np.isfinite(values)].astype(float)
    if values.empty:
        return {"n": 0, "median": np.nan, "positive": np.nan, "p_value": np.nan}
    nonzero = values[values != 0]
    p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
    return {
        "n": int(len(values)),
        "median": float(values.median()),
        "positive": float((values > 0).mean()),
        "p_value": p_value,
    }


def validate_audits(
    audits: list[dict],
    primary_mode: str,
    expected_pair_ids: set[str],
    allow_incomplete: bool,
    allow_mixed_signatures: bool,
    max_run_signatures: int,
) -> None:
    relevant = {
        audit["directory"]: audit
        for audit in audits
        if audit["directory"] in {
            f"split_family__{primary_mode}", f"split_random__{primary_mode}"
        }
    }
    problems = []
    for name, audit in relevant.items():
        if audit["failed"]:
            problems.append(f"{name}: {len(audit['failed'])} unreadable/failed files")
        if None in audit["signatures"] or not audit["signatures"]:
            problems.append(f"{name}: missing run signatures")
        elif len(audit["signatures"]) != 1 and not allow_mixed_signatures:
            problems.append(f"{name}: mixed or missing run signatures")
        elif len(audit["signatures"]) > max_run_signatures:
            problems.append(
                f"{name}: {len(audit['signatures'])} run signatures exceed the "
                f"documented maximum of {max_run_signatures}"
            )
        if audit["duplicate_pair_ids"]:
            problems.append(
                f"{name}: {len(audit['duplicate_pair_ids'])} duplicate canonical pairs"
            )
        missing = expected_pair_ids - audit["pair_ids"]
        if missing:
            problems.append(f"{name}: {len(missing)} expected pairs missing")
        if audit["extra_pair_ids"]:
            print(
                f"  {name}: ignoring {len(audit['extra_pair_ids'])} out-of-panel "
                "legacy pairs"
            )
        if len(audit["signatures"]) > 1 and allow_mixed_signatures:
            print(
                f"  {name}: accepted {len(audit['signatures'])} run signatures "
                "for the documented panel extension"
            )
    if len(relevant) == 2:
        family = relevant[f"split_family__{primary_mode}"]["pair_ids"]
        random = relevant[f"split_random__{primary_mode}"]["pair_ids"]
        if family != random:
            problems.append(
                f"family/random pair sets differ ({len(family)} versus {len(random)})"
            )
    elif not allow_incomplete:
        problems.append(
            f"both split_family__{primary_mode} and split_random__{primary_mode} are required"
        )
    if problems and not allow_incomplete:
        raise RuntimeError("result validation failed:\n - " + "\n - ".join(problems))
    for problem in problems:
        print(f"  PILOT WARNING: {problem}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--primary-input-mode", choices=["pca", "raw"], default=None)
    parser.add_argument(
        "--primary-split",
        choices=["split_family", "split_random"],
        default="split_family",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    panel, manifest_path = load_manifest(config, args.config)
    aliases = id_aliases(config)
    primary_mode = args.primary_input_mode or config.get("input_mode", "pca")
    out = Path(os.path.expanduser(config["train_output"]))
    root = out / "sweep_results"
    pca_template = os.path.expanduser(str(config["pca_output"]))
    if "{split_column}" in pca_template:
        pca_family = Path(pca_template.format(split_column="split_family"))
    else:
        pca_family = Path(pca_template) / "split_family"
    pca_summary = pd.read_csv(pca_family / "pca_summary.tsv", sep="\t")
    pca_summary["embedding_id"] = pca_summary.embedding_id.astype(str).map(
        lambda value: canonicalize_id(value, aliases)
    )
    embedding_ids = sorted(panel.embedding_id.tolist())
    missing_pca = sorted(set(embedding_ids) - set(pca_summary.embedding_id))
    if missing_pca:
        raise RuntimeError(f"family PCA summary is missing manifest IDs: {missing_pca}")
    expected_pair_ids = {
        f"{left}__VS__{right}"
        for position, left in enumerate(embedding_ids)
        for right in embedding_ids[position + 1:]
    }

    frames, audits = [], []
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if not directory.is_dir():
            continue
        frame, audit = load_directory(directory, expected_pair_ids, aliases)
        audits.append(audit)
        if not frame.empty:
            frames.append(frame)
            print(f"  {directory.name:<34} {audit['n_pairs']:,} pairs; {len(frame):,} rows")
    if not frames:
        raise RuntimeError(f"no sweep results under {root}")
    validate_audits(
        audits,
        primary_mode,
        expected_pair_ids,
        args.allow_incomplete,
        bool(config.get("allow_panel_extension_mixed_signatures", False)),
        int(config.get("max_panel_extension_run_signatures", 1)),
    )

    long = add_direction_labels(pd.concat(frames, ignore_index=True), panel)
    long.to_csv(out / "sweep_long.tsv", sep="\t", index=False)
    panel.to_csv(out / "embedding_manifest_used.tsv", sep="\t", index=False)
    depths = [f"depth{int(depth)}" for depth in config["depths"]]
    nonlinear = [model for model in depths if model != "depth0"]
    primary = long[
        long.split_column.eq(args.primary_split) & long.input_mode.eq(primary_mode)
    ].copy()

    print("\n" + "=" * 72)
    print("DP v3: VALIDATION-SELECTED SWEEP SUMMARY")
    print("=" * 72)
    print(f"primary input mode: {primary_mode}")

    print("\n[1] fixed-depth full-target test performance (no model selection)")
    print(f"    {'model':<14}{'n':>8}{'full R2@K':>13}{'retained R2':>14}{'unstable':>12}")
    fixed_rows = []
    for model in ["ridge"] + depths:
        block = primary[primary.model.eq(model)]
        if block.empty:
            continue
        stable = block[block.fit_status.eq("PASS") & np.isfinite(block.r2)]
        unstable = int((block.fit_status != "PASS").sum())
        median = float(stable.r2.median()) if len(stable) else np.nan
        retained_median = (
            float(stable.r2_retained_subspace.median()) if len(stable) else np.nan
        )
        fixed_rows.append({
            "model": model, "n_stable": len(stable), "n_unstable": unstable,
            "median_full_target_test_r2_at_k": median,
            "median_retained_subspace_test_r2": retained_median,
        })
        print(f"    {model:<14}{len(stable):>8,}{median:>13.4f}"
              f"{retained_median:>14.4f}{unstable:>12,}")
    pd.DataFrame(fixed_rows).to_csv(
        out / "fixed_depth_summary.tsv", sep="\t", index=False
    )

    print("\n[2] nonlinear depth selected on validation full-target R2@K")
    selected = select_by_validation(long, nonlinear)
    selected.to_csv(out / "validation_selected_depths.tsv", sep="\t", index=False)
    selected_primary = selected[
        selected.split_column.eq(args.primary_split)
        & selected.input_mode.eq(primary_mode)
    ].copy()
    counts = selected_primary.selected_model.value_counts()
    for model in nonlinear:
        count = int(counts.get(model, 0))
        share = count / len(selected_primary) if len(selected_primary) else np.nan
        print(f"    {model:<10} selected for {count:>5,} directions ({share:.1%})")

    linear = primary[
        primary.model.eq("depth0")
        & primary.fit_status.eq("PASS")
        & np.isfinite(primary.r2)
    ][KEY + ["r2", "val_r2"]].rename(columns={
        "r2": "linear_test_r2", "val_r2": "linear_val_r2"
    })
    gain = selected_primary.merge(linear, on=KEY, how="inner")
    gain["test_gain_over_linear"] = gain.selected_test_r2 - gain.linear_test_r2
    gain["validation_gain_over_linear"] = gain.selected_val_r2 - gain.linear_val_r2
    gain.to_csv(out / "nonlinear_gain_over_linear.tsv", sep="\t", index=False)
    stats = paired_summary(gain.test_gain_over_linear)
    print("\n    selected nonlinear architecture minus stable linear depth0")
    print(f"    n={stats['n']:,}  median test gain={stats['median']:+.4f}  "
          f"positive={stats['positive']:.1%}  paired Wilcoxon p={stats['p_value']:.3g}")
    print("    Interpretation: this is an added-hidden-architecture gain; LayerNorm, ")
    print("    GELU and hidden-layer regularisation change together, so it is not a")
    print("    pure activation-function effect.")

    print("\n[3] family-aware versus random split (full-target R2@K)")
    family = selected[
        selected.split_column.eq("split_family")
        & selected.input_mode.eq(primary_mode)
    ][["pair_id", "source", "target", "selected_model", "selected_test_r2"]]
    random = selected[
        selected.split_column.eq("split_random")
        & selected.input_mode.eq(primary_mode)
    ][["pair_id", "source", "target", "selected_model", "selected_test_r2"]]
    comparison = family.merge(
        random, on=["pair_id", "source", "target"], suffixes=("_family", "_random")
    )
    if not comparison.empty:
        comparison["random_minus_family"] = (
            comparison.selected_test_r2_random - comparison.selected_test_r2_family
        )
        comparison.to_csv(out / "split_comparison.tsv", sep="\t", index=False)
        split_stats = paired_summary(comparison.random_minus_family)
        print(f"    directions={len(comparison):,}")
        print(f"    median family R2={comparison.selected_test_r2_family.median():+.4f}")
        print(f"    median random R2={comparison.selected_test_r2_random.median():+.4f}")
        print(f"    median random-family={split_stats['median']:+.4f}; "
              f"p={split_stats['p_value']:.3g}")
    else:
        print("    both complete splits are needed")

    print("\n[4] incremental fixed-depth gains (full-target R2@K)")
    stable_depths = primary[
        primary.model.isin(nonlinear)
        & primary.fit_status.eq("PASS")
        & np.isfinite(primary.r2)
    ]
    wide = stable_depths.pivot_table(index=KEY, columns="model", values="r2")
    increment_rows = []
    for previous, current in zip(nonlinear, nonlinear[1:]):
        if previous not in wide or current not in wide:
            continue
        difference = (wide[current] - wide[previous]).dropna()
        depth_stats = paired_summary(difference)
        increment_rows.append({
            "comparison": f"{current}_minus_{previous}", **depth_stats,
        })
        print(f"    {current} - {previous}: median={depth_stats['median']:+.4f}; "
              f"positive={depth_stats['positive']:.1%}; p={depth_stats['p_value']:.3g}")
    pd.DataFrame(increment_rows).to_csv(
        out / "fixed_depth_increments.tsv", sep="\t", index=False
    )

    print("\n[5] retained-subspace spectra for validation-selected nonlinear models")
    selection_map = {
        (row.pair_id, row.source, row.target): row.selected_model
        for row in selected_primary.itertuples()
    }
    spectra_rows = []
    primary_dir = root / f"{args.primary_split}__{primary_mode}"
    for result_path in sorted(primary_dir.glob("*.json")):
        spectra_path = result_path.with_name(result_path.stem + "__spectra.npz")
        if not spectra_path.is_file():
            continue
        record = json.loads(result_path.read_text())
        if record.get("status") != "PASS":
            continue
        embedding_x = canonicalize_id(record.get("embedding_x", ""), aliases)
        embedding_y = canonicalize_id(record.get("embedding_y", ""), aliases)
        pair_id = canonical_pair(embedding_x, embedding_y)
        if pair_id not in expected_pair_ids:
            continue
        with np.load(spectra_path) as archive:
            for direction, source, target in (
                ("x_to_y", embedding_x, embedding_y),
                ("y_to_x", embedding_y, embedding_x),
            ):
                map_key = (pair_id, source, target)
                model = selection_map.get(map_key)
                archive_key = f"{direction}__{model}" if model else None
                if not archive_key or archive_key not in archive:
                    continue
                values = archive[archive_key]
                spectra_rows.append({
                    "pair_id": pair_id,
                    "source": source, "target": target,
                    "selected_model": model,
                    "n_components": int(values.size),
                    "n_above_0p50": int(np.nansum(values > 0.50)),
                    "n_above_0p25": int(np.nansum(values > 0.25)),
                    "first_component_r2": float(values[0]) if values.size else np.nan,
                })
    if spectra_rows:
        spectra = add_direction_labels(pd.DataFrame(spectra_rows), panel)
        spectra.to_csv(out / "selected_spectrum_summary.tsv", sep="\t", index=False)
        print(f"    directions={len(spectra):,}; median components above 0.50="
              f"{spectra.n_above_0p50.median():.0f}")
    else:
        print("    no matching spectra found")

    audit_rows = [
        {
            k: v for k, v in audit.items()
            if k not in {
                "pair_ids", "failed", "extra_pair_ids", "duplicate_pair_ids",
                "signatures",
            }
        }
        | {
            "n_failed": len(audit["failed"]),
            "n_run_signatures": len(audit["signatures"]),
            "n_ignored_out_of_panel_pairs": len(audit["extra_pair_ids"]),
            "n_duplicate_pairs": len(audit["duplicate_pair_ids"]),
        }
        for audit in audits
    ]
    pd.DataFrame(audit_rows).to_csv(out / "result_audit.tsv", sep="\t", index=False)
    (out / "summary_metadata.json").write_text(json.dumps({
        "manifest_file": str(manifest_path),
        "panel_sha256": panel_sha256(panel),
        "n_embeddings": len(panel),
        "n_expected_unordered_pairs": len(expected_pair_ids),
        "n_expected_directions_per_split": 2 * len(expected_pair_ids),
        "primary_split": args.primary_split,
        "primary_input_mode": primary_mode,
        "status": "PASS",
    }, indent=2))
    print(f"\nDONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
