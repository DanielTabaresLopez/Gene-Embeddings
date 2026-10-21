#!/usr/bin/env python3
"""Run the publication consensus workflow from the repository's analysis/ directory.

This launcher wires the original computational scripts to the YAML. The paper
uses calibrated linear CKA and calibrated gene-neighbour overlap at k=100,
followed by an embedding-level union kNN graph at k=10. Other overlap scales
are retained only for the original concordance/sensitivity diagnostics.

See `python scripts/consensus_graph/cs_pipeline.py --help` for stages. On Slurm,
use `bash slurm/consensus_graph/submit_full_workflow.sh` instead of running the
compute stages on a login node. Provide CS_LIBRARY_ROOT and CS_PFI_TABLE.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pandas as pd
import yaml

from cs_manifest import load_manifest, resolve_path


SCRIPTS = Path(__file__).resolve().parent
STAGES = ("check", "check-inputs", "download", "select", "freeze", "prepare",
          "neighbors", "pair", "pairs", "aggregate", "downstream", "figures",
          "heatmaps", "panel", "audit", "test")
FILES = {i: next(SCRIPTS.glob(f"cs_{i:02}_*.py")) for i in range(1, 14)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("task_index", type=int, nargs="?")
    ap.add_argument("--root", type=Path, default=Path(os.environ.get(
        "CS_PROJECT_ROOT", SCRIPTS.parents[1])))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("CS_PAIR_WORKERS", "8")))
    ap.add_argument("--threads-per-pair", type=int,
                    default=int(os.environ.get("CS_THREADS_PER_PAIR", "2")))
    ap.add_argument("--dry-run", action="store_true", help="Print commands; do not execute stages.")
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    config_path = (args.config or Path(os.environ.get(
        "CS_CONFIG", root / "config/consensu_graph.yaml"))).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text())
    manifest, manifest_path = load_manifest(cfg, config_path)

    def path(key, env=None):
        value = os.environ.get(env) if env else None
        return resolve_path(value or cfg[key], config_path)

    # These scripts intentionally reproduce one frozen publication analysis.
    # Refuse YAML changes that would otherwise be ignored by fixed constants.
    fixed = {"expected_retained_embedding_count": 56, "expected_common_gene_count": 14039,
             "expected_pair_count": 1540, "primary_local_k": 100, "graph_k": 10,
             "atlas_seed": 20260725, "atlas_null_shuffles": 5,
             "atlas_k_values": [10, 25, 50, 100],
             "primary_platonic_view": "platonic_linear_calibrated",
             "primary_local_view": "aristotelian_local_k100_calibrated",
             "consensus_weights": [0.5, 0.5], "consensus_configurations": ["primary"]}
    for key, expected in fixed.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Publication preset requires {key}={expected!r}, got {cfg.get(key)!r}")
    workflow = root / "results/consensus_graph_v2"
    outputs = {"frozen_panel_output": "01_fixed_panel", "atlas_cache": "02_common_panel_atlas_cache",
               "atlas_output": "02_common_panel_atlas", "pairwise_evidence_output": "03_pairwise_evidence",
               "views_output": "04_views", "concordance_output": "05_view_concordance",
               "consensus_output": "06_consensus", "modalities_output": "07_modalities",
               "graph_output": "08_graph_structure", "interpretation_output": "09_community_interpretation",
               "figures_output": "10_figures", "final_audit_output": "11_final_audit"}
    for key, relative in outputs.items():
        if path(key) != workflow / relative:
            raise ValueError(f"{key} must be {workflow / relative}; use --root to relocate the workflow")
    frozen = path("frozen_panel_source")
    for filename, key in (("retained_embeddings.txt", "expected_retained_file_sha256"),
                          ("common_genes.txt", "expected_common_genes_sha256")):
        if sha(frozen / filename) != cfg[key]:
            raise ValueError(f"Frozen publication input changed: {filename}")
    frozen_meta = pd.read_csv(frozen / "retained_embeddings.tsv", sep="\t", dtype=str)
    expected_meta = manifest.set_index("embedding_id").loc[frozen_meta.embedding_id]
    for col in ("display_name", "modality"):
        if frozen_meta[col].tolist() != expected_meta[col].tolist():
            raise ValueError(f"Frozen panel {col} differs from the publication manifest")

    library = path("library_root", "CS_LIBRARY_ROOT")
    pfi = path("pfi_pair_results", "CS_PFI_TABLE")
    panel = path("frozen_panel_output")
    cache = path("atlas_cache")
    atlas = path("atlas_output")

    def run(number, *arguments, env=None, log=None):
        cmd = [sys.executable, str(FILES[number]), *map(str, arguments)]
        if args.dry_run:
            print(shlex.join(cmd), flush=True)
            return
        if log is None:
            subprocess.run(cmd, check=True, env=env)
        else:
            with log.open("w") as handle:
                try:
                    subprocess.run(cmd, check=True, env=env, stdout=handle, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(f"Pair calculation failed; see {log}") from error

    def atlas_args(stage, index=None):
        values = [stage, "--panel-dir", panel]
        if stage in {"prepare", "neighbors", "pair"}:
            values += ["--cache-dir", cache, "--task-index", index]
        if stage == "prepare":
            values += ["--embeddings-root", library, "--gene-column", cfg["id_column"],
                       "--matrix-key", cfg["array_key"], "--feature-block", cfg["atlas_feature_block"]]
        if stage == "neighbors":
            values += ["--row-block", cfg["atlas_neighbour_row_block"], "--k-values", *cfg["atlas_k_values"]]
        if stage == "pair":
            values += ["--output-dir", atlas, "--null-shuffles", cfg["atlas_null_shuffles"],
                       "--feature-block", cfg["atlas_feature_block"], "--k-values", *cfg["atlas_k_values"]]
        if stage == "aggregate":
            values += ["--output-dir", atlas, "--dpi", cfg["atlas_dpi"]]
        return values

    def figures():
        run(10, "--root", root, "--seed", cfg["figure_layout_seed"],
            "--layout-iterations", cfg["figure_layout_iterations"],
            "--top-bridge-labels", cfg["figure_top_bridge_labels"])

    def heatmaps():
        run(12, "--root", root, "--manifest", manifest_path,
            "--order-file", path("heatmap_order_file"), "--dpi", cfg["atlas_dpi"])

    def assemble():
        run(13, "--root", root)

    def audit():
        run(11, "--root", root, "--manifest", manifest_path)

    if args.stage in {"check", "check-inputs"}:
        if args.stage == "check-inputs":
            # Presence check only; stage 1 audits numerical validity of all rows.
            from cs_02_common_panel_atlas import resolve_embeddings_root
            embroot = resolve_embeddings_root(library, manifest.embedding_id)
            for embedding in manifest.embedding_id:
                for filename in ("genes.tsv", "embeddings.npz"):
                    if not (embroot / embedding / filename).is_file():
                        raise FileNotFoundError(embroot / embedding / filename)
            if not pfi.is_file():
                raise FileNotFoundError(f"Set CS_PFI_TABLE to the completed 67-embedding PFI table: {pfi}")
        print("PASS: k=100 publication preset, manifest names and frozen panel hashes")
    elif args.stage == "download":
        if args.dry_run:
            print(f"Download {cfg['hf_repo_id']} at {cfg['hf_revision']} to {library}")
        else:
            from huggingface_hub import snapshot_download
            patterns = [f"{cfg['embeddings_subdir']}/{item}/{name}"
                        for item in manifest.embedding_id for name in ("genes.tsv", "embeddings.npz")]
            snapshot_download(repo_id=cfg["hf_repo_id"], repo_type=cfg["hf_repo_type"],
                              revision=cfg["hf_revision"], local_dir=str(library), allow_patterns=patterns)
    elif args.stage == "select":
        run(1, "--config", config_path, "--library-root", library)
    elif args.stage == "freeze":
        source = path("frozen_panel_source", "CS_SELECTED_PANEL")
        # Allow a repeat only when the identities/order of an existing panel match.
        for name in ("retained_embeddings.txt", "common_genes.txt"):
            if (panel / name).exists() and sha(panel / name) != sha(source / name):
                raise ValueError(f"Refusing to change a previously frozen panel: {panel / name}")
        run(2, "freeze", "--source-panel", source, "--panel-dir", panel, "--allow-existing")
    elif args.stage in {"prepare", "neighbors", "pair"}:
        limit = 1540 if args.stage == "pair" else 56
        if args.task_index is None or not 0 <= args.task_index < limit:
            ap.error(f"{args.stage} requires a task index between 0 and {limit - 1}")
        run(2, *atlas_args(args.stage, args.task_index))
    elif args.stage == "pairs":
        if args.workers < 1 or args.threads_per_pair < 1:
            ap.error("workers and threads-per-pair must be positive")
        if "SLURM_CPUS_PER_TASK" in os.environ:
            if args.workers * args.threads_per_pair > int(os.environ["SLURM_CPUS_PER_TASK"]):
                ap.error("workers × threads-per-pair exceeds allocated Slurm CPUs")
        if args.dry_run:
            run(2, *atlas_args("pair", 0))
            print(f"Repeat for pair indices 0–1539, up to {args.workers} workers; existing pairs are skipped.")
            return
        logs = root / "logs/consensus_pair_workers"
        logs.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(args.threads_per_pair)
        env["PYTHONUNBUFFERED"] = "1"
        def compute(index):
            run(2, *atlas_args("pair", index), env=env, log=logs / f"pair_{index:04d}.log")
        # The executor runs subprocesses; numerical work is outside Python's GIL.
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for completed, _ in enumerate(executor.map(compute, range(1540)), 1):
                if completed % 25 == 0 or completed == 1540:
                    print(f"Pairs complete or cached: {completed}/1540", flush=True)
    elif args.stage == "aggregate":
        run(2, *atlas_args("aggregate"))
    elif args.stage == "downstream":
        if not args.dry_run and not pfi.is_file():
            raise FileNotFoundError(f"Set CS_PFI_TABLE to the completed PFI pair_results.tsv: {pfi}")
        run(3, "--root", root, "--pfi", pfi)
        run(4, "--root", root)
        run(5, "--root", root, "--n-permutations", cfg["concordance_permutations"], "--seed", cfg["concordance_seed"])
        run(6, "--root", root)
        run(7, "--root", root)
        run(8, "--root", root, "--graph-k", cfg["graph_k"], "--permutations", cfg["graph_permutations"], "--seed", cfg["graph_seed"])
        run(9, "--root", root, "--permutations", cfg["interpretation_permutations"], "--seed", cfg["interpretation_seed"])
        figures()
        heatmaps()
        assemble()
        audit()
    elif args.stage == "figures":
        figures()
        heatmaps()
        assemble()
        audit()
    elif args.stage == "heatmaps":
        heatmaps()
        assemble()
    elif args.stage == "panel":
        assemble()
    elif args.stage == "audit":
        audit()
    elif args.stage == "test":
        run(2, "self-test")
        for number in (4, 5, 6, 8, 9, 10):
            run(number, "--self-test")


if __name__ == "__main__":
    main()
