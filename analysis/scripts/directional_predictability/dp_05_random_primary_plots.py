#!/usr/bin/env python3
"""Create random-primary figures for DP3 directional predictability.

The script produces:
  1. A source-embedding -> target-embedding heatmap of test R2 for the
     validation-selected nonlinear model (depth1--depth4).
  2. A source-embedding x target-domain heatmap of median cross-domain R2,
     ranked by each source embedding's cross-domain median.
  3. CKA/nearest-neighbour/MLP comparison figures, plus embedding- and
     domain-level summaries of nonlinear gain over stable depth0.

It expects the long table written by the DP3 summarizer and a wide pairwise
similarity table. Column names and table paths are auto-detected where this is
unambiguous; every important choice can also be supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import spearmanr

from dp_manifest import (
    canonicalize_id,
    id_aliases,
    load_manifest,
    manifest_maps,
    panel_sha256,
)


NONLINEAR_MODELS = ("depth1", "depth2", "depth3", "depth4")
LINEAR_MODEL = "depth0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="DP3 random-primary plots and plot-input audit.",
    )
    parser.add_argument("--dp-root", required=True, type=Path)
    parser.add_argument("--pair-metrics", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--domain-table", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dp-long-table", type=Path, default=None)
    parser.add_argument("--dp-selected-table", type=Path, default=None)
    parser.add_argument("--split", default="split_random")
    parser.add_argument("--input-mode", default="raw")
    parser.add_argument("--aggregate", choices=("median", "mean"), default="median")
    parser.add_argument("--expected-embeddings", type=int, default=0)
    parser.add_argument("--top-pairs", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=220)

    # DP-table overrides.
    parser.add_argument("--source-col")
    parser.add_argument("--target-col")
    parser.add_argument("--model-col")
    parser.add_argument("--test-r2-col")
    parser.add_argument("--validation-r2-col")
    parser.add_argument("--fit-status-col")
    parser.add_argument("--split-col")
    parser.add_argument("--input-mode-col")
    parser.add_argument("--selected-model-col")

    # Pairwise metric-table overrides.
    parser.add_argument("--pair-a-col")
    parser.add_argument("--pair-b-col")
    parser.add_argument("--cka-col")
    parser.add_argument("--nn-col")
    return parser.parse_args()


def norm_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        return frame if nrows is None else frame.head(nrows)
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
    raise ValueError(f"Unsupported table format: {path}")


def table_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for pattern in ("*.tsv", "*.csv", "*.parquet", "*.pq"):
        result.extend(root.rglob(pattern))
    return sorted(set(result))


def normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for col in frame.columns:
        key = norm_name(col)
        if key in mapping and mapping[key] != col:
            raise ValueError(f"Columns become ambiguous after normalization: {mapping[key]!r}, {col!r}")
        mapping[key] = col
    return mapping


def choose_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
    label: str,
    *,
    required: bool = True,
    regex: str | None = None,
) -> str | None:
    cols = normalized_columns(frame)
    if explicit:
        if explicit in frame.columns:
            return explicit
        key = norm_name(explicit)
        if key in cols:
            return cols[key]
        raise ValueError(f"Requested {label} column {explicit!r} not found. Columns: {list(frame.columns)}")
    for candidate in candidates:
        key = norm_name(candidate)
        if key in cols:
            return cols[key]
    if regex:
        hits = [original for key, original in cols.items() if re.search(regex, key)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(
                f"Ambiguous {label} columns {hits}. Select one explicitly with the relevant CLI option."
            )
    if required:
        raise ValueError(f"Could not detect {label} column. Columns: {list(frame.columns)}")
    return None


def looks_like_long_table(frame: pd.DataFrame) -> bool:
    cols = set(normalized_columns(frame))
    has_source = bool(cols & {"source", "source_embedding", "embedding_source"})
    has_target = bool(cols & {"target", "target_embedding", "embedding_target"})
    has_model = bool(cols & {"model", "model_name", "architecture", "depth"})
    has_r2 = "r2" in cols or any("r2" in col and "test" in col for col in cols)
    return has_source and has_target and has_model and has_r2


def looks_like_selected_table(frame: pd.DataFrame) -> bool:
    cols = set(normalized_columns(frame))
    return (
        bool(cols & {"source", "source_embedding", "embedding_source"})
        and bool(cols & {"target", "target_embedding", "embedding_target"})
        and bool(cols & {"selected_model", "selected_architecture", "best_model"})
    )


def autodetect_table(root: Path, kind: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in table_files(root):
        try:
            sample = read_table(path, nrows=8)
        except Exception:
            continue
        good = looks_like_long_table(sample) if kind == "long" else looks_like_selected_table(sample)
        if not good:
            continue
        name = norm_name(path.stem)
        score = 0
        if kind in name:
            score += 4
        if "summary" in name:
            score += 2
        if "direction" in name or "sweep" in name:
            score += 1
        candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1])))
    best_score = candidates[0][0]
    best = [path for score, path in candidates if score == best_score]
    if len(best) > 1:
        print(
            "WARNING: multiple candidate " + kind + " tables; using " + str(best[0]) +
            ". Override with --dp-" + kind + "-table if needed.",
            file=sys.stderr,
        )
    return best[0]


def normalize_model(series: pd.Series) -> pd.Series:
    def one(value: object) -> str:
        text = norm_name(value)
        match = re.fullmatch(r"(?:mlp_?)?depth_?([0-4])", text)
        if match:
            return f"depth{match.group(1)}"
        if text in {"linear", "linear_mlp", "no_hidden", "zero_hidden"}:
            return "depth0"
        return text
    return series.map(one)


def filter_context(
    frame: pd.DataFrame,
    split_col: str | None,
    input_col: str | None,
    split: str,
    input_mode: str,
) -> pd.DataFrame:
    result = frame.copy()
    if split_col:
        normalized = result[split_col].map(norm_name)
        wanted = norm_name(split)
        result = result.loc[normalized.eq(wanted)].copy()
    if input_col:
        normalized = result[input_col].map(norm_name)
        wanted = norm_name(input_mode)
        result = result.loc[normalized.eq(wanted)].copy()
    return result


def prepare_dp_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    long_path = args.dp_long_table or autodetect_table(args.dp_root, "long")
    if long_path is None:
        available = "\n".join(str(p) for p in table_files(args.dp_root)) or "[none]"
        raise FileNotFoundError(
            "Could not auto-detect the summarizer's long table. Use --dp-long-table. "
            f"Delimited tables found under {args.dp_root}:\n{available}"
        )
    long = read_table(long_path)
    source_col = choose_column(
        long, args.source_col, ("source", "source_embedding", "embedding_source"), "DP source"
    )
    target_col = choose_column(
        long, args.target_col, ("target", "target_embedding", "embedding_target"), "DP target"
    )
    model_col = choose_column(
        long, args.model_col, ("model", "model_name", "architecture", "depth"), "DP model"
    )
    r2_col = choose_column(
        long,
        args.test_r2_col,
        ("r2", "test_r2", "test_full_target_r2_at_k", "full_target_test_r2_at_k", "test_r2_at_k"),
        "test R2",
        regex=r"(?:^test.*r2|r2.*test)",
    )
    split_col = choose_column(
        long, args.split_col, ("split_column", "split", "split_name"), "split", required=False
    )
    input_col = choose_column(
        long, args.input_mode_col, ("input_mode", "mode", "representation_mode"), "input mode", required=False
    )
    fit_col = choose_column(
        long, args.fit_status_col, ("fit_status", "status", "training_status"), "fit status", required=False
    )
    val_col = choose_column(
        long,
        args.validation_r2_col,
        (
            "validation_r2",
            "val_r2",
            "validation_full_target_r2_at_k",
            "val_full_target_r2_at_k",
            "validation_r2_at_k",
            "val_r2_at_k",
        ),
        "validation R2",
        required=False,
        regex=r"(?:^val|validation).*r2|r2.*(?:val|validation)",
    )
    long = filter_context(long, split_col, input_col, args.split, args.input_mode)
    tidy = pd.DataFrame(
        {
            "source": long[source_col].astype(str).str.strip(),
            "target": long[target_col].astype(str).str.strip(),
            "model": normalize_model(long[model_col]),
            "test_r2": pd.to_numeric(long[r2_col], errors="coerce"),
            "fit_status": long[fit_col].astype(str).str.upper() if fit_col else "PASS",
        }
    )
    if val_col:
        tidy["validation_r2"] = pd.to_numeric(long[val_col], errors="coerce")

    selected_path = args.dp_selected_table or autodetect_table(args.dp_root, "selected")
    selected_keys: pd.DataFrame | None = None
    selected_col: str | None = None
    if selected_path:
        selected_raw = read_table(selected_path)
        ss = choose_column(
            selected_raw, args.source_col, ("source", "source_embedding", "embedding_source"), "selected source"
        )
        st = choose_column(
            selected_raw, args.target_col, ("target", "target_embedding", "embedding_target"), "selected target"
        )
        selected_col = choose_column(
            selected_raw,
            args.selected_model_col,
            ("selected_model", "selected_architecture", "best_model"),
            "selected model",
        )
        ssplit = choose_column(
            selected_raw, args.split_col, ("split_column", "split", "split_name"), "selected split", required=False
        )
        sinput = choose_column(
            selected_raw, args.input_mode_col, ("input_mode", "mode", "representation_mode"),
            "selected input mode", required=False
        )
        selected_raw = filter_context(selected_raw, ssplit, sinput, args.split, args.input_mode)
        selected_keys = pd.DataFrame(
            {
                "source": selected_raw[ss].astype(str).str.strip(),
                "target": selected_raw[st].astype(str).str.strip(),
                "model": normalize_model(selected_raw[selected_col]),
            }
        ).drop_duplicates()

    nonlinear = tidy.loc[tidy.model.isin(NONLINEAR_MODELS)].copy()
    if selected_keys is not None:
        selected = nonlinear.merge(selected_keys, on=["source", "target", "model"], how="inner")
    else:
        if "validation_r2" not in nonlinear:
            raise ValueError(
                "No selected-model table or validation R2 column was detected. Supply "
                "--dp-selected-table or --validation-r2-col."
            )
        candidates = nonlinear.loc[
            nonlinear.fit_status.eq("PASS") & np.isfinite(nonlinear.validation_r2)
        ].copy()
        candidates = candidates.sort_values(
            ["source", "target", "validation_r2", "model"],
            ascending=[True, True, False, True],
        )
        selected = candidates.drop_duplicates(["source", "target"], keep="first")

    selected = selected.loc[
        selected.fit_status.eq("PASS") & np.isfinite(selected.test_r2)
    ].copy()
    depth0 = tidy.loc[
        tidy.model.eq(LINEAR_MODEL) & tidy.fit_status.eq("PASS") & np.isfinite(tidy.test_r2),
        ["source", "target", "test_r2"],
    ].rename(columns={"test_r2": "depth0_r2"})

    if selected.duplicated(["source", "target"]).any():
        examples = selected.loc[selected.duplicated(["source", "target"], keep=False), ["source", "target", "model"]]
        raise ValueError(f"More than one selected nonlinear row per direction. Examples:\n{examples.head(20)}")
    if depth0.duplicated(["source", "target"]).any():
        raise ValueError("More than one stable depth0 row per direction after filtering.")
    selected = selected.merge(depth0, on=["source", "target"], how="left", validate="one_to_one")
    selected["nonlinear_gain"] = selected.test_r2 - selected.depth0_r2

    metadata = {
        "dp_long_table": str(long_path.resolve()),
        "dp_selected_table": str(selected_path.resolve()) if selected_path else None,
        "columns": {
            "source": source_col,
            "target": target_col,
            "model": model_col,
            "test_r2": r2_col,
            "validation_r2": val_col,
            "fit_status": fit_col,
            "split": split_col,
            "input_mode": input_col,
            "selected_model": selected_col,
        },
    }
    return tidy, selected, metadata


PAIR_ID_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("embedding_x", "embedding_y"),
    ("embedding_a", "embedding_b"),
    ("source", "target"),
    ("embedding_1", "embedding_2"),
    ("embedding1", "embedding2"),
    ("method_a", "method_b"),
    ("representation_a", "representation_b"),
    ("model_a", "model_b"),
)


def choose_pair_columns(
    frame: pd.DataFrame, explicit_a: str | None, explicit_b: str | None
) -> tuple[str, str]:
    if bool(explicit_a) != bool(explicit_b):
        raise ValueError("Supply both --pair-a-col and --pair-b-col, or neither.")
    if explicit_a and explicit_b:
        a = choose_column(frame, explicit_a, (), "pair A")
        b = choose_column(frame, explicit_b, (), "pair B")
        return str(a), str(b)
    cols = normalized_columns(frame)
    for a, b in PAIR_ID_CANDIDATES:
        if a in cols and b in cols:
            return cols[a], cols[b]
    raise ValueError(f"Could not detect pair ID columns. Columns: {list(frame.columns)}")


def canonical_pair(a: object, b: object) -> str:
    left, right = sorted((str(a).strip(), str(b).strip()))
    return left + "\x1f" + right


def prepare_pair_metrics(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = read_table(args.pair_metrics)
    a_col, b_col = choose_pair_columns(raw, args.pair_a_col, args.pair_b_col)
    cka_col = choose_column(
        raw,
        args.cka_col,
        (
            "unbiased_linear_cka_row_normalized",
            "linear_cka", "cka_linear", "cka", "cka_similarity",
            "centered_kernel_alignment",
        ),
        "CKA",
        regex=r"(?:^|_)cka(?:_|$)",
    )
    nn_col = choose_column(
        raw,
        args.nn_col,
        (
            "local_overlap_raw_k50",
            "knn_jaccard",
            "knn_overlap",
            "nearest_neighbor_overlap",
            "nearest_neighbour_overlap",
            "neighbor_overlap",
            "neighbour_overlap",
            "nn_overlap",
        ),
        "nearest-neighbour similarity",
        regex=r"(?:knn|nearest.*neigh|nn_overlap|neigh.*overlap)",
    )
    result = pd.DataFrame(
        {
            "embedding_a": raw[a_col].astype(str).str.strip(),
            "embedding_b": raw[b_col].astype(str).str.strip(),
            "cka": pd.to_numeric(raw[cka_col], errors="coerce"),
            "nn": pd.to_numeric(raw[nn_col], errors="coerce"),
        }
    )
    result["pair_key"] = [canonical_pair(a, b) for a, b in zip(result.embedding_a, result.embedding_b)]
    result = result.loc[
        result.embedding_a.ne(result.embedding_b) & np.isfinite(result.cka) & np.isfinite(result.nn)
    ].copy()
    if result.duplicated("pair_key").any():
        duplicate = result.loc[result.duplicated("pair_key", keep=False)].sort_values("pair_key")
        ranges = duplicate.groupby("pair_key")[["cka", "nn"]].nunique(dropna=False)
        inconsistent = ranges.loc[(ranges > 1).any(axis=1)]
        if not inconsistent.empty:
            raise ValueError("Pair metric table contains inconsistent duplicate unordered pairs.")
        result = result.drop_duplicates("pair_key")
    metadata = {
        "pair_metrics_table": str(args.pair_metrics.resolve()),
        "columns": {"embedding_a": a_col, "embedding_b": b_col, "cka": cka_col, "nn": nn_col},
    }
    return raw, result, metadata


def add_mapping(mapping: dict[str, str], embedding: object, domain: object, origin: str) -> None:
    emb = str(embedding).strip()
    dom = str(domain).strip()
    if not emb or emb.lower() == "nan" or not dom or dom.lower() == "nan":
        return
    if emb in mapping and norm_name(mapping[emb]) != norm_name(dom):
        raise ValueError(
            f"Conflicting domain labels for {emb!r}: {mapping[emb]!r} versus {dom!r} ({origin})"
        )
    mapping[emb] = dom


def extract_domain_mapping(frame: pd.DataFrame, origin: str) -> dict[str, str]:
    cols = normalized_columns(frame)
    mapping: dict[str, str] = {}

    single_ids = ("embedding", "embedding_name", "name", "model", "representation")
    single_domains = ("domain", "modality", "category", "embedding_domain", "embedding_modality")
    id_col = next((cols[x] for x in single_ids if x in cols), None)
    domain_col = next((cols[x] for x in single_domains if x in cols), None)
    if id_col and domain_col:
        for emb, dom in zip(frame[id_col], frame[domain_col]):
            add_mapping(mapping, emb, dom, origin)

    sides = (
        ("embedding_a", "domain_a"), ("embedding_b", "domain_b"),
        ("embedding_a", "modality_a"), ("embedding_b", "modality_b"),
        ("embedding_a", "embedding_a_domain"), ("embedding_b", "embedding_b_domain"),
        ("embedding_a", "embedding_a_modality"), ("embedding_b", "embedding_b_modality"),
        ("embedding_1", "domain_1"), ("embedding_2", "domain_2"),
        ("embedding_1", "modality_1"), ("embedding_2", "modality_2"),
        ("source", "source_domain"), ("target", "target_domain"),
        ("source", "source_modality"), ("target", "target_modality"),
        ("source", "source_category"), ("target", "target_category"),
        ("source_embedding", "source_domain"), ("target_embedding", "target_domain"),
        ("method_a", "method_a_domain"), ("method_b", "method_b_domain"),
        ("method_a", "method_a_modality"), ("method_b", "method_b_modality"),
        ("representation_a", "domain_a"), ("representation_b", "domain_b"),
    )
    for id_key, domain_key in sides:
        if id_key in cols and domain_key in cols:
            for emb, dom in zip(frame[cols[id_key]], frame[cols[domain_key]]):
                add_mapping(mapping, emb, dom, origin)
    return mapping


def build_domain_mapping(
    args: argparse.Namespace,
    long_tidy: pd.DataFrame,
    pair_raw: pd.DataFrame,
    embeddings: set[str],
) -> dict[str, str]:
    sources: list[tuple[pd.DataFrame, str]] = [(pair_raw, str(args.pair_metrics))]
    # The standardized DP table normally lacks domains, but this keeps the
    # extraction route explicit and harmless.
    sources.append((long_tidy, "standardized DP long table"))
    if args.domain_table:
        sources.insert(0, (read_table(args.domain_table), str(args.domain_table)))
    mapping: dict[str, str] = {}
    for frame, origin in sources:
        for emb, dom in extract_domain_mapping(frame, origin).items():
            add_mapping(mapping, emb, dom, origin)
    missing = sorted(embeddings - set(mapping))
    if missing:
        preview = ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else "")
        raise ValueError(
            f"Domain labels are missing for {len(missing)} embeddings: {preview}. "
            "Pass --domain-table pointing to a table with embedding/domain columns, or a pair table "
            "with embedding_a/domain_a and embedding_b/domain_b columns."
        )
    return {emb: mapping[emb] for emb in sorted(embeddings)}


def robust_limits(values: Iterable[float], include_zero: bool = True) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return (-1.0, 1.0)
    low, high = np.quantile(array, [0.01, 0.99])
    if include_zero:
        low = min(float(low), 0.0)
        high = max(float(high), 0.0)
    if math.isclose(float(low), float(high)):
        high = float(low) + 1e-6
    return float(low), float(high)


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_directional_heatmap(
    selected: pd.DataFrame,
    display: dict[str, str],
    out: Path,
    dpi: int,
) -> tuple[list[str], list[str]]:
    matrix = selected.pivot(index="source", columns="target", values="test_r2")
    source_order = matrix.median(axis=1, skipna=True).sort_values(ascending=False).index.tolist()
    # Use the same embedding order on both axes so the undefined self-prediction
    # cells form a visible diagonal. Rows still carry the requested best-to-worst
    # source ranking; an independent target order would scatter that diagonal and
    # obscure domain/block structure.
    target_order = source_order.copy()
    matrix = matrix.reindex(index=source_order, columns=target_order)
    low, high = robust_limits(matrix.to_numpy().ravel())
    fig, ax = plt.subplots(figsize=(22, 19))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=low,
        vmax=high,
        square=False,
        xticklabels=[display[value] for value in matrix.columns],
        yticklabels=[display[value] for value in matrix.index],
        cbar_kws={"label": "Test full-target $R^2@K$"},
    )
    ax.set_title("Random split: validation-selected nonlinear prediction", pad=14, weight="bold")
    ax.set_xlabel("Target embedding (same order as source axis)")
    ax.set_ylabel("Source embedding (ranked by median outgoing prediction $R^2$)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    save_figure(fig, out / "01_random_mlp_r2_source_to_target_heatmap", dpi)
    return source_order, target_order


def plot_cross_domain_heatmap(
    selected: pd.DataFrame,
    domains: dict[str, str],
    display: dict[str, str],
    aggregate: str,
    out: Path,
    dpi: int,
) -> pd.DataFrame:
    data = selected.copy()
    data["source_domain"] = data.source.map(domains)
    data["target_domain"] = data.target.map(domains)
    cross = data.loc[data.source_domain.ne(data.target_domain)].copy()
    grouped = (
        cross.groupby(["source", "target_domain"], as_index=False)
        .agg(r2=("test_r2", aggregate), n_targets=("target", "nunique"))
    )
    matrix = grouped.pivot(index="source", columns="target_domain", values="r2")
    row_score = cross.groupby("source").test_r2.agg(aggregate).sort_values(ascending=False)
    order = row_score.index.tolist()
    matrix = matrix.reindex(index=order)
    labels = [f"{display[embedding]}  [{domains[embedding]}]" for embedding in matrix.index]
    low, high = robust_limits(matrix.to_numpy().ravel())
    height = max(12, 0.27 * len(matrix) + 2)
    width = max(10, 1.15 * len(matrix.columns) + 6)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="vlag",
        center=0,
        vmin=low,
        vmax=high,
        mask=matrix.isna(),
        linewidths=0.25,
        linecolor="white",
        yticklabels=labels,
        cbar_kws={"label": f"{aggregate.title()} test full-target $R^2@K$"},
    )
    ax.set_title(
        "Random split: cross-domain prediction by source embedding",
        pad=14,
        weight="bold",
    )
    ax.set_xlabel("Target embedding domain (source's own domain excluded)")
    ax.set_ylabel(f"Source embedding (best-to-worst by cross-domain {aggregate} $R^2$)")
    ax.tick_params(axis="x", labelrotation=35, labelsize=9)
    ax.tick_params(axis="y", labelsize=6)
    save_figure(fig, out / "02_random_mlp_r2_by_target_domain_heatmap", dpi)
    grouped.to_csv(out / "02_random_mlp_r2_by_target_domain.tsv", sep="\t", index=False)
    ranking = row_score.rename("cross_domain_r2").reset_index()
    ranking["source_domain"] = ranking.source.map(domains)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(out / "02_source_embedding_cross_domain_ranking.tsv", sep="\t", index=False)
    return ranking


def pairwise_mlp(selected: pd.DataFrame) -> pd.DataFrame:
    work = selected.copy()
    work["pair_key"] = [canonical_pair(a, b) for a, b in zip(work.source, work.target)]
    result = (
        work.groupby("pair_key", as_index=False)
        .agg(
            embedding_a=("source", lambda x: sorted(set(x))[0]),
            embedding_b=("source", lambda x: sorted(set(x))[-1]),
            mlp_r2=("test_r2", "mean"),
            depth0_r2=("depth0_r2", "mean"),
            nonlinear_gain=("nonlinear_gain", "mean"),
            n_directions=("source", "size"),
            stable_gain_directions=("nonlinear_gain", "count"),
        )
    )
    # Derive IDs from pair_key so they remain correct even if source order is unusual.
    split_ids = result.pair_key.str.split("\x1f", n=1, expand=True)
    result["embedding_a"] = split_ids[0]
    result["embedding_b"] = split_ids[1]
    return result


def rho_label(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return (np.nan, np.nan, int(mask.sum()))
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p == 0:
        return "<1e-300"
    return f"{p:.2g}"


def scatter_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str | None = None,
    norm: TwoSlopeNorm | None = None,
) -> object | None:
    clean = data.loc[np.isfinite(data[x]) & np.isfinite(data[y])].copy()
    kwargs = dict(s=18, alpha=0.55, linewidths=0, rasterized=True)
    artist = None
    if color and norm is not None:
        artist = ax.scatter(clean[x], clean[y], c=clean[color], cmap="coolwarm", norm=norm, **kwargs)
    else:
        ax.scatter(clean[x], clean[y], color="#325f88", **kwargs)
    rho, p, n = rho_label(clean[x], clean[y])
    ax.text(
        0.03,
        0.97,
        f"Spearman $\\rho$={rho:+.3f}\n$p$={p_text(p)}; n={n:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    ax.set_title(title, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.7", linewidth=0.7, zorder=0)
    sns.despine(ax=ax)
    return artist


def plot_metric_comparison(joined: pd.DataFrame, out: Path, dpi: int) -> list[dict]:
    gains = joined.nonlinear_gain.to_numpy(dtype=float)
    finite_gain = gains[np.isfinite(gains)]
    bound = max(abs(np.quantile(finite_gain, 0.01)), abs(np.quantile(finite_gain, 0.99)))
    bound = max(float(bound), 1e-6)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    artist = scatter_panel(
        axes[0, 0], joined, "cka", "mlp_r2", "Linear CKA", "Bidirectional mean selected-MLP test $R^2$",
        "CKA versus nonlinear prediction", "nonlinear_gain", norm,
    )
    scatter_panel(
        axes[0, 1], joined, "nn", "mlp_r2", "Nearest-neighbour overlap", "Bidirectional mean selected-MLP test $R^2$",
        "NN overlap versus nonlinear prediction", "nonlinear_gain", norm,
    )
    scatter_panel(
        axes[1, 0], joined, "cka", "nonlinear_gain", "Linear CKA", "Bidirectional mean nonlinear gain in test $R^2$",
        "CKA versus benefit from nonlinearity",
    )
    scatter_panel(
        axes[1, 1], joined, "nn", "nonlinear_gain", "Nearest-neighbour overlap", "Bidirectional mean nonlinear gain in test $R^2$",
        "NN overlap versus benefit from nonlinearity",
    )
    if artist is not None:
        cbar = fig.colorbar(artist, ax=axes[0, :].tolist(), shrink=0.82, pad=0.02)
        cbar.set_label("Selected nonlinear minus stable depth0 test $R^2$")
    fig.suptitle("Same-pair comparison: geometry, neighbourhoods and prediction", fontsize=15, weight="bold")
    save_figure(fig, out / "03a_same_pair_cka_nn_mlp_comparison", dpi)

    correlations: list[dict] = []
    for x, y in (("cka", "mlp_r2"), ("nn", "mlp_r2"), ("cka", "nonlinear_gain"), ("nn", "nonlinear_gain"), ("cka", "nn")):
        rho, p, n = rho_label(joined[x], joined[y])
        correlations.append({"x": x, "y": y, "spearman_rho": rho, "p_value": p, "n_pairs": n})
    pd.DataFrame(correlations).to_csv(out / "03a_same_pair_spearman_correlations.tsv", sep="\t", index=False)
    return correlations


def domain_palette(domains: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    unique = sorted(set(domains), key=str.casefold)
    colors = sns.color_palette("husl", n_colors=max(1, len(unique)))
    return dict(zip(unique, colors))


def plot_embedding_gain(
    selected: pd.DataFrame,
    domains: dict[str, str],
    display: dict[str, str],
    aggregate: str,
    out: Path,
    dpi: int,
) -> pd.DataFrame:
    stable = selected.loc[np.isfinite(selected.nonlinear_gain)].copy()
    summary = (
        stable.groupby("source", as_index=False)
        .agg(
            nonlinear_gain=("nonlinear_gain", aggregate),
            median_selected_mlp_r2=("test_r2", "median"),
            n_targets=("target", "nunique"),
        )
        .sort_values("nonlinear_gain", ascending=True)
    )
    summary["domain"] = summary.source.map(domains)
    summary["rank_best_to_worst"] = summary.nonlinear_gain.rank(method="first", ascending=False).astype(int)
    palette = domain_palette(summary.domain.tolist())
    fig, ax = plt.subplots(figsize=(11, max(12, 0.28 * len(summary) + 2)))
    y = np.arange(len(summary))
    ax.hlines(y, 0, summary.nonlinear_gain, color="0.82", linewidth=1)
    ax.scatter(summary.nonlinear_gain, y, c=[palette[d] for d in summary.domain], s=34, zorder=3)
    ax.axvline(0, color="0.35", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(summary.source.map(display), fontsize=6)
    ax.set_xlabel(f"{aggregate.title()} outgoing nonlinear gain in test $R^2$")
    ax.set_ylabel("Source embedding")
    ax.set_title("Which source embeddings benefit most from nonlinearity?", weight="bold", pad=12)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=domain, markersize=6) for domain, color in palette.items()]
    ax.legend(handles=handles, title="Source domain", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    sns.despine(ax=ax)
    save_figure(fig, out / "03b_nonlinearity_gain_by_source_embedding", dpi)
    summary.sort_values("nonlinear_gain", ascending=False).to_csv(
        out / "03b_nonlinearity_gain_by_source_embedding.tsv", sep="\t", index=False
    )
    return summary


def plot_domain_gain(
    selected: pd.DataFrame,
    domains: dict[str, str],
    aggregate: str,
    out: Path,
    dpi: int,
) -> pd.DataFrame:
    stable = selected.loc[np.isfinite(selected.nonlinear_gain)].copy()
    stable["source_domain"] = stable.source.map(domains)
    stable["target_domain"] = stable.target.map(domains)
    grouped = (
        stable.groupby(["source_domain", "target_domain"], as_index=False)
        .agg(nonlinear_gain=("nonlinear_gain", aggregate), n_directions=("target", "size"))
    )
    matrix = grouped.pivot(index="source_domain", columns="target_domain", values="nonlinear_gain")
    domain_order = stable.groupby("source_domain").nonlinear_gain.agg(aggregate).sort_values(ascending=False).index
    matrix = matrix.reindex(index=domain_order, columns=domain_order)
    low, high = robust_limits(matrix.to_numpy().ravel())
    bound = max(abs(low), abs(high), 1e-6)
    fig, ax = plt.subplots(figsize=(max(9, len(matrix.columns) * 1.1 + 4), max(7, len(matrix) * 0.9 + 3)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-bound,
        vmax=bound,
        annot=len(matrix) <= 12,
        fmt=".3f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"{aggregate.title()} selected nonlinear minus depth0 test $R^2$"},
    )
    ax.set_title("Nonlinearity benefit by directional domain combination", weight="bold", pad=12)
    ax.set_xlabel("Target domain")
    ax.set_ylabel("Source domain")
    ax.tick_params(axis="x", labelrotation=35)
    ax.tick_params(axis="y", labelrotation=0)
    save_figure(fig, out / "03c_nonlinearity_gain_by_domain_combination", dpi)
    grouped.to_csv(out / "03c_nonlinearity_gain_by_domain_combination.tsv", sep="\t", index=False)
    return grouped


def plot_top_pairs(
    joined: pd.DataFrame,
    domains: dict[str, str],
    display: dict[str, str],
    top_n: int,
    out: Path,
    dpi: int,
) -> pd.DataFrame:
    top = joined.loc[np.isfinite(joined.nonlinear_gain)].nlargest(top_n, "nonlinear_gain").copy()
    top["domain_a"] = top.embedding_a.map(domains)
    top["domain_b"] = top.embedding_b.map(domains)
    top["display_a"] = top.embedding_a.map(display)
    top["display_b"] = top.embedding_b.map(display)
    top["label"] = top.display_a + " ↔ " + top.display_b
    ordered = top.sort_values("nonlinear_gain", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(7, 0.38 * len(ordered) + 2)))
    y = np.arange(len(ordered))
    ax.hlines(y, 0, ordered.nonlinear_gain, color="0.82", linewidth=1)
    ax.scatter(ordered.nonlinear_gain, y, color="#b54a4a", s=38, zorder=3)
    ax.axvline(0, color="0.35", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered.label, fontsize=7)
    ax.set_xlabel("Bidirectional mean selected nonlinear minus stable depth0 test $R^2$")
    ax.set_ylabel("Embedding pair")
    ax.set_title(f"Top {len(ordered)} embedding pairs benefiting from nonlinearity", weight="bold", pad=12)
    sns.despine(ax=ax)
    save_figure(fig, out / "03d_top_nonlinearity_gain_pairs", dpi)
    top.drop(columns="label").to_csv(out / "03d_top_nonlinearity_gain_pairs.tsv", sep="\t", index=False)
    return top


def validate_and_audit(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    metrics: pd.DataFrame,
    joined: pd.DataFrame,
    metadata: dict,
) -> dict:
    embeddings = sorted(set(selected.source) | set(selected.target))
    expected_pairs = len(embeddings) * (len(embeddings) - 1) // 2
    pair_keys = {canonical_pair(a, b) for a, b in zip(selected.source, selected.target)}
    reverse_keys = set(zip(selected.target, selected.source))
    directions = set(zip(selected.source, selected.target))
    missing_reverse = len(directions - reverse_keys)
    expected_directions = len(embeddings) * (len(embeddings) - 1)
    if len(embeddings) != args.expected_embeddings:
        raise ValueError(f"Found {len(embeddings)} embeddings; expected {args.expected_embeddings}.")
    if len(selected) != expected_directions:
        raise ValueError(f"Found {len(selected)} selected directions; expected {expected_directions}.")
    if len(pair_keys) != expected_pairs or missing_reverse:
        raise ValueError(
            f"Directional pair coverage is incomplete: {len(pair_keys)}/{expected_pairs} unordered pairs; "
            f"{missing_reverse} directions lack their reverse."
        )
    overlap = len(pair_keys & set(metrics.pair_key))
    if overlap != expected_pairs:
        raise ValueError(
            f"CKA/NN table overlaps {overlap}/{expected_pairs} DP pairs. This must be complete before "
            "making same-pair comparisons. Check embedding names and table provenance."
        )
    if len(joined) != expected_pairs:
        raise ValueError(f"Joined same-pair table has {len(joined)} rows; expected {expected_pairs}.")
    audit = {
        "primary_split": args.split,
        "input_mode": args.input_mode,
        "aggregation": args.aggregate,
        "n_embeddings": len(embeddings),
        "n_selected_directions": len(selected),
        "expected_directions": expected_directions,
        "n_unordered_pairs": len(pair_keys),
        "expected_unordered_pairs": expected_pairs,
        "directions_with_stable_depth0": int(selected.nonlinear_gain.notna().sum()),
        "metric_pairs": int(len(metrics)),
        "joined_metric_pairs": int(len(joined)),
        "selected_model_counts": {str(k): int(v) for k, v in selected.model.value_counts().sort_index().items()},
        "test_r2": {
            "median": float(selected.test_r2.median()),
            "mean": float(selected.test_r2.mean()),
            "minimum": float(selected.test_r2.min()),
            "maximum": float(selected.test_r2.max()),
        },
        "nonlinear_gain": {
            "n": int(selected.nonlinear_gain.notna().sum()),
            "median": float(selected.nonlinear_gain.median()),
            "mean": float(selected.nonlinear_gain.mean()),
            "positive_fraction": float((selected.nonlinear_gain.dropna() > 0).mean()),
        },
        "input_detection": metadata,
    }
    return audit


def main() -> None:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.dp_root = args.dp_root.expanduser().resolve()
    args.pair_metrics = args.pair_metrics.expanduser().resolve()
    args.domain_table = args.domain_table.expanduser().resolve() if args.domain_table else None
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dp_long_table = args.dp_long_table.expanduser().resolve() if args.dp_long_table else None
    args.dp_selected_table = args.dp_selected_table.expanduser().resolve() if args.dp_selected_table else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text())
    panel, manifest_path = load_manifest(config, args.config)
    aliases = id_aliases(config)
    display, domains = manifest_maps(panel)
    if args.expected_embeddings and args.expected_embeddings != len(panel):
        raise ValueError(
            f"--expected-embeddings={args.expected_embeddings} but manifest has {len(panel)} rows"
        )
    args.expected_embeddings = len(panel)

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

    tidy, selected, dp_metadata = prepare_dp_tables(args)
    pair_raw, pair_metrics, metric_metadata = prepare_pair_metrics(args)
    for frame in (tidy, selected):
        frame["source"] = frame.source.map(lambda value: canonicalize_id(value, aliases))
        frame["target"] = frame.target.map(lambda value: canonicalize_id(value, aliases))
    pair_metrics["embedding_a"] = pair_metrics.embedding_a.map(
        lambda value: canonicalize_id(value, aliases)
    )
    pair_metrics["embedding_b"] = pair_metrics.embedding_b.map(
        lambda value: canonicalize_id(value, aliases)
    )
    pair_metrics["pair_key"] = [
        canonical_pair(a, b)
        for a, b in zip(pair_metrics.embedding_a, pair_metrics.embedding_b)
    ]
    embeddings = set(selected.source) | set(selected.target)
    if embeddings != set(panel.embedding_id):
        raise ValueError(
            "directional results do not match the manifest panel; "
            f"missing={sorted(set(panel.embedding_id) - embeddings)}, "
            f"extra={sorted(embeddings - set(panel.embedding_id))}"
        )

    mlp_pairs = pairwise_mlp(selected)
    joined = mlp_pairs.merge(
        pair_metrics[["pair_key", "cka", "nn"]], on="pair_key", how="inner", validate="one_to_one"
    )
    metadata = {
        "dp": dp_metadata,
        "pair_metrics": metric_metadata,
        "manifest_file": str(manifest_path),
        "panel_sha256": panel_sha256(panel),
    }
    audit = validate_and_audit(args, selected, pair_metrics, joined, metadata)

    selected.assign(
        source_display=selected.source.map(display),
        target_display=selected.target.map(display),
        source_domain=selected.source.map(domains),
        target_domain=selected.target.map(domains),
    ).to_csv(args.output_dir / "00_selected_random_directional_plot_data.tsv", sep="\t", index=False)
    joined.assign(
        display_a=joined.embedding_a.map(display),
        display_b=joined.embedding_b.map(display),
        domain_a=joined.embedding_a.map(domains),
        domain_b=joined.embedding_b.map(domains),
    ).to_csv(args.output_dir / "00_same_pair_metric_plot_data.tsv", sep="\t", index=False)
    panel.rename(columns={"embedding_id": "embedding", "modality": "domain"}).to_csv(
        args.output_dir / "00_embedding_domains.tsv", sep="\t", index=False
    )

    source_order, target_order = plot_directional_heatmap(selected, display, args.output_dir, args.dpi)
    pd.DataFrame({
        "rank": range(1, len(source_order) + 1),
        "source_embedding": source_order,
        "source_display": [display[value] for value in source_order],
    }).to_csv(
        args.output_dir / "01_source_order.tsv", sep="\t", index=False
    )
    pd.DataFrame({
        "rank": range(1, len(target_order) + 1),
        "target_embedding": target_order,
        "target_display": [display[value] for value in target_order],
    }).to_csv(
        args.output_dir / "01_target_order.tsv", sep="\t", index=False
    )
    plot_cross_domain_heatmap(selected, domains, display, args.aggregate, args.output_dir, args.dpi)
    correlations = plot_metric_comparison(joined, args.output_dir, args.dpi)
    plot_embedding_gain(selected, domains, display, args.aggregate, args.output_dir, args.dpi)
    plot_domain_gain(selected, domains, args.aggregate, args.output_dir, args.dpi)
    plot_top_pairs(joined, domains, display, args.top_pairs, args.output_dir, args.dpi)
    audit["same_pair_spearman_correlations"] = correlations

    with (args.output_dir / "plot_input_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)

    print("PLOT AUDIT: PASS")
    print(f"Primary split: {args.split}; input mode: {args.input_mode}")
    print(f"Embeddings: {audit['n_embeddings']}")
    print(f"Directions: {audit['n_selected_directions']}")
    print(f"Unordered pairs: {audit['n_unordered_pairs']}")
    print(f"Stable depth0 comparisons: {audit['directions_with_stable_depth0']}")
    print(f"Median selected nonlinear test R2: {audit['test_r2']['median']:+.4f}")
    print(f"Median nonlinear gain: {audit['nonlinear_gain']['median']:+.4f}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
