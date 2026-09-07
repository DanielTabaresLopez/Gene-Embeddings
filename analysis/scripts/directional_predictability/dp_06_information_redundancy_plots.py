#!/usr/bin/env python3
"""Plot modality-balanced information exportability versus redundancy.

The input must be the selected directional-prediction table: one held-out-test
R^2 value for every ordered source -> target embedding pair.  The primary
scores are:

  information(e) = mean over target modalities t != modality(e) of
                   median_j_in_t R^2(e -> j)

  redundancy(e)  = mean over source modalities s != modality(e) of
                   median_i_in_s R^2(i -> e)

Thus, every modality receives one vote regardless of how many embeddings it
contains.  Same-modality comparisons are excluded from the primary plot.

Two sensitivity analyses are also written:
  1. The same modality-balanced calculation including the embedding's own
     modality.
  2. Leave-one-embedding-out residual scores relative to the relevant
     source-modality x target-modality baseline.

The script never clips negative test R^2 values.

By default, stable IDs, display names, and modalities come from the embedding
manifest referenced by ``--config``. The old 65-entry mapping remains available
only as an explicit legacy fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from dp_manifest import canonicalize_id, id_aliases, load_manifest, panel_sha256


SOURCE_CANDIDATES = (
    "source_embedding",
    "embedding_source",
    "source",
    "src_embedding",
    "source_name",
    "embedding_x",
)
TARGET_CANDIDATES = (
    "target_embedding",
    "embedding_target",
    "target",
    "dst_embedding",
    "target_name",
    "embedding_y",
)
SCORE_CANDIDATES = (
    "selected_test_r2",
    "selected_nonlinear_test_r2",
    "test_full_target_r2_at_k",
    "full_target_test_r2_at_k",
    "test_r2_at_k",
    "test_r2_selected",
    "selected_r2",
    "best_test_r2",
    "test_r2",
    "r2_test",
    "r2",
    "score",
)
SPLIT_CANDIDATES = (
    "split",
    "split_column",
    "split_name",
    "evaluation_split",
    "data_split",
)
INPUT_MODE_CANDIDATES = (
    "input_mode",
    "representation_mode",
    "normalization",
    "normalisation",
)
EMBEDDING_CANDIDATES = (
    "embedding",
    "embedding_name",
    "name",
    "source_embedding",
)
DOMAIN_CANDIDATES = ("domain", "modality", "embedding_domain", "category")
SELECTED_FLAG_CANDIDATES = ("is_selected", "selected", "best_model")
SELECTION_RANK_CANDIDATES = ("selection_rank", "rank")


PREFERRED_DOMAIN_COLORS = {
    "Protein sequence": "#3B6FB6",
    "Biomedical text": "#D97732",
    "Integrated / multimodal": "#6F57A5",
    "Transcriptomics / cell state": "#2C9A86",
    "DNA / RNA sequence": "#B64C5C",
    "Regulatory genomics": "#8A6D3B",
    "Knowledge graph context": "#C25A9B",
    "Networks / ontologies": "#6B8E3D",
    "Functional / phenotypic": "#6C7680",
    "Multimodal": "#6F57A5",
    "Single-cell": "#2C9A86",
    "DNA sequence": "#B64C5C",
    "RNA sequence": "#B65F76",
    "Functional": "#6C7680",
    "Expression": "#D65DB1",
    "Knowledge graph": "#C25A9B",
    "Networks": "#6B8E3D",
}


# Authoritative modality mapping transcribed from the final MLP heatmap
# `02_random_mlp_r2_by_target_domain_heatmap (2)(2).pdf`.  These are the
# categories shown in brackets next to the 65 source-embedding labels.
FINAL_MLP_DOMAIN_BY_EMBEDDING = {
    "alphagenome_tss1mb_regulatory_tracksummary_pca256": "Regulatory genomics",
    "basenji2_tss_131kb_human_central3": "Regulatory genomics",
    "biobert_gene_protein_summary": "Biomedical text",
    "bioconceptvec_cbow": "Biomedical text",
    "bioconceptvec_fasttext": "Biomedical text",
    "bioconceptvec_glove": "Biomedical text",
    "bioconceptvec_skipgram": "Biomedical text",
    "deepnf_string_v12_ppmi_svd_fusion": "Networks / ontologies",
    "deepsea_tss_1kb_epigenomic_probs": "Regulatory genomics",
    "depmap_crispr_gene_effect_pca256": "Functional / phenotypic",
    "dnabert2_117m_cds": "DNA / RNA sequence",
    "enformer_tss_393kb_human_central3": "Regulatory genomics",
    "esm2": "Protein sequence",
    "frogs_archs4_256": "Functional / phenotypic",
    "gene2vec": "Integrated / multimodal",
    "genecompass_base_gene_token": "Transcriptomics / cell state",
    "geneformer_v2_316m": "Transcriptomics / cell state",
    "genept_ada_conservative": "Biomedical text",
    "genept_model3_gene_protein_conservative": "Biomedical text",
    "gtex_tissue_median_log1p_zscore": "Transcriptomics / cell state",
    "helix_mrna_full_transcript": "DNA / RNA sequence",
    "hetionet_nonppi_biomedical_context_tfidf_svd256": "Knowledge graph context",
    "hpa_normal_ihc_tissue_level_zscore": "Protein sequence",
    "hpa_subcellular_location_multihot_zscore": "Protein sequence",
    "hyenadna_medium_160k_cds": "DNA / RNA sequence",
    "mahi_all_contexts_mean": "Integrated / multimodal",
    "mahi_global": "Integrated / multimodal",
    "mashup_string": "Networks / ontologies",
    "msa_transformer_depth1_uniprot_human": "Protein sequence",
    "newt_archs4_256": "Integrated / multimodal",
    "newt_cellnet": "Integrated / multimodal",
    "newt_go_256": "Integrated / multimodal",
    "newt_go_graph": "Integrated / multimodal",
    "newt_msigdb_bundle": "Integrated / multimodal",
    "node2vec_consensus_ppi": "Networks / ontologies",
    "nt_v2_500m_multispecies_cds": "DNA / RNA sequence",
    "opa2vec_goa_human": "Knowledge graph context",
    "orthrus_base_4track_full_transcript": "DNA / RNA sequence",
    "primekg_nonppi_biomedical_context_tfidf_svd256": "Knowledge graph context",
    "probe_aac": "Protein sequence",
    "probe_albert": "Protein sequence",
    "probe_apaac": "Protein sequence",
    "probe_bert_bfd": "Protein sequence",
    "probe_bert_pfam": "Protein sequence",
    "probe_blast": "Protein sequence",
    "probe_cpc_prot": "Protein sequence",
    "probe_esmb1": "Protein sequence",
    "probe_gene2vec_uniprot": "Protein sequence",
    "probe_hmmer": "Protein sequence",
    "probe_learned_vec": "Protein sequence",
    "probe_mut2vec": "Protein sequence",
    "probe_protvec": "Protein sequence",
    "probe_seqvec": "Protein sequence",
    "probe_t5": "Protein sequence",
    "probe_tcga_embedding": "Protein sequence",
    "probe_unirep": "Protein sequence",
    "probe_xlnet": "Protein sequence",
    "pubmedbert_gene_protein_summary": "Biomedical text",
    "rnafm_full_transcript": "DNA / RNA sequence",
    "scfoundation_gene_pos": "Transcriptomics / cell state",
    "scgpt_pancancer": "Transcriptomics / cell state",
    "scgpt_whole_human": "Transcriptomics / cell state",
    "scprint_medium_v1_5_gene": "Transcriptomics / cell state",
    "spliceai_canonical_splice_site_profile": "Regulatory genomics",
    "uce_33l": "Transcriptomics / cell state",
}

FINAL_MLP_EXPECTED_DOMAIN_COUNTS = {
    "Protein sequence": 22,
    "Biomedical text": 8,
    "Integrated / multimodal": 8,
    "Transcriptomics / cell state": 8,
    "DNA / RNA sequence": 6,
    "Regulatory genomics": 5,
    "Knowledge graph context": 3,
    "Networks / ontologies": 3,
    "Functional / phenotypic": 2,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Create modality-balanced information-versus-redundancy scatter "
            "plots from selected directional MLP test R^2 values."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Directional-predictability YAML. Its embedding manifest supplies "
            "stable IDs, display names, and modalities."
        ),
    )
    parser.add_argument(
        "--direction-table",
        type=Path,
        help="TSV/CSV containing one selected result per ordered pair.",
    )
    parser.add_argument(
        "--dp-root",
        type=Path,
        help=(
            "Directional-predictability result root. Used only to find "
            "a selected directional R^2 table when --direction-table is omitted."
        ),
    )
    parser.add_argument(
        "--domain-source",
        choices=("manifest", "final-mlp", "table"),
        default="manifest",
        help=(
            "Use the configured embedding manifest (recommended), the legacy "
            "65-entry built-in mapping, or an external mapping table."
        ),
    )
    parser.add_argument(
        "--domain-table",
        type=Path,
        help=(
            "TSV/CSV mapping every embedding to one modality/domain. Required "
            "only with --domain-source table."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots_information_redundancy_v1")
    )
    parser.add_argument("--split", default="split_random")
    parser.add_argument("--input-mode", default="raw")
    parser.add_argument("--source-col")
    parser.add_argument("--target-col")
    parser.add_argument("--score-col")
    parser.add_argument("--split-col")
    parser.add_argument("--input-mode-col")
    parser.add_argument("--embedding-col")
    parser.add_argument("--domain-col")
    parser.add_argument(
        "--within-modality",
        choices=("median", "mean"),
        default="median",
        help="Aggregate individual embedding-pair R^2 values within a modality.",
    )
    parser.add_argument(
        "--across-modalities",
        choices=("mean", "median"),
        default="mean",
        help="Combine modality-level cells with equal modality weights.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("all", "extremes", "none"),
        default="all",
        help="Which embedding names to draw next to points.",
    )
    parser.add_argument(
        "--extreme-label-count",
        type=int,
        default=5,
        help="Number of high/low points per axis when --label-mode=extremes.",
    )
    parser.add_argument(
        "--expected-embeddings",
        type=int,
        default=0,
        help="Expected panel size; 0 derives it from the manifest or input.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a synthetic 65-embedding test and exit.",
    )
    args = parser.parse_args(argv)

    if not args.self_test:
        if args.direction_table is None and args.dp_root is None and args.config is None:
            parser.error("provide --direction-table, --dp-root, or --config")
        if args.domain_source == "table" and args.domain_table is None:
            parser.error("--domain-source table requires --domain-table")
        if args.domain_source == "manifest" and args.config is None:
            parser.error("--domain-source manifest requires --config")
    return args


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=None, engine="python")


def choose_direction_table(args: argparse.Namespace) -> Path:
    if args.direction_table is not None:
        return args.direction_table
    if args.dp_root is None and args.config is not None:
        config = yaml.safe_load(args.config.expanduser().read_text())
        args.dp_root = Path(os.path.expanduser(str(config["train_output"])))
    assert args.dp_root is not None
    preferred = (
        args.dp_root
        / "plots_random_primary_v1"
        / "00_selected_random_directional_plot_data.tsv",
        args.dp_root / "validation_selected_depths.tsv",
        args.dp_root / "nonlinear_gain_over_linear.tsv",
        args.dp_root / "selected_summary.tsv",
        args.dp_root / "directional_selected_summary.tsv",
    )
    for path in preferred:
        if path.is_file():
            return path
    found = sorted(args.dp_root.glob("*selected*direction*.tsv"))
    if len(found) == 1:
        return found[0]
    if found:
        choices = "\n  ".join(str(p) for p in found)
        raise ValueError(
            "Several selected summary tables were found. Pass the intended one "
            f"with --direction-table:\n  {choices}"
        )
    raise FileNotFoundError(
        f"No selected directional summary table found under {args.dp_root}. "
        "Pass it explicitly with --direction-table."
    )


def final_mlp_domain_data() -> pd.DataFrame:
    domains = pd.DataFrame(
        sorted(FINAL_MLP_DOMAIN_BY_EMBEDDING.items()),
        columns=["embedding", "domain"],
    )
    observed_counts = domains["domain"].value_counts().to_dict()
    if len(domains) != 65 or observed_counts != FINAL_MLP_EXPECTED_DOMAIN_COUNTS:
        raise AssertionError(
            "Internal final-MLP modality mapping failed validation. "
            f"Embeddings={len(domains)}; counts={observed_counts}"
        )
    return domains


def normalized_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def detect_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: Iterable[str],
    role: str,
    required: bool = True,
) -> str | None:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(
                f"Requested {role} column {explicit!r} is absent. Columns are: "
                + ", ".join(map(str, frame.columns))
            )
        return explicit
    by_normalized = {normalized_name(col): str(col) for col in frame.columns}
    for candidate in candidates:
        if normalized_name(candidate) in by_normalized:
            return by_normalized[normalized_name(candidate)]
    if required:
        raise ValueError(
            f"Could not detect the {role} column. Use the corresponding --*-col "
            "argument. Columns are: "
            + ", ".join(map(str, frame.columns))
        )
    return None


def filter_requested_value(
    frame: pd.DataFrame, column: str | None, requested: str, role: str
) -> pd.DataFrame:
    if column is None:
        return frame
    observed = frame[column].dropna().astype(str)
    normalized_requested = normalized_name(requested)
    keep = observed.map(normalized_name) == normalized_requested
    mask = pd.Series(False, index=frame.index)
    mask.loc[observed.index] = keep.values
    if not mask.any():
        available = sorted(observed.unique().tolist())
        raise ValueError(
            f"No rows matched {role}={requested!r} in column {column!r}. "
            f"Available values: {available}"
        )
    return frame.loc[mask].copy()


def truthy_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).map(normalized_name)
    return normalized.isin({"1", "true", "yes", "y", "selected", "best"})


def select_one_row_per_direction(
    frame: pd.DataFrame, source_col: str, target_col: str
) -> pd.DataFrame:
    duplicate_mask = frame.duplicated([source_col, target_col], keep=False)
    if not duplicate_mask.any():
        return frame

    selected_col = detect_column(
        frame, None, SELECTED_FLAG_CANDIDATES, "selected-model flag", required=False
    )
    if selected_col is not None:
        selected = frame.loc[truthy_mask(frame[selected_col])].copy()
        if not selected.duplicated([source_col, target_col]).any():
            return selected

    rank_col = detect_column(
        frame, None, SELECTION_RANK_CANDIDATES, "selection-rank", required=False
    )
    if rank_col is not None:
        rank = pd.to_numeric(frame[rank_col], errors="coerce")
        selected = frame.loc[rank.eq(1)].copy()
        if not selected.duplicated([source_col, target_col]).any():
            return selected

    example = (
        frame.loc[duplicate_mask, [source_col, target_col]]
        .value_counts()
        .head(5)
        .to_string()
    )
    raise ValueError(
        "More than one row remains per ordered source-target pair. Use the "
        "selected summary table, or provide a table with a detectable selected "
        f"flag/rank. Example duplicates:\n{example}"
    )


def prepare_directional_data(
    path: Path, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    frame = read_table(path)
    source_col = detect_column(
        frame, args.source_col, SOURCE_CANDIDATES, "source embedding"
    )
    target_col = detect_column(
        frame, args.target_col, TARGET_CANDIDATES, "target embedding"
    )
    score_col = detect_column(frame, args.score_col, SCORE_CANDIDATES, "test R^2")
    split_col = detect_column(
        frame, args.split_col, SPLIT_CANDIDATES, "split", required=False
    )
    input_mode_col = detect_column(
        frame,
        args.input_mode_col,
        INPUT_MODE_CANDIDATES,
        "input mode",
        required=False,
    )

    frame = filter_requested_value(frame, split_col, args.split, "split")
    frame = filter_requested_value(
        frame, input_mode_col, args.input_mode, "input mode"
    )
    frame = select_one_row_per_direction(frame, source_col, target_col)

    data = frame[[source_col, target_col, score_col]].copy()
    data.columns = ["source_embedding", "target_embedding", "test_r2"]
    data["source_embedding"] = data["source_embedding"].astype(str).str.strip()
    data["target_embedding"] = data["target_embedding"].astype(str).str.strip()
    data["test_r2"] = pd.to_numeric(data["test_r2"], errors="coerce")

    if data[["source_embedding", "target_embedding"]].eq("").any().any():
        raise ValueError("Blank source or target embedding names were found.")
    if data["test_r2"].isna().any() or not np.isfinite(data["test_r2"]).all():
        bad = int((data["test_r2"].isna() | ~np.isfinite(data["test_r2"])).sum())
        raise ValueError(f"Found {bad} non-finite test R^2 values.")
    if (data["source_embedding"] == data["target_embedding"]).any():
        raise ValueError("Self-prediction rows are present; expected only i -> j, i != j.")
    if data.duplicated(["source_embedding", "target_embedding"]).any():
        raise ValueError("Duplicate ordered source-target pairs remain after filtering.")

    columns = {
        "source": source_col,
        "target": target_col,
        "score": score_col,
        "split": split_col,
        "input_mode": input_mode_col,
    }
    return data, columns


def prepare_domain_data(
    path: Path, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = read_table(path)
    embedding_col = detect_column(
        frame, args.embedding_col, EMBEDDING_CANDIDATES, "domain-table embedding"
    )
    domain_col = detect_column(
        frame, args.domain_col, DOMAIN_CANDIDATES, "domain/modality"
    )
    data = frame[[embedding_col, domain_col]].copy()
    data.columns = ["embedding", "domain"]
    data["embedding"] = data["embedding"].astype(str).str.strip()
    data["domain"] = data["domain"].astype(str).str.strip()
    if data.eq("").any().any():
        raise ValueError("Blank embedding or domain values were found in the domain table.")
    if data["embedding"].duplicated().any():
        names = sorted(data.loc[data["embedding"].duplicated(False), "embedding"].unique())
        raise ValueError(f"Duplicate embeddings in domain table: {names}")
    return data, {"embedding": embedding_col, "domain": domain_col}


def validate_complete_directional_matrix(
    directional: pd.DataFrame,
    domains: pd.DataFrame,
    expected_embeddings: int,
) -> pd.DataFrame:
    source_set = set(directional["source_embedding"])
    target_set = set(directional["target_embedding"])
    if source_set != target_set:
        only_source = sorted(source_set - target_set)
        only_target = sorted(target_set - source_set)
        raise ValueError(
            "Source and target embedding sets differ. "
            f"Only as source: {only_source}; only as target: {only_target}"
        )
    embeddings = sorted(source_set)
    n_embeddings = len(embeddings)
    if expected_embeddings and n_embeddings != expected_embeddings:
        raise ValueError(
            f"Found {n_embeddings} embeddings, expected {expected_embeddings}. "
            "Use --expected-embeddings 0 only if this is intentional."
        )
    expected_rows = n_embeddings * (n_embeddings - 1)
    if len(directional) != expected_rows:
        observed_pairs = set(
            zip(directional["source_embedding"], directional["target_embedding"])
        )
        expected_pairs = {(a, b) for a in embeddings for b in embeddings if a != b}
        missing = sorted(expected_pairs - observed_pairs)[:10]
        extra = sorted(observed_pairs - expected_pairs)[:10]
        raise ValueError(
            f"Directional table has {len(directional):,} rows; a complete "
            f"{n_embeddings}-embedding matrix requires {expected_rows:,}. "
            f"First missing pairs: {missing}; first extras: {extra}"
        )

    mapped = set(domains["embedding"])
    missing_domains = sorted(source_set - mapped)
    if missing_domains:
        raise ValueError(f"Embeddings missing from domain table: {missing_domains}")
    unused_domains = sorted(mapped - source_set)
    if unused_domains:
        print(
            f"Warning: ignoring {len(unused_domains)} domain-table rows not present "
            "in the directional table.",
            file=sys.stderr,
        )
    domains = domains.loc[domains["embedding"].isin(source_set)].copy()
    if domains["domain"].nunique() < 2:
        raise ValueError("At least two modalities/domains are required.")
    return domains


def aggregate_values(series: pd.Series, method: str) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    if method == "median":
        return float(clean.median())
    if method == "mean":
        return float(clean.mean())
    raise ValueError(f"Unknown aggregation method: {method}")


def grouped_cells(
    directional: pd.DataFrame,
    group_columns: list[str],
    within_method: str,
) -> pd.DataFrame:
    grouped = directional.groupby(group_columns, sort=False, observed=True)["test_r2"]
    if within_method == "median":
        score = grouped.median()
    elif within_method == "mean":
        score = grouped.mean()
    else:
        raise ValueError(within_method)
    count = grouped.size()
    return pd.concat([score.rename("cell_score"), count.rename("n_pairs")], axis=1).reset_index()


def leave_one_embedding_out_baseline(
    cells: pd.DataFrame,
    group_columns: list[str],
    value_col: str = "cell_score",
) -> pd.Series:
    result = pd.Series(np.nan, index=cells.index, dtype=float)
    for _, group in cells.groupby(group_columns, sort=False, observed=True):
        values = group[value_col].to_numpy(dtype=float)
        indices = group.index.to_numpy()
        for position, index in enumerate(indices):
            others = np.delete(values, position)
            others = others[np.isfinite(others)]
            if len(others):
                result.loc[index] = float(np.median(others))
    return result


def calculate_scores(
    directional: pd.DataFrame,
    domains: pd.DataFrame,
    within_method: str,
    across_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain_by_embedding = domains.set_index("embedding")["domain"].to_dict()
    data = directional.copy()
    data["source_domain"] = data["source_embedding"].map(domain_by_embedding)
    data["target_domain"] = data["target_embedding"].map(domain_by_embedding)

    outgoing = grouped_cells(
        data, ["source_embedding", "source_domain", "target_domain"], within_method
    )
    outgoing["baseline"] = leave_one_embedding_out_baseline(
        outgoing, ["source_domain", "target_domain"]
    )
    outgoing["residual"] = outgoing["cell_score"] - outgoing["baseline"]
    outgoing["perspective"] = "outgoing_information"
    outgoing = outgoing.rename(
        columns={
            "source_embedding": "embedding",
            "source_domain": "embedding_domain",
            "target_domain": "comparison_domain",
        }
    )

    incoming = grouped_cells(
        data, ["target_embedding", "target_domain", "source_domain"], within_method
    )
    incoming["baseline"] = leave_one_embedding_out_baseline(
        incoming, ["target_domain", "source_domain"]
    )
    incoming["residual"] = incoming["cell_score"] - incoming["baseline"]
    incoming["perspective"] = "incoming_redundancy"
    incoming = incoming.rename(
        columns={
            "target_embedding": "embedding",
            "target_domain": "embedding_domain",
            "source_domain": "comparison_domain",
        }
    )

    cell_columns = [
        "perspective",
        "embedding",
        "embedding_domain",
        "comparison_domain",
        "cell_score",
        "n_pairs",
        "baseline",
        "residual",
    ]
    cells = pd.concat(
        [outgoing[cell_columns], incoming[cell_columns]], ignore_index=True
    )
    cells["included_in_cross_domain_score"] = (
        cells["comparison_domain"] != cells["embedding_domain"]
    )

    rows: list[dict[str, object]] = []
    for embedding in sorted(domain_by_embedding):
        own_domain = domain_by_embedding[embedding]
        out_all = outgoing.loc[outgoing["embedding"] == embedding]
        in_all = incoming.loc[incoming["embedding"] == embedding]
        out_cross = out_all.loc[out_all["comparison_domain"] != own_domain]
        in_cross = in_all.loc[in_all["comparison_domain"] != own_domain]

        pair_out = data.loc[data["source_embedding"] == embedding, "test_r2"]
        pair_in = data.loc[data["target_embedding"] == embedding, "test_r2"]
        rows.append(
            {
                "embedding": embedding,
                "domain": own_domain,
                "information_cross_domain": aggregate_values(
                    out_cross["cell_score"], across_method
                ),
                "redundancy_cross_domain": aggregate_values(
                    in_cross["cell_score"], across_method
                ),
                "information_all_domains": aggregate_values(
                    out_all["cell_score"], across_method
                ),
                "redundancy_all_domains": aggregate_values(
                    in_all["cell_score"], across_method
                ),
                "information_modality_residual": aggregate_values(
                    out_cross["residual"], across_method
                ),
                "redundancy_modality_residual": aggregate_values(
                    in_cross["residual"], across_method
                ),
                "raw_outgoing_pair_median_r2": float(pair_out.median()),
                "raw_incoming_pair_median_r2": float(pair_in.median()),
                "n_cross_target_modalities": int(out_cross["cell_score"].notna().sum()),
                "n_cross_source_modalities": int(in_cross["cell_score"].notna().sum()),
                "n_all_target_modalities": int(out_all["cell_score"].notna().sum()),
                "n_all_source_modalities": int(in_all["cell_score"].notna().sum()),
            }
        )
    scores = pd.DataFrame(rows)
    required = [
        "information_cross_domain",
        "redundancy_cross_domain",
        "information_all_domains",
        "redundancy_all_domains",
    ]
    if scores[required].isna().any().any():
        bad = scores.loc[scores[required].isna().any(axis=1), "embedding"].tolist()
        raise ValueError(f"Could not calculate complete scores for: {bad}")
    return scores, cells


def domain_palette(domains: Sequence[str]) -> dict[str, object]:
    palette: dict[str, object] = {}
    fallback = plt.get_cmap("tab20")
    for index, domain in enumerate(sorted(domains)):
        palette[domain] = PREFERRED_DOMAIN_COLORS.get(
            domain, fallback(index % fallback.N)
        )
    return palette


def selected_labels(
    data: pd.DataFrame, x_col: str, y_col: str, mode: str, count: int
) -> pd.DataFrame:
    if mode == "none":
        return data.iloc[0:0]
    if mode == "all":
        return data
    count = max(1, min(count, len(data)))
    indices: set[int] = set()
    for column in (x_col, y_col):
        indices.update(data.nlargest(count, column).index.tolist())
        indices.update(data.nsmallest(count, column).index.tolist())
    return data.loc[sorted(indices)]


def add_labels(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    mode: str,
    count: int,
) -> str:
    subset = selected_labels(data, x_col, y_col, mode, count)
    if subset.empty:
        return "none"
    texts = [
        ax.text(
            float(row[x_col]),
            float(row[y_col]),
            str(row.get("display_name", row["embedding"])),
            fontsize=5.4 if mode == "all" else 6.5,
            color="#263238",
            zorder=4,
        )
        for _, row in subset.iterrows()
    ]
    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            ax=ax,
            expand=(1.03, 1.12),
            force_text=(0.30, 0.45),
            arrowprops={"arrowstyle": "-", "color": "#A7ADB2", "lw": 0.35},
        )
        return f"{mode} (adjustText)"
    except ImportError:
        for index, text in enumerate(texts):
            x, y = text.get_position()
            offset = 4 + 2 * (index % 3)
            text.set_position((x, y))
            text.set_transform(ax.transData)
            text.set_ha("left" if index % 2 == 0 else "right")
            text.set_va("bottom" if index % 3 else "top")
            text.set_fontsize(5.0 if mode == "all" else 6.2)
            text.set_path_effects([])
            text.set_clip_on(False)
            # Offset is applied in display units by wrapping the existing text.
            text.set_transform(
                matplotlib.transforms.offset_copy(
                    ax.transData,
                    fig=ax.figure,
                    x=offset if index % 2 == 0 else -offset,
                    y=offset if index % 3 else -offset,
                    units="points",
                )
            )
        print(
            "Warning: adjustText is not installed; labels use fixed offsets and "
            "may overlap. Install it or use --label-mode extremes.",
            file=sys.stderr,
        )
        return f"{mode} (fixed-offset fallback)"


def square_limits(x: pd.Series, y: pd.Series, symmetric_zero: bool) -> tuple[float, float]:
    values = np.concatenate([x.to_numpy(float), y.to_numpy(float)])
    values = values[np.isfinite(values)]
    if symmetric_zero:
        bound = max(abs(float(values.min())), abs(float(values.max())))
        pad = 0.09 * bound if bound else 0.05
        return -bound - pad, bound + pad
    low = float(values.min())
    high = float(values.max())
    span = high - low
    pad = 0.08 * span if span else 0.05
    return low - pad, high + pad


def draw_scatter(
    scores: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    output_stem: Path,
    dpi: int,
    label_mode: str,
    extreme_label_count: int,
    zero_reference: bool = False,
) -> str:
    palette = domain_palette(scores["domain"].unique().tolist())
    fig, ax = plt.subplots(figsize=(12.4, 9.4))
    ax.set_facecolor("#FAFAF8")
    ax.grid(True, color="#D9DDDF", linewidth=0.55, alpha=0.8, zorder=0)

    for domain in sorted(scores["domain"].unique()):
        group = scores.loc[scores["domain"] == domain]
        ax.scatter(
            group[x_col],
            group[y_col],
            s=58,
            color=palette[domain],
            edgecolor="white",
            linewidth=0.75,
            alpha=0.93,
            label=f"{domain} (n={len(group)})",
            zorder=3,
        )

    lower, upper = square_limits(scores[x_col], scores[y_col], zero_reference)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle=(0, (2, 3)),
        color="#ABB1B5",
        linewidth=0.8,
        zorder=1,
    )
    label_status = add_labels(
        ax, scores, x_col, y_col, label_mode, extreme_label_count
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label, fontsize=11, labelpad=10)
    ax.set_ylabel(y_label, fontsize=11, labelpad=10)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)
    if subtitle:
        ax.set_title(subtitle, fontsize=9.4, color="#4F5961", pad=12)
    ax.legend(
        title="Embedding modality",
        bbox_to_anchor=(1.015, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=9.6,
        title_fontsize=10.4,
        borderaxespad=0,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#8B9297")
    ax.spines["bottom"].set_color("#8B9297")
    ax.tick_params(labelsize=8.5, colors="#3F474D")
    fig.subplots_adjust(left=0.11, right=0.74, bottom=0.11, top=0.91)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return label_status


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    direction_path = choose_direction_table(args).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    directional, directional_columns = prepare_directional_data(direction_path, args)
    manifest_path: Path | None = None
    manifest_hash: str | None = None
    aliases: dict[str, str] = {}
    if args.domain_source == "manifest":
        assert args.config is not None
        config_path = args.config.expanduser().resolve()
        config = yaml.safe_load(config_path.read_text())
        panel, manifest_path = load_manifest(config, config_path)
        manifest_hash = panel_sha256(panel)
        aliases = id_aliases(config)
        directional["source_embedding"] = directional.source_embedding.map(
            lambda value: canonicalize_id(value, aliases)
        )
        directional["target_embedding"] = directional.target_embedding.map(
            lambda value: canonicalize_id(value, aliases)
        )
        domains = panel.rename(columns={
            "embedding_id": "embedding",
            "modality": "domain",
        })[["embedding", "display_name", "domain"]]
        domain_path = manifest_path
        domain_columns = {
            "embedding": config.get("manifest_columns", {}).get(
                "embedding_id", "Files name"
            ),
            "display_name": config.get("manifest_columns", {}).get(
                "display_name", "Embedding display name"
            ),
            "domain": config.get("manifest_columns", {}).get(
                "modality", "Final modality"
            ),
        }
    elif args.domain_source == "final-mlp":
        if args.domain_table is not None:
            print(
                "Note: --domain-table is ignored because --domain-source "
                "defaults to final-mlp. Use --domain-source table to override.",
                file=sys.stderr,
            )
        domains = final_mlp_domain_data()
        domain_path: Path | None = None
        domain_columns = {
            "embedding": "built-in final MLP mapping",
            "domain": "built-in final MLP mapping",
        }
    else:
        assert args.domain_table is not None
        domain_path = args.domain_table.resolve()
        domains, domain_columns = prepare_domain_data(domain_path, args)
    domains = validate_complete_directional_matrix(
        directional,
        domains,
        args.expected_embeddings or (len(domains) if args.domain_source == "manifest" else 0),
    )
    scores, cells = calculate_scores(
        directional,
        domains,
        within_method=args.within_modality,
        across_method=args.across_modalities,
    )
    if "display_name" in domains.columns:
        display = domains.set_index("embedding")["display_name"].to_dict()
        scores["display_name"] = scores.embedding.map(display)
        cells["display_name"] = cells.embedding.map(display)

    scores = scores.sort_values(
        ["information_cross_domain", "embedding"], ascending=[False, True]
    ).reset_index(drop=True)
    scores.to_csv(output_dir / "information_redundancy_scores.tsv", sep="\t", index=False)
    domains.sort_values("embedding").to_csv(
        output_dir / "information_redundancy_domain_mapping_used.tsv",
        sep="\t",
        index=False,
    )
    cells.sort_values(
        ["perspective", "embedding", "comparison_domain"]
    ).to_csv(output_dir / "information_redundancy_domain_cells.tsv", sep="\t", index=False)

    label_status: dict[str, str] = {}
    label_status["primary"] = draw_scatter(
        scores,
        x_col="redundancy_cross_domain",
        y_col="information_cross_domain",
        title="Information contained versus redundancy",
        subtitle="",
        x_label="Predictability",
        y_label="Predictive capacity",
        output_stem=output_dir / "information_redundancy_cross_domain",
        dpi=args.dpi,
        label_mode=args.label_mode,
        extreme_label_count=args.extreme_label_count,
    )
    label_status["all_domains"] = draw_scatter(
        scores,
        x_col="redundancy_all_domains",
        y_col="information_all_domains",
        title="Information contained versus redundancy",
        subtitle="",
        x_label="Predictability",
        y_label="Predictive capacity",
        output_stem=output_dir / "information_redundancy_all_domains_sensitivity",
        dpi=args.dpi,
        label_mode=args.label_mode,
        extreme_label_count=args.extreme_label_count,
    )
    domain_counts = (
        domains["domain"].value_counts().sort_index().astype(int).to_dict()
    )
    audit: dict[str, object] = {
        "direction_table": str(direction_path),
        "domain_source": args.domain_source,
        "domain_table": str(domain_path) if domain_path is not None else None,
        "manifest_file": str(manifest_path) if manifest_path is not None else None,
        "panel_sha256": manifest_hash,
        "domain_mapping_output": str(
            output_dir / "information_redundancy_domain_mapping_used.tsv"
        ),
        "final_mlp_mapping_reference": (
            "02_random_mlp_r2_by_target_domain_heatmap (2)(2).pdf"
            if args.domain_source == "final-mlp"
            else None
        ),
        "output_dir": str(output_dir),
        "detected_direction_columns": directional_columns,
        "detected_domain_columns": domain_columns,
        "requested_split": args.split,
        "requested_input_mode": args.input_mode,
        "n_embeddings": int(domains["embedding"].nunique()),
        "n_modalities": int(domains["domain"].nunique()),
        "n_directional_rows": int(len(directional)),
        "domain_counts": domain_counts,
        "within_modality_aggregation": args.within_modality,
        "across_modality_aggregation": args.across_modalities,
        "primary_definition": {
            "information": (
                "For each embedding, aggregate R^2(source embedding -> targets) "
                "within each other target modality, then aggregate those modality "
                "cells with equal weights."
            ),
            "redundancy": (
                "For each embedding, aggregate R^2(sources -> target embedding) "
                "within each other source modality, then aggregate those modality "
                "cells with equal weights."
            ),
            "same_modality_excluded": True,
            "negative_r2_clipped": False,
        },
        "residual_definition": (
            "Each embedding-by-comparison-modality cell minus the leave-one-"
            "embedding-out median cell for the same source-modality x target-"
            "modality pair; cross-modality residual cells are then equally "
            "aggregated."
        ),
        "label_rendering": label_status,
        "median_primary_information": float(scores["information_cross_domain"].median()),
        "median_primary_redundancy": float(scores["redundancy_cross_domain"].median()),
    }
    with (output_dir / "information_redundancy_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Direction table: {direction_path}")
    if domain_path is None:
        print("Domain mapping:  built-in final MLP heatmap categories")
    else:
        print(f"Domain table:    {domain_path}")
    print(f"Embeddings:      {len(domains):,}")
    print(f"Directional rows:{len(directional):,}")
    print("Domain counts:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count}")
    print(f"Outputs written to: {output_dir}")
    return audit


def run_self_test(args: argparse.Namespace) -> None:
    counts = {
        "Protein sequence": 22,
        "Biomedical text": 8,
        "Integrated / multimodal": 8,
        "Transcriptomics / cell state": 8,
        "DNA / RNA sequence": 6,
        "Regulatory genomics": 5,
        "Knowledge graph context": 3,
        "Networks / ontologies": 3,
        "Functional / phenotypic": 2,
    }
    rng = np.random.default_rng(20260814)
    rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, str]] = []
    embedding_domains: dict[str, str] = {}
    index = 0
    for domain, count in counts.items():
        for _ in range(count):
            name = f"synthetic_embedding_{index:02d}"
            embedding_domains[name] = domain
            domain_rows.append({"embedding": name, "domain": domain})
            index += 1
    names = sorted(embedding_domains)
    source_effect = dict(zip(names, rng.normal(0.0, 0.07, len(names))))
    target_effect = dict(zip(names, rng.normal(0.0, 0.06, len(names))))
    domain_effect = {
        domain: value
        for domain, value in zip(counts, rng.normal(0.0, 0.04, len(counts)))
    }
    for source in names:
        for target in names:
            if source == target:
                continue
            same = embedding_domains[source] == embedding_domains[target]
            score = (
                0.16
                + source_effect[source]
                + target_effect[target]
                + domain_effect[embedding_domains[target]]
                + (0.08 if same else 0.0)
                + rng.normal(0.0, 0.025)
            )
            rows.append(
                {
                    "source_embedding": source,
                    "target_embedding": target,
                    "split": "split_random",
                    "input_mode": "raw",
                    "selected_test_r2": score,
                }
            )

    with tempfile.TemporaryDirectory(prefix="information_redundancy_selftest_") as tmp:
        root = Path(tmp)
        direction_path = root / "selected_directional_scores.tsv"
        domain_path = root / "domain_mapping.tsv"
        output_dir = root / "plots"
        pd.DataFrame(rows).to_csv(direction_path, sep="\t", index=False)
        pd.DataFrame(domain_rows).to_csv(domain_path, sep="\t", index=False)
        test_args = argparse.Namespace(**vars(args))
        test_args.self_test = False
        test_args.direction_table = direction_path
        test_args.dp_root = None
        test_args.domain_source = "table"
        test_args.domain_table = domain_path
        test_args.output_dir = output_dir
        test_args.label_mode = "extremes"
        audit = run_analysis(test_args)
        expected = [
            "information_redundancy_scores.tsv",
            "information_redundancy_domain_mapping_used.tsv",
            "information_redundancy_domain_cells.tsv",
            "information_redundancy_audit.json",
            "information_redundancy_cross_domain.pdf",
            "information_redundancy_cross_domain.png",
            "information_redundancy_all_domains_sensitivity.pdf",
            "information_redundancy_all_domains_sensitivity.png",
        ]
        missing = [name for name in expected if not (output_dir / name).is_file()]
        if missing:
            raise AssertionError(f"Self-test outputs missing: {missing}")
        if audit["n_embeddings"] != 65 or audit["n_directional_rows"] != 4160:
            raise AssertionError("Self-test audit counts are incorrect.")
    print(
        "SELF-TEST PASSED: 65 embeddings, 4,160 directional rows, "
        "two plot families created."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            run_self_test(args)
        else:
            run_analysis(args)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
