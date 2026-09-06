#!/usr/bin/env python3
"""Manifest and path utilities shared by the PFI scripts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


INTERNAL_COLUMNS = ("embedding_id", "display_name", "modality")
DEFAULT_COLUMNS = {
    "embedding_id": "Files name",
    "display_name": "Embedding display name",
    "modality": "Final modality",
}
SAFE_EMBEDDING_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | os.PathLike[str], config_path: Path) -> Path:
    """Resolve ~ and environment variables; relative paths use the YAML folder."""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return path.resolve()


def _read_table(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet, dtype=str)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", dtype=str)
    raise ValueError(
        f"unsupported manifest format {suffix!r}; use .xlsx, .csv, or .tsv"
    )


def load_manifest(config: dict[str, Any], config_path: Path) -> tuple[pd.DataFrame, Path]:
    """Read, normalize, and strictly validate the configured embedding manifest."""
    if not config.get("manifest_path"):
        raise KeyError("config must define manifest_path")
    path = resolve_path(config["manifest_path"], config_path)
    if not path.is_file():
        raise FileNotFoundError(f"embedding manifest not found: {path}")

    source = _read_table(path, sheet=config.get("manifest_sheet", 0))
    columns = dict(DEFAULT_COLUMNS)
    columns.update(config.get("manifest_columns") or {})
    missing_columns = [
        columns[key] for key in INTERNAL_COLUMNS if columns[key] not in source.columns
    ]
    if missing_columns:
        raise ValueError(
            f"manifest {path.name} is missing required columns: {missing_columns}; "
            f"found {list(source.columns)}"
        )

    panel = source[[columns[key] for key in INTERNAL_COLUMNS]].copy()
    panel.columns = list(INTERNAL_COLUMNS)
    for column in INTERNAL_COLUMNS:
        panel[column] = panel[column].fillna("").astype(str).str.strip()

    blank_rows = []
    for index, row in panel.iterrows():
        missing = [column for column in INTERNAL_COLUMNS if not row[column]]
        if missing:
            blank_rows.append(f"row {index + 2}: {', '.join(missing)}")
    if blank_rows:
        raise ValueError("manifest contains blank required cells: " + "; ".join(blank_rows))

    for column in ("embedding_id", "display_name"):
        duplicates = sorted(panel.loc[panel[column].duplicated(keep=False), column].unique())
        if duplicates:
            raise ValueError(f"manifest has duplicate {column} values: {duplicates}")

    unsafe = [
        value for value in panel.embedding_id
        if not SAFE_EMBEDDING_ID.fullmatch(value) or "__VS__" in value
    ]
    if unsafe:
        raise ValueError(
            "embedding IDs must contain only letters, numbers, '.', '_', or '-': "
            + ", ".join(unsafe)
        )

    expected = config.get("expected_embedding_count")
    if expected is not None and len(panel) != int(expected):
        raise ValueError(
            f"manifest contains {len(panel)} embeddings; expected_embedding_count={expected}"
        )
    if len(panel) < 2:
        raise ValueError("manifest must contain at least two embeddings")

    panel.insert(0, "manifest_order", range(1, len(panel) + 1))
    panel = panel.reset_index(drop=True)
    expected_hash = config.get("expected_panel_sha256")
    actual_hash = panel_sha256(panel)
    if expected_hash and actual_hash != str(expected_hash):
        raise ValueError(
            "normalized manifest content does not match expected_panel_sha256; "
            f"expected {expected_hash}, found {actual_hash}. If this change is "
            "intentional, review the full panel diff and update the frozen hash."
        )
    return panel, path


def panel_sha256(panel: pd.DataFrame) -> str:
    """Hash normalized semantic content, independent of Excel formatting."""
    records = panel[["manifest_order", *INTERNAL_COLUMNS]].to_dict(orient="records")
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_maps(panel: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    display = dict(zip(panel.embedding_id, panel.display_name))
    modality = dict(zip(panel.embedding_id, panel.modality))
    return display, modality


def annotate_pairs(frame: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add paper labels while preserving stable package IDs."""
    result = frame.copy()
    display, modality = manifest_maps(panel)
    for side in ("x", "y"):
        id_column = f"embedding_{side}"
        if id_column not in result.columns:
            continue
        result[f"embedding_{side}_display"] = result[id_column].map(display)
        result[f"modality_{side}"] = result[id_column].map(modality)
        missing = sorted(
            result.loc[result[f"embedding_{side}_display"].isna(), id_column].unique()
        )
        if missing:
            raise ValueError(f"results contain embedding IDs absent from the manifest: {missing}")
    return result


def write_normalized_manifest(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, sep="\t", index=False)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without Pandas' optional tabulate dependency."""
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)
