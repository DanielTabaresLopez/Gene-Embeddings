#!/usr/bin/env python3
"""Manifest utilities shared by the directional-predictability workflow.

Stable repository folder IDs are used for all computation and joins.  Display
names and modalities are metadata applied only to exported tables and plots.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = {
    "embedding_id": "Files name",
    "display_name": "Embedding display name",
    "modality": "Final modality",
}


def resolve_path(value: str | Path, config_path: Path) -> Path:
    """Resolve ``~``, environment variables, and YAML-relative paths."""
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if expanded.is_absolute():
        return expanded
    return (config_path.resolve().parent / expanded).resolve()


def id_aliases(config: dict) -> dict[str, str]:
    aliases = {
        str(old).strip(): str(new).strip()
        for old, new in (config.get("embedding_id_aliases") or {}).items()
    }
    if any(not old or not new for old, new in aliases.items()):
        raise ValueError("embedding_id_aliases contains a blank ID")
    return aliases


def canonicalize_id(value: object, aliases: dict[str, str]) -> str:
    current = str(value).strip()
    seen: set[str] = set()
    while current in aliases:
        if current in seen:
            raise ValueError(f"cyclic embedding ID alias involving {current!r}")
        seen.add(current)
        current = aliases[current]
    return current


def panel_sha256(panel: pd.DataFrame) -> str:
    records = panel[["embedding_id", "display_name", "modality"]].to_dict("records")
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(config: dict, config_path: Path) -> tuple[pd.DataFrame, Path]:
    if not config.get("manifest_path"):
        raise ValueError("config must define manifest_path")
    path = resolve_path(config["manifest_path"], config_path)
    if not path.is_file():
        raise FileNotFoundError(f"embedding manifest not found: {path}")

    suffix = path.suffix.lower()
    sheet = config.get("manifest_sheet", 0)
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name=sheet)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"unsupported manifest format: {path.suffix}")

    columns = {**DEFAULT_COLUMNS, **(config.get("manifest_columns") or {})}
    missing = [source for source in columns.values() if source not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing manifest columns {missing}")
    panel = frame[[columns[key] for key in ("embedding_id", "display_name", "modality")]].copy()
    panel.columns = ["embedding_id", "display_name", "modality"]
    for column in panel.columns:
        panel[column] = panel[column].astype("string").str.strip()
    if panel.isna().any().any() or panel.eq("").any().any():
        raise ValueError(f"{path}: manifest contains blank IDs, display names, or modalities")

    aliases = id_aliases(config)
    panel["embedding_id"] = panel.embedding_id.map(lambda value: canonicalize_id(value, aliases))
    for column in ("embedding_id", "display_name"):
        duplicated = panel.loc[panel[column].duplicated(False), column].tolist()
        if duplicated:
            raise ValueError(f"{path}: duplicate {column} values: {sorted(set(duplicated))}")

    expected_count = config.get("expected_embedding_count")
    if expected_count is not None and len(panel) != int(expected_count):
        raise ValueError(
            f"{path}: expected {int(expected_count)} embeddings, found {len(panel)}"
        )
    observed_hash = panel_sha256(panel)
    expected_hash = config.get("expected_panel_sha256")
    if expected_hash and observed_hash != str(expected_hash):
        raise ValueError(
            "normalized manifest hash differs from expected_panel_sha256: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    return panel.reset_index(drop=True), path


def manifest_maps(panel: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    display = dict(zip(panel.embedding_id, panel.display_name))
    modality = dict(zip(panel.embedding_id, panel.modality))
    return display, modality


def add_direction_labels(frame: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add paper-facing labels while retaining stable ``source``/``target`` IDs."""
    display, modality = manifest_maps(panel)
    result = frame.copy()
    result["source_display"] = result.source.map(display)
    result["target_display"] = result.target.map(display)
    result["source_modality"] = result.source.map(modality)
    result["target_modality"] = result.target.map(modality)
    missing = result.loc[
        result[["source_display", "target_display"]].isna().any(axis=1),
        ["source", "target"],
    ]
    if not missing.empty:
        values = sorted(set(missing.to_numpy().ravel()))
        raise ValueError(f"directional table contains IDs absent from manifest: {values}")
    return result

