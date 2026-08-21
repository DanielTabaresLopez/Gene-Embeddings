#!/usr/bin/env python3
"""
Enrich a master gene table with Entrez and UniProt identifiers using MyGene.info.

Input:  master_gene_table_v1.csv
Output: master_gene_table_v1_mygene_enriched.csv

This script:
- uses Ensembl gene IDs as the query key
- fills/updates entrez_id and uniprot_id columns
- adds mapping status/source columns
- writes a small mapping report

Run on a machine with internet access:
    python 02_enrich_ids_with_mygene.py --input master_gene_table_v1.csv --out master_gene_table_v1_mygene_enriched.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

MYGENE_URL = "https://mygene.info/v3/query"


def chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def normalize_to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for v in value:
            out.extend(normalize_to_list(v))
        return out
    if isinstance(value, dict):
        out: List[str] = []
        for v in value.values():
            out.extend(normalize_to_list(v))
        return out
    return [str(value)]


def unique_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        v = str(v).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def extract_uniprot(hit: Dict[str, Any]) -> Tuple[str, str]:
    """Return (pipe-separated UniProt IDs, mapping_status). Prefer reviewed Swiss-Prot IDs."""
    uni = hit.get("uniprot")
    if not uni:
        return "", "missing"

    reviewed: List[str] = []
    unreviewed: List[str] = []
    other: List[str] = []

    if isinstance(uni, dict):
        reviewed = normalize_to_list(uni.get("Swiss-Prot"))
        unreviewed = normalize_to_list(uni.get("TrEMBL"))
        # Keep anything else just in case MyGene changes field names.
        for key, val in uni.items():
            if key not in {"Swiss-Prot", "TrEMBL"}:
                other.extend(normalize_to_list(val))
    else:
        other = normalize_to_list(uni)

    if reviewed:
        return "|".join(unique_keep_order(reviewed)), "mapped_reviewed_swissprot"
    if unreviewed:
        return "|".join(unique_keep_order(unreviewed)), "mapped_unreviewed_trembl"
    if other:
        return "|".join(unique_keep_order(other)), "mapped_other"
    return "", "missing"


def hit_matches_query(hit: Dict[str, Any], query_id: str) -> bool:
    ens = hit.get("ensembl")
    if not ens:
        return False
    entries = ens if isinstance(ens, list) else [ens]
    for entry in entries:
        if isinstance(entry, dict) and entry.get("gene") == query_id:
            return True
    return False


def query_mygene(ensembl_ids: List[str], batch_size: int = 1000, sleep_s: float = 0.2) -> Dict[str, Dict[str, str]]:
    """Batch-query MyGene.info and return mapping by Ensembl gene ID."""
    results: Dict[str, Dict[str, str]] = {}

    total = len(ensembl_ids)
    for batch_idx, batch in enumerate(chunks(ensembl_ids, batch_size), start=1):
        data = urllib.parse.urlencode({
            "q": ",".join(batch),
            "scopes": "ensembl.gene",
            "fields": "entrezgene,uniprot,ensembl.gene,symbol,name,taxid",
            "species": "human",
            "returnall": "false",
            "size": "1",
        }).encode("utf-8")
        req = urllib.request.Request(
            MYGENE_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"ERROR: MyGene query failed for batch {batch_idx}: {exc}", file=sys.stderr)
            raise

        for item in payload:
            query_id = item.get("query", "")
            if item.get("notfound"):
                results[query_id] = {
                    "entrez_id": "",
                    "uniprot_id": "",
                    "id_mapping_status": "not_found",
                    "id_mapping_source": "mygene.info",
                }
                continue

            # MyGene normally returns one best hit because size=1. Still verify if possible.
            if not hit_matches_query(item, query_id):
                status_prefix = "mapped_without_exact_ensembl_confirmation"
            else:
                status_prefix = "mapped_exact_ensembl"

            entrez = item.get("entrezgene")
            entrez_str = str(entrez).strip() if entrez is not None else ""
            uniprot_str, uniprot_status = extract_uniprot(item)

            if not entrez_str and not uniprot_str:
                status = "found_but_missing_entrez_and_uniprot"
            elif not entrez_str:
                status = f"{status_prefix}; missing_entrez; {uniprot_status}"
            elif not uniprot_str:
                status = f"{status_prefix}; mapped_entrez; missing_uniprot"
            else:
                status = f"{status_prefix}; mapped_entrez; {uniprot_status}"

            results[query_id] = {
                "entrez_id": entrez_str,
                "uniprot_id": uniprot_str,
                "id_mapping_status": status,
                "id_mapping_source": "mygene.info",
            }

        done = min(batch_idx * batch_size, total)
        print(f"Queried {done}/{total} Ensembl gene IDs...")
        time.sleep(sleep_s)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="master_gene_table_v1.csv", help="Input master gene table CSV")
    parser.add_argument("--out", default="master_gene_table_v1_mygene_enriched.csv", help="Output enriched CSV")
    parser.add_argument("--report", default="master_gene_table_v1_mygene_enrichment_report.txt", help="Text report path")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    report_path = Path(args.report)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "ensembl_gene_id" not in fieldnames:
        raise ValueError("Input CSV must contain an 'ensembl_gene_id' column")

    # Ensure columns exist.
    for col in ["entrez_id", "uniprot_id", "id_mapping_status", "id_mapping_source"]:
        if col not in fieldnames:
            fieldnames.append(col)

    ensembl_ids = [row["ensembl_gene_id"].strip() for row in rows if row.get("ensembl_gene_id", "").strip()]
    if len(ensembl_ids) != len(set(ensembl_ids)):
        print("WARNING: duplicate Ensembl gene IDs detected in input", file=sys.stderr)

    mappings = query_mygene(ensembl_ids, batch_size=args.batch_size)

    counts = {
        "rows": len(rows),
        "mapped_entrez": 0,
        "mapped_uniprot": 0,
        "mapped_both": 0,
        "not_found": 0,
        "missing_entrez": 0,
        "missing_uniprot": 0,
    }

    for row in rows:
        gid = row.get("ensembl_gene_id", "").strip()
        m = mappings.get(gid, {
            "entrez_id": "",
            "uniprot_id": "",
            "id_mapping_status": "not_queried_or_no_result",
            "id_mapping_source": "mygene.info",
        })
        # Fill even if original cells exist; this is an enrichment script.
        row["entrez_id"] = m["entrez_id"]
        row["uniprot_id"] = m["uniprot_id"]
        row["id_mapping_status"] = m["id_mapping_status"]
        row["id_mapping_source"] = m["id_mapping_source"]

        has_entrez = bool(row["entrez_id"].strip())
        has_uniprot = bool(row["uniprot_id"].strip())
        if has_entrez:
            counts["mapped_entrez"] += 1
        else:
            counts["missing_entrez"] += 1
        if has_uniprot:
            counts["mapped_uniprot"] += 1
        else:
            counts["missing_uniprot"] += 1
        if has_entrez and has_uniprot:
            counts["mapped_both"] += 1
        if row["id_mapping_status"] == "not_found":
            counts["not_found"] += 1

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("Master gene table ID enrichment report\n")
        f.write("======================================\n\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Output: {out_path}\n")
        f.write("Mapping source: MyGene.info\n\n")
        for key, value in counts.items():
            f.write(f"{key}: {value}\n")

    print(f"Wrote enriched table: {out_path}")
    print(f"Wrote report: {report_path}")
    print(counts)


if __name__ == "__main__":
    main()
