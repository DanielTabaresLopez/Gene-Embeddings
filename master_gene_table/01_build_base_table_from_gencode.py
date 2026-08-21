#!/usr/bin/env python3
"""
Create a v1 master gene table for the gene-embedding project.

Default reference:
- Human GENCODE release 49, GRCh38
- Core universe: protein-coding genes
- Primary identifier: Ensembl gene ID without version
- Canonical transcript priority:
  1) MANE_Select
  2) Ensembl_canonical
  3) APPRIS principal tags
  4) transcript_support_level 1
  5) longest exon-summed transcript length

Run:
  python 01_build_base_table_from_gencode.py --out master_gene_table_v1.csv

Requires internet access unless --gtf is provided.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Tuple, Any

DEFAULT_RELEASE = "49"
DEFAULT_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    f"release_{DEFAULT_RELEASE}/gencode.v{DEFAULT_RELEASE}.annotation.gtf.gz"
)

ATTR_RE = re.compile(r'([A-Za-z0-9_]+) "([^"]*)";')


def strip_version(x: str | None) -> str:
    if not x:
        return ""
    return x.split(".")[0]


def parse_attrs(attr_text: str) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    tags = []
    for key, value in ATTR_RE.findall(attr_text):
        if key == "tag":
            tags.append(value)
        elif key in attrs:
            # Keep repeated non-tag fields just in case.
            if isinstance(attrs[key], list):
                attrs[key].append(value)
            else:
                attrs[key] = [attrs[key], value]
        else:
            attrs[key] = value
    attrs["tag"] = tags
    return attrs


def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def download(url: str) -> str:
    print(f"Downloading {url}", file=sys.stderr)
    tmp = NamedTemporaryFile(delete=False, suffix=".gtf.gz")
    tmp.close()
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


def transcript_priority(t: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Lower is better."""
    tags = set(t.get("tags", []))

    if "MANE_Select" in tags:
        tag_rank = 0
    elif "Ensembl_canonical" in tags:
        tag_rank = 1
    elif any(x.startswith("appris_principal") for x in tags):
        tag_rank = 2
    else:
        tag_rank = 3

    tsl = t.get("transcript_support_level", "")
    try:
        tsl_rank = int(str(tsl).split()[0])
    except Exception:
        tsl_rank = 99

    # Prefer longer transcripts after main biological priority criteria.
    length_rank = -int(t.get("exon_length", 0))

    # Stable deterministic tie-breaker.
    tx_id = t.get("transcript_id", "")
    tie = sum(ord(c) for c in tx_id) % 100000
    return (tag_rank, tsl_rank, length_rank, tie)


def selection_rule(t: Dict[str, Any]) -> str:
    tags = set(t.get("tags", []))
    if "MANE_Select" in tags:
        return "MANE_Select"
    if "Ensembl_canonical" in tags:
        return "Ensembl_canonical"
    appris = sorted(x for x in tags if x.startswith("appris_principal"))
    if appris:
        return appris[0]
    tsl = t.get("transcript_support_level", "")
    if str(tsl).startswith("1"):
        return "transcript_support_level_1"
    return "longest_exon_sum_fallback"


def build_table(gtf_path: str) -> list[dict[str, str]]:
    genes: Dict[str, Dict[str, Any]] = {}
    transcripts: Dict[str, Dict[str, Any]] = {}
    tx_by_gene: Dict[str, set[str]] = defaultdict(set)

    with open_maybe_gzip(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attr_text = fields
            attrs = parse_attrs(attr_text)
            gene_id_v = attrs.get("gene_id", "")
            gene_id = strip_version(gene_id_v)
            gene_type = attrs.get("gene_type", attrs.get("gene_biotype", ""))
            if not gene_id or gene_type != "protein_coding":
                continue

            if feature == "gene":
                genes[gene_id] = {
                    "ensembl_gene_id": gene_id,
                    "ensembl_gene_id_versioned": gene_id_v,
                    "gene_symbol": attrs.get("gene_name", ""),
                    "gene_type": gene_type,
                    "chromosome": chrom,
                    "gene_start": start,
                    "gene_end": end,
                    "strand": strand,
                    "gene_source": attrs.get("gene_source", source),
                    "havana_gene": attrs.get("havana_gene", ""),
                    "level": attrs.get("level", ""),
                }

            tx_id_v = attrs.get("transcript_id", "")
            tx_id = strip_version(tx_id_v)
            if tx_id:
                tx_by_gene[gene_id].add(tx_id)
                if tx_id not in transcripts:
                    transcripts[tx_id] = {
                        "gene_id": gene_id,
                        "transcript_id": tx_id,
                        "transcript_id_versioned": tx_id_v,
                        "transcript_name": attrs.get("transcript_name", ""),
                        "transcript_type": attrs.get("transcript_type", attrs.get("transcript_biotype", "")),
                        "transcript_support_level": attrs.get("transcript_support_level", ""),
                        "tags": attrs.get("tag", []),
                        "ccdsid": attrs.get("ccdsid", ""),
                        "havana_transcript": attrs.get("havana_transcript", ""),
                        "protein_id_versioned": "",
                        "protein_id": "",
                        "exon_length": 0,
                    }
                # Merge tags seen across features.
                transcripts[tx_id]["tags"] = sorted(set(transcripts[tx_id].get("tags", [])) | set(attrs.get("tag", [])))
                if attrs.get("protein_id") and not transcripts[tx_id].get("protein_id_versioned"):
                    transcripts[tx_id]["protein_id_versioned"] = attrs.get("protein_id", "")
                    transcripts[tx_id]["protein_id"] = strip_version(attrs.get("protein_id", ""))
                if attrs.get("ccdsid") and not transcripts[tx_id].get("ccdsid"):
                    transcripts[tx_id]["ccdsid"] = attrs.get("ccdsid", "")
                if feature == "exon":
                    transcripts[tx_id]["exon_length"] += int(end) - int(start) + 1

    rows = []
    for gene_id in sorted(genes):
        gene = genes[gene_id]
        txs = [transcripts[tx] for tx in tx_by_gene.get(gene_id, []) if tx in transcripts]
        txs = [t for t in txs if t.get("transcript_type") == "protein_coding"] or txs
        if txs:
            chosen = sorted(txs, key=transcript_priority)[0]
            rule = selection_rule(chosen)
            notes = ""
        else:
            chosen = {}
            rule = "no_transcript_found"
            notes = "No protein-coding transcript parsed for this protein-coding gene."

        rows.append({
            "ensembl_gene_id": gene.get("ensembl_gene_id", ""),
            "ensembl_gene_id_versioned": gene.get("ensembl_gene_id_versioned", ""),
            "gene_symbol": gene.get("gene_symbol", ""),
            "gene_type": gene.get("gene_type", ""),
            "chromosome": gene.get("chromosome", ""),
            "gene_start": gene.get("gene_start", ""),
            "gene_end": gene.get("gene_end", ""),
            "strand": gene.get("strand", ""),
            "canonical_transcript_id": chosen.get("transcript_id", ""),
            "canonical_transcript_id_versioned": chosen.get("transcript_id_versioned", ""),
            "canonical_transcript_name": chosen.get("transcript_name", ""),
            "canonical_transcript_type": chosen.get("transcript_type", ""),
            "canonical_protein_id": chosen.get("protein_id", ""),
            "canonical_protein_id_versioned": chosen.get("protein_id_versioned", ""),
            "ccdsid": chosen.get("ccdsid", ""),
            "canonical_selection_rule": rule,
            "canonical_transcript_tags": ";".join(chosen.get("tags", [])),
            "transcript_support_level": chosen.get("transcript_support_level", ""),
            "exon_length": str(chosen.get("exon_length", "")),
            "entrez_id": "",
            "uniprot_id": "",
            "havana_gene": gene.get("havana_gene", ""),
            "havana_transcript": chosen.get("havana_transcript", ""),
            "gencode_release": DEFAULT_RELEASE,
            "assembly": "GRCh38",
            "mapping_notes": notes,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gtf", help="Path to local GENCODE GTF/GTF.gz. If omitted, v49 is downloaded.")
    p.add_argument("--url", default=DEFAULT_GTF_URL, help="GENCODE GTF URL to download if --gtf is omitted.")
    p.add_argument("--out", default="master_gene_table_v1.csv")
    args = p.parse_args()

    gtf_path = args.gtf or download(args.url)
    rows = build_table(gtf_path)
    if not rows:
        raise SystemExit("No rows parsed. Check the GTF path/release.")

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} genes to {args.out}")
    counts = defaultdict(int)
    for r in rows:
        counts[r["canonical_selection_rule"]] += 1
    for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
