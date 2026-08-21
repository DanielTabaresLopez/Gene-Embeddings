#!/usr/bin/env python3
"""
Cross-check and fill Entrez / UniProt IDs in a master gene table without pandas.

Inputs:
  - master_gene_table_v1_mygene_enriched.csv
Outputs:
  - master_gene_table_v1_crosschecked.csv
  - master_gene_table_v1_crosscheck_report.txt

Uses only Python standard library. Downloads mappings from:
  - HGNC complete set
  - Ensembl BioMart
"""

import argparse
import csv
import io
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
BIOMART_URL = "https://www.ensembl.org/biomart/martservice"


def clean(value):
    if value is None:
        return ""
    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return value


def strip_ensembl_version(gene_id):
    gene_id = clean(gene_id)
    return gene_id.split(".")[0]


def split_ids(value):
    value = clean(value)
    if not value:
        return set()
    for sep in ["|", ",", " "]:
        value = value.replace(sep, ";")
    return {x.strip() for x in value.split(";") if x.strip()}


def join_ids(ids):
    ids = sorted({clean(x) for x in ids if clean(x)})
    return ";".join(ids)


def download_text(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def load_input_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError("Input CSV is empty or could not be read.")
    if "ensembl_gene_id" not in fieldnames:
        raise ValueError("Input CSV must contain column 'ensembl_gene_id'.")
    return rows, fieldnames


def fetch_hgnc_mappings():
    print("Downloading HGNC complete set...", file=sys.stderr)
    text = download_text(HGNC_URL)
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")

    entrez_by_gene = defaultdict(set)
    uniprot_by_gene = defaultdict(set)

    for row in reader:
        ens = strip_ensembl_version(row.get("ensembl_gene_id"))
        if not ens:
            continue
        entrez = clean(row.get("entrez_id"))
        if entrez:
            entrez_by_gene[ens].add(entrez)

        # HGNC often stores multiple UniProt accessions as pipe-separated.
        for uid in split_ids(row.get("uniprot_ids")):
            uniprot_by_gene[ens].add(uid)

    return entrez_by_gene, uniprot_by_gene


def biomart_query():
    query = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Filter name="biotype" value="protein_coding"/>
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="entrezgene_id"/>
    <Attribute name="uniprotswissprot"/>
    <Attribute name="uniprotsptrembl"/>
  </Dataset>
</Query>"""
    url = BIOMART_URL + "?query=" + urllib.parse.quote(query)
    return url


def fetch_biomart_mappings():
    print("Downloading Ensembl BioMart mappings...", file=sys.stderr)
    text = download_text(biomart_query())
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")

    entrez_by_gene = defaultdict(set)
    uniprot_by_gene = defaultdict(set)

    for row in reader:
        # BioMart headers can vary slightly; handle common variants robustly.
        ens = clean(row.get("Gene stable ID") or row.get("ensembl_gene_id") or row.get("Ensembl gene ID"))
        ens = strip_ensembl_version(ens)
        if not ens:
            continue

        entrez = clean(row.get("NCBI gene (formerly Entrezgene) ID") or row.get("entrezgene_id") or row.get("EntrezGene ID"))
        if entrez:
            entrez_by_gene[ens].add(entrez)

        swiss = clean(row.get("UniProtKB/Swiss-Prot ID") or row.get("uniprotswissprot") or row.get("UniProt/SwissProt ID"))
        trembl = clean(row.get("UniProtKB/TrEMBL ID") or row.get("uniprotsptrembl") or row.get("UniProt/TrEMBL ID"))
        for uid in split_ids(swiss) | split_ids(trembl):
            uniprot_by_gene[ens].add(uid)

    return entrez_by_gene, uniprot_by_gene


def status_for_entrez(current, hgnc_set, biomart_set):
    current_set = split_ids(current)
    external = set(hgnc_set) | set(biomart_set)

    if current_set:
        if not external:
            return current, "current_only_no_external_mapping", ""
        if current_set & external:
            return current, "agreement", ""
        return current, "conflict", "current=" + join_ids(current_set) + "; external=" + join_ids(external)

    # missing current value: fill only if unambiguous
    if len(external) == 1:
        return next(iter(external)), "filled_from_external", ""
    if len(external) > 1:
        return "", "missing_ambiguous_external", "external=" + join_ids(external)
    return "", "still_missing", ""


def status_for_uniprot(current, hgnc_set, biomart_set):
    current_set = split_ids(current)
    external = set(hgnc_set) | set(biomart_set)

    if current_set:
        if not external:
            return join_ids(current_set), "current_only_no_external_mapping", ""
        if current_set & external:
            # Keep the original current value, but normalize separators.
            return join_ids(current_set), "agreement", ""
        return join_ids(current_set), "conflict", "current=" + join_ids(current_set) + "; external=" + join_ids(external)

    # missing current value: for UniProt, multiple isoform/accession mappings can be valid.
    # We fill all external accessions if any are available.
    if external:
        return join_ids(external), "filled_from_external", ""
    return "", "still_missing", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input enriched master CSV")
    parser.add_argument("--out", required=True, help="Output crosschecked CSV")
    parser.add_argument("--report", required=True, help="Output text report")
    args = parser.parse_args()

    rows, fieldnames = load_input_csv(args.input)

    hgnc_entrez, hgnc_uniprot = fetch_hgnc_mappings()
    biomart_entrez, biomart_uniprot = fetch_biomart_mappings()

    new_fields = [
        "entrez_id_hgnc",
        "entrez_id_biomart",
        "entrez_agreement_status",
        "uniprot_id_hgnc",
        "uniprot_id_biomart",
        "uniprot_agreement_status",
        "mapping_review_flag",
        "mapping_notes_crosscheck",
    ]

    # Preserve existing columns and append new audit columns if not already present.
    out_fields = list(fieldnames)
    for col in new_fields:
        if col not in out_fields:
            out_fields.append(col)

    # Ensure these columns exist even if input somehow lacks them.
    for col in ["entrez_id", "uniprot_id"]:
        if col not in out_fields:
            out_fields.append(col)

    entrez_status_counts = Counter()
    uniprot_status_counts = Counter()
    review_counts = Counter()

    initial_missing_entrez = 0
    initial_missing_uniprot = 0
    final_missing_entrez = 0
    final_missing_uniprot = 0
    filled_entrez = 0
    filled_uniprot = 0

    for row in rows:
        ens = strip_ensembl_version(row.get("ensembl_gene_id"))
        row["ensembl_gene_id"] = ens

        current_entrez = clean(row.get("entrez_id"))
        current_uniprot = clean(row.get("uniprot_id"))
        if not current_entrez:
            initial_missing_entrez += 1
        if not current_uniprot:
            initial_missing_uniprot += 1

        h_entrez = hgnc_entrez.get(ens, set())
        b_entrez = biomart_entrez.get(ens, set())
        h_uniprot = hgnc_uniprot.get(ens, set())
        b_uniprot = biomart_uniprot.get(ens, set())

        row["entrez_id_hgnc"] = join_ids(h_entrez)
        row["entrez_id_biomart"] = join_ids(b_entrez)
        row["uniprot_id_hgnc"] = join_ids(h_uniprot)
        row["uniprot_id_biomart"] = join_ids(b_uniprot)

        new_entrez, e_status, e_note = status_for_entrez(current_entrez, h_entrez, b_entrez)
        new_uniprot, u_status, u_note = status_for_uniprot(current_uniprot, h_uniprot, b_uniprot)

        if not current_entrez and new_entrez:
            filled_entrez += 1
        if not current_uniprot and new_uniprot:
            filled_uniprot += 1

        row["entrez_id"] = new_entrez
        row["uniprot_id"] = new_uniprot
        row["entrez_agreement_status"] = e_status
        row["uniprot_agreement_status"] = u_status

        notes = []
        if e_note:
            notes.append("entrez: " + e_note)
        if u_note:
            notes.append("uniprot: " + u_note)
        row["mapping_notes_crosscheck"] = " | ".join(notes)

        flags = []
        if e_status == "conflict" or u_status == "conflict":
            flags.append("conflict")
        if e_status == "missing_ambiguous_external":
            flags.append("ambiguous_entrez")
        if clean(row.get("entrez_id")) == "":
            flags.append("missing_entrez")
        if clean(row.get("uniprot_id")) == "":
            flags.append("missing_uniprot")
        row["mapping_review_flag"] = ";".join(flags) if flags else "ok"

        entrez_status_counts[e_status] += 1
        uniprot_status_counts[u_status] += 1
        review_counts[row["mapping_review_flag"]] += 1

        if not clean(row.get("entrez_id")):
            final_missing_entrez += 1
        if not clean(row.get("uniprot_id")):
            final_missing_uniprot += 1

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    report_lines = []
    report_lines.append("Master gene table ID cross-check report")
    report_lines.append("=" * 48)
    report_lines.append(f"Input: {args.input}")
    report_lines.append(f"Output: {args.out}")
    report_lines.append(f"Rows: {len(rows)}")
    report_lines.append("")
    report_lines.append("Missing values")
    report_lines.append("--------------")
    report_lines.append(f"Initial missing Entrez: {initial_missing_entrez}")
    report_lines.append(f"Final missing Entrez:   {final_missing_entrez}")
    report_lines.append(f"Entrez filled:          {filled_entrez}")
    report_lines.append(f"Initial missing UniProt: {initial_missing_uniprot}")
    report_lines.append(f"Final missing UniProt:   {final_missing_uniprot}")
    report_lines.append(f"UniProt filled:          {filled_uniprot}")
    report_lines.append("")
    report_lines.append("Entrez status counts")
    report_lines.append("--------------------")
    for k, v in entrez_status_counts.most_common():
        report_lines.append(f"{k}: {v}")
    report_lines.append("")
    report_lines.append("UniProt status counts")
    report_lines.append("---------------------")
    for k, v in uniprot_status_counts.most_common():
        report_lines.append(f"{k}: {v}")
    report_lines.append("")
    report_lines.append("Review flag counts")
    report_lines.append("------------------")
    for k, v in review_counts.most_common():
        report_lines.append(f"{k}: {v}")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"Wrote: {args.out}")
    print(f"Wrote report: {args.report}")
    print({
        "rows": len(rows),
        "initial_missing_entrez": initial_missing_entrez,
        "final_missing_entrez": final_missing_entrez,
        "entrez_filled": filled_entrez,
        "initial_missing_uniprot": initial_missing_uniprot,
        "final_missing_uniprot": final_missing_uniprot,
        "uniprot_filled": filled_uniprot,
        "entrez_status": dict(entrez_status_counts),
        "uniprot_status": dict(uniprot_status_counts),
    })


if __name__ == "__main__":
    main()
