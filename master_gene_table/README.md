# Master gene table

This directory contains the pipeline used to construct the versioned human master gene table for the Gene Embeddings resource.

The master table defines the protein-coding gene universe and provides the Ensembl, Entrez, UniProt, transcript, protein, HGNC and alternative identifiers used to standardize the individual embeddings.

## Pipeline

The scripts must be executed in numerical order.

### 1. Build the base table from GENCODE

`01_build_base_table_from_gencode.py` constructs the initial protein-coding gene table from GENCODE release 49 on GRCh38.

It selects one canonical transcript per gene using the following priority:

1. MANE Select
2. Ensembl canonical
3. APPRIS principal
4. Transcript support level 1
5. Longest exon-summed transcript

The GENCODE GTF is downloaded automatically unless a local GTF is supplied.

Run from the repository root:

    python master_gene_table/01_build_base_table_from_gencode.py \
      --out ~/metadata/master_gene_table_v1.csv

### 2. Enrich identifiers using MyGene.info

`02_enrich_ids_with_mygene.py` queries MyGene.info using Ensembl gene IDs and adds Entrez and UniProt identifiers.

    python master_gene_table/02_enrich_ids_with_mygene.py \
      --input ~/metadata/master_gene_table_v1.csv \
      --out ~/metadata/master_gene_table_v1_mygene_enriched.csv \
      --report ~/metadata/reports/master_gene_table_v1_mygene_enrichment_report.txt

### 3. Cross-check identifiers

`03_crosscheck_ids_with_hgnc_biomart.py` compares the identifier mappings against the HGNC complete set and Ensembl BioMart. It fills unambiguous missing identifiers and records conflicts and review flags.

    python master_gene_table/03_crosscheck_ids_with_hgnc_biomart.py \
      --input ~/metadata/master_gene_table_v1_mygene_enriched.csv \
      --out ~/metadata/master_gene_table_v1_crosschecked.csv \
      --report ~/metadata/reports/master_gene_table_v1_crosscheck_report.txt

### 4. Finalize the master table

`04_finalize_table_with_hgnc_ensembl.py` adds detailed HGNC annotations, Ensembl GRCh38 and GRCh37 transcript/protein identifiers, and consolidated mapping fields used by the embedding-processing scripts.

    python master_gene_table/04_finalize_table_with_hgnc_ensembl.py

The final output is:

    ~/metadata/master_gene_table_v1_1_enriched.csv

## Reference files

Step 4 expects these files under `~/metadata/reference/`:

| Local filename | Source | Version |
|---|---|---|
| `hgnc_complete_set.txt` | HGNC complete set | Retrieved 2026-07-03 |
| `Homo_sapiens.GRCh38.current.gtf.gz` | Ensembl GTF | Release 116, GRCh38 |
| `Homo_sapiens.GRCh37.87.gtf.gz` | Ensembl GTF | Release 87, GRCh37 |

The Ensembl files can be downloaded with:

    mkdir -p ~/metadata/reference

    curl -L \
      https://ftp.ensembl.org/pub/release-116/gtf/homo_sapiens/Homo_sapiens.GRCh38.116.gtf.gz \
      -o ~/metadata/reference/Homo_sapiens.GRCh38.current.gtf.gz

    curl -L \
      https://ftp.ensembl.org/pub/grch37/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.gtf.gz \
      -o ~/metadata/reference/Homo_sapiens.GRCh37.87.gtf.gz

The HGNC complete set is available from:

https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt

The HGNC URL is updated over time. The table used in this project was retrieved on 2026-07-03 and is identified by the checksum below.

## Reference checksums

SHA-256 checksums of the exact reference files used:

| File | SHA-256 |
|---|---|
| `hgnc_complete_set.txt` | `5a2ed2d1ec244d1ad8a096edc04839c83d635ec886f5bc4b0dd77c00964af697` |
| `Homo_sapiens.GRCh38.current.gtf.gz` | `ed992f0eac7197d9627bda618f8f831ba355c95bd5d0796af785387d462828b6` |
| `Homo_sapiens.GRCh37.87.gtf.gz` | `56bc520f5e9cbc14ab4d03b472355b572818729fa422b905ba62d58291590058` |

## Requirements

- Python 3.10 or later
- `pandas` for step 4
- Internet access for steps 1–3 unless the corresponding sources are already available locally

The large reference files and generated master tables are not stored in this GitHub repository.
