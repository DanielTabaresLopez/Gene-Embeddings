# Gene Embeddings

This repository contains the processing scripts used to construct the standardized gene embeddings available in the Gene Embeddings Hugging Face resource.

The collection brings together human gene representations derived from multiple biological modalities, including protein sequence, DNA, RNA, gene expression, single-cell expression, biomedical literature, biological networks, knowledge graphs, ontologies, and functional annotations.

## Hugging Face resource

- Dataset: I have to add the link after changing SSRF organization to BoevaLab
- Interactive Space: I have to add the link after changing SSRF organization to BoevaLab

The processed embedding matrices are hosted on Hugging Face and are not duplicated in this GitHub repository.

## Repository structure

Processing scripts are organized under `embeddings/`.

Each subdirectory uses the same identifier as the corresponding Hugging Face package:

    embeddings/<embedding_id>/

A directory may contain one complete processing script or several numbered scripts when multiple processing steps are required.

## Standardized outputs

The processing pipelines generate the standardized files published on Hugging Face:

- `embeddings.npz`: embedding matrix stored under the key `embeddings`
- `genes.tsv`: standardized gene identifiers in the same row order as the matrix
- `metadata.json`: provenance and processing metadata
- `README.md`: embedding-level documentation
