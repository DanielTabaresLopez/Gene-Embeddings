# Gene Embeddings

This repository contains the processing scripts used to construct the standardized gene embeddings available in the Gene Embeddings Hugging Face resource.

The collection brings together human gene representations derived from multiple biological modalities, including protein sequence, DNA, RNA, gene expression, single-cell expression, biomedical literature, biological networks, knowledge graphs, ontologies, and functional annotations.

## Hugging Face resource

- Dataset: https://huggingface.co/datasets/embeddingsSSRF/gene_embeddings
- Interactive Space: public link to be added before publication

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

These generated files should not be committed to this GitHub repository.

## Reproducing an embedding

Each processing script should document:

1. The original source of the embedding or source data.
2. The required input files.
3. Gene identifier mapping and filtering.
4. Pooling, aggregation, or dimensionality-reduction steps.
5. The command used to generate the standardized package.

## Contributing

New processing pipelines should be added under:

    embeddings/<embedding_id>/

The directory name must match the identifier used in the Hugging Face dataset.

## Citation

Citation information for the Gene Embeddings resource will be added following publication.

## Licensing

The code in this repository documents processing performed by the Gene Embeddings project. Original embeddings, models, and source datasets remain subject to their respective licenses and terms of use.
