#!/usr/bin/env bash
# From the repository's analysis/ directory:
#   python -m pip install -r scripts/consensus_graph/requirements.txt
#   bash scripts/consensus_graph/run_consensus.sh check
#   bash scripts/consensus_graph/run_consensus.sh download  # if inputs are absent
#   bash slurm/consensus_graph/submit_full_workflow.sh
# CS_LIBRARY_ROOT: pinned dataset checkout (or its data/embeddings directory).
# CS_PFI_TABLE: completed 67-embedding pair_results.tsv from the PFI analysis.
# Results: analysis/results/consensus_graph_v2/.
set -euo pipefail
CS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${CS_PYTHON_BIN:-python}" "${CS_SCRIPT_DIR}/cs_pipeline.py" "$@"
