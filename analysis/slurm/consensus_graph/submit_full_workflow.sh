#!/usr/bin/env bash
# Submit the entire publication workflow. Use --frozen-panel to begin from
# the included, hash-checked 56-embedding/14,039-gene panel instead of stage 1.
# Example from analysis/:
#   export CS_VENV="$HOME/envs/platonic_gene_repr"
#   export CS_LIBRARY_ROOT="/path/to/Gene-Embedding-Hub-dataset"
#   export CS_PFI_TABLE="/path/to/pair_results.tsv"
#   bash slurm/consensus_graph/submit_full_workflow.sh
set -euo pipefail
CS_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CS_PROJECT_ROOT="${CS_PROJECT_ROOT:-$(cd "${CS_SLURM_DIR}/../.." && pwd)}"
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash $0 [--frozen-panel]"
    echo "Set CS_VENV (or CS_PYTHON_BIN), CS_LIBRARY_ROOT and CS_PFI_TABLE."
    exit 0
fi
if (( $# > 1 )) || [[ -n "${1:-}" && "$1" != "--frozen-panel" ]]; then
    echo "Usage: bash $0 [--frozen-panel]" >&2
    exit 2
fi
source "${CS_SLURM_DIR}/cs_env.sh"
mkdir -p logs
bash "${CS_PROJECT_ROOT}/scripts/consensus_graph/run_consensus.sh" check-inputs
JOB_FILE="$(mktemp logs/consensus_submission_XXXXXXXX.tsv)"
printf 'stage\tjob_id\n' > "${JOB_FILE}"

submit() {
    local stage="$1" file="$2" dependency="${3:-}" job
    local -a options=(--parsable --export=ALL --kill-on-invalid-dep=yes)
    [[ -z "${dependency}" ]] || options+=(--dependency="afterok:${dependency}")
    job="$(sbatch "${options[@]}" "${CS_SLURM_DIR}/${file}")"
    job="${job%%;*}"
    [[ "${job}" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: ${job}" >&2; return 2; }
    printf '%s\t%s\n' "${stage}" "${job}" >> "${JOB_FILE}"
    printf '%s: %s\n' "${stage}" "${job}" >&2
    printf '%s\n' "${job}"
}

selection=""
if [[ "${1:-}" == "--frozen-panel" ]]; then
    export CS_SELECTED_PANEL="${CS_PROJECT_ROOT}/config/consensus_graph_frozen_panel"
else
    selection="$(submit selection 01_select_common_panel.sbatch)"
fi
frozen="$(submit freeze 02_freeze_selected_panel.sbatch "${selection}")"
prepared="$(submit prepare 03_prepare_embeddings.sbatch "${frozen}")"
neighbors="$(submit neighbors 04_build_neighbors.sbatch "${prepared}")"
final="$(submit pairs_and_finalize 05_compute_all_pairs_and_finalize.sbatch "${neighbors}")"
printf '%s\n' "${final}" > logs/all_pairs_and_finalize_job_id.txt
printf 'Submitted. Job IDs: %s\nResults: %s/results/consensus_graph_v2\n' "${JOB_FILE}" "${CS_PROJECT_ROOT}"
