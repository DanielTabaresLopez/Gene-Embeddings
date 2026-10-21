#!/usr/bin/env bash
#SBATCH --job-name=cs_select_panel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
CS_JOB_ROOT="${CS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Set CS_PROJECT_ROOT or submit from the project root}}"
source "${CS_JOB_ROOT}/slurm/consensus_graph/cs_env.sh"

[[ -n "${CS_LIBRARY_ROOT:-}" ]] || {
  echo "Set CS_LIBRARY_ROOT to the Hugging Face dataset checkout." >&2
  exit 2
}

bash "${CS_PROJECT_ROOT}/scripts/consensus_graph/run_consensus.sh" select
