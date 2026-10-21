#!/usr/bin/env bash
#SBATCH --job-name=cs_neighbors
#SBATCH --array=0-55
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
CS_JOB_ROOT="${CS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Set CS_PROJECT_ROOT or submit from the project root}}"
source "${CS_JOB_ROOT}/slurm/consensus_graph/cs_env.sh"

bash "${CS_PROJECT_ROOT}/scripts/consensus_graph/run_consensus.sh" neighbors "${SLURM_ARRAY_TASK_ID}"
