#!/usr/bin/env bash
#SBATCH --job-name=cs_all_pairs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
CS_JOB_ROOT="${CS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the analysis directory or set CS_PROJECT_ROOT}}"
source "${CS_JOB_ROOT}/slurm/consensus_graph/cs_env.sh"
RUNNER="${CS_PROJECT_ROOT}/scripts/consensus_graph/run_consensus.sh"

# One allocation, eight pair processes with two numerical threads each.
# Set CS_PAIR_WORKERS / CS_THREADS_PER_PAIR to fit a different allocation.
bash "${RUNNER}" pairs
bash "${RUNNER}" aggregate
bash "${RUNNER}" downstream
