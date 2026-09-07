#!/usr/bin/env bash
#SBATCH --job-name=pfi_metrics_v2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PFI_CONFIG:-${ROOT}/config/pairwise_full_intersection_v2.yaml}"

if [[ -n "${PFI_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${PFI_CONDA_ENV}"
fi

export PFI_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PFI_PYTHON_BIN="${PFI_PYTHON_BIN:-python}"

cd "${ROOT}"
srun "${PFI_PYTHON_BIN}" \
  scripts/pairwise_full_intersection/pfi_02_run_pair_metrics.py \
  --config "${CONFIG}"
