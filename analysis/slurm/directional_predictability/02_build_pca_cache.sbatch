#!/bin/bash
#SBATCH --job-name=dp3_pca
#SBATCH --output=logs/dp3_pca_%j.out
#SBATCH --error=logs/dp3_pca_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00

set -euo pipefail
SPLIT="${1:?usage: sbatch 02_build_pca_cache.sbatch split_family|split_random}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dp_env.sh"
cd "$PROJECT"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
python -u scripts/directional_predictability/dp_02_build_pca_cache.py \
  --config "$DP_CONFIG" \
  --split-column "$SPLIT"
