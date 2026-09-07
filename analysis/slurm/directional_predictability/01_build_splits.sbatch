#!/bin/bash
#SBATCH --job-name=dp3_splits
#SBATCH --output=logs/dp3_splits_%j.out
#SBATCH --error=logs/dp3_splits_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dp_env.sh"
cd "$PROJECT"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
python -u scripts/directional_predictability/dp_01_build_splits.py \
  --config "$DP_CONFIG"
