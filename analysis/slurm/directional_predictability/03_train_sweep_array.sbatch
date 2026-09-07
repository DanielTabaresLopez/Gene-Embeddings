#!/bin/bash
#SBATCH --job-name=dp3_sweep
#SBATCH --output=logs/dp3_%x_%A_%a.out
#SBATCH --error=logs/dp3_%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00

set -euo pipefail
SPLIT="${1:?usage: sbatch --array=0-79 03_train_sweep_array.sbatch split_family|split_random}"
N_CHUNKS="${SLURM_ARRAY_TASK_COUNT:?set --array when submitting}"
CHUNK="${SLURM_ARRAY_TASK_ID:?}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dp_env.sh"
cd "$PROJECT"
mkdir -p logs

export DP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$DP_THREADS"
export MKL_NUM_THREADS="$DP_THREADS"
export OPENBLAS_NUM_THREADS="$DP_THREADS"

python -u scripts/directional_predictability/dp_03_train_sweep.py \
  --config "$DP_CONFIG" \
  --split-column "$SPLIT" \
  --input-mode raw \
  --n-chunks "$N_CHUNKS" \
  --chunk-index "$CHUNK"
