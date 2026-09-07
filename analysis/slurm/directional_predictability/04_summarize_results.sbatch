#!/bin/bash
#SBATCH --job-name=dp3_summary
#SBATCH --output=logs/dp3_summary_%j.out
#SBATCH --error=logs/dp3_summary_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dp_env.sh"
cd "$PROJECT"
mkdir -p logs
python -u scripts/directional_predictability/dp_04_summarize_sweep.py \
  --config "$DP_CONFIG" \
  --primary-input-mode raw
