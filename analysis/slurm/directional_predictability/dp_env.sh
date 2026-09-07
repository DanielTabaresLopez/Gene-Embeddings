# Sourced by every directional-predictability Slurm job.
# Adjust the module line for the local cluster.
module load stack/2024-06 python/3.11.6 2>/dev/null || true

DP_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DP_DEFAULT_PROJECT="$(cd "${DP_SLURM_DIR}/../.." && pwd)"

export PROJECT="${DP_PROJECT_ROOT:-$DP_DEFAULT_PROJECT}"
export DP_VENV="${DP_VENV:-$PROJECT/.venv}"
export DP_CONFIG="${DP_CONFIG:-$PROJECT/config/directional_predictability_v3.yaml}"
source "$DP_VENV/bin/activate"
