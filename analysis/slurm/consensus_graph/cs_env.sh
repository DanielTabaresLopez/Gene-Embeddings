#!/usr/bin/env bash
# Sourced by the job scripts using an absolute path, never Slurm's spool path.
# CS_PROJECT_ROOT is the repository's analysis/ directory, not its top level.
# On other clusters, adapt the module line to the locally available Python.
if type module >/dev/null 2>&1; then
    module load stack/2024-06 python/3.11.6
fi
CS_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CS_PROJECT_ROOT="${CS_PROJECT_ROOT:-$(cd "${CS_SLURM_DIR}/../.." && pwd)}"
export CS_VENV="${CS_VENV:-${CS_PROJECT_ROOT}/.venv}"
export CS_CONFIG="${CS_CONFIG:-${CS_PROJECT_ROOT}/config/consensu_graph.yaml}"
export CS_PYTHON_BIN="${CS_PYTHON_BIN:-${CS_VENV}/bin/python}"
if [[ ! -x "${CS_PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${CS_PYTHON_BIN}. Set CS_VENV or CS_PYTHON_BIN." >&2
    return 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
cd "${CS_PROJECT_ROOT}"
