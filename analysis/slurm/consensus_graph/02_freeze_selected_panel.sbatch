#!/usr/bin/env bash
#SBATCH --job-name=cs_freeze_panel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
CS_JOB_ROOT="${CS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Set CS_PROJECT_ROOT or submit from the project root}}"
source "${CS_JOB_ROOT}/slurm/consensus_graph/cs_env.sh"

# The full workflow freezes the panel just recomputed by stage 1. To reproduce
# only downstream stages, override CS_SELECTED_PANEL=config/consensus_graph_frozen_panel.
export CS_SELECTED_PANEL="${CS_SELECTED_PANEL:-${CS_PROJECT_ROOT}/results/consensus_graph_v2/00_panel_selection/candidate_panels/target_14000__alternative_01}"
bash "${CS_PROJECT_ROOT}/scripts/consensus_graph/run_consensus.sh" freeze
