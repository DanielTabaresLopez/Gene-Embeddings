#!/usr/bin/env bash
#SBATCH --job-name=dp32_information_plots
#SBATCH --output=logs/dp32_information_plots_%j.out
#SBATCH --error=logs/dp32_information_plots_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dp_env.sh"
cd "$PROJECT"
mkdir -p logs

ANALYSIS_ROOT="${DP_ANALYSIS_ROOT:-$(python -c 'import os, sys, yaml; print(os.path.expanduser(yaml.safe_load(open(sys.argv[1]))["train_output"]))' "$DP_CONFIG")}"
PAIR_METRICS="${DP_PAIR_METRICS:?Set DP_PAIR_METRICS to the PFI pair_results.tsv file}"
PLOT_BASE="${DP_PLOT_BASE:-$ANALYSIS_ROOT}"
RANDOM_PRIMARY_ROOT="$PLOT_BASE/plots_random_primary_manifest"
PLOT_ROOT="$PLOT_BASE/plots_information_redundancy_manifest"
PRESENTATION_ROOT="$PLOT_BASE/plots_information_predictability_presentation"

python -u scripts/directional_predictability/dp_05_random_primary_plots.py \
  --dp-root "$ANALYSIS_ROOT" \
  --dp-long-table "$ANALYSIS_ROOT/sweep_long.tsv" \
  --dp-selected-table "$ANALYSIS_ROOT/validation_selected_depths.tsv" \
  --pair-metrics "$PAIR_METRICS" \
  --pair-a-col embedding_x \
  --pair-b-col embedding_y \
  --cka-col unbiased_linear_cka_row_normalized \
  --nn-col local_overlap_raw_k50 \
  --config "$DP_CONFIG" \
  --output-dir "$RANDOM_PRIMARY_ROOT"

python -u scripts/directional_predictability/dp_06_information_redundancy_plots.py \
  --config "$DP_CONFIG" \
  --direction-table "$ANALYSIS_ROOT/validation_selected_depths.tsv" \
  --split split_random \
  --input-mode raw \
  --output-dir "$PLOT_ROOT"

python -u scripts/directional_predictability/dp_07_information_predictability_plots.py \
  --scores-table "$PLOT_ROOT/information_redundancy_scores.tsv" \
  --output-dir "$PRESENTATION_ROOT"
