#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage 0 (part B): after manual cluster selection → finish data construction
# and bridge SkinRationale into model_training/data/trajectory_v2/.
#
# Usage:
#   # First invocation (after run_stage0_pre_cluster.sh + label review):
#   bash scripts/run_stage0_post_cluster.sh bbc_<timestamp> "02_01 02_03 01 00 03"
#
#   # After interactive run_filter.sh (refined_data.csv must exist):
#   bash scripts/run_stage0_post_cluster.sh --finish
#
# Environment:
#   RUN_DIR — set automatically after continue; or source data/outputs/.env_run
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR/.."
# shellcheck source=_run_dir_hint.sh
source "$SCRIPT_DIR/_run_dir_hint.sh"

DATA_DIR=${SKIN_R1_DATA_DIR:-"$REPO_ROOT/data"}

finish_only=false
if [[ "${1:-}" == "--finish" ]]; then
  finish_only=true
  shift
fi

run_finish_pipeline() {
  if [[ -z "${RUN_DIR:-}" ]]; then
    if [[ -f "$DATA_DIR/outputs/.env_run" ]]; then
      # shellcheck source=/dev/null
      source "$DATA_DIR/outputs/.env_run"
    fi
  fi
  if [[ -z "${RUN_DIR:-}" ]]; then
    echo "Error: RUN_DIR is not set. Source data/outputs/.env_run or export RUN_DIR."
    exit 1
  fi
  if [[ ! -f "$RUN_DIR/refined_data.csv" ]]; then
    echo "Error: $RUN_DIR/refined_data.csv not found."
    echo "Run the interactive curation first:"
    echo "  export RUN_DIR=\"$RUN_DIR\""
    echo "  bash scripts/run_filter.sh   # complete menu steps 1 and 2"
    echo "Then: bash scripts/run_stage0_post_cluster.sh --finish"
    exit 1
  fi

  export RUN_DIR
  echo "=== Stage 0 finish: taxonomy → SkinRationale → model_training layout ==="
  bash scripts/run_taxonomy_generate.sh
  bash scripts/generate_sft_samples.sh
  bash "$REPO_ROOT/model_training/scripts/prepare_sft_from_construction.sh"
}

if $finish_only; then
  run_finish_pipeline
  exit 0
fi

SOURCE_RUN=${1:-}
EXCLUDE_LABELS=${2:-}
if [[ -z "$SOURCE_RUN" || -z "$EXCLUDE_LABELS" ]]; then
  echo "Usage:"
  echo "  bash scripts/run_stage0_post_cluster.sh <bbc_run_name> \"<exclude labels>\""
  echo "  bash scripts/run_stage0_post_cluster.sh --finish"
  exit 1
fi

echo "=== Re-filter + match with chosen cluster labels ==="
bash scripts/process_bbc_pdf_continue.sh "$SOURCE_RUN" "$EXCLUDE_LABELS"
# shellcheck source=/dev/null
source "$DATA_DIR/outputs/.env_run"
export RUN_DIR

echo "=== LLM rephrase + DDx graph ==="
bash scripts/run_llm_diagnosis.sh
bash scripts/run_llm_ddx.sh

echo ""
echo "================================================================"
echo "  Manual step: interactive image curation (required)"
echo ""
echo "    export RUN_DIR=\"$RUN_DIR\""
echo "    bash scripts/run_filter.sh    # menu: 1 → curate → 2"
echo ""
echo "  Then finish Stage 0 and copy SkinRationale for SFT:"
echo "    bash scripts/run_stage0_post_cluster.sh --finish"
echo "================================================================"
