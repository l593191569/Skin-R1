#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build standardized evaluation JSON files for scripts/run_eval.sh.
#
# Prerequisites:
#   bash scripts/build_rl_dataset.sh     → RL_dataset_verl_test.json
#   cp ../data_construction/synonym_and_subtype2.json data/
#   OmniMedVQA download (for omnimedvqa):
#     huggingface-cli download foreverbeliever/OmniMedVQA \
#       --repo-type dataset --local-dir "$SKIN_R1_DATA_ROOT/OmniMedVQA"
#   DDx graph from data_construction (for ddx):
#     export SKIN_R1_DDX_GRAPH=/path/to/data/outputs/<run>/ddx_graph_merged.json
#
# Output: $SKIN_R1_DATA_ROOT/standardized_datasets/*_standardized.json
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=_data_root.sh
source "$SCRIPT_DIR/_data_root.sh"
skin_r1_set_data_root "$PROJECT_DIR"

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "SKIN_R1_DATA_ROOT: $SKIN_R1_DATA_ROOT"
echo "SKIN_R1_DDX_GRAPH:  ${SKIN_R1_DDX_GRAPH:-<not set; required for ddx>}"
echo

python -m src.organize_eval_datasets \
  --data-root "$SKIN_R1_DATA_ROOT" \
  ${SKIN_R1_DDX_GRAPH:+--ddx-graph "$SKIN_R1_DDX_GRAPH"} \
  "$@"

echo
echo "Done. Evaluation JSON written under:"
echo "  $SKIN_R1_DATA_ROOT/standardized_datasets/"
