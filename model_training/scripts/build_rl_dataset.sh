#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build RL GRPO training data from raw public dermatology datasets.
#
# Step 1: load_RL_data_raw.py   — load $SKIN_R1_DATA_ROOT/RL/, resolve taxonomy
# Step 2: RL_data_construct.py  — MCQ options, hierarchical scores, VERL prompts
#
# Prerequisites (user-provided, not in repo):
#   $SKIN_R1_DATA_ROOT/RL/<dataset>/...   raw public datasets (see README)
#   $SKIN_R1_DATA_ROOT/synonym_and_subtype2.json
#
# Output (consumed by scripts/run_rl.sh):
#   $SKIN_R1_DATA_ROOT/RL_dataset_prompt_format_4/RL_dataset_verl_train.json
#   ... valid.json, test.json
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=_data_root.sh
source "$SCRIPT_DIR/_data_root.sh"
skin_r1_set_data_root "$PROJECT_DIR"
export SKIN_R1_RL_PROMPT_FORMAT=${SKIN_R1_RL_PROMPT_FORMAT:-4}

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

FILTERED_JSON="${FILTERED_JSON:-$SKIN_R1_DATA_ROOT/RL_dataset_filtered.json}"

echo "SKIN_R1_DATA_ROOT:      $SKIN_R1_DATA_ROOT"
echo "SKIN_R1_RL_PROMPT_FORMAT: $SKIN_R1_RL_PROMPT_FORMAT"
echo

echo "=== Step 1/2: load raw RL datasets and filter ==="
python -m src.load_RL_data_raw --output-dir "$SKIN_R1_DATA_ROOT"

echo
echo "=== Step 2/2: build prompts, options, and scores ==="
python -m src.RL_data_construct \
  --input "$FILTERED_JSON" \
  --output-dir "$SKIN_R1_DATA_ROOT" \
  --prompt-type "$SKIN_R1_RL_PROMPT_FORMAT"

echo
echo "Done. RL training data under:"
echo "  $SKIN_R1_DATA_ROOT/RL_dataset_prompt_format_${SKIN_R1_RL_PROMPT_FORMAT}/"
echo "Run GRPO with: bash scripts/run_rl.sh"
