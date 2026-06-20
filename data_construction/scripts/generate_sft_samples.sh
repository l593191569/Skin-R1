#!/usr/bin/env bash
# Stage 5 of data construction: generate SkinRationale — hierarchy-aware / DDx
# reasoning trajectories (train_type{1..5}.jsonl) from the refined data,
# synonym/subtype config, taxonomy tree and merged DDx graph.
#
# Run from data_construction/. Requires OPENAI_API_KEY (see .env) for the
# DDx-reasoning sample types. Adjust the input paths to your pipeline outputs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# training_sample_generator imports resolve_path_and_bm_only from model_training,
# which reads $SKIN_R1_DATA_ROOT/synonym_and_subtype2.json. Point it at this
# directory (which ships synonym_and_subtype2.json) unless already set.
export SKIN_R1_DATA_ROOT=${SKIN_R1_DATA_ROOT:-$(pwd)}
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export SKIN_R1_MODEL_TRAINING_ROOT="${SKIN_R1_MODEL_TRAINING_ROOT:-$REPO_ROOT/model_training}"
export PYTHONPATH="$SKIN_R1_MODEL_TRAINING_ROOT:${PYTHONPATH:-}"

# Unified data directory and RUN_DIR (data/outputs/<run> holds taxonomy_tree.json / ddx_graph_merged.json / refined_data.csv)
DATA_DIR=${SKIN_R1_DATA_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)/data"}
RUN_DIR=${RUN_DIR:-$DATA_DIR/outputs/run}

python training_sample_generator.py \
  --csv "${RUN_DIR}/refined_data.csv" \
  --synonyms synonym_and_subtype2.json \
  --taxonomy "${RUN_DIR}/taxonomy_tree.json" \
  --ddxgraph "${RUN_DIR}/ddx_graph_merged.json" \
  --output "$DATA_DIR/sft_dataset" \
  --format jsonl \
  --max_per_diag_type2 5 \
  --max_per_diag_type3 5 \
  --max_per_diag_type4 5 \
  --max_per_diag_type5 5 \
  --limit_type2_examples 1000 \
  --limit_type3_examples 1000 \
  --limit_type4_examples 1000 \
  --limit_type5_examples 1000 \
  --min_repeats_if_single 3 \
  --openai-model gpt-4.1-mini
