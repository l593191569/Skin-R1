#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluation: run the trained Skin-R1 model across dermatology benchmarks and
#             print an accuracy summary.
#
# Entry point: model_training/src/test_performance.py
# Datasets:    $SKIN_R1_DATA_ROOT/standardized_datasets/<name>_standardized.json
#
# Edit the CONFIG block below, then run:
#     bash scripts/run_eval.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=skin_r1_eval
#SBATCH --output=logs/output/skin_r1_eval_%j.out
#SBATCH --error=logs/error/skin_r1_eval_%j.err
#SBATCH --partition=standard
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

# ============================ CONFIG (edit me) =============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}

# Trained adapter to evaluate (RL output of run_rl.sh, or an SFT checkpoint).
# e.g. $PROJECT_DIR/output/RL_openr1_<TIMESTAMP>/lora_checkpoint/lora_step_1500
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$PROJECT_DIR/output/RL_openr1/lora_checkpoint/lora_step_1500}

BATCH_SIZE=${BATCH_SIZE:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

# Benchmarks to evaluate (each maps to $SKIN_R1_DATA_ROOT/standardized_datasets/<name>_standardized.json).
DATASETS=(${DATASETS:-indomain_b_or_m omnimedvqa indomain hierarchical ddx})

export SKIN_R1_CACHE_DIR=${SKIN_R1_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}
export HF_HOME=$SKIN_R1_CACHE_DIR
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME

# Evaluation benchmarks live under SKIN_R1_DATA_ROOT/standardized_datasets/ by default.
# shellcheck source=_data_root.sh
source "$SCRIPT_DIR/_data_root.sh"
skin_r1_set_data_root "$PROJECT_DIR"
DATASET_DIR=${DATASET_DIR:-$SKIN_R1_DATA_ROOT/standardized_datasets}

# Activate your python environment here, e.g.:
#   source /path/to/conda/etc/profile.d/conda.sh && conda activate skin_r1
# ===========================================================================

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

declare -a FINAL_RESULTS=()
for datasource in "${DATASETS[@]}"; do
    echo "================================================================"
    echo "### Evaluating on: $datasource"
    echo "================================================================"
    comment="skin_r1_${datasource}"
    set +e
    output=$(python src/test_performance.py \
        --batch_size "$BATCH_SIZE" \
        --dataset_path "$DATASET_DIR/${datasource}_standardized.json" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --comment "$comment" \
        --model_name_or_path "$BASE_MODEL" \
        --checkpoint_path "$CHECKPOINT_PATH" 2>&1 | tee /dev/stderr)
    status=$?
    set -e
    if [ $status -eq 0 ]; then
        accuracy=$(echo "$output" | grep -oP 'Test completed with accuracy: \K[0-9.]+' || echo "Parse_Failed")
    else
        accuracy="FAILED"
    fi
    FINAL_RESULTS+=("${comment},${accuracy}")
done

echo "################################################################"
echo "###                  Final Accuracy Summary                  ###"
echo "################################################################"
( echo "Test_Name,Accuracy"; printf "%s\n" "${FINAL_RESULTS[@]}" ) | column -t -s ','
