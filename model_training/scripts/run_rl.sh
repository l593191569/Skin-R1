#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stage 2: Reinforcement learning (GRPO) with a hierarchy-aware reward.
#          Starts from the SFT LoRA adapter and generalizes grounded
#          diagnostic reasoning to large, sparsely-labeled datasets.
#
# Entry point: model_training/src/train_rl_grpo.py
# Requires:    an SFT checkpoint produced by scripts/run_sft.sh
# Output:      $OUTPUT_DIR/lora_checkpoint/lora_step_{N}
#
# Edit the CONFIG block below, then run:
#     bash scripts/run_rl.sh
# (Or submit on a cluster with `sbatch scripts/run_rl.sh`.)
# ---------------------------------------------------------------------------
#SBATCH --job-name=skin_r1_rl
#SBATCH --output=logs/output/skin_r1_rl_%j.out
#SBATCH --error=logs/error/skin_r1_rl_%j.err
#SBATCH --partition=standard
#SBATCH --time=160:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2

set -euo pipefail

# ============================ CONFIG (edit me) =============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Base VLM (must match the base used during SFT).
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}

# SFT LoRA adapter to start RL from (output of run_sft.sh). REQUIRED.
# e.g. $PROJECT_DIR/output/SFT_trajectory_<TIMESTAMP>/module_checkpoint/module_epoch_4
SFT_CHECKPOINT=${SFT_CHECKPOINT:-$PROJECT_DIR/output/SFT_trajectory/module_checkpoint/module_epoch_4}

# Root that holds the prepared RL datasets (RL_dataset_prompt_format_4/, etc.).
# shellcheck source=_data_root.sh
source "$SCRIPT_DIR/_data_root.sh"
skin_r1_set_data_root "$PROJECT_DIR"

export SKIN_R1_CACHE_DIR=${SKIN_R1_CACHE_DIR:-$PROJECT_DIR/cache/huggingface}
export HF_HOME=$SKIN_R1_CACHE_DIR
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME

# Activate your python environment here, e.g.:
#   source /path/to/conda/etc/profile.d/conda.sh && conda activate skin_r1
# ===========================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PROJECT_DIR/logs/output" "$PROJECT_DIR/logs/error" "$SKIN_R1_CACHE_DIR"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

DATASET_SOURCE=${DATASET_SOURCE:-RL_dataset_verl_train}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/output/RL_openr1_${TIMESTAMP}}
export WANDB_PROJECT=${WANDB_PROJECT:-Skin-R1-RL}
mkdir -p "$OUTPUT_DIR"

echo "Model:      $MODEL_NAME_OR_PATH"
echo "SFT ckpt:   $SFT_CHECKPOINT"
echo "Dataset:    $DATASET_SOURCE (root: $SKIN_R1_DATA_ROOT)"
echo "Output:     $OUTPUT_DIR"

python -m src.train_rl_grpo \
  --task train_RL \
  --task_type train_RL \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --cache_dir "$SKIN_R1_CACHE_DIR" \
  --dataset_source "$DATASET_SOURCE" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-5 \
  --logging_steps 1 \
  --max_seq_length 1024 \
  --num_generations 4 \
  --seed 42 \
  --lora_r 64 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --max_train_samples 100000 \
  --max_eval_samples 10 \
  --save_merger_every_n_epochs 1 \
  --timestamp "$TIMESTAMP" \
  --checkpoint_path "$SFT_CHECKPOINT" \
  --task_name "Skin-R1"

echo "Done. LoRA checkpoints under: $OUTPUT_DIR/lora_checkpoint/"
